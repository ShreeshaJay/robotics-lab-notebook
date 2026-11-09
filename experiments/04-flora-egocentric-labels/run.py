#!/usr/bin/env python3
"""04 · Flora smoke test: timestamped captions + progress/outcome rewards on one public clip.

Runs on the VM after setup.sh. Uses only the standard library itself; the models run inside the
two venvs that setup.sh built. Everything heavy (weights, video, outputs) lives under $WORK,
never inside this notebook repo.

Stages (run in order; each can be re-run on its own):
  fetch     download the public EPIC-KITCHENS clip and make a Flora-compatible MP4 (<=120 s, <=256 MiB)
  prefetch  download all weights up front (~185 GB) so model timings exclude downloads
  validate  CPU-only checks: captioner --validate-only, rewarder --dry-run (downloads ~30 MB of processor files)
  caption   TempSeg -> Captioner pipeline (downloads ~130 GB on first run)
  reward    dense (process) reward every 0.5 s + outcome reward with progress (downloads ~56 GB)
  summary   collect a small results/summary.json for the write-up

Modes:
  --mode l4         (default) one 24 GB L4. The released 27B Captioner/Rewarder need ~55 GB in BF16, so
                    this mode loads those two in 4-bit (bitsandbytes NF4). That is NOT the reference
                    precision; treat outputs as "does it run / roughly sensible", not as quality numbers.
  --mode reference  unmodified upstream commands, for an 80 GB GPU (A100 80GB / H100).

Example:
  export FLORA_LICENSE_ACK=noncommercial-research-eval   # after reading Flora/LICENSE yourself
  python3 run.py all --mode l4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

WORK = Path(os.environ.get("WORK", str(Path.home() / "flora-lab"))).resolve()
FLORA = WORK / "Flora"
VENV_ANN = WORK / ".venv-annotation" / "bin" / "python"
VENV_REW = WORK / ".venv-rewarder" / "bin" / "python"
DATA = WORK / "data"
RESULTS = WORK / "results"
PATCHES = WORK / "patches"

# Public test clip: EPIC-KITCHENS-55 video P08_02 (107 s, 1920x1080 @ 59.94 fps, kitchen-cleaning
# actions: "clean counter top", "clean sink", "clean sponge"). Licence: non-commercial
# (CC BY-NC 4.0 per epic-kitchens.github.io; data.bris lists a Non-Commercial Government Licence).
# Direct download, no sign-up. 405,011,907 bytes.
CLIP_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P08/P08_02.MP4"
CLIP_BYTES = 405_011_907
CLIP_RAW = DATA / "raw" / "P08_02.MP4"
CLIP = DATA / "P08_02_720p.mp4"
TASK = "Clean the kitchen counter top and the sink."

REWARDER_REPO = "nvidia/Flora-Rewarder"
REWARDER_REV = "b64cee902fd6937184d9bea5afa043cd55650e45"  # HF main on 2026-11-07
ACK_VALUE = "noncommercial-research-eval"


def log(msg: str) -> None:
    print(f"[run.py {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env() -> dict[str, str]:
    e = dict(os.environ)
    e.setdefault("HF_HOME", str(WORK / "hf"))
    e.setdefault("FLORA_MODEL_CACHE", str(WORK / "flora-models"))
    e["FLORA_REWARDER_REPO"] = REWARDER_REPO
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    # vLLM 0.33.0 JIT-builds a FlashInfer sampler that needs ninja + a CUDA 13 nvcc (lesson from experiment 05).
    e.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Child Python output is block-buffered into the pipe otherwise, so log lines arrive late (seen in the first run).
    e.setdefault("PYTHONUNBUFFERED", "1")
    return e


def run(cmd: list[str], *, cwd: Path, logfile: Path, stamp: bool = False) -> int:
    """Run a command, tee output to a log file, sample GPU memory while it runs."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log("$ " + " ".join(cmd))
    peak = {"mib": 0}
    stop = threading.Event()

    trace = logfile.with_suffix(".gpu.csv")  # UTC time, MiB used: lets us split peak VRAM per model

    def sample() -> None:
        with trace.open("w") as tf:
            while not stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10,
                    ).stdout.split()
                    if out:
                        peak["mib"] = max(peak["mib"], int(out[0]))
                        tf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())},{out[0]}\n")
                        tf.flush()
                except Exception:
                    pass
                stop.wait(2.0)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    start = time.monotonic()
    with logfile.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            # stamp=True prefixes each log line with UTC time, so batch runs can be split per clip afterwards.
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ", time.gmtime()) + line if stamp else line)
            fh.flush()
        code = proc.wait()
    stop.set()
    t.join(timeout=5)
    meta = {"cmd": cmd, "exit_code": code, "elapsed_s": round(time.monotonic() - start, 1), "peak_gpu_mib": peak["mib"]}
    logfile.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    log(f"exit={code} elapsed={meta['elapsed_s']}s peak_gpu={peak['mib']} MiB -> {logfile}")
    return code


def require_ack() -> None:
    if os.environ.get("FLORA_LICENSE_ACK") != ACK_VALUE:
        sys.exit(
            "Flora code and weights are under the NVIDIA License: non-commercial research and evaluation only.\n"
            "Read Flora/LICENSE (and LICENSE.pdf in each model repo) yourself, then\n"
            f"  export FLORA_LICENSE_ACK={ACK_VALUE}\n"
            "and re-run. This script does not accept any terms on your behalf."
        )


# ----------------------------------------------------------------------------- stages

def stage_fetch(_: argparse.Namespace) -> None:
    CLIP_RAW.parent.mkdir(parents=True, exist_ok=True)
    if not CLIP_RAW.exists() or CLIP_RAW.stat().st_size != CLIP_BYTES:
        log(f"downloading {CLIP_URL} ({CLIP_BYTES / 1e6:.0f} MB)")
        tmp = CLIP_RAW.with_suffix(".part")
        with urllib.request.urlopen(CLIP_URL, timeout=60) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=8 << 20)
        if tmp.stat().st_size != CLIP_BYTES:
            sys.exit(f"unexpected size {tmp.stat().st_size}, expected {CLIP_BYTES}")
        tmp.rename(CLIP_RAW)
    # Flora annotation limits: <=120 s, <=256 MiB, <=4096 px per side, >=4 FPS, one video stream.
    # The raw file is 405 MB, so re-encode once: drop audio, scale to 720p, keep every native frame
    # timestamp (no fps change, no trim) so Flora's absolute times still line up with the source.
    if not CLIP.exists():
        cmd = [
            "ffmpeg", "-hide_banner", "-y", "-i", str(CLIP_RAW), "-map", "0:v:0", "-an",
            "-vf", "scale=1280:-2", "-fps_mode", "passthrough",
            "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(CLIP),
        ]
        if run(cmd, cwd=WORK, logfile=RESULTS / "logs" / "ffmpeg.log") != 0:
            sys.exit("ffmpeg failed")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets:format=duration,size",
         "-of", "json", str(CLIP)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(probe)
    (RESULTS / "clip_probe.json").write_text(json.dumps(info, indent=2))
    size_mib = int(info["format"]["size"]) / 2**20
    dur = float(info["format"]["duration"])
    log(f"clip ready: {CLIP} {dur:.1f}s {size_mib:.0f} MiB {info['streams'][0]}")
    if dur > 120 or size_mib > 256:
        sys.exit("clip exceeds Flora limits (120 s / 256 MiB)")
    (DATA / "ATTRIBUTION.txt").write_text(textwrap.dedent(f"""\
        P08_02 from EPIC-KITCHENS-55 (Damen et al., 2018), {CLIP_URL}
        DOI 10.5523/bris.3h91syskeag572hl6tvuovwv4d. Non-commercial licence. Re-encoded to 720p H.264, audio removed.
        """))


PREFETCH = '''\
"""Download all weights up front with Flora's own snapshot() helper and the same revisions/patterns,
so the timed model stages measure loading + inference, not downloads."""
import os, sys, time
sys.path.insert(0, os.path.join(sys.argv[1], "captioner"))
from huggingface_hub import snapshot_download
from batch_utils.config import MODELS
from batch_utils.models import snapshot
for kind in ("tempseg", "captioner"):
    t = time.monotonic(); p = snapshot(kind); print("PREFETCH", kind, p, round(time.monotonic() - t, 1), "s", flush=True)
b = MODELS["tempseg_base"]
t = time.monotonic()
p = snapshot_download(b["repo"], revision=b["revision"], allow_patterns=["*.safetensors", "*.json", "*.jinja"])
print("PREFETCH tempseg_base", p, round(time.monotonic() - t, 1), "s", flush=True)
t = time.monotonic(); p = snapshot_download(sys.argv[2], revision=sys.argv[3])
print("PREFETCH rewarder", p, round(time.monotonic() - t, 1), "s", flush=True)
'''


def stage_prefetch(_: argparse.Namespace) -> None:
    require_ack()
    PATCHES.mkdir(parents=True, exist_ok=True)
    (PATCHES / "prefetch.py").write_text(PREFETCH)
    code = run([str(VENV_ANN), "-I", str(PATCHES / "prefetch.py"), str(FLORA), REWARDER_REPO, REWARDER_REV],
               cwd=WORK, logfile=RESULTS / "logs" / "prefetch.log")
    if code != 0:
        sys.exit("prefetch failed (if this is a 401/403, the repos are gated: stop, do not sign in)")


def stage_validate(_: argparse.Namespace) -> None:
    code = run([str(VENV_ANN), "captioner/infer.py", "--video", str(CLIP), "--profile", "human", "--validate-only"],
               cwd=FLORA, logfile=RESULTS / "logs" / "validate_caption.log")
    if code != 0:
        sys.exit("captioner --validate-only failed")
    code = run([str(VENV_REW), "-m", "rewarder.batch_utils.dense_reward", "--model", REWARDER_REPO,
                "--revision", REWARDER_REV, "--video", str(CLIP), "--task", TASK,
                "--output", str(RESULTS / "dense_dryrun.jsonl"), "--overwrite", "--dry-run"],
               cwd=FLORA, logfile=RESULTS / "logs" / "validate_reward.log")
    if code != 0:
        sys.exit("rewarder --dry-run failed")


CAPTION_4BIT = '''\
"""Experimental: run the unmodified Flora annotation CLI, but load the 27B Captioner in 4-bit NF4.

Only the Captioner (snapshot dir named 'captioner') is quantized; TempSeg (9B + LoRA) loads as released.
GDN A_log/dt_bias are nn.Parameters, not nn.Linear, so they stay FP32 and Flora's own GDN check still runs.
"""
import sys
from pathlib import Path

FLORA = Path(sys.argv[1])
sys.path.insert(0, str(FLORA / "captioner"))

import torch
import transformers
from transformers import BitsAndBytesConfig

_orig = transformers.AutoModelForImageTextToText.from_pretrained

def _patched(path, *args, **kwargs):
    if Path(str(path)).name == "captioner":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head", "visual"],
        )
        kwargs["device_map"] = {"": 0}
        kwargs.pop("max_memory", None)
        print("FLORA_L4_EXPERIMENT captioner loaded in 4-bit NF4 (not reference precision)", flush=True)
    return _orig(path, *args, **kwargs)

transformers.AutoModelForImageTextToText.from_pretrained = _patched

from batch_utils.cli import main
raise SystemExit(main(sys.argv[2:]))
'''

REWARD_4BIT = '''\
"""Experimental: run the unmodified Flora rewarder scripts with vLLM in-flight bitsandbytes 4-bit."""
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from rewarder.batch_utils import dense_reward, engine, terminal_reward

_orig = engine.load_llm

def _quantized(*args, **kwargs):
    kwargs.setdefault("quantization", "bitsandbytes")
    print("FLORA_L4_EXPERIMENT rewarder loaded with vLLM bitsandbytes 4-bit (not reference precision)", flush=True)
    return _orig(*args, **kwargs)

dense_reward.load_llm = _quantized
terminal_reward.load_llm = _quantized
module = {"dense": dense_reward, "terminal": terminal_reward}[sys.argv[2]]
raise SystemExit(module.main(sys.argv[3:]))
'''


def write_patches() -> None:
    PATCHES.mkdir(parents=True, exist_ok=True)
    (PATCHES / "caption_4bit.py").write_text(CAPTION_4BIT)
    (PATCHES / "reward_4bit.py").write_text(REWARD_4BIT)


def stage_caption(a: argparse.Namespace) -> None:
    require_ack()
    out = RESULTS / f"captions_{a.mode}.json"
    args = ["--video", str(CLIP), "--profile", "human", "--output", str(out), "--overwrite"]
    if a.mode == "reference":
        cmd = [str(VENV_ANN), "captioner/infer.py", *args]
    else:
        write_patches()
        cmd = [str(VENV_ANN), "-I", str(PATCHES / "caption_4bit.py"), str(FLORA), *args]
    code = run(cmd, cwd=FLORA, logfile=RESULTS / "logs" / f"caption_{a.mode}.log")
    if code != 0:
        log("caption stage failed; see the log. On L4, an OOM in TempSeg (BF16 9B) or a refused "
            "Captioner load are the expected failure points and are findings in themselves.")


def stage_reward(a: argparse.Namespace) -> None:
    require_ack()
    common = ["--model", REWARDER_REPO, "--revision", REWARDER_REV, "--video", str(CLIP), "--task", TASK, "--overwrite"]
    jobs = {
        "dense": [*common, "--output", str(RESULTS / f"dense_{a.mode}.jsonl")],
        "terminal": [*common, "--with-rationale", "--output", str(RESULTS / f"terminal_{a.mode}.jsonl")],
    }
    for name, args in jobs.items():
        if a.mode == "reference":
            cmd = [str(VENV_REW), "-m", f"rewarder.batch_utils.{name}_reward", *args]
        else:
            write_patches()
            # One sequence at a time and a little more of the 24 GB for weights + KV cache.
            cmd = [str(VENV_REW), "-I", str(PATCHES / "reward_4bit.py"), str(FLORA), name, *args,
                   "--max-num-seqs", "1", "--gpu-memory-utilization", "0.92"]
        if run(cmd, cwd=FLORA, logfile=RESULTS / "logs" / f"reward_{name}_{a.mode}.log") != 0:
            log(f"{name} reward failed; see the log (vLLM bitsandbytes support for Qwen3.5 GDN layers is unverified).")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def stage_summary(a: argparse.Namespace) -> None:
    s: dict = {"mode": a.mode, "clip": str(CLIP.name), "task": TASK}
    cap = RESULTS / f"captions_{a.mode}.json"
    if cap.exists():
        d = json.loads(cap.read_text())
        segs = d.get("segments", [])
        s["captions"] = {
            "segments": len(segs),
            "complete": sum(x.get("status") == "complete" for x in segs),
            "first_10": [{k: x.get(k) for k in ("start_s", "end_s", "caption", "status")} for x in segs[:10]],
        }
    dense = read_jsonl(RESULTS / f"dense_{a.mode}.jsonl")
    if dense:
        r = dense[0]
        s["dense"] = {k: r.get(k) for k in ("mean_score", "worst_score", "fraction", "invalid_clips", "per_second")}
        s["dense"]["non_normal_segments"] = r.get("segments", [])[:10]
    term = read_jsonl(RESULTS / f"terminal_{a.mode}.jsonl")
    if term:
        s["terminal"] = {k: term[0].get(k) for k in ("score", "expected_score", "progress", "probabilities", "rationale")}
    s["runs"] = {}
    for meta in sorted((RESULTS / "logs").glob("*.meta.json")):
        m = json.loads(meta.read_text())
        s["runs"][meta.name.replace(".meta.json", "")] = {k: m[k] for k in ("exit_code", "elapsed_s", "peak_gpu_mib")}
    # Flora prints FLORA_RUNTIME {json} events, including peak_allocated_gib per generation.
    peaks = []
    for lg in (RESULTS / "logs").glob("caption_*.log"):
        for line in lg.read_text(errors="replace").splitlines():
            m = re.match(r"FLORA_RUNTIME (\{.*\})", line)
            if m:
                ev = json.loads(m.group(1))
                if "peak_allocated_gib" in ev:
                    peaks.append(ev["peak_allocated_gib"])
    if peaks:
        s["caption_peak_allocated_gib"] = max(peaks)
    out = RESULTS / f"summary_{a.mode}.json"
    out.write_text(json.dumps(s, indent=2))
    log(f"wrote {out}. Copy this JSON (not the video) into the notebook's results/ folder.")


# ----------------------------------------------------------------------------- several clips (--clips)
# One Flora process per stage for all clips: the annotation CLI's --input-jsonl and the rewarder's --jobs.
# Inside one annotation process Flora remembers which model manifests it has already SHA-256-verified
# (keyed on path, inode, size, mtime, ctime), so the 109 GB Captioner hash is paid once, not per clip.

MORE = RESULTS / "more_clips"


def load_clips(a: argparse.Namespace) -> dict:
    return json.loads(Path(a.clips).read_text())


def clip_path(cid: str) -> Path:
    return DATA / f"{cid}_720p.mp4"


ENCODE = ["-map", "0:v:0", "-an", "-vf", "scale=1280:-2", "-fps_mode", "passthrough", "-c:v", "libx264",
          "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def download(url: str, size: int, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    # Resume a partial download (a Spot VM with ON_END=STOP keeps the disk through a preemption).
    have = tmp.stat().st_size if tmp.exists() else 0
    if have > size:
        tmp.unlink()
        have = 0
    log(f"downloading {url} ({size / 1e6:.0f} MB, resuming at {have / 1e6:.0f} MB)")
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        with tmp.open("ab" if have else "wb") as f:
            shutil.copyfileobj(r, f, length=8 << 20)
    if tmp.stat().st_size != size:
        sys.exit(f"unexpected size {tmp.stat().st_size} for {url}, expected {size}")
    tmp.rename(dest)


def probe(p: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,avg_frame_rate,nb_read_packets:format=duration,size", "-of", "json",
                          str(p)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def stage_fetch_many(a: argparse.Namespace) -> None:
    m = load_clips(a)
    whole = [c for c in m["clips"] if c["start_s"] is None]
    # Whole videos: plain HTTPS downloads in parallel (data.bris was about 1 MB/s per stream in the first run).
    threads = [threading.Thread(target=download, args=(c["url"], c["bytes"], DATA / "raw" / Path(c["url"]).name))
               for c in whole]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    probes = {}
    for c in m["clips"]:
        out = clip_path(c["id"])
        if not out.exists():
            if c["start_s"] is None:
                src = ["-i", str(DATA / "raw" / Path(c["url"]).name)]
            else:
                # A window of a long video: ffmpeg reads only the needed byte ranges over HTTPS (the server
                # sends Accept-Ranges: bytes). Input-side -ss with re-encoding is frame-accurate; output starts at 0.
                src = ["-ss", f"{c['start_s']:.3f}", "-t", f"{c['dur_s']:.3f}", "-i", c["url"]]
            part = out.with_suffix(".part.mp4")  # rename only when complete, so a preempted encode is redone
            if run(["ffmpeg", "-hide_banner", "-y", *src, *ENCODE, str(part)], cwd=WORK,
                   logfile=MORE / "logs" / f"ffmpeg_{c['id']}.log") != 0:
                sys.exit(f"ffmpeg failed for {c['id']}")
            part.rename(out)
        probes[c["id"]] = probe(out)
    n = m["negative_control"]
    neg = clip_path(n["id"])
    if not neg.exists():
        part = neg.with_suffix(".part.mp4")
        if run(["ffmpeg", "-hide_banner", "-y", "-i", str(clip_path(n["from"])), "-t", f"{n['dur_s']:.3f}", *ENCODE,
                str(part)], cwd=WORK, logfile=MORE / "logs" / f"ffmpeg_{n['id']}.log") != 0:
            sys.exit("ffmpeg failed for the negative control")
        part.rename(neg)
    probes[n["id"]] = probe(neg)
    for cid, info in probes.items():
        dur, mib = float(info["format"]["duration"]), int(info["format"]["size"]) / 2**20
        log(f"{cid}: {dur:.1f}s {mib:.0f} MiB {info['streams'][0]}")
        if dur > 120 or mib > 256:
            sys.exit(f"{cid} exceeds Flora limits (120 s / 256 MiB)")
    (MORE / "clip_probe.json").write_text(json.dumps(probes, indent=2))


def stage_caption_many(a: argparse.Namespace) -> None:
    require_ack()
    m = load_clips(a)
    jobs = MORE / "caption_jobs.jsonl"
    jobs.write_text("".join(json.dumps({"id": c["id"], "video": str(clip_path(c["id"])), "profile": "human"}) + "\n"
                            for c in m["clips"]))
    cmd = [str(VENV_ANN), "captioner/infer.py", "--input-jsonl", str(jobs), "--output-dir", str(MORE / "captions"),
           "--overwrite"]
    if a.validate_only:
        cmd = [str(VENV_ANN), "captioner/infer.py", "--input-jsonl", str(jobs), "--validate-only"]
    code = run(cmd, cwd=FLORA, logfile=MORE / "logs" / ("validate_caption.log" if a.validate_only else "caption.log"),
               stamp=True)
    if code != 0:
        log("caption stage failed; see the log")


def stage_reward_many(a: argparse.Namespace) -> None:
    require_ack()
    m = load_clips(a)
    rows = [{"id": c["id"], "task": c["task"], "videos": [str(clip_path(c["id"]))]} for c in m["clips"]]
    n = m["negative_control"]
    (MORE / "dense_jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows.append({"id": n["id"], "task": n["task"], "videos": [str(clip_path(n["id"]))]})
    (MORE / "terminal_jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    common = ["--model", REWARDER_REPO, "--revision", REWARDER_REV, "--overwrite"]
    jobs = {
        "dense": ["--jobs", str(MORE / "dense_jobs.jsonl"), *common, "--output", str(MORE / "dense.jsonl")],
        "terminal": ["--jobs", str(MORE / "terminal_jobs.jsonl"), *common, "--with-rationale",
                     "--output", str(MORE / "terminal.jsonl")],
    }
    for name, args in jobs.items():
        if a.validate_only:
            args = [*args, "--dry-run"]
        cmd = [str(VENV_REW), "-m", f"rewarder.batch_utils.{name}_reward", *args]
        tag = f"validate_{name}" if a.validate_only else f"reward_{name}"
        if run(cmd, cwd=FLORA, logfile=MORE / "logs" / f"{tag}.log", stamp=True) != 0:
            log(f"{name} reward failed; see the log")


STAGES = {"fetch": stage_fetch, "prefetch": stage_prefetch, "validate": stage_validate, "caption": stage_caption,
          "reward": stage_reward, "summary": stage_summary}
MANY = {"fetch": stage_fetch_many, "caption": stage_caption_many, "reward": stage_reward_many}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--mode", choices=["l4", "reference"], default="l4")
    p.add_argument("--clips", help="JSON manifest of several clips (see more_clips.json); reference mode only. "
                                   "Stages fetch, caption and reward then run all clips in one process each.")
    p.add_argument("--validate-only", action="store_true",
                   help="with --clips: caption runs --validate-only, reward runs --dry-run")
    a = p.parse_args()
    if a.clips and a.mode != "reference":
        sys.exit("--clips is for --mode reference only")
    if a.clips:
        MORE.mkdir(parents=True, exist_ok=True)
        if a.stage == "fetch":  # needs only ffmpeg, so it can run while setup.sh is still building venvs
            log("== stage fetch (--clips)")
            stage_fetch_many(a)
            return
    if not FLORA.is_dir() or not VENV_ANN.exists() or not VENV_REW.exists():
        sys.exit(f"run setup.sh first (expected {FLORA}, {VENV_ANN}, {VENV_REW})")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.clips:
        for name in (MANY if a.stage == "all" else [a.stage]):
            if name not in MANY:
                sys.exit(f"stage {name} has no --clips variant (use prefetch without --clips)")
            log(f"== stage {name} (--clips)")
            MANY[name](a)
        return
    for name in (STAGES if a.stage == "all" else [a.stage]):
        log(f"== stage {name} ({a.mode})")
        STAGES[name](a)


if __name__ == "__main__":
    main()
