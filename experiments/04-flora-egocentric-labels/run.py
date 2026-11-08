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
    return e


def run(cmd: list[str], *, cwd: Path, logfile: Path) -> int:
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
            fh.write(line)
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


STAGES = {"fetch": stage_fetch, "prefetch": stage_prefetch, "validate": stage_validate, "caption": stage_caption,
          "reward": stage_reward, "summary": stage_summary}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--mode", choices=["l4", "reference"], default="l4")
    a = p.parse_args()
    if not FLORA.is_dir() or not VENV_ANN.exists() or not VENV_REW.exists():
        sys.exit(f"run setup.sh first (expected {FLORA}, {VENV_ANN}, {VENV_REW})")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        log(f"== stage {name} ({a.mode})")
        STAGES[name](a)


if __name__ == "__main__":
    main()
