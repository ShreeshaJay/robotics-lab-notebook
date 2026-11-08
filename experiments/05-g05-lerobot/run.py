#!/usr/bin/env python3
"""05 · G0.5 in LeRobot: one LIBERO rollout (video saved) + auto-annotation of 3 public episodes.

Runs on the VM after setup.sh. Uses only the standard library itself; the work happens inside the
venvs that setup.sh built. Weights, datasets, videos and outputs live under $WORK, never in this repo.

Stages:
  fetch     pinned downloads: lerobot/g05_libero (11.6 GB), LIBERO assets, lerobot/svla_so101_pickplace (86 MB)
  rollout   lerobot-eval: G0.5-LIBERO, libero_10 task 0, 1 episode, MP4 saved under results/rollout/videos/
  annotate  lerobot-annotate on episodes 0-2: subtasks (plan module) + bounding boxes (vqa module),
            served by an open VLM through a local vLLM server (no paid API, no Hub push)
  summary   small results/summary.json (success, timings, peak GPU memory, subtasks, boxes)

Example:
  export G05_LICENSE_ACK=reviewed-g05-community-license   # only after a human has read and accepted it
  python3 run.py all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

WORK = Path(os.environ.get("WORK", str(Path.home() / "g05-lab"))).resolve()
PY = WORK / ".venv-lerobot" / "bin" / "python"
BIN = WORK / ".venv-lerobot" / "bin"
VLLM = WORK / ".venv-vllm" / "bin" / "vllm"
RESULTS = WORK / "results"
CKPT = WORK / "checkpoints" / "g05_libero"
DATASET = WORK / "datasets" / "svla_so101_pickplace"

# Pinned revisions (HF main on 2026-11-07).
G05_LIBERO = ("lerobot/g05_libero", "9b7ee7d7286b2744e7cfae855dd71aa3cf4c075a")
LIBERO_ASSETS = ("lerobot/libero-assets", "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6")  # same pin as lerobot's Dockerfile.sim
DATASET_REPO = ("lerobot/svla_so101_pickplace", "f641879e22172be7e8161d5e6c1503c2d2feb657")  # Apache-2.0, v3.0, 50 eps
# Annotation VLM. The upstream default, Qwen/Qwen3.6-27B (BF16, ~55 GB), does not fit a 24 GB L4,
# and neither does Qwen3.5-27B-GPTQ-Int4 (30 GB on disk). Qwen3.5-4B (Apache-2.0, 9.3 GB) does.
VLM = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
EPISODES = [0, 1, 2]
ACK_VALUE = "reviewed-g05-community-license"


def log(msg: str) -> None:
    print(f"[run.py {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env() -> dict[str, str]:
    e = dict(os.environ)
    e.setdefault("HF_HOME", str(WORK / "hf"))
    e["MUJOCO_GL"] = "egl"
    e["PYOPENGL_PLATFORM"] = "egl"
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    return e


def run(cmd: list[str], *, logfile: Path, cwd: Path = WORK) -> int:
    """Run a command, tee output to a log, and sample GPU memory while it runs."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log("$ " + " ".join(cmd))
    peak = {"mib": 0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.split()
                if out:
                    peak["mib"] = max(peak["mib"], int(out[0]))
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
    if os.environ.get("G05_LICENSE_ACK") != ACK_VALUE:
        sys.exit(textwrap.dedent(f"""\
            G0.5 code (src/lerobot/policies/g05) and checkpoints are under the G0.5 Community License
            (non-commercial). Its first paragraph says that DOWNLOADING the materials counts as accepting it.
            A human must read https://huggingface.co/OpenGalaxea/G05/blob/main/licenses/LICENSE-G0.5 and decide;
            this script will not download the checkpoint until you
              export G05_LICENSE_ACK={ACK_VALUE}
            """))


def py(code: str, logname: str) -> int:
    return run([str(PY), "-I", "-c", code], logfile=RESULTS / "logs" / logname)


# ----------------------------------------------------------------------------- stages

def stage_fetch(_: argparse.Namespace) -> None:
    require_ack()
    code = f"""
from huggingface_hub import snapshot_download
snapshot_download({G05_LIBERO[0]!r}, revision={G05_LIBERO[1]!r}, local_dir={str(CKPT)!r})
snapshot_download({LIBERO_ASSETS[0]!r}, repo_type='dataset', revision={LIBERO_ASSETS[1]!r},
                  local_dir={str(Path.home() / '.libero' / 'assets')!r})
snapshot_download({DATASET_REPO[0]!r}, repo_type='dataset', revision={DATASET_REPO[1]!r}, local_dir={str(DATASET)!r})
print('fetched')
"""
    if py(code, "fetch.log") != 0:
        sys.exit("fetch failed")
    # Keep the pristine dataset: lerobot-annotate rewrites parquet shards in place.
    pristine = DATASET.with_name(DATASET.name + "_pristine")
    if not pristine.exists():
        shutil.copytree(DATASET, pristine, ignore=shutil.ignore_patterns(".cache"))


def stage_rollout(_: argparse.Namespace) -> None:
    require_ack()
    out = RESULTS / "rollout"
    if out.exists():
        shutil.rmtree(out)  # lerobot-eval refuses to reuse an output dir
    # The documented G0.5 LIBERO command, cut down to one task and one episode.
    cmd = [
        str(BIN / "lerobot-eval"),
        f"--policy.path={CKPT}",
        "--policy.device=cuda",
        "--env.type=libero",
        "--env.task=libero_10",
        "--env.task_ids=[0]",
        "--env.control_mode=relative",
        "--env.observation_height=512",
        "--env.observation_width=512",
        '--env.camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}',
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--seed=0",
        f"--output_dir={out}",
    ]
    if run(cmd, logfile=RESULTS / "logs" / "rollout.log") != 0:
        log("rollout failed; see log")
    videos = sorted(out.rglob("*.mp4"))
    log(f"rollout videos: {[str(v) for v in videos] or 'none found'}")


def stage_annotate(_: argparse.Namespace) -> None:
    serve = (f"{VLLM} serve {VLM[0]} --revision {VLM[1]} --tensor-parallel-size 1 "
             "--max-model-len 32768 --gpu-memory-utilization 0.85 --uvicorn-log-level warning --port {port}")
    cmd = [
        str(BIN / "lerobot-annotate"),
        f"--root={DATASET}",
        f"--only_episodes=[{','.join(map(str, EPISODES))}]",
        "--push_to_hub=false",
        f"--vlm.model_id={VLM[0]}",
        f"--vlm.serve_command={serve}",
        "--vlm.serve_ready_timeout_s=1800",
        '--vlm.chat_template_kwargs={"enable_thinking": false}',
        "--vlm.camera_key=observation.images.up",
        "--vqa.restrict_to_default_camera=true",
        "--interjections.enabled=false",  # scope: subtasks + boxes only
    ]
    if run(cmd, logfile=RESULTS / "logs" / "annotate.log") != 0:
        log("annotation failed; see log")
        return
    # Pull the new language rows for the annotated episodes out of the rewritten parquet.
    code = f"""
import json, glob
import pandas as pd
files = sorted(glob.glob({str(DATASET)!r} + '/data/chunk-*/file-*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files])
df = df[df['episode_index'].isin({EPISODES!r})]
def rows(cell):
    # A language cell is a list of row structs; be tolerant of None / numpy arrays / dicts.
    if cell is None:
        return []
    try:
        return [dict(r) for r in list(cell)]
    except Exception:
        return [{{'raw': repr(cell)[:500]}}]
out = {{'columns': list(df.columns)}}
for ep, g in df.groupby('episode_index'):
    persistent = rows(g.iloc[0]['language_persistent']) if 'language_persistent' in g else []
    events = [e for cell in (g['language_events'] if 'language_events' in g else []) for e in rows(cell)]
    out[int(ep)] = {{
        'frames': int(len(g)),
        'subtasks': [r for r in persistent if r.get('style') == 'subtask'],
        'plan': [r for r in persistent if r.get('style') == 'plan'][:2],
        'vqa': [e for e in events if e.get('style') == 'vqa'][:12],
    }}
json.dump(out, open({str(RESULTS / 'annotations.json')!r}, 'w'), indent=2, default=str)
print({{k: (len(v['subtasks']), len(v['vqa'])) for k, v in out.items() if k != 'columns'}})
"""
    py(code, "annotate_extract.log")


def stage_summary(_: argparse.Namespace) -> None:
    s: dict = {"rollout": {}, "annotate": {}, "runs": {}}
    for meta in sorted((RESULTS / "logs").glob("*.meta.json")):
        m = json.loads(meta.read_text())
        s["runs"][meta.name.replace(".meta.json", "")] = {k: m[k] for k in ("exit_code", "elapsed_s", "peak_gpu_mib")}
    info = sorted((RESULTS / "rollout").rglob("eval_info.json"))
    if info:
        d = json.loads(info[0].read_text())
        s["rollout"]["eval_info_overall"] = d.get("overall", d.get("aggregated", {}))
    s["rollout"]["videos"] = [str(p.relative_to(RESULTS)) for p in sorted((RESULTS / "rollout").rglob("*.mp4"))]
    ann = RESULTS / "annotations.json"
    if ann.exists():
        s["annotate"] = json.loads(ann.read_text())
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(s, indent=2, default=str))
    log(f"wrote {out}. Copy the JSON (not the MP4) into the notebook's results/ folder.")


STAGES = {"fetch": stage_fetch, "rollout": stage_rollout, "annotate": stage_annotate, "summary": stage_summary}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "all"])
    a = p.parse_args()
    if not PY.exists() or not VLLM.exists():
        sys.exit(f"run setup.sh first (expected {PY} and {VLLM})")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        log(f"== stage {name}")
        STAGES[name](a)


if __name__ == "__main__":
    main()
