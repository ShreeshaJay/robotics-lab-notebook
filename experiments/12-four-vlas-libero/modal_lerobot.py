"""Tutorial 1: G0.5 and SmolVLA re-runs for the extensions that need new conditions. Modal app t3-tut01-lerobot.

Same stack as the baseline (experiments/05-g05-lerobot/results/rerun50/freeze-lerobot.txt: lerobot 0.7.0, torch
2.11.0+cu128, hf-libero 0.1.4, mujoco 3.3.1, robosuite 1.4.0), same checkpoints (g05_libero@9b7ee7d7,
smolvla_libero@a24f1f38), same GPU type (L4), same lerobot-eval flags; one change: --eval.use_async_envs=false so the
env patches of harness/lerobot_tut_eval.py apply in-process.

  modal run modal_lerobot.py::fetch
  modal run modal_lerobot.py --jobs "g05:states,smolvla:states"
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = modal.App("t3-tut01-lerobot")
VOL = modal.Volume.from_name("t3-tut01-cache", create_if_missing=True)
CACHE = "/cache"
CKPTS = {"g05": ("lerobot/g05_libero", "9b7ee7d7286b2744e7cfae855dd71aa3cf4c075a"),
         "smolvla": ("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d")}
ASSETS = ("lerobot/libero-assets", "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "libglx0", "libgles2", "libglib2.0-0", "libosmesa6", "libxrender1", "libsm6", "cmake",
                 "build-essential")
    .add_local_file(str(HERE / "harness" / "freeze-lerobot.txt"), "/opt/freeze-lerobot.txt", copy=True)
    .run_commands("pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /opt/freeze-lerobot.txt")
    .env({"HF_HOME": f"{CACHE}/hf", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(HERE / "harness"), "/root/harness")
)


def _hf_secret() -> modal.Secret:
    # HF_TOKEN from the environment, or the HF_TOKEN_ASPIRE line of the dotenv file named by T3_DOTENV
    tok = os.environ.get("HF_TOKEN", "")
    env = Path(os.environ.get("T3_DOTENV", "") or ".")
    if not tok and env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN_ASPIRE="):
                tok = line.split("=", 1)[1].strip().strip('"')
    return modal.Secret.from_dict({"HF_TOKEN": tok})


@APP.function(image=image, volumes={CACHE: VOL}, timeout=3600, cpu=4, secrets=[_hf_secret()])
def fetch() -> dict:
    from huggingface_hub import snapshot_download

    for name, (repo, rev) in CKPTS.items():
        snapshot_download(repo, revision=rev, local_dir=f"{CACHE}/models/{name}_libero")
    snapshot_download(ASSETS[0], repo_type="dataset", revision=ASSETS[1], local_dir=f"{CACHE}/libero-assets")
    VOL.commit()
    return {"ok": True}


def _libero_config() -> str:
    import importlib.util

    spec = importlib.util.find_spec("libero")
    libero_dir = os.path.join(os.path.dirname(spec.origin), "libero")
    home = os.path.expanduser("~/.libero")
    os.makedirs(home, exist_ok=True)
    Path(home, "config.yaml").write_text(
        f"assets: {CACHE}/libero-assets\nbddl_files: {libero_dir}/bddl_files\ndatasets: {libero_dir}/../datasets\n"
        f"init_states: {libero_dir}/init_files\n")
    os.makedirs(os.path.expanduser("~/.cache/libero"), exist_ok=True)
    link = os.path.expanduser("~/.cache/libero/assets")
    if not os.path.exists(link):
        os.symlink(f"{CACHE}/libero-assets", link)
    return libero_dir


POLICY_ARGS = {
    "g05": ["--env.control_mode=relative", "--env.observation_height=512", "--env.observation_width=512",
            '--env.camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}'],
    "smolvla": ["--env.control_mode=relative",
                '--rename_map={"observation.images.image": "observation.images.camera1", '
                '"observation.images.image2": "observation.images.camera2"}'],
}


@APP.function(image=image, gpu=["L4", "A10"], volumes={CACHE: VOL}, timeout=3 * 3600, cpu=8, memory=32768, secrets=[_hf_secret()])
def evaluate(model: str, mode: str, tag: str) -> dict:
    """mode: states | horizon:<n> | cf:<n_eps> | perturb"""
    import sys

    t_container = time.time()
    libero_dir = _libero_config()
    sys.path.insert(0, "/root/harness")
    import libero_common as lc

    out_dir = Path(f"{CACHE}/results/{model}/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    task_ids, n_eps, extra = list(range(10)), 5, []
    env["T3_TRACK_OUT"] = str(out_dir / "track.jsonl")
    Path(env["T3_TRACK_OUT"]).write_text("")
    if mode.startswith("horizon"):
        n_eps = 2
        extra = [f"--policy.n_action_steps={int(mode.split(':')[1])}"]
    elif mode.startswith("cf"):
        task_ids, n_eps = sorted(lc.COUNTERFACTUALS), int(mode.split(":")[1])
        env["T3_CF"] = "1"
    elif mode == "perturb":
        n_eps = 1
        env["T3_PERTURB"] = "1"
    elif mode != "states":
        raise ValueError(mode)
    cmd = [sys.executable, "-I", "/root/harness/lerobot_tut_eval.py", str(out_dir / "timing.json"),
           f"--policy.path={CACHE}/models/{model}_libero", *POLICY_ARGS[model], *extra, "--policy.device=cuda",
           "--env.type=libero", "--env.task=libero_10", f"--env.task_ids=[{','.join(map(str, task_ids))}]",
           "--env.init_states=true", "--eval.batch_size=1", f"--eval.n_episodes={n_eps}", "--eval.use_async_envs=false",
           "--seed=0", f"--output_dir=/tmp/out_{tag}"]
    # -I drops PYTHONPATH and the script dir from sys.path; the script inserts its own dir.
    peak = {"mib": 0}
    stop = threading.Event()

    def sample():
        with open(out_dir / "nvsmi.csv", "w") as fh:
            fh.write("t,mib,util\n")
            t0 = time.time()
            while not stop.is_set():
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                    mib, util = [int(x) for x in o.split(",")]
                    peak["mib"] = max(peak["mib"], mib)
                    fh.write(f"{time.time()-t0:.0f},{mib},{util}\n")
                except Exception:
                    pass
                stop.wait(2.0)

    threading.Thread(target=sample, daemon=True).start()
    with open(out_dir / "eval.log", "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    info = sorted(Path(f"/tmp/out_{tag}").rglob("eval_info.json"))
    if info:
        (out_dir / "eval_info.json").write_text(info[0].read_text())
    track = [json.loads(l) for l in Path(env["T3_TRACK_OUT"]).read_text().splitlines() if l.strip()]
    eps_path = out_dir / "timing.episodes.jsonl"
    eps = [json.loads(l) for l in eps_path.read_text().splitlines() if l.strip()] if eps_path.exists() else []
    timing = json.loads((out_dir / "timing.json").read_text()) if (out_dir / "timing.json").exists() else {}
    meta = {"model": model, "mode": mode, "tag": tag, "exit": proc.returncode, "container_s": round(time.time() - t_container, 1),
            "nvidia_smi_peak_mib": peak["mib"], "timing": timing, "libero_dir": libero_dir,
            "log_tail": (out_dir / "eval.log").read_text()[-3000:] if proc.returncode else ""}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    VOL.commit()
    return {"meta": meta, "track": track, "episodes": eps,
            "eval_info": json.loads(info[0].read_text()) if info else None}


@APP.local_entrypoint()
def main(jobs: str = "smolvla:states"):
    t0 = time.time()
    args = []
    for j in jobs.split(","):
        model, mode = j.split(":", 1)
        tag = mode.replace(":", "_")
        args.append((model, mode, tag))
    out = HERE / "results" / "raw" / "lerobot"
    out.mkdir(parents=True, exist_ok=True)
    for res in evaluate.starmap(args, return_exceptions=True):
        if isinstance(res, Exception):
            print("job failed:", repr(res)[:500])
            continue
        m = res["meta"]
        name = f"{m['model']}_{m['tag']}"
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        n = len(res["track"])
        print(f"{name}: exit={m['exit']} track_eps={n} success={sum(r['success'] for r in res['track'])}/{n} "
              f"container_s={m['container_s']} {m['log_tail'][-800:]}")
    print(f"wall {time.time()-t0:.0f}s")
