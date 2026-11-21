"""Tutorial 3 (ACT vs Diffusion Policy from scratch) on Modal.

One generic GPU function, `job(spec)`, trains (optionally) one LeRobot policy and then runs any number
of `lerobot-eval` passes on the checkpoint. Checkpoints stay on the Modal Volume `t3-tut03-ckpts`;
only JSON results come back to the laptop.

Usage (from this folder, with the pinned Modal client):
    modal run modal_app.py::prefetch
    modal run modal_app.py --phase pilot
    modal run --detach modal_app.py --phase main      # etc., see runs.py
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "t3-tut03-actdiff"
LEROBOT_VERSION = "0.6.1"  # last PyPI release before 0.7.0 (0.7.0 bundles G0.5-licensed code)
DATASETS = {
    "lerobot/pusht": "7628202a2180972f291ba1bc6723834921e72c19",
    "lerobot/aloha_sim_insertion_scripted": "8ab660912970111cbb26738b11458e6fc4a4aed1",
}

# $/s, modal.com/pricing, checked 2026-11-20
PRICE = {"L4": 0.000222, "A10": 0.000306, "T4": 0.000164, "cpu_core": 0.0000131, "mem_gib": 0.00000222}
GPU = "L4"
MAX_GPUS = 6  # coordinator cap (2026-11-20): the workspace allows 10 GPUs at once, shared with tutorial 1
GPU_SPEC = ["A10", "L4"]  # A10G measured cheaper per training step than L4 (pilot3 vs pilot); L4 fallback. Cost uses the GPU actually seen
CPU_CORES = 4.0  # physical cores (8 vCPU)
MEM_GIB = 16

HERE = Path(__file__).parent
EGL_ICD = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libegl1", "libopengl0", "libgles2", "libglvnd0", "libglib2.0-0", "libosmesa6", "git")
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        f"echo '{EGL_ICD}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(f"lerobot[training,aloha,pusht,diffusion]=={LEROBOT_VERSION}", "matplotlib==3.10.7", "scikit-learn==1.7.2")
    .env({
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root/t3",
    })
    .add_local_file(str(HERE / "t3_heads.py"), "/root/t3/t3_heads.py")
    .add_local_file(str(HERE / "t3_launch.py"), "/root/t3/t3_launch.py")
    .add_local_file(str(HERE / "t3_probe.py"), "/root/t3/t3_probe.py")
    .add_local_file(str(HERE / "t3_envfix.py"), "/root/t3/t3_envfix.py")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("t3-tut03-ckpts", create_if_missing=True)
VOL = Path("/ckpts")
LOCAL_LR_HOME = Path("/root/lr_home")


def _rate():
    g = "A10" if "A10" in _gpu_name() else GPU
    return PRICE[g] + CPU_CORES * PRICE["cpu_core"] + MEM_GIB * PRICE["mem_gib"]


def _stage_datasets():
    """Copy the pinned datasets from the volume to local disk (avoids concurrent writes to the volume)."""
    src = VOL / "lerobot_home"
    if src.exists() and not LOCAL_LR_HOME.exists():
        shutil.copytree(src, LOCAL_LR_HOME)
    os.environ["HF_LEROBOT_HOME"] = str(LOCAL_LR_HOME)


def _gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def _run(cmd, log_path, env_extra=None, mode="w"):
    env = dict(os.environ)
    env.update(env_extra or {})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, mode) as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd="/root")
    dt = time.time() - t0
    tail = log_path.read_text(errors="replace").splitlines()[-40:]
    return p.returncode, dt, tail


def _parse_train_log(log_path):
    rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "loss:" in line and "updt_s:" in line:
            d = {}
            for tok in line.split():
                if ":" in tok:
                    k, _, v = tok.partition(":")
                    d[k] = v
            rows.append(d)
    return rows


def _committer(stop):
    """Commit the volume every 5 min so a preempted container loses at most a few minutes of checkpoints."""
    while not stop.wait(300):
        try:
            vol.commit()
        except Exception as e:  # noqa: BLE001
            print("background commit failed:", e, flush=True)


def _log_attempt(name, t_start):
    """Append this container attempt to /ckpts/attempts/<name>.json (preemptions restart the same input)."""
    p = VOL / "attempts" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(p.read_text()) if p.exists() else []
    rows.append({"start": t_start, "gpu": _gpu_name()})
    p.write_text(json.dumps(rows))
    vol.commit()
    return len(rows)


def _job(spec):
    import threading

    t_start = time.time()
    _stage_datasets()
    name = spec["name"]
    attempt = _log_attempt(name, t_start)
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    try:
        return _job_inner(spec, t_start, name, attempt)
    finally:
        stop.set()


def _job_inner(spec, t_start, name, attempt):
    res = {"name": name, "spec": spec, "gpu": _gpu_name(), "lerobot": LEROBOT_VERSION, "evals": [], "attempt": attempt,
           "rate_usd_per_s": _rate(), "gl": os.environ.get("MUJOCO_GL")}
    if spec.get("gl"):
        os.environ["MUJOCO_GL"] = spec["gl"]
        os.environ["PYOPENGL_PLATFORM"] = spec["gl"]
        res["gl"] = spec["gl"]
    run_dir = VOL / "runs" / name
    ckpt = run_dir / "checkpoints" / "last" / "pretrained_model"
    tr = spec.get("train")
    if tr:
        head = tr.get("head", "diffusion")
        if ckpt.exists() and (run_dir / "t3_train_meta.json").exists():
            res["train"] = json.loads((run_dir / "t3_train_meta.json").read_text())
            res["train"]["reused"] = True
        else:
            log = VOL / "logs" / f"{name}.train.log"
            prev_cfg = ckpt / "train_config.json"
            resumed = False
            if prev_cfg.exists():  # a preempted attempt left an intermediate checkpoint: resume from it
                cmd = ["python", "/root/t3/t3_launch.py", "train", f"--config_path={prev_cfg}", "--resume=true"]
                resumed = True
            else:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                cmd = ["python", "/root/t3/t3_launch.py", "train", *tr["args"],
                       f"--output_dir={run_dir}", f"--job_name={name}", "--wandb.enable=false",
                       "--policy.push_to_hub=false", "--policy.device=cuda"]
            rc, dt, tail = _run(cmd, log, {"T3_HEAD": head}, mode="a")
            rows = _parse_train_log(log)
            meta = {"returncode": rc, "wall_s": dt, "cmd": cmd, "head": head, "log_rows": rows[-200:],
                    "tail": tail, "resumed": resumed, "attempt": attempt}
            if rc != 0:
                res["train"] = meta
                res["error"] = "train failed"
                vol.commit()
                _write(res, t_start)
                return res
            (run_dir / "t3_head.txt").write_text(head)
            (run_dir / "t3_train_meta.json").write_text(json.dumps(meta))
            res["train"] = meta
            vol.commit()
    for ev in spec.get("evals", []):
        path = Path(ev.get("ckpt", str(ckpt)))
        out = VOL / "evals" / name / ev["tag"]
        done_p = out / "t3_eval_meta.json"
        if done_p.exists():  # finished in an earlier (preempted) attempt
            res["evals"].append(json.loads(done_p.read_text()))
            continue
        if out.exists():
            shutil.rmtree(out)
        cmd = ["python", "/root/t3/t3_launch.py", "eval", f"--policy.path={path}", f"--env.type={ev['env']}",
               f"--eval.n_episodes={ev['n_episodes']}", f"--eval.batch_size={ev['batch_size']}",
               f"--seed={ev.get('seed', 1000)}", f"--output_dir={out}", "--policy.device=cuda",
               *ev.get("overrides", [])]
        log = VOL / "logs" / f"{name}.{ev['tag']}.eval.log"
        rc, dt, tail = _run(cmd, log, ev.get("env_extra"))
        e = {"tag": ev["tag"], "returncode": rc, "wall_s": dt, "cmd": cmd}
        info_p = out / "eval_info.json"
        if rc == 0 and info_p.exists():
            info = json.loads(info_p.read_text())
            e["info"] = info
            done_p.write_text(json.dumps(e))
        else:
            e["tail"] = tail
        res["evals"].append(e)
        vol.commit()
    _write(res, t_start)
    return res


def _write(res, t_start):
    res["job_wall_s"] = time.time() - t_start
    res["job_cost_usd"] = res["job_wall_s"] * _rate()
    (VOL / "results").mkdir(parents=True, exist_ok=True)
    (VOL / "results" / f"{res['name']}.json").write_text(json.dumps(res, indent=1, default=str))
    vol.commit()


@app.function(gpu=GPU_SPEC, cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=3 * 3600, volumes={"/ckpts": vol},
              max_containers=MAX_GPUS)
def job(spec: dict):
    if "chain" in spec:  # several sub-jobs back to back in one container (used by the pilot)
        t0 = time.time()
        subs = [_job(s) for s in spec["chain"]]
        res = {"name": spec["name"], "chain": subs, "gpu": _gpu_name(), "job_wall_s": time.time() - t0}
        res["job_cost_usd"] = res["job_wall_s"] * _rate()
        return res
    return _job(spec)


@app.function(gpu=GPU_SPEC, cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=3600, volumes={"/ckpts": vol},
              max_containers=MAX_GPUS)
def probe(spec: dict):
    t0 = time.time()
    _stage_datasets()
    import sys
    sys.path.insert(0, "/root/t3")
    import t3_probe
    out = t3_probe.run(spec, VOL)
    out["job_wall_s"] = time.time() - t0
    out["job_cost_usd"] = out["job_wall_s"] * _rate()
    out["gpu"] = _gpu_name()
    (VOL / "results").mkdir(parents=True, exist_ok=True)
    (VOL / "results" / f"{spec['name']}.json").write_text(json.dumps(out, indent=1, default=str))
    vol.commit()
    return out


@app.function(gpu=GPU, cpu=2.0, memory=8192, timeout=600)
def smoke(gl: str = "egl"):
    """Check that both sim envs import, reset and render headless with the given MuJoCo GL backend."""
    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    code = "\n".join([
        "import time, gymnasium as gym, gym_aloha, gym_pusht",
        "t=time.time(); e=gym.make('gym_aloha/AlohaInsertion-v0', obs_type='pixels_agent_pos'); o,_=e.reset(seed=0)",
        "for _ in range(50): o,*_=e.step(e.action_space.sample())",
        "print('aloha ok', o['pixels']['top'].shape, '50 steps s=', round(time.time()-t,2))",
        "e=gym.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos'); o,_=e.reset(seed=0); print('pusht ok', o['pixels'].shape)",
    ])
    print("smoke start", flush=True)
    p = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=300)
    return p.stdout + p.stderr[-3000:]


@app.function(cpu=2.0, memory=4096, timeout=1800, volumes={"/ckpts": vol})
def prefetch():
    """Download the two pinned datasets once into the volume; record the frozen environment."""
    os.environ["HF_LEROBOT_HOME"] = str(VOL / "lerobot_home")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = {}
    for rid, rev in DATASETS.items():
        ds = LeRobotDataset(rid, revision=rev)
        out[rid] = {"revision": rev, "episodes": ds.num_episodes, "frames": ds.num_frames, "fps": ds.fps,
                    "features": {k: v.get("shape") for k, v in ds.meta.features.items()}}
    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    (VOL / "meta").mkdir(parents=True, exist_ok=True)
    (VOL / "meta" / "freeze.txt").write_text(freeze)
    du = subprocess.run(["du", "-sh", str(VOL / "lerobot_home")], capture_output=True, text=True).stdout
    out["du"] = du
    vol.commit()
    return {"datasets": out, "freeze": freeze}


@app.local_entrypoint()
def main(phase: str = "pilot", only: str = ""):
    import runs

    specs = runs.PHASES[phase]()
    if only:
        keep = set(only.split(","))
        specs = [s for s in specs if s["name"] in keep]
    print(f"phase={phase}: {len(specs)} jobs: {[s['name'] for s in specs]}")
    raw = HERE / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    fn = probe if phase.startswith("probe") else job
    for r in fn.map(specs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("JOB EXCEPTION:", repr(r)[:500])
            continue
        (raw / f"{r['name']}.json").write_text(json.dumps(r, indent=1, default=str))
        evs = [(e["tag"], e.get("info", {}).get("aggregated", {}).get("pc_success"), round(e["wall_s"]))
               for e in r.get("evals", [])]
        print(r["name"], "train_s=", round(r.get("train", {}).get("wall_s", 0)), "cost=",
              round(r.get("job_cost_usd", 0), 3), evs, r.get("error", ""))


@app.local_entrypoint()
def smoke_main(gl: str = "egl"):
    print(smoke.remote(gl))
