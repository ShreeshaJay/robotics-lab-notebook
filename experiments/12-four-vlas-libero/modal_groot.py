"""Tutorial 1: GR00T-N1.7-LIBERO on the fixed LIBERO-10 episodes. Modal app t3-tut01-groot.

  modal run modal_groot.py::fetch                      # CPU: weights into the t3-tut01-cache volume
  modal run modal_groot.py --tag main --plan main      # GPU (L4): the 50 episodes, fanned out over containers

Pins: Isaac-GR00T@708617f5 (Apache-2.0), nvidia/GR00T-N1.7-LIBERO@65854032 libero_10 (NVIDIA Open Model License),
Qwen/Qwen3-VL-2B-Instruct@89644892 (Apache-2.0; config + processor stand in for the gated nvidia/Cosmos-Reason2-2B,
see README), LIBERO@8f1084e3 (MIT).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = modal.App("t3-tut01-groot")
VOL = modal.Volume.from_name("t3-tut01-cache", create_if_missing=True)
CACHE = "/cache"

GROOT_COMMIT = "708617f5716c63fc0a327391492ea2b91977ae90"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
GROOT_REPO, GROOT_REV = "nvidia/GR00T-N1.7-LIBERO", "658540327860f93dd2cb9c1ed3e3b10fb792a0eb"
QWEN_REPO, QWEN_REV = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
CKPT = f"{CACHE}/models/GR00T-N1.7-LIBERO/libero_10"
QWEN_DIR = f"{CACHE}/models/Qwen/Qwen3-VL-2B-Instruct"
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
              "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgl1", "libglib2.0-0", "libosmesa6", "ffmpeg")
    .pip_install(
        "torch==2.9.0", "torchvision==0.24.0", "transformers==4.57.3", "numpy==1.26.4", "diffusers==0.35.1",
        "albumentations==1.4.18", "opencv-python-headless==4.11.0.86", "dm-tree==0.1.9", "msgpack==1.1.0",
        "msgpack-numpy==0.4.8", "pandas==2.2.3", "peft==0.17.1", "termcolor==3.2.0", "tyro==0.9.17",
        "einops==0.8.1", "gymnasium==1.2.2", "omegaconf==2.3.0", "scipy==1.15.3", "pyzmq==27.0.1",
        "huggingface-hub==0.36.0", "safetensors==0.6.2", "accelerate==1.10.1", "click==8.1.8", "lmdb==1.7.5",
        "jsonlines==4.0.0", "av==15.1.0", "torchcodec==0.8.0",
        # LIBERO runtime (versions as Isaac-GR00T's setup_libero.sh / lerobot's Dockerfile.sim)
        "robosuite==1.4.0", "mujoco==3.3.1", "bddl==1.0.1", "easydict==1.13", "future==1.0.0", "gym==0.26.2",
        "imageio==2.37.0", "matplotlib==3.10.1", "cloudpickle==3.1.1", "h5py==3.14.0", "pyyaml==6.0.2", "tqdm==4.67.1",
    )
    .pip_install(FLASH_ATTN)
    .run_commands(
        f"git init /opt/Isaac-GR00T && cd /opt/Isaac-GR00T && git remote add origin https://github.com/NVIDIA/Isaac-GR00T.git"
        f" && git fetch -q --depth 1 origin {GROOT_COMMIT} && git checkout -q FETCH_HEAD",
        f"git init /opt/LIBERO && cd /opt/LIBERO && git remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git"
        f" && git fetch -q --depth 1 origin {LIBERO_COMMIT} && git checkout -q FETCH_HEAD",
        "python -c \"import sysconfig,pathlib; pathlib.Path(sysconfig.get_path('purelib'),'t3.pth').write_text('/opt/Isaac-GR00T\\n/opt/LIBERO\\n')\"",
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false"})
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

    snapshot_download(GROOT_REPO, revision=GROOT_REV, local_dir=f"{CACHE}/models/GR00T-N1.7-LIBERO",
                      allow_patterns=[f"libero_10/{p}" for p in ("config.json", "embodiment_id.json", "model-*.safetensors",
                                                                  "model.safetensors.index.json", "processor_config.json",
                                                                  "statistics.json")])
    snapshot_download(QWEN_REPO, revision=QWEN_REV, local_dir=QWEN_DIR)
    changed = {}
    for name in ("config.json", "processor_config.json"):
        p = Path(CKPT) / name
        orig = p.with_suffix(".json.orig")
        if not orig.exists():
            orig.write_text(p.read_text())
        txt = orig.read_text()
        changed[name] = txt.count("nvidia/Cosmos-Reason2-2B")
        txt = txt.replace("nvidia/Cosmos-Reason2-2B", QWEN_DIR)
        if name == "processor_config.json":  # the processor otherwise defaults model_name to the gated repo
            d = json.loads(txt)
            d["processor_kwargs"].setdefault("model_name", QWEN_DIR)
            changed["processor_model_name_set"] = d["processor_kwargs"]["model_name"]
            txt = json.dumps(d, indent=2)
        p.write_text(txt)
    VOL.commit()
    return {"replaced_model_name": changed, "files": sorted(os.listdir(CKPT))}


def _sampler(stop, path):
    import subprocess

    peak = 0
    with open(path, "w") as fh:
        fh.write("t,mib,util\n")
        t0 = time.time()
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                mib, util = [int(x) for x in out.split(",")]
                peak = max(peak, mib)
                fh.write(f"{time.time()-t0:.0f},{mib},{util}\n")
            except Exception:
                pass
            stop.wait(2.0)
    return peak


@APP.function(image=image, gpu=["L4", "A10"], volumes={CACHE: VOL}, timeout=3 * 3600, cpu=4, memory=24576)
def evaluate(specs: list[dict], tag: str, shard: int) -> dict:
    import sys
    import threading

    sys.path.insert(0, "/root/harness")
    import libero_common as lc

    lc.write_libero_config("/opt/LIBERO/libero/libero")
    os.environ["HF_HUB_OFFLINE"] = "1"
    t_container = time.time()
    stop = threading.Event()
    peak = {}
    nv_path = f"/tmp/nvsmi_{tag}_{shard}.csv"
    th = threading.Thread(target=lambda: peak.update(mib=_sampler(stop, nv_path)), daemon=True)
    th.start()
    from groot_runner import GrootRunner

    runner = GrootRunner(CKPT)
    out_dir = Path(f"{CACHE}/results/groot/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    with open(out_dir / f"shard{shard}.jsonl", "a") as fh:
        for spec in specs:
            try:
                rec = runner.run(spec)
            except Exception as e:  # keep going; the record says what failed
                import traceback

                rec = {**spec, "model": "groot", "error": repr(e)[:500], "traceback": traceback.format_exc()[-2000:]}
            recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[groot {tag}/{shard}] t{spec['task_id']} e{spec['episode']} succ={rec.get('success')} "
                  f"steps={rec.get('steps')} wall={rec.get('wall_s')} err={rec.get('error')}", flush=True)
    stop.set()
    th.join(timeout=5)
    import torch

    meta = {"tag": tag, "shard": shard, "gpu": torch.cuda.get_device_name(0), "load_s": runner.load_s,
            "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "nvidia_smi_peak_mib": peak.get("mib"), "container_s": round(time.time() - t_container, 1)}
    (out_dir / f"shard{shard}.meta.json").write_text(json.dumps(meta, indent=2))
    VOL.commit()
    return {"meta": meta, "records": recs}


def plan(name: str) -> list[dict]:
    import sys

    sys.path.insert(0, str(HERE / "harness"))
    import libero_common as lc

    if name == "smoke":
        return [{"task_id": 0, "episode": 0, "condition": "main"}]
    if name == "latency":  # same-GPU latency check: episode 0 of tasks 0-4
        return [{"task_id": t, "episode": 0, "condition": "latency"} for t in range(5)]
    if name == "main":
        return [{"task_id": t, "episode": e, "condition": "main"} for t in range(10) for e in range(5)]
    if name.startswith("horizon"):  # e.g. horizon:1,4,16 -> the 20-episode subset (tasks 0-9, episodes 0-1)
        hs = name.split(":")[1].split(",")
        return [{"task_id": t, "episode": e, "condition": f"horizon_{h}", "n_action_steps": (h if h == "full" else int(h))}
                for h in hs for t in range(10) for e in range(2)]
    if name.startswith("cf"):  # cf:<episodes per task>
        n = int(name.split(":")[1])
        return [{"task_id": t, "episode": e, "condition": "counterfactual", "instruction": instr, "alt_bddl": bddl}
                for t, (instr, bddl) in lc.COUNTERFACTUALS.items() for e in range(n)]
    if name == "perturb":
        return [{"task_id": t, "episode": 0, "condition": "perturb", "perturb": True} for t in range(10)]
    raise ValueError(name)


@APP.local_entrypoint()
def main(tag: str = "smoke", plan_name: str = "smoke", shards: int = 1, gpu: str = ""):
    specs = plan(plan_name)
    chunks = [specs[i::shards] for i in range(shards)]
    t0 = time.time()
    fn = evaluate.with_options(gpu=gpu) if gpu else evaluate
    results = list(fn.starmap([(c, tag, i) for i, c in enumerate(chunks) if c]))
    out = HERE / "results" / "raw" / "groot"
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for res in results for r in res["records"]]
    (out / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / f"{tag}.meta.json").write_text(json.dumps([res["meta"] for res in results], indent=2))
    ok = [r for r in recs if "success" in r]
    print(f"{tag}: {sum(r['success'] for r in ok)}/{len(ok)} success, errors={len(recs)-len(ok)}, "
          f"wall={time.time()-t0:.0f}s, container_s={[res['meta']['container_s'] for res in results]}")
