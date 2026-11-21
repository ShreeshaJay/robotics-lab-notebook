"""Tutorial 1: PEAR (Qwen3.8-27B-PEAR, BF16) on the fixed LIBERO-10 episodes. Modal app t3-tut01-pear.

  modal run modal_pear.py::fetch                                         # CPU: 55.7 GB of weights into the volume
  modal run modal_pear.py --tag pilot --plan-name pilot --conc 5 --gpu B200   # 5-episode pilot
  modal run modal_pear.py --tag main --plan-name main_rest --conc 12 --gpu B200

One container: a vLLM 0.28.0 OpenAI server (PEAR's server pin) on the GPU, and `conc` LIBERO episodes in parallel
CPU processes (harness/pear_runner.py) talking to it. Pins: mit-han-lab/Qwen3.8-27B-PEAR@65176e43 (Apache-2.0),
pear@2a81b2ee (Apache-2.0, controller and prompts vendored in harness/pear_vendor), LIBERO@8f1084e3 (MIT),
robosuite 1.4.0, mujoco 3.2.3 (PEAR's LIBERO pins).
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
APP = modal.App("t3-tut01-pear")
VOL = modal.Volume.from_name("t3-tut01-cache", create_if_missing=True)
CACHE = "/cache"
PEAR_REPO, PEAR_REV = "mit-han-lab/Qwen3.8-27B-PEAR", "65176e435e5fa80a059ebfafaf8215e6ad00a83e"
MODEL_DIR = f"{CACHE}/models/Qwen3.8-27B-PEAR"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
# FlashInfer's TRT-LLM decode kernels are JIT-built on Blackwell and need nvcc, which this image does not carry;
# vLLM's own FlashAttention backend ships prebuilt kernels.
ATTN_ARGS = ["--attention-backend", "FLASH_ATTN"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libegl1", "libgl1", "libglib2.0-0", "libosmesa6", "curl")
    .pip_install("uv==0.9.5", "huggingface-hub==0.36.0")
    .run_commands(
        "uv venv /opt/vllm --python 3.12",
        "uv pip install --python /opt/vllm/bin/python vllm==0.28.0 transformers==5.10.4",
        "uv venv /opt/sim --python 3.12",
        "uv pip install --python /opt/sim/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.9.0",
        "uv pip install --python /opt/sim/bin/python numpy==1.26.4 robosuite==1.4.0 mujoco==3.2.3 bddl==1.0.1 "
        "easydict==1.13 future==1.0.0 gym==0.26.2 imageio==2.37.0 matplotlib==3.10.1 cloudpickle==3.1.1 h5py==3.14.0 "
        "pyyaml==6.0.2 tqdm==4.67.1 termcolor==3.2.0 pillow==11.3.0 openai==2.8.1",
        f"git init /opt/LIBERO && cd /opt/LIBERO && git remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git"
        f" && git fetch -q --depth 1 origin {LIBERO_COMMIT} && git checkout -q FETCH_HEAD",
        "echo /opt/LIBERO > $(/opt/sim/bin/python -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')/libero.pth",
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "FLASHINFER_WORKSPACE_BASE": f"{CACHE}/flashinfer", "VLLM_CACHE_ROOT": f"{CACHE}/vllm_cache", "HF_HOME": f"{CACHE}/hf", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "TOKENIZERS_PARALLELISM": "false"})
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


@APP.function(image=image, volumes={CACHE: VOL}, timeout=3 * 3600, cpu=8, memory=16384, secrets=[_hf_secret()])
def fetch() -> dict:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(PEAR_REPO, revision=PEAR_REV, local_dir=MODEL_DIR, max_workers=16)
    VOL.commit()
    files = {p.name: p.stat().st_size for p in Path(MODEL_DIR).iterdir() if p.is_file()}
    return {"seconds": round(time.time() - t0), "files": files}


def _sample(stop: threading.Event, path: str, peak: dict) -> None:
    with open(path, "w") as fh:
        fh.write("t,mib,util\n")
        t0 = time.time()
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
                mib, util = [int(x) for x in out.split(",")]
                peak["mib"] = max(peak.get("mib", 0), mib)
                fh.write(f"{time.time()-t0:.0f},{mib},{util}\n")
                fh.flush()
            except Exception:
                pass
            stop.wait(5.0)


@APP.function(image=image, gpu="B200", volumes={CACHE: VOL}, timeout=3 * 3600, cpu=16, memory=65536)
def evaluate(specs: list[dict], tag: str, conc: int) -> dict:
    import sys

    t_container = time.time()
    out_dir = f"{CACHE}/results/pear/{tag}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stop, peak = threading.Event(), {}
    threading.Thread(target=_sample, args=(stop, f"{out_dir}/nvsmi.csv", peak), daemon=True).start()

    def committer():
        while not stop.is_set():
            stop.wait(240)
            try:
                VOL.commit()
            except Exception:
                pass

    threading.Thread(target=committer, daemon=True).start()
    server_log = open(f"{out_dir}/vllm_server.log", "w")
    server = subprocess.Popen(
        ["/opt/vllm/bin/vllm", "serve", MODEL_DIR, "--served-model-name", "pear", "--port", "8000",
         "--max-model-len", "260096", "--gpu-memory-utilization", "0.88", "--enable-prefix-caching",
         "--reasoning-parser", "qwen3", "--limit-mm-per-prompt", json.dumps({"image": 1016}),
         "--max-num-seqs", str(max(conc, 1)), "--max-num-batched-tokens", "16384", "--seed", "0",
         "--dtype", "bfloat16", "--uvicorn-log-level", "warning", *ATTN_ARGS],
        stdout=server_log, stderr=subprocess.STDOUT, env={**os.environ, "VLLM_LOGGING_LEVEL": "INFO"})
    t_ready = None
    for _ in range(1800):
        if server.poll() is not None:
            break
        try:
            if subprocess.run(["curl", "-sf", "http://127.0.0.1:8000/health"], capture_output=True, timeout=5).returncode == 0:
                t_ready = time.time()
                break
        except Exception:
            pass
        time.sleep(2)
    if t_ready is None:
        stop.set()
        server_log.flush()
        VOL.commit()
        return {"meta": {"tag": tag, "error": "vLLM did not start", "server_log_tail": Path(f"{out_dir}/vllm_server.log").read_text()[-4000:],
                         "container_s": round(time.time() - t_container, 1)}, "records": []}
    print(f"[pear {tag}] vLLM ready after {t_ready - t_container:.0f}s", flush=True)
    sys.path.insert(0, "/root/harness")
    import libero_common as lc

    lc.write_libero_config("/opt/LIBERO/libero/libero")
    spec_path = f"{out_dir}/specs.json"
    Path(spec_path).write_text(json.dumps(specs))
    t_run = time.time()
    run = subprocess.run(["/opt/sim/bin/python", "/root/harness/pear_runner.py", spec_path, out_dir, str(conc),
                          "http://127.0.0.1:8000/v1"], env={**os.environ, "LIBERO_CONFIG_PATH": os.environ["LIBERO_CONFIG_PATH"],
                                                             "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    # The simulators render on the CPU (as PEAR's own LIBERO servers do, with Mesa): one EGL context per episode
    # process on the GPU took the memory vLLM had reserved and crashed the engine at 23 concurrent episodes.
    t_end = time.time()
    server.terminate()
    try:
        server.wait(timeout=60)
    except Exception:
        server.kill()
    stop.set()
    recs = [json.loads(l) for l in Path(f"{out_dir}/episodes.jsonl").read_text().splitlines() if l.strip()] \
        if Path(f"{out_dir}/episodes.jsonl").exists() else []
    import re

    log_txt = Path(f"{out_dir}/vllm_server.log").read_text()
    kv = re.findall(r"(GPU KV cache size: [\d,]+ tokens|Maximum concurrency for [^\n]+)", log_txt)
    meta = {"tag": tag, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
            "startup_s": round(t_ready - t_container, 1), "episodes_s": round(t_end - t_run, 1),
            "container_s": round(time.time() - t_container, 1), "runner_exit": run.returncode, "conc": conc,
            "nvidia_smi_peak_mib": peak.get("mib"), "vllm_kv_lines": kv[:4]}
    Path(f"{out_dir}/meta.json").write_text(json.dumps(meta, indent=2))
    VOL.commit()
    return {"meta": meta, "records": recs}


def plan(name: str) -> list[dict]:
    import sys

    sys.path.insert(0, str(HERE / "harness"))
    import libero_common as lc

    main = [{"task_id": t, "episode": e, "condition": "main"} for t in range(10) for e in range(5)]
    # pilot: 5 distinct tasks, episode 0 (tasks 0, 2, 4, 6, 8)
    pilot = [{"task_id": t, "episode": 0, "condition": "main"} for t in (0, 2, 4, 6, 8)]
    if name == "pilot":
        return pilot
    if name == "main_rest":
        done = {(s["task_id"], s["episode"]) for s in pilot}
        return [s for s in main if (s["task_id"], s["episode"]) not in done]
    if name.startswith("cf"):
        n = int(name.split(":")[1])
        return [{"task_id": t, "episode": e, "condition": "counterfactual", "instruction": i, "alt_bddl": b}
                for t, (i, b) in lc.COUNTERFACTUALS.items() for e in range(n)]
    if name == "perturb":
        return [{"task_id": t, "episode": 0, "condition": "perturb", "perturb": True} for t in range(10)]
    if name.startswith("ext:"):  # ext:<cf episodes per task> -> counterfactual + perturbation in one container
        return plan(f"cf:{name.split(':')[1]}") + plan("perturb")
    if name == "resume":  # the main episodes not yet completed, then the extension episodes stopped unfinished
        done = set()
        for f in ("pilot", "main_a"):
            p = HERE / "results" / "raw" / "pear" / f"{f}.jsonl"
            if p.exists():
                done |= {(r["task_id"], r["episode"]) for r in map(json.loads, p.read_text().splitlines()) if "success" in r}
        rest = [s for s in main if (s["task_id"], s["episode"]) not in done]
        stopped = json.loads((HERE / "results" / "raw" / "pear" / "ext_stopped.meta.json").read_text())["unfinished_at_stop"]
        return rest + stopped
    if name.startswith("list:"):  # list:t-e,t-e
        return [{"task_id": int(x.split("-")[0]), "episode": int(x.split("-")[1]), "condition": "main"}
                for x in name.split(":")[1].split(",")]
    raise ValueError(name)


@APP.local_entrypoint()
def main(tag: str = "pilot", plan_name: str = "pilot", conc: int = 5, gpu: str = "B200"):
    specs = plan(plan_name)
    t0 = time.time()
    res = evaluate.with_options(gpu=gpu).remote(specs, tag, conc)
    out = HERE / "results" / "raw" / "pear"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["records"]))
    if plan_name == "resume":
        for cond, fname in (("main", "main_b"), ("counterfactual", "cf_b"), ("perturb", "perturb_b")):
            (out / f"{fname}.jsonl").write_text("".join(json.dumps(r) + chr(10) for r in res["records"] if r.get("condition") == cond))
    if plan_name.startswith("ext:"):  # split into the files analysis/analyze.py reads
        for cond, fname in (("counterfactual", "cf"), ("perturb", "perturb")):
            (out / f"{fname}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["records"] if r.get("condition") == cond))
    (out / f"{tag}.meta.json").write_text(json.dumps(res["meta"], indent=2))
    ok = [r for r in res["records"] if "success" in r]
    print(json.dumps(res["meta"], indent=2)[:3000])
    print(f"{tag}: {sum(r['success'] for r in ok)}/{len(ok)} success, errors={len(res['records'])-len(ok)}, wall={time.time()-t0:.0f}s")


@APP.function(image=image, timeout=1800, cpu=4, memory=16384)
def dry(specs: list[dict]) -> list[dict]:
    """CPU-only check of the episode loop with a scripted stand-in for the model (no GPU, no weights)."""
    import sys

    sys.path.insert(0, "/root/harness")
    import libero_common as lc

    lc.write_libero_config("/opt/LIBERO/libero/libero")
    Path("/tmp/dry").mkdir(exist_ok=True)
    Path("/tmp/dry/specs.json").write_text(json.dumps(specs))
    subprocess.run(["/opt/sim/bin/python", "/root/harness/pear_runner.py", "/tmp/dry/specs.json", "/tmp/dry", "2", "dry"],
                   env={**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}, check=False)
    return [json.loads(l) for l in Path("/tmp/dry/episodes.jsonl").read_text().splitlines() if l.strip()]


@APP.local_entrypoint()
def dry_main():
    import sys

    sys.path.insert(0, str(HERE / "harness"))
    import libero_common as lc

    specs = [{"task_id": 3, "episode": 0, "condition": "dry"},
             {"task_id": 0, "episode": 1, "condition": "dry_cf", "instruction": lc.COUNTERFACTUALS[0][0], "alt_bddl": lc.COUNTERFACTUALS[0][1],
              "perturb": True}]
    for r in dry.remote(specs):
        r.pop("turn_log", None)
        print(json.dumps(r)[:1500])
