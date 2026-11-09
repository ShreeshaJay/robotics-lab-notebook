#!/usr/bin/env python3
"""Run `lerobot-eval` unchanged, but time every action-chunk inference call and record peak VRAM.

Usage (inside the lerobot venv):  python timed_eval.py <timings.json> <lerobot-eval args...>

It wraps the one method per policy that produces a fresh action chunk:
  G05Policy._run_inference        (called when the action queue is empty)
  SmolVLAPolicy._get_action_chunk (called when the action queue is empty)
Each call is bracketed with torch.cuda.synchronize(), so the time is real GPU latency.
It also wraps lerobot_eval.rollout (one call = one batch; batch size 1 = one episode) and appends one line per
episode to <timings>.episodes.jsonl as it finishes: wall time, steps, success, chunk calls and their latencies.
Peak VRAM is torch's max_memory_allocated/reserved in the policy process (simulator rendering in
worker processes is not included; run.py also samples nvidia-smi for the whole-GPU peak).
"""

from __future__ import annotations

import json
import statistics
import sys
import time


def main() -> None:
    out_path, eval_args = sys.argv[1], sys.argv[2:]
    import torch

    from lerobot.policies.g05.modeling_g05 import G05Policy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    times: list[float] = []
    episodes: list[dict] = []
    ep_path = out_path.rsplit(".json", 1)[0] + ".episodes.jsonl"
    open(ep_path, "w").close()

    def wrap(cls: type, name: str) -> None:
        orig = getattr(cls, name)

        def timed(self, *a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(self, *a, **k)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            return out

        setattr(cls, name, timed)

    wrap(G05Policy, "_run_inference")
    wrap(SmolVLAPolicy, "_get_action_chunk")

    from lerobot.scripts import lerobot_eval

    # Workaround for lerobot 0.7.0: validate_checkpoint compares the raw LIBERO state (8-D: eef pos 3 +
    # axis-angle 3 + both gripper qpos) with g05_libero's declared 7-D state and refuses to start, although
    # G0.5's own preprocessor drops the second gripper qpos (processor_g05.py, "raw_state[..., :7]").
    # Tolerate exactly that one mismatch for G0.5; every other check still runs.
    orig_validate = lerobot_eval.validate_checkpoint

    def validate(descriptor, config, rename_map, **kw):
        try:
            orig_validate(descriptor, config, rename_map, **kw)
        except ValueError as e:
            if getattr(config, "type", None) == "g05" and "observation.state: expected (7,), got (8,)" in str(e):
                print("[timed_eval] tolerating known G0.5 LIBERO 8->7 state check:", e, flush=True)
                return
            raise

    lerobot_eval.validate_checkpoint = validate

    orig_rollout = lerobot_eval.rollout

    def timed_rollout(*a, **k):
        n0, t0 = len(times), time.perf_counter()
        ret = orig_rollout(*a, **k)
        rec = {"episode_seq": len(episodes), "wall_s": round(time.perf_counter() - t0, 2),
               "chunk_calls": len(times) - n0,
               "chunk_ms": [round(1000 * x, 1) for x in times[n0:]]}
        try:
            succ = ret["success"]  # (batch, steps); done/success are padded after the episode ends
            rec["success"] = bool(succ.any(dim=1)[0].item())
            done = ret["done"]
            rec["steps"] = int(done.shape[1]) if not bool(done[0].any()) else int(done[0].float().argmax().item()) + 1
        except Exception as e:  # keep the timing even if the return format differs
            rec["parse_error"] = repr(e)[:200]
        episodes.append(rec)
        with open(ep_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[timed_eval] episode {rec['episode_seq']}: success={rec.get('success')} "
              f"wall={rec['wall_s']}s steps={rec.get('steps')}", flush=True)
        return ret

    lerobot_eval.rollout = timed_rollout

    sys.argv = ["lerobot-eval", *eval_args]
    try:
        lerobot_eval.main()
    finally:
        steady = times[1:] if len(times) > 1 else times  # first call includes compile / cache warm-up
        res = {
            "n_chunk_calls": len(times),
            "first_call_s": round(times[0], 4) if times else None,
            "chunk_latency_ms_mean": round(1000 * statistics.mean(steady), 1) if steady else None,
            "chunk_latency_ms_median": round(1000 * statistics.median(steady), 1) if steady else None,
            "chunk_latency_ms_p90": round(1000 * sorted(steady)[int(0.9 * (len(steady) - 1))], 1) if steady else None,
            "n_episodes_timed": len(episodes),
            "episode_wall_s_mean": round(statistics.mean(e["wall_s"] for e in episodes), 2) if episodes else None,
            "episode_wall_s_median": round(statistics.median(e["wall_s"] for e in episodes), 2) if episodes else None,
            "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "gpu": torch.cuda.get_device_name(0),
        }
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print("[timed_eval]", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
