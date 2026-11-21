#!/usr/bin/env python3
"""`lerobot-eval` (lerobot 0.7.0) for G0.5 and SmolVLA, with the baseline's timing wrapper plus Tutorial 1's hooks.

Usage (inside the lerobot venv):  python lerobot_tut_eval.py <timings.json> <lerobot-eval args...>

Same as experiments/05-g05-lerobot/timed_eval.py (chunk latency of G05Policy._run_inference and
SmolVLAPolicy._get_action_chunk, per-episode JSONL, the G0.5 8->7 state-check workaround), plus patches on lerobot's
LiberoEnv driven by environment variables:
  T3_TRACK_OUT  JSONL path: one record per episode with every BDDL goal predicate after every step (extension 3)
  T3_CF=1       counterfactual instruction (libero_common.COUNTERFACTUALS) and its goal; the episode ends when either
                goal holds (extension 2)
  T3_PERTURB=1  camera shift + lighting change after reset (extension 4)
The physics, seeds, init states and settle steps are lerobot's own (identical to the baseline run).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_common as lc  # noqa: E402


def _libero_env_class():
    for mod in ("lerobot.envs.libero", "lerobot.sim.integrations.libero"):
        try:
            m = __import__(mod, fromlist=["LiberoEnv"])
            return m.LiberoEnv
        except (ImportError, AttributeError):
            continue
    raise ImportError("LiberoEnv not found in lerobot")


def patch_env() -> None:
    out = os.environ.get("T3_TRACK_OUT")
    cf = os.environ.get("T3_CF") == "1"
    perturb = os.environ.get("T3_PERTURB") == "1"
    if not (out or cf or perturb):
        return
    LiberoEnv = _libero_env_class()
    orig_init, orig_reset, orig_step, orig_close = LiberoEnv.__init__, LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

    def flush(self):
        tr = getattr(self, "_t3_tracker", None)
        if tr is None or getattr(self, "_t3_flushed", True):
            return
        self._t3_flushed = True
        if out:
            rec = {"task_id": self.task_id, "episode": self._t3_episode, "steps": tr.n_steps,
                   "success": tr.default_done_step is not None, "instruction_used": self.task_description,
                   "tracker": tr.summary(), "outcome": lc.episode_outcome(tr) if tr.alt_goal is not None else None,
                   "perturb_info": getattr(self, "_t3_pinfo", None)}
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._t3_alt = lc.COUNTERFACTUALS.get(self.task_id) if cf else None
        if self._t3_alt:
            self.task_description = self._t3_alt[0]

    def reset(self, seed=None, **kw):
        flush(self)
        episode = self.init_state_id  # the init state this reset uses (lerobot increments it inside reset)
        observation, info = orig_reset(self, seed=seed, **kw)
        self._t3_episode = int(episode)
        self._t3_pinfo = None
        if perturb:
            self._t3_pinfo = lc.apply_perturbation(self._env, lc.PERTURBATION)
            raw = self._env.env._get_observations(force_update=True)
            observation = self._format_raw_obs(raw)
        alt_goal = lc.parse_goal(self._t3_alt[1]) if self._t3_alt else None
        self._t3_tracker = lc.StepTracker(self.task_id, lc.inner(self._env).parsed_problem["goal_state"], alt_goal)
        self._t3_tracker.start(self._env)
        self._t3_flushed = False
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = orig_step(self, action)
        tr = self._t3_tracker
        tr.update(self._env)
        if tr.alt_goal is not None:
            stop = tr.alt_done_step is not None or tr.default_done_step is not None
            terminated = bool(stop)
            info["is_success"] = tr.alt_done_step is not None  # lerobot's metric = instructed task in this mode
        if terminated or tr.n_steps >= lc.MAX_STEPS:
            flush(self)
        return observation, reward, terminated, truncated, info

    def close(self):
        flush(self)
        return orig_close(self)

    LiberoEnv.__init__, LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = init, reset, step, close


def main() -> None:
    out_path, eval_args = sys.argv[1], sys.argv[2:]
    import torch

    from lerobot.policies.g05.modeling_g05 import G05Policy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    patch_env()
    times: list[float] = []
    episodes: list[dict] = []
    ep_path = out_path.rsplit(".json", 1)[0] + ".episodes.jsonl"
    open(ep_path, "w").close()

    def wrap(cls: type, name: str) -> None:
        orig = getattr(cls, name)

        def timed(self, *a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = orig(self, *a, **k)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            return res

        setattr(cls, name, timed)

    wrap(G05Policy, "_run_inference")
    wrap(SmolVLAPolicy, "_get_action_chunk")

    from lerobot.scripts import lerobot_eval

    orig_validate = lerobot_eval.validate_checkpoint

    def validate(descriptor, config, rename_map, **kw):
        try:
            orig_validate(descriptor, config, rename_map, **kw)
        except ValueError as e:
            if getattr(config, "type", None) == "g05" and "observation.state: expected (7,), got (8,)" in str(e):
                print("[tut_eval] tolerating known G0.5 LIBERO 8->7 state check:", e, flush=True)
                return
            raise

    lerobot_eval.validate_checkpoint = validate
    orig_rollout = lerobot_eval.rollout

    def timed_rollout(*a, **k):
        n0, t0 = len(times), time.perf_counter()
        ret = orig_rollout(*a, **k)
        rec = {"episode_seq": len(episodes), "wall_s": round(time.perf_counter() - t0, 2), "chunk_calls": len(times) - n0,
               "chunk_ms": [round(1000 * x, 1) for x in times[n0:]]}
        try:
            succ = ret["success"]
            rec["success"] = bool(succ.any(dim=1)[0].item())
            done = ret["done"]
            rec["steps"] = int(done.shape[1]) if not bool(done[0].any()) else int(done[0].float().argmax().item()) + 1
            act = ret.get("action")
            if act is not None:
                a = act[0, : rec["steps"], :6].float()
                rec["action_jerk"] = round(float((a[1:] - a[:-1]).norm(dim=1).mean().item()), 5) if len(a) > 2 else None
        except Exception as e:
            rec["parse_error"] = repr(e)[:200]
        episodes.append(rec)
        with open(ep_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[tut_eval] episode {rec['episode_seq']}: success={rec.get('success')} wall={rec['wall_s']}s steps={rec.get('steps')}", flush=True)
        return ret

    lerobot_eval.rollout = timed_rollout
    sys.argv = ["lerobot-eval", *eval_args]
    try:
        lerobot_eval.main()
    finally:
        steady = times[1:] if len(times) > 1 else times
        res = {
            "n_chunk_calls": len(times), "first_call_s": round(times[0], 4) if times else None,
            "chunk_latency_ms_median": round(1000 * statistics.median(steady), 1) if steady else None,
            "n_episodes_timed": len(episodes),
            "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "gpu": torch.cuda.get_device_name(0),
        }
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=2)
        print("[tut_eval]", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
