"""GR00T-N1.7-LIBERO (libero_10 checkpoint) on our fixed LIBERO-10 episodes, in one process.

Observation and action handling are copied from NVIDIA's own LIBERO wrapper (Isaac-GR00T@708617f5,
gr00t/eval/sim/LIBERO/libero_env.py): 256x256 frames rotated 180 degrees, state = eef xyz + axis-angle + 2 gripper
qpos, gripper action mapped [0,1] -> sign(2x-1) and inverted. The policy is NVIDIA's Gr00tPolicy wrapped by
Gr00tSimPolicyWrapper, exactly as gr00t/eval/rollout_policy.py builds it without the client/server split.

Differences from NVIDIA's rollout_policy.py (on purpose, to match the G0.5/SmolVLA baseline):
- LIBERO's fixed init state i with seed i and 10 no-op settle steps (NVIDIA's script uses env.reset() only).
- 520 steps budget (NVIDIA's README uses 720).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libero_common as lc  # noqa: E402


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def groot_obs(obs: dict, instruction: str) -> dict:
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    grip = obs["robot0_gripper_qpos"]

    def s(v):
        return np.asarray(v, dtype=np.float32).reshape(1, 1, -1)

    return {
        "video.image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])[None, None].astype(np.uint8),
        "video.wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])[None, None].astype(np.uint8),
        "state.x": s([xyz[0]]), "state.y": s([xyz[1]]), "state.z": s([xyz[2]]),
        "state.roll": s([rpy[0]]), "state.pitch": s([rpy[1]]), "state.yaw": s([rpy[2]]),
        "state.gripper": s(grip),
        "annotation.human.action.task_description": (instruction,),
    }


KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def to_native(chunk: dict) -> np.ndarray:
    a = np.concatenate([np.asarray(chunk[f"action.{k}"])[0] for k in KEYS], axis=-1).astype(np.float64)  # (H, 7)
    a[:, -1] = np.sign(2.0 * a[:, -1] - 1.0)  # normalize_gripper_action(binarize=True)
    a[:, -1] *= -1.0  # invert_gripper_action
    return a


class GrootRunner:
    def __init__(self, ckpt_dir: str):
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        self.torch = torch
        t0 = time.time()
        self.policy = Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA, model_path=ckpt_dir, device=0))
        self.load_s = round(time.time() - t0, 1)
        self.horizon = len(self.policy.policy.modality_configs["action"].delta_indices)
        from libero.libero import benchmark

        self.suite = benchmark.get_benchmark_dict()[lc.SUITE]()
        self._envs: dict = {}
        self.first_call = True

    def env(self, task_id: int):
        if task_id not in self._envs:
            from libero.libero.envs import OffScreenRenderEnv

            task = self.suite.get_task(task_id)
            bddl = os.path.join(lc.bddl_path(""), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            self._envs[task_id] = (env, lc.init_states(self.suite, task_id), task.language)
        return self._envs[task_id]

    def run(self, spec: dict) -> dict:
        torch = self.torch
        task_id, ep = spec["task_id"], spec["episode"]
        n_exec = spec.get("n_action_steps") or 8
        n_exec = self.horizon if n_exec == "full" else int(n_exec)
        env, init_states, default_instr = self.env(task_id)
        instr = spec.get("instruction") or default_instr
        alt_goal = lc.parse_goal(spec["alt_bddl"]) if spec.get("alt_bddl") else None
        t_start = time.time()
        obs, info = lc.reset_episode(env, init_states, ep, lc.PERTURBATION if spec.get("perturb") else None)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        goal = lc.inner(env).parsed_problem["goal_state"]
        if alt_goal is not None and not lc.resolvable(env, alt_goal):
            return {**spec, "error": "alt goal not resolvable in this scene"}
        tr = lc.StepTracker(task_id, goal, alt_goal)
        tr.start(env)
        chunk_ms, actions = [], []
        steps, success, first_ms, n_calls = 0, False, None, 0
        while steps < lc.MAX_STEPS:
            o = groot_obs(obs, instr)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk, _ = self.policy.get_action(o)
            torch.cuda.synchronize()
            dt = 1000 * (time.perf_counter() - t0)
            n_calls += 1
            if self.first_call:
                self.first_call = False
                first_ms = dt
            else:
                chunk_ms.append(round(dt, 1))
            native = to_native(chunk)
            for k in range(min(n_exec, len(native))):
                obs, _, _, _ = env.step(native[k].tolist())
                actions.append(native[k])
                steps += 1
                tr.update(env)
                if tr.default_done_step is not None or tr.alt_done_step is not None or steps >= lc.MAX_STEPS:
                    break
            if tr.default_done_step is not None or tr.alt_done_step is not None:
                break
        success = tr.default_done_step is not None
        acts = np.asarray(actions)
        jerk = float(np.mean(np.linalg.norm(np.diff(acts[:, :6], axis=0), axis=1))) if len(acts) > 2 else None
        rec = {
            **spec, "model": "groot", "instruction_used": instr, "success": bool(success), "steps": steps,
            "wall_s": round(time.time() - t_start, 2), "chunk_calls": n_calls,
            "chunk_ms": chunk_ms, "n_action_steps_used": n_exec, "chunk_horizon": self.horizon,
            "action_jerk": round(jerk, 5) if jerk is not None else None, "tracker": tr.summary(),
            "outcome": lc.episode_outcome(tr) if alt_goal is not None else None, "reset_info": info,
        }
        if first_ms is not None:
            rec["first_call_ms"] = round(first_ms, 1)
        return rec


def main() -> None:
    """CLI used inside the container: python groot_runner.py <ckpt_dir> <specs.json> <out.jsonl>"""
    ckpt, specs_path, out_path = sys.argv[1:4]
    runner = GrootRunner(ckpt)
    specs = json.loads(Path(specs_path).read_text())
    with open(out_path, "a") as fh:
        for spec in specs:
            rec = runner.run(spec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[groot] {spec} success={rec.get('success')} steps={rec.get('steps')} wall={rec.get('wall_s')}", flush=True)
    torch = runner.torch
    print("[groot] vram", json.dumps({"torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                                      "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                                      "load_s": runner.load_s, "gpu": torch.cuda.get_device_name(0)}), flush=True)


if __name__ == "__main__":
    main()
