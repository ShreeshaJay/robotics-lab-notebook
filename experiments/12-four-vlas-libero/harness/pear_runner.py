"""PEAR (mit-han-lab/Qwen3.8-27B-PEAR) on our fixed LIBERO-10 episodes, against a local vLLM OpenAI server.

Faithful to PEAR's LIBERO environment family (pear@2a81b2ee, pear/envs/libero/env.py, pear/envs/tool.py,
pear/agent_loop/agent_loop.py), re-implemented without Ray/verl:
- system prompt = pear/envs/system_prompt.md with the LIBERO ENVIRONMENT section and the instruction, with
  "left"/"right" swapped (the frames are rotated 180 degrees, as PEAR does);
- each user turn = the arm/gripper quantities as JSON + the reminder, then "[agentview]" frame, "[wrist]" frame,
  512x512 RGB rotated 180 degrees, sent losslessly (PNG);
- the answer's last ```json block is the action, carried out by PEAR's own MotionController (vendored verbatim);
- the whole conversation, reasoning included (preserve_thinking), stays in context; xhigh effort (the chat
  template's default, the paper's default); temperature 1.0, top_p 0.95, top_k -1 (pear/configs/embodied.yaml);
- the episode ends on success, after 520 native steps, on an unusable answer, or when the response budget of
  256,000 tokens after the first prompt (generated + injected) runs out.
Episode = (task t, init state i), seeded and settled as libero_common.reset_episode does for every model.

CLI (inside the container): python pear_runner.py <specs.json> <out_dir> <concurrency> <base_url>
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import traceback
import zlib
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESPONSE_BUDGET = 256_000
MAX_MODEL_LEN = 260_096  # 4,096 prompt + 256,000 response, as examples/eval/eval_libero.sh
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."
JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
SCHEMA = {"arm": ("x", "y", "z", "roll", "pitch", "yaw"), "gripper": ("width",)}
MIRROR = {"left": "right", "right": "left"}
FRAME = 512


def system_prompt(instruction: str) -> str:
    template = (HERE / "pear_vendor" / "system_prompt.md").read_text()
    env = (HERE / "pear_vendor" / "environment_prompt.md").read_text().strip()
    return template.replace("{environment}", f"## ENVIRONMENT\n\n{env}\n\n").replace("{instruction}", instruction)


def mirrored(instruction: str) -> str:
    return re.sub(r"\b(left|right)\b", lambda m: MIRROR[m.group(1)], instruction)


def check_action(action) -> str | None:
    if not isinstance(action, dict) or set(action) != set(SCHEMA):
        return "parts"
    for part, qs in action.items():
        if not isinstance(qs, dict) or set(qs) != set(SCHEMA[part]):
            return f"quantities of {part}"
        for v in qs.values():
            if type(v) not in (int, float) or not abs(v) <= sys.float_info.max:
                return "non-finite"
    return None


def png_url(arr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@contextmanager
def no_render(scene):
    """PEAR's _no_render: the controller's native steps skip the camera renders (their frames are never shown)."""
    import numpy as np

    sim = scene.sim
    render = sim.render

    def noop(width=None, height=None, *, camera_name=None, depth=False, mode="offscreen", device_id=-1, segmentation=False):
        h, w = height or 1, width or 1
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        return (frame, np.zeros((h, w), dtype=np.float32)) if depth else frame

    sim.render = noop
    try:
        yield
    finally:
        sim.render = render


class TrackedScene:
    """Proxy of the OffScreenRenderEnv for PEAR's MotionController: every native step updates the step tracker, and
    the step reports done once the default goal or the alternative (counterfactual) goal holds."""

    def __init__(self, scene, tracker, step_log):
        self._scene, self._tracker, self._log = scene, tracker, step_log

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self, action):
        obs, reward, done, info = self._scene.step(action)
        self._tracker.update(self._scene)
        self._log.append(action)
        stop = self._tracker.default_done_step is not None or self._tracker.alt_done_step is not None
        return obs, (1.0 if self._tracker.default_done_step is not None else reward), bool(done or stop), info


class DryClient:
    """Stands in for the server in a CPU-only check of the episode loop: always moves the tool 1 cm down."""

    class _NS(dict):
        __getattr__ = dict.get

    def __init__(self):
        self.chat = self._NS(completions=self._NS(create=self.create))
        self.n = 0

    def create(self, **kw):
        self.n += 1
        fence = "`" * 3
        act = {"arm": {"x": 0, "y": 0, "z": -1, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": -1}}
        ans = "ok" + chr(10) + fence + "json" + chr(10) + json.dumps(act) + chr(10) + fence
        NS = self._NS
        n_img = sum(1 for m in kw["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url")
        return NS(choices=[NS(message=NS(content=ans, reasoning_content="dry"), finish_reason="stop")],
                  usage=NS(prompt_tokens=1000 + 600 * n_img, completion_tokens=50, prompt_tokens_details=None))


def run_episode(spec: dict, out_dir: str, base_url: str) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    from openai import OpenAI

    import libero_common as lc
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from pear_vendor.controller import MotionController

    task_id, ep = spec["task_id"], spec["episode"]
    suite = benchmark.get_benchmark_dict()[lc.SUITE]()
    task = suite.get_task(task_id)
    bddl = os.path.join(lc.bddl_path(""), task.problem_folder, task.bddl_file)
    scene = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=FRAME, camera_widths=FRAME)
    instr = spec.get("instruction") or task.language
    alt_goal = lc.parse_goal(spec["alt_bddl"]) if spec.get("alt_bddl") else None
    t_start = time.time()
    obs, reset_info = lc.reset_episode(scene, lc.init_states(suite, task_id), ep,
                                       lc.PERTURBATION if spec.get("perturb") else None)
    tracker = lc.StepTracker(task_id, lc.inner(scene).parsed_problem["goal_state"], alt_goal)
    tracker.start(scene)
    native: list = []
    proxy = TrackedScene(scene, tracker, native)

    def frames(observation):
        o = scene.env._get_observations(force_update=True)
        return [np.ascontiguousarray(o[k][::-1, ::-1], dtype=np.uint8) for k in ("agentview_image", "robot0_eye_in_hand_image")]

    def user_turn(observation):
        state = MotionController.state(scene, observation)
        imgs = frames(observation)
        return [{"type": "text", "text": json.dumps(state) + "\n\n" + REMINDER},
                {"type": "text", "text": "[agentview]"}, {"type": "image_url", "image_url": {"url": png_url(imgs[0])}},
                {"type": "text", "text": "[wrist]"}, {"type": "image_url", "image_url": {"url": png_url(imgs[1])}}]

    client = DryClient() if base_url == "dry" else OpenAI(base_url=base_url, api_key="unused", timeout=3600, max_retries=0)
    messages = [{"role": "system", "content": system_prompt(mirrored(instr))}, {"role": "user", "content": user_turn(obs)}]
    turns, end_reason, p0 = [], None, None
    sim_s = 0.0
    while True:
        if p0 is None:
            max_tokens = RESPONSE_BUDGET
        else:
            last = turns[-1]["prompt_tokens"] + turns[-1]["completion_tokens"]
            max_tokens = min(RESPONSE_BUDGET - (last - p0), MAX_MODEL_LEN - last - 1200)
            if max_tokens <= 0:
                end_reason = "budget"
                break
        t0 = time.time()
        resp = client.chat.completions.create(
            model="pear", messages=messages, temperature=1.0, top_p=0.95,
            max_tokens=max_tokens,
            seed=zlib.crc32(f"{spec.get('condition','main')}|{task_id}|{ep}|{len(turns)}".encode()) % (2**31),
            extra_body={"top_k": -1, "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True,
                                                                "reasoning_effort": spec.get("effort", "xhigh")}},
        )
        gen_s = time.time() - t0
        choice = resp.choices[0]
        answer = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""
        if p0 is None:
            p0 = resp.usage.prompt_tokens
        turn = {"turn": len(turns), "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                "cached_tokens": getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", None),
                "gen_s": round(gen_s, 2), "finish_reason": choice.finish_reason, "reasoning_chars": len(reasoning),
                "answer_tail": answer[-400:]}
        turns.append(turn)
        messages.append({"role": "assistant", "content": answer, "reasoning_content": reasoning})
        if turn["prompt_tokens"] + turn["completion_tokens"] - p0 >= RESPONSE_BUDGET or choice.finish_reason == "length":
            end_reason = "budget"
            break
        blocks = JSON_BLOCK.findall(answer)
        action = None
        if blocks:
            try:
                action = json.loads(blocks[-1])
            except json.JSONDecodeError:
                action = None
        if action is None or check_action(action):
            end_reason = "parse"
            turn["parse_error"] = "no json block" if action is None else check_action(action)
            break
        turn["action"] = action
        t1 = time.time()
        with no_render(scene):
            obs, acts, rewards, done, outcome = MotionController(proxy, obs, action).move(lc.MAX_STEPS - len(native))
        sim_s += time.time() - t1
        turn.update({"executed": len(acts), "outcome": outcome, "steps_total": len(native)})
        if done or len(native) >= lc.MAX_STEPS:
            end_reason = "env" if (tracker.default_done_step or tracker.alt_done_step) else "steps"
            break
        messages.append({"role": "user", "content": user_turn(obs)})
        # the next user turn's tokens count against the budget too; the server reports them in the next prompt
    scene.close()
    success = tracker.default_done_step is not None
    rec = {**spec, "model": "pear", "instruction_used": instr, "instruction_sent": mirrored(instr), "success": success,
           "steps": len(native), "end_reason": end_reason, "turns": len(turns), "wall_s": round(time.time() - t_start, 1),
           "sim_s": round(sim_s, 1), "gen_s_total": round(sum(t["gen_s"] for t in turns), 1),
           "completion_tokens": sum(t["completion_tokens"] for t in turns),
           "final_context_tokens": turns[-1]["prompt_tokens"] + turns[-1]["completion_tokens"],
           "response_tokens_used": turns[-1]["prompt_tokens"] + turns[-1]["completion_tokens"] - p0,
           "turn_log": turns, "tracker": tracker.summary(), "reset_info": reset_info,
           "outcome": lc.episode_outcome(tracker) if alt_goal is not None else None}
    # full conversation text (no images) for later reading, outside the small results
    convo = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
              [p for p in m["content"] if p["type"] == "text"], "reasoning": m.get("reasoning_content")} for m in messages]
    tag = f"{spec.get('condition','main')}_t{task_id}_e{ep}"
    Path(out_dir, "convos").mkdir(parents=True, exist_ok=True)
    import gzip

    with gzip.open(Path(out_dir, "convos", f"{tag}.json.gz"), "wt") as fh:
        json.dump(convo, fh)
    return rec


def _safe(spec, out_dir, base_url):
    try:
        return run_episode(spec, out_dir, base_url)
    except Exception as e:
        return {**spec, "model": "pear", "error": repr(e)[:500], "traceback": traceback.format_exc()[-3000:]}


def main() -> None:
    specs_path, out_dir, conc, base_url = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    specs = json.loads(Path(specs_path).read_text())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / "episodes.jsonl"
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=conc, mp_context=mp.get_context("spawn")) as pool, open(out, "a") as fh:
        futs = {pool.submit(_safe, s, out_dir, base_url): s for s in specs}
        for f in as_completed(futs):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[pear] t{rec['task_id']} e{rec['episode']} {rec.get('condition')} success={rec.get('success')} "
                  f"steps={rec.get('steps')} turns={rec.get('turns')} end={rec.get('end_reason')} "
                  f"tok={rec.get('completion_tokens')} wall={rec.get('wall_s')} err={rec.get('error')}", flush=True)


if __name__ == "__main__":
    main()
