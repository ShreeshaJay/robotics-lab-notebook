"""Shared LIBERO-10 episode logic for every model in Tutorial 1.

Used inside the Modal containers by the GR00T runner, the PEAR runner and the lerobot-eval patch, so all four models
see the same thing:

- Episode (task t, episode i) = LIBERO's fixed initial state ``init_states[i]`` of task t, numpy seed i, then 10 no-op
  settle steps (gripper open). This is what lerobot 0.7.0's LiberoEnv does with ``--env.init_states=true --seed=0`` and
  batch size 1, the setting of the G0.5 / SmolVLA baseline (experiments/05-g05-lerobot, rerun50).
- Budget: 520 native 20 Hz control steps (lerobot's libero_10 budget).
- Extension 3: every goal predicate of the task's BDDL is evaluated after every native step (``StepTracker``).
- Extension 2: an alternative instruction valid in the same scene, with its own BDDL goal evaluated in the same sim.
- Extension 4: camera shift + lighting change applied to the MuJoCo model after reset.

Standard library + numpy only at import time; LIBERO is imported lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:  # the local Modal client venv only needs the constants
    np = None

SUITE = "libero_10"
MAX_STEPS = 520
SETTLE_STEPS = 10
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

# Human-readable step labels per libero_10 task, as (label, index of the predicate in the BDDL goal), in the order
# the instruction states them. Tasks 0, 1, 7 ("put both X and Y in the basket") have no prescribed order.
STEP_LABELS: dict[int, list[tuple[str, int]]] = {
    0: [("alphabet soup in basket", 0), ("tomato sauce in basket", 1)],
    1: [("cream cheese in basket", 0), ("butter in basket", 1)],
    2: [("stove turned on", 0), ("moka pot on stove", 1)],
    3: [("bowl in bottom drawer", 1), ("drawer closed", 0)],
    4: [("white mug on left plate", 0), ("yellow-white mug on right plate", 1)],
    5: [("book in back compartment", 0)],
    6: [("white mug on plate", 0), ("pudding right of plate", 1)],
    7: [("alphabet soup in basket", 0), ("cream cheese in basket", 1)],
    8: [("moka pot 1 on stove", 0), ("moka pot 2 on stove", 1), ("stove on (already on at start)", 2)],
    9: [("mug in microwave", 0), ("microwave closed", 1)],
}
# Only tasks 3 and 9 are physically sequential (object into a container, then close it): closing an empty drawer or
# microwave is not progress. In every other task the goal predicates can be met in any order.
ORDERED_TASKS = {3, 9}
UNORDERED_TASKS = set(range(10)) - ORDERED_TASKS

# Extension 2: one alternative instruction per task that is valid in the same scene (same objects), taken from the
# LIBERO-90 / LIBERO-10 task that shares the scene, with a different target object or location. Task 8's scene has
# no such task (its LIBERO-90 siblings are a subset of its own goal or "turn off the stove", already true at start).
COUNTERFACTUALS: dict[int, tuple[str, str]] = {
    0: ("put both the cream cheese box and the butter in the basket",
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl"),
    1: ("put both the alphabet soup and the tomato sauce in the basket",
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"),
    2: ("put the frying pan on the stove", "libero_90/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.bddl"),
    3: ("put the wine bottle on the wine rack", "libero_90/KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl"),
    4: ("put the red mug on the left plate", "libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate.bddl"),
    5: ("pick up the book and place it in the front compartment of the caddy",
        "libero_90/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl"),
    6: ("put the red mug on the plate", "libero_90/LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate.bddl"),
    7: ("pick up the ketchup and put it in the basket",
        "libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl"),
    9: ("put the yellow and white mug to the front of the white mug",
        "libero_90/KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.bddl"),
}

# Extension 4: one combined perturbation. The fixed third-person camera moves 3 cm sideways and 2 cm up (world
# frame), and every light's diffuse intensity is halved with a warm tint. The wrist camera is untouched.
PERTURBATION = {"agentview_shift_m": [0.0, 0.03, 0.02], "light_diffuse_scale": [0.55, 0.5, 0.4]}


def write_libero_config(libero_pkg_dir: str, config_dir: str | None = None) -> str:
    """Writes LIBERO's config.yaml (LIBERO calls input() on import without one); returns its directory."""
    config_dir = config_dir or os.path.expanduser("~/.libero")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "config.yaml"), "w") as fh:
        fh.write(
            f"benchmark_root: {libero_pkg_dir}\nbddl_files: {libero_pkg_dir}/bddl_files\n"
            f"init_states: {libero_pkg_dir}/init_files\ndatasets: {libero_pkg_dir}/../datasets\n"
            f"assets: {libero_pkg_dir}/assets\n"
        )
    os.environ["LIBERO_CONFIG_PATH"] = config_dir
    return config_dir


def inner(env: Any) -> Any:
    """The BDDL problem env (has parsed_problem, object_states_dict, _eval_predicate) behind an OffScreenRenderEnv."""
    return env.env if hasattr(env, "env") and hasattr(env.env, "parsed_problem") else env


def bddl_path(rel: str) -> str:
    from libero.libero import get_libero_path

    return os.path.join(get_libero_path("bddl_files"), rel)


def parse_goal(rel_bddl: str) -> list:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    return robosuite_parse_problem(bddl_path(rel_bddl))["goal_state"]


def resolvable(env: Any, goal: list) -> bool:
    names = inner(env).object_states_dict
    return all(all(a in names for a in state[1:]) for state in goal)


def eval_goal(env: Any, goal: list) -> list[bool]:
    e = inner(env)
    return [bool(e._eval_predicate(state)) for state in goal]


def apply_perturbation(env: Any, spec: dict | None) -> dict:
    """Moves the agentview camera and dims/tints the lights in the live MuJoCo model; returns what changed."""
    if not spec:
        return {}
    sim = env.sim
    model = sim.model
    cam_id = model.camera_name2id("agentview")
    before = np.array(model.cam_pos[cam_id]).copy()
    model.cam_pos[cam_id] = before + np.asarray(spec["agentview_shift_m"])
    scale = np.asarray(spec["light_diffuse_scale"])
    model.light_diffuse[:] = np.asarray(model.light_diffuse) * scale
    sim.forward()
    return {"agentview_pos_before": before.round(4).tolist(), "agentview_pos_after": np.array(model.cam_pos[cam_id]).round(4).tolist(),
            "n_lights": int(model.nlight)}


def init_states(suite: Any, task_id: int) -> Any:
    """LIBERO's fixed initial states of a task. Its .init files are pickled numpy arrays from the LIBERO repo, which
    torch>=2.6's weights_only default refuses; load them as PEAR's env does (weights_only=False for this call)."""
    import torch

    orig = torch.load

    def permissive(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = permissive
    try:
        return suite.get_task_init_states(task_id)
    finally:
        torch.load = orig


def reset_episode(env: Any, init_states: Any, episode: int, perturb: dict | None = None) -> tuple[dict, dict]:
    """seed(i) -> reset -> set_init_state(init_states[i]) -> perturbation -> 10 no-op settle steps. Returns the last
    observation and a small info dict."""
    env.seed(episode)
    env.reset()
    obs = env.set_init_state(init_states[episode])
    info = {"perturbation": apply_perturbation(env, perturb)}
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    return obs, info


@dataclass
class StepTracker:
    """Evaluates the default goal (and an optional alternative goal) after every native step."""

    task_id: int
    goal: list
    alt_goal: list | None = None
    n_steps: int = 0
    initial: list[bool] = field(default_factory=list)
    ever: list[bool] = field(default_factory=list)
    first_step: list[int | None] = field(default_factory=list)
    max_simultaneous: int = 0
    alt_initial: list[bool] = field(default_factory=list)
    alt_ever: list[bool] = field(default_factory=list)
    alt_done_step: int | None = None
    default_done_step: int | None = None
    last: list[bool] = field(default_factory=list)
    alt_last: list[bool] = field(default_factory=list)

    def start(self, env: Any) -> None:
        self.initial = eval_goal(env, self.goal)
        self.ever = [False] * len(self.goal)
        self.first_step = [None] * len(self.goal)
        self.last = list(self.initial)
        if self.alt_goal is not None:
            self.alt_initial = eval_goal(env, self.alt_goal)
            self.alt_ever = [False] * len(self.alt_goal)
            self.alt_last = list(self.alt_initial)

    def update(self, env: Any) -> None:
        self.n_steps += 1
        now = eval_goal(env, self.goal)
        self.last = now
        for k, v in enumerate(now):
            if v and not self.ever[k]:
                self.ever[k], self.first_step[k] = True, self.n_steps
        self.max_simultaneous = max(self.max_simultaneous, sum(v and not i for v, i in zip(now, self.initial)))
        if all(now) and self.default_done_step is None:
            self.default_done_step = self.n_steps
        if self.alt_goal is not None:
            alt = eval_goal(env, self.alt_goal)
            self.alt_last = alt
            self.alt_ever = [a or b for a, b in zip(self.alt_ever, alt)]
            if all(alt) and self.alt_done_step is None:
                self.alt_done_step = self.n_steps

    def summary(self) -> dict:
        labels = STEP_LABELS[self.task_id]
        ordered = [self.last[idx] for _, idx in labels]
        counted = [v and not self.initial[idx] for v, (_, idx) in zip(ordered, labels)]
        n_done_end = sum(counted)
        if self.task_id in UNORDERED_TASKS:
            # order-free goals: the first step is whichever predicate was met first
            prefix = n_done_end
        else:
            prefix = 0
            for c in counted:
                if not c:
                    break
                prefix += 1
        out = {
            "n_predicates": len(self.goal),
            "predicates_initial": self.initial,
            "predicates_end": self.last,
            "predicates_ever": self.ever,
            "predicate_first_step": self.first_step,
            "steps_done_end": n_done_end,
            "ordered_prefix_done_end": prefix,
            "partial_credit": round((prefix if self.task_id in ORDERED_TASKS else n_done_end) / len(labels), 4),
            "max_simultaneous": self.max_simultaneous,
            "default_done_step": self.default_done_step,
        }
        if self.alt_goal is not None:
            out.update({"alt_initial": self.alt_initial, "alt_end": self.alt_last, "alt_ever": self.alt_ever,
                        "alt_done_step": self.alt_done_step})
        return out


def episode_outcome(tracker: StepTracker) -> str:
    """Extension 2 label: which task the robot executed."""
    alt = tracker.alt_done_step is not None
    default = tracker.default_done_step is not None
    if alt and default:
        return "both"
    return "instructed" if alt else ("default" if default else "neither")
