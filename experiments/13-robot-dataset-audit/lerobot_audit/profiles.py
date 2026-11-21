"""Per-dataset profiles: which columns mean what. Everything else in the audit is dataset-agnostic."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass
class Gripper:
    name: str
    action_dim: int                 # index into the action vector
    state_dims: tuple[int, ...]     # (i,) -> state[i]; (i, j) -> state[i] - state[j] (finger width)
    action_closed_high: bool        # larger command = more closed
    state_closed_high: bool         # larger state value = more closed


@dataclass
class Profile:
    name: str
    repo: str
    revision: str
    license: str
    kind: str                       # "sim scripted/teleop", "lab teleop", "in-the-wild teleop"
    root: Path
    arm_state_dims: list[int]
    arm_action_dims: list[int]
    action_kind: str                # "delta" (velocity-like command) or "absolute" (position target)
    grippers: list[Gripper]
    action_key: str = "action"
    state_key: str = "observation.state"
    axis_angle_state_dims: list[int] | None = None   # unwrap these (axis-angle flips near pi)
    eef_pos: tuple[str, list[int]] | None = None     # (column, xyz dims) in metres, for a physical idle cross-check
    success_key: str | None = None                   # per-frame bool column marking a successful episode
    extra_columns: list[str] = field(default_factory=list)
    data_files: list[str] | None = None              # None = every local data parquet
    notes: list[str] = field(default_factory=list)


PROFILES: dict[str, Profile] = {
    "libero_10": Profile(
        name="libero_10", repo="lerobot/libero_10", revision="551d7d86f25edd0ffeda8b60053c15438cfd1d6a",
        license="apache-2.0 (HF tag)", kind="sim (human teleop demos, LIBERO)",
        root=DATA / "libero_10",
        arm_state_dims=[0, 1, 2, 3, 4, 5], arm_action_dims=[0, 1, 2, 3, 4, 5], action_kind="delta",
        axis_angle_state_dims=[3, 4, 5], eef_pos=("observation.state", [0, 1, 2]),
        grippers=[Gripper("gripper", 6, (6, 7), action_closed_high=True, state_closed_high=False)],
        notes=["info.json names state dims x,y,z,rx,ry,rz,rw,gripper; the values are xyz + axis-angle (3) + two finger "
               "joint positions (+0.039/-0.039 open). The names do not match the data."],
    ),
    "fineart": Profile(
        name="fineart", repo="lerobot/FineART", revision="15746b76ddc21ffd38d73888fefe2e7a1496385b",
        license="cc-by-4.0 (HF tag)", kind="lab teleop (bimanual ALOHA-style)",
        root=DATA / "fineart",
        arm_state_dims=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        arm_action_dims=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], action_kind="absolute",
        grippers=[Gripper("left", 6, (6,), action_closed_high=False, state_closed_high=False),
                  Gripper("right", 13, (13,), action_closed_high=False, state_closed_high=False)],
        notes=["Dims 5/12 are named *_gripper but span about -1.8..0.9 and behave like a rotary wrist joint; dims 6/13 "
               "(*_gripper_finger, 0..0.04) behave like finger opening. The audit treats 6/13 as the grippers."],
    ),
    "droid": Profile(
        name="droid", repo="lerobot/droid_1.0.1", revision="71da9dcba338f53889907ac5762a2ebc6c2e55ff",
        license="apache-2.0 (HF tag)", kind="in-the-wild teleop (Franka, many labs)",
        root=DATA / "droid",
        arm_state_dims=[0, 1, 2, 3, 4, 5, 6], arm_action_dims=[0, 1, 2, 3, 4, 5, 6], action_kind="absolute",
        eef_pos=("observation.state.cartesian_position", [0, 1, 2]),
        grippers=[Gripper("gripper", 7, (7,), action_closed_high=True, state_closed_high=True)],
        success_key="is_episode_successful",
        extra_columns=["language_instruction", "language_instruction_2", "language_instruction_3"],
    ),
}


def get_profile(name: str, root: str | Path | None = None) -> Profile:
    """Profile by name; `root` overrides the data folder (used for the corrupted copy)."""
    import copy
    base = name.replace("_corrupt", "")
    p = copy.deepcopy(PROFILES[base])
    if root is not None:
        p.root = Path(root)
        p.name = name
    return p
