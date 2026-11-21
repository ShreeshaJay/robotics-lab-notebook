"""Vendor-delivery acceptance gate for a LeRobot v3 dataset.

    pytest tests/test_acceptance.py --dataset libero_10
    pytest tests/test_acceptance.py --dataset libero_10 --root data/libero_10_corrupt --spec libero_10

Each test is one check from lerobot_audit.acceptance; a failure message lists how many episodes are flagged."""
import pytest

CHECKS = ["schema", "frame_index_contiguous", "timestamps_regular", "meta_length_matches", "no_nan",
          "dynamic_continuity", "gripper_sign_convention", "action_bounds", "task_labels_clean", "task_min_episodes",
          "no_exact_duplicates", "no_near_duplicates", "task_trajectory_consistency", "idle_fraction", "episode_length"]


@pytest.mark.parametrize("name", CHECKS)
def test_check(gate, name):
    if name not in gate:
        pytest.skip(f"{name} not applicable to this dataset")
    c = gate[name]
    assert c.passed, f"{name}: {c.detail} First flagged: {c.flagged[:10]}"
