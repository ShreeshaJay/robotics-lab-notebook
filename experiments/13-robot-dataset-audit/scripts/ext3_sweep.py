"""Ext 3b: harder single-fault variants on LIBERO-10 to find where the gate stops catching faults."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_audit.acceptance import run_gate
from lerobot_audit.corrupt import make_single_fault_copy
from lerobot_audit.profiles import DATA, ROOT

VARIANTS = [("drop_frames_renumbered", p, "dynamic_continuity") for p in (0.01, 0.02, 0.05)] + \
           [("mislabel_confusable", p, "task_trajectory_consistency") for p in (0, 1, 2)] + \
           [("duplicate_near", p, "no_near_duplicates") for p in (0.01, 0.02, 0.05, 0.10)] + \
           [("duplicate_time_shift", p, "no_near_duplicates") for p in (5, 20)] + \
           [("flip_gripper_partial", p, "gripper_sign_convention") for p in (0.25, 0.5)]
_, clean = run_gate("libero_10", log=lambda *a: None)
clean_flags = {n: set(c.flagged) for n, c in clean.items()}
rows = []
for kind, param, intended in VARIANTS:
    dst = DATA / "sweep" / f"{kind}_{param}"
    faulty = set(make_single_fault_copy(DATA / "libero_10", dst, kind, param, n=8, seed=1))
    _, gate = run_gate(f"libero_10_corrupt", dst, spec_name="libero_10", log=lambda *a: None)
    new_flags = {n: set(c.flagged) - clean_flags.get(n, set()) for n, c in gate.items()}
    any_new = set().union(*new_flags.values())
    rows.append({"fault": kind, "param": param, "intended_check": intended, "n_injected": len(faulty),
                 "caught_by_intended": len(faulty & set(gate[intended].flagged)),
                 "caught_by_any_new_flag": len(faulty & any_new), "gate_failed": not gate[intended].passed,
                 "false_pos_unfaulted": len(any_new - faulty)})
    print(rows[-1])
(ROOT / "results" / "corruption").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "corruption" / "sweep.json").write_text(json.dumps(rows, indent=2))
