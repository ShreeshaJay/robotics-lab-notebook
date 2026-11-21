"""Ext 3: run the acceptance gate on clean LIBERO-10 and on the corrupted copy; score what it catches and misses."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_audit.acceptance import run_gate
from lerobot_audit.corrupt import make_corrupt_copy
from lerobot_audit.profiles import ROOT, DATA

INTENDED = {"drop_frames_naive": "timestamps_regular", "drop_frames_renumbered": "dynamic_continuity",
            "flip_gripper_sign": "gripper_sign_convention", "mislabel_task": "task_trajectory_consistency",
            "typo_task_string": "task_labels_clean", "duplicate_exact": "no_exact_duplicates",
            "duplicate_near": "no_near_duplicates"}
out = ROOT / "results" / "corruption"; out.mkdir(parents=True, exist_ok=True)
manifest = make_corrupt_copy(DATA / "libero_10", DATA / "libero_10_corrupt", seed=0)
_, clean = run_gate("libero_10")
_, bad = run_gate("libero_10_corrupt", DATA / "libero_10_corrupt", spec_name="libero_10")
faulty_all = {e for v in manifest["faulty_episodes"].values() for e in v}
rows = []
for kind, eps in manifest["faulty_episodes"].items():
    eps = set(eps)
    by_check = {n: sorted(eps & set(c.flagged)) for n, c in bad.items() if eps & set(c.flagged)}
    caught_any = set().union(*by_check.values()) if by_check else set()
    intended = INTENDED[kind]
    rows.append({"fault": kind, "n_injected": len(eps), "intended_check": intended,
                 "caught_by_intended": len(eps & set(bad[intended].flagged)) if intended in bad else 0,
                 "caught_by_any": len(caught_any), "missed": sorted(eps - caught_any),
                 "checks_that_fired": {k: len(v) for k, v in by_check.items()}})
false_pos = {n: sorted(set(c.flagged) - faulty_all) for n, c in bad.items() if set(c.flagged) - faulty_all}
res = {"manifest": manifest,
       "clean_gate": {n: {"passed": c.passed, "n_flagged": len(c.flagged), "detail": c.detail} for n, c in clean.items()},
       "corrupt_gate": {n: {"passed": c.passed, "n_flagged": len(c.flagged), "detail": c.detail, "flagged": c.flagged} for n, c in bad.items()},
       "per_fault": rows, "false_positives_on_unfaulted_episodes": false_pos}
(out / "corruption_eval.json").write_text(json.dumps(res, indent=2))
print("clean:", {n: c.passed for n, c in clean.items()})
print("corrupt:", {n: c.passed for n, c in bad.items()})
for r in rows: print(r)
print("FP:", false_pos)
