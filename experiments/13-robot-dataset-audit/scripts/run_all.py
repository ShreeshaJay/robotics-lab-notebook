"""Audit each dataset, run its acceptance gate, write results/<name>/{audit.json, episodes.csv, gate.json, report.md, report.png}."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_audit.acceptance import run_gate
from lerobot_audit.audit import save
from lerobot_audit.profiles import ROOT, get_profile
from lerobot_audit.report import write_report

hub = json.loads((ROOT / "results" / "hub_file_consistency.json").read_text())
FINDINGS = {  # verified by hand during this run (see RESULTS.md for how)
    "fineart": ["Episodes 30424 and 30449 (task 88) are byte-identical in action and state; the meta-stats scan finds "
                "519 such copies across the full dataset.",
                "The publisher's own motion scan (`meta/sampling/sampling_report.json`) strips 3.1% of frames as idle "
                "(chunk mode, threshold 0.002 rad/frame), the same order as this audit's idle share."],
    "droid": [f"meta/episodes points to {hub['droid']['referenced_by_meta']} of {hub['droid']['hub_data_files']} data files on the Hub; "
              f"{hub['droid']['unreferenced_files']} files ({hub['droid']['unreferenced_gb']:.1f} GB, file-086 to file-155) are unreferenced. "
              "Sampled orphan file-154 reuses episode ids 94870+ with lengths that disagree with meta in 606/607 episodes. "
              "The audit skips unreferenced files.",
              "Episodes 80912 and 95021 are byte-identical (same instruction, building and collector); the meta-stats scan finds 96 copies in all.",
              "Column `date` equals `collector_id` in 100% of sampled episodes, and `task_category` holds the lab name (Gates, TRI, BWW): "
              "these metadata columns look shifted."],
}
for name in sys.argv[1:] or ["libero_10", "fineart", "droid"]:
    t = time.time()
    subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "make_specs.py"), name], check=True)
    res, gate = run_gate(name)
    out = ROOT / "results" / name
    save(res, out)
    (out / "gate.json").write_text(json.dumps({k: {"passed": c.passed, "detail": c.detail, "n_flagged": len(c.flagged),
                                                  "flagged_first_50": c.flagged[:50]} for k, c in gate.items()}, indent=2))
    write_report(res, get_profile(name).root, out, gate, FINDINGS.get(name))
    print(name, "done in", round(time.time() - t, 1), "s; gate:", sum(c.passed for c in gate.values()), "/", len(gate))
