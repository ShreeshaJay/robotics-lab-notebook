"""Build notebooks/audit_dataset.ipynb (then execute it with nbclient)."""
import nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Audit a LeRobot-format robot dataset\n\nRuns `lerobot_audit` on a local LeRobot v3 folder (meta/ + data/ parquet, no videos). "
       "Download first with `python -I scripts/download.py <name>`. Change `DATASET` to `fineart` or `droid`."),
    code("import sys, json\nfrom pathlib import Path\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\nfrom lerobot_audit.profiles import get_profile\nfrom lerobot_audit.audit import run_audit\n"
         "from lerobot_audit.acceptance import run_checks, load_spec\nfrom lerobot_audit.report import write_report\nDATASET = 'libero_10'"),
    md("## 1. Audit\nEpisode lengths, idle time, smoothness, gripper events and re-grasps, duplicates, label hygiene, coverage, usable hours."),
    code("p = get_profile(DATASET)\nres = run_audit(p)\ns = res.summary\n"
         "{k: s[k] for k in ['slice', 'idle', 'smoothness', 'gripper', 'duplicates', 'usable']}"),
    md("## 2. Per-episode table (worst idle first)"),
    code("res.episodes.sort_values('idle_frac', ascending=False)[['episode_index', 'task', 'duration_s', 'idle_frac', 'sparc', 'n_close', 'n_regrasp', 'nn_dist']].head(10)"),
    md("## 3. Label hygiene and coverage"),
    code("{k: v for k, v in s['hygiene'].items() if not isinstance(v, (list, dict)) or len(str(v)) < 400}, s['coverage']"),
    md("## 4. Acceptance gate and one-page report"),
    code("import json\ninfo = json.loads((p.root / 'meta' / 'info.json').read_text())\nchecks = run_checks(res, load_spec(DATASET), info)\n"
         "{n: (c.passed, len(c.flagged)) for n, c in checks.items()}"),
    code("out = ROOT / 'results' / 'notebook' / DATASET\nwrite_report(res, p.root, out, checks)\nfrom IPython.display import Image, Markdown\n"
         "display(Image(filename=str(out / 'report.png')))\nMarkdown((out / 'report.md').read_text())"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "audit_dataset.ipynb")
print("notebook written")
