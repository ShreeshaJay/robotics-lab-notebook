"""Write specs/<dataset>.json: one shared threshold template + per-dataset schema, bounds and frozen scales."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_audit.profiles import get_profile, ROOT
from lerobot_audit.load import load_episodes
from lerobot_audit import metrics as M

TEMPLATE = {  # shared vendor-gate thresholds, identical for every dataset
    "max_jump_rate": 0.02, "min_jumps": 3, "min_gripper_sign_corr": 0.0, "min_task_episodes": 5, "max_task_knn_flags": 0,
    "max_idle_frac": 0.25, "min_episode_s": 2.0, "max_episode_s": 600.0, "allowed_typo_words": [],
}
for name in sys.argv[1:]:
    p = get_profile(name)
    info = json.loads((p.root / "meta" / "info.json").read_text())
    stats = json.loads((p.root / "meta" / "stats.json").read_text())
    sc = M.compute_scales(p, load_episodes(p))
    spec = dict(dataset=name, **TEMPLATE,
                required_features={k: info["features"][k]["shape"] for k in (p.action_key, p.state_key)},
                fps=info["fps"], action_min=stats[p.action_key]["min"], action_max=stats[p.action_key]["max"],
                scales=sc.to_json(), scales_source=f"computed from the clean local slice of {p.repo}@{p.revision[:8]}")
    (ROOT / "specs").mkdir(exist_ok=True)
    (ROOT / "specs" / f"{name}.json").write_text(json.dumps(spec, indent=1))
    print(name, "spec written")
