"""Compare the data parquet files present on the Hub with the files meta/episodes points to."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi
from lerobot_audit.load import load_meta, referenced_files
from lerobot_audit.profiles import PROFILES, ROOT
api = HfApi(); out = {}
for name, p in PROFILES.items():
    files = [f for f in api.list_repo_tree(p.repo, repo_type="dataset", revision=p.revision, path_in_repo="data", recursive=True) if hasattr(f, "size")]
    ref = referenced_files(load_meta(p.root))
    key = lambda f: (int(f.path.split("/")[1].split("-")[1]), int(f.path.split("/")[2].split("-")[1].split(".")[0]))
    orph = [f for f in files if key(f) not in ref]
    missing = sorted(ref - {key(f) for f in files})
    out[name] = {"hub_data_files": len(files), "hub_data_gb": sum(f.size for f in files) / 1e9, "referenced_by_meta": len(ref),
                 "unreferenced_files": len(orph), "unreferenced_gb": sum(f.size for f in orph) / 1e9,
                 "unreferenced_first_last": [orph[0].path, orph[-1].path] if orph else [], "referenced_but_missing": missing}
    print(name, json.dumps(out[name]))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "hub_file_consistency.json").write_text(json.dumps(out, indent=2))
