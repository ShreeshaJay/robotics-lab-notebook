import sys
from huggingface_hub import HfApi
api = HfApi()
repo, rev = sys.argv[1], sys.argv[2]
for sub in ["meta", "data"]:
    files = [f for f in api.list_repo_tree(repo, repo_type="dataset", revision=rev, path_in_repo=sub, recursive=True) if hasattr(f, "size")]
    tot = sum(f.size for f in files)
    print(sub, len(files), f"{tot/1e9:.2f} GB")
    for f in files[:6]: print("  ", f.path, f"{f.size/1e6:.1f} MB")
    if len(files) > 6: print("   ...", files[-1].path, f"{files[-1].size/1e6:.1f} MB")
