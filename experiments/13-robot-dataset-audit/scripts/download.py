"""Download metadata + a slice of data parquet files (no videos) for the three audited datasets.
Each dataset goes into its own directory under data/ (untrusted downloads, kept apart from code)."""
import json, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASETS = {
    "libero_10": dict(repo="lerobot/libero_10", rev="551d7d86f25edd0ffeda8b60053c15438cfd1d6a",
                      patterns=["meta/*", "meta/**", "data/**"]),  # whole dataset is ~10 MB of parquet
    "fineart": dict(repo="lerobot/FineART", rev="15746b76ddc21ffd38d73888fefe2e7a1496385b",
                    patterns=["meta/*", "meta/**"] + [f"data/chunk-000/file-{i:03d}.parquet" for i in (0, 52, 105, 157, 210)]),
    "droid": dict(repo="lerobot/droid_1.0.1", rev="71da9dcba338f53889907ac5762a2ebc6c2e55ff",
                  patterns=["meta/*", "meta/**"] + [f"data/chunk-000/file-{i:03d}.parquet" for i in (0, 42, 77, 85, 154)]),  # 154 is kept only as evidence: meta never references it
}
if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        d = DATASETS[n]; t = time.time()
        p = snapshot_download(d["repo"], repo_type="dataset", revision=d["rev"], allow_patterns=d["patterns"],
                              local_dir=DATA / n)
        size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file() and ".cache" not in f.parts)
        print(json.dumps({"name": n, "repo": d["repo"], "rev": d["rev"], "bytes": size, "seconds": round(time.time() - t, 1)}))
