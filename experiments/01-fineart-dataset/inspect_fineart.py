"""Inspect the schema and subtask labels of the FineART robot dataset (lerobot/FineART).

Downloads only metadata (meta/info.json, meta/stats.json, meta/tasks.parquet,
meta/episodes/...) and reads the first N episodes from ONE data parquet file
using HTTP range reads (only the needed row groups are fetched). No videos.

Usage:
    python -I inspect.py --cache-dir C:/path/outside/repo [--episodes 3]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "lerobot/FineART"
REVISION = "15746b76ddc21ffd38d73888fefe2e7a1496385b"  # commit inspected on 2026-11-07
META_FILES = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
]


def section(title):
    print(f"\n=== {title} ===")


def fetch_meta(cache_dir):
    paths = {}
    for f in META_FILES:
        paths[f] = Path(
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, cache_dir=cache_dir)
        )
    return paths


def describe_info(info):
    section("meta/info.json")
    for k in ["codebase_version", "robot_type", "fps", "total_episodes", "total_frames",
              "total_tasks", "chunks_size", "data_files_size_in_mb", "video_files_size_in_mb",
              "splits", "data_path", "video_path"]:
        print(f"{k}: {info.get(k)}")
    section("features")
    for name, spec in info["features"].items():
        extra = ""
        if spec.get("names") is not None:
            extra = f" names={spec['names']}"
        if spec.get("dtype") == "video":
            vi = spec.get("info", {})
            extra = f" codec={vi.get('video.codec')} fps={vi.get('video.fps')}"
        print(f"{name}: dtype={spec.get('dtype')} shape={spec.get('shape')}{extra}")


def describe_tasks(tasks):
    section("meta/tasks.parquet")
    print("columns:", list(tasks.columns), "rows:", len(tasks))
    col = "task" if "task" in tasks.columns else tasks.columns[0]
    for t in tasks[col].head(8):
        print("  task:", t)


def describe_episodes(eps):
    section("meta/episodes (per-episode metadata)")
    print("rows:", len(eps))
    print("columns:", [c for c in eps.columns if not c.startswith("stats/")])
    print("stats/* columns:", sum(c.startswith("stats/") for c in eps.columns))
    if "length" in eps.columns:
        sec = eps["length"] / 30.0
        print(f"episode length (s): mean={sec.mean():.1f} median={sec.median():.1f} "
              f"min={sec.min():.1f} max={sec.max():.1f}")
    if "tasks" in eps.columns:
        per_task = Counter(t for ts in eps["tasks"] for t in ts)
        print("distinct tasks referenced:", len(per_task))
        print("top 5 tasks by episode count:", per_task.most_common(5))
        print("bottom 3 tasks by episode count:", per_task.most_common()[-3:])
    if "subtask_count" in eps.columns:
        sc = eps["subtask_count"]
        print(f"subtask_count: total={int(sc.sum())} mean={sc.mean():.1f} median={sc.median():.0f} "
              f"min={sc.min()} max={sc.max()} episodes with 0={int((sc == 0).sum())}")
    for c in eps.columns:
        if any(k in c for k in ("language", "annot", "success", "quality", "summary")):
            print(f"annotation-like column {c!r} example:", eps[c].iloc[0])


def sample_labels_across_tasks(eps, keywords):
    """For one episode per keyword-matched task, read a single row group (range read)
    from that episode's data file and print its full subtask list."""
    fs = HfFileSystem()
    section("label lists sampled across tasks (one row group each)")
    first_words = Counter()
    for kw in keywords:
        hits = eps[eps["tasks"].apply(lambda ts: any(kw.lower() in t.lower() for t in ts))]
        if hits.empty:
            print(f"[{kw}] no task match")
            continue
        ep = hits.iloc[len(hits) // 2]
        path = (f"datasets/{REPO}@{REVISION}/data/chunk-{int(ep['data/chunk_index']):03d}/"
                f"file-{int(ep['data/file_index']):03d}.parquet")
        with fs.open(path, "rb", block_size=1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            # Row-group statistics are absent in some files, so locate the episode's
            # row inside its file from meta/episodes offsets (no extra data reads).
            same_file = eps[(eps["data/chunk_index"] == ep["data/chunk_index"])
                            & (eps["data/file_index"] == ep["data/file_index"])]
            first_row = int(ep["dataset_from_index"] - same_file["dataset_from_index"].min())
            start = 0
            for rg in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                if start <= first_row < start + n:
                    tbl = pf.read_row_group(rg, columns=["episode_index", "language_persistent"])
                    break
                start += n
        df = tbl.to_pandas()
        rows = sorted(df[df["episode_index"] == ep["episode_index"]]["language_persistent"].iloc[0],
                      key=lambda r: r["timestamp"])
        print(f"\n[{kw}] episode {int(ep['episode_index'])} task={ep['tasks'][0]!r} "
              f"length={ep['length'] / 30:.0f}s subtasks={len(rows)}")
        for r in rows:
            print(f"  {r['timestamp']:7.1f}s  {r['content']}")
            first_words[r["content"].split()[0].lower()] += 1
    print("\nfirst-word (verb) counts in sampled labels:", first_words.most_common(15))


def read_first_episodes(n_episodes):
    """Read only the row groups covering the first n episodes of data file-000."""
    fs = HfFileSystem()
    path = f"datasets/{REPO}@{REVISION}/data/chunk-000/file-000.parquet"
    with fs.open(path, "rb", block_size=4 * 1024 * 1024) as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        section("data/chunk-000/file-000.parquet (footer only)")
        print("rows:", md.num_rows, "row groups:", md.num_row_groups)
        print("arrow schema:")
        print(pf.schema_arrow)
        ep_col = pf.schema_arrow.get_field_index("episode_index")
        frames, seen = [], set()
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ep_col).statistics
            if st is not None and st.has_min_max and len(seen) >= n_episodes and st.min > max(seen):
                break
            tbl = pf.read_row_group(rg)
            df = tbl.to_pandas()
            frames.append(df)
            seen.update(df["episode_index"].unique().tolist())
            if len(seen) > n_episodes:
                break
        df = pd.concat(frames, ignore_index=True)
        keep = sorted(seen)[:n_episodes]
        return df[df["episode_index"].isin(keep)].reset_index(drop=True)


def describe_frames(df, task_by_idx):
    section("frames read")
    print("columns:", list(df.columns))
    print("frames:", len(df), "episodes:", sorted(df["episode_index"].unique().tolist()))
    row = df.iloc[0]
    for c in ["observation.state", "action"]:
        if c in df.columns:
            print(f"{c} len={len(row[c])} first frame={[round(float(x), 3) for x in row[c]]}")
    print("language_events non-empty frames:",
          int(sum(len(x) > 0 for x in df["language_events"])) if "language_events" in df else "n/a")

    all_labels = []
    for ep, g in df.groupby("episode_index"):
        lp = g["language_persistent"]
        same = all(json.dumps(list(x), default=str) == json.dumps(list(lp.iloc[0]), default=str)
                   for x in lp.iloc[:: max(1, len(lp) // 50)])
        rows = sorted(lp.iloc[0], key=lambda r: r["timestamp"])
        dur = float(g["timestamp"].max())
        task = task_by_idx.get(int(g["task_index"].iloc[0]))
        section(f"episode {ep}: task={task!r}  frames={len(g)}  duration={dur:.1f}s")
        print("label list identical on sampled frames:", same,
              "| styles:", Counter(r["style"] for r in rows),
              "| roles:", Counter(r["role"] for r in rows))
        for i, r in enumerate(rows):
            end = rows[i + 1]["timestamp"] if i + 1 < len(rows) else dur
            print(f"  {r['timestamp']:7.2f}s -> {end:7.2f}s ({end - r['timestamp']:5.1f}s)  {r['content']}")
            all_labels.append((end - r["timestamp"], len(r["content"].split())))
        first = rows[0]["timestamp"] if rows else None
        print("unlabeled lead-in (s):", first, "| other row keys:",
              {k: rows[0][k] for k in rows[0] if k not in ("content", "timestamp")} if rows else None)
    if all_labels:
        d = pd.Series([a for a, _ in all_labels])
        w = pd.Series([b for _, b in all_labels])
        section("subtask label stats (these episodes)")
        print(f"labels={len(all_labels)} seg duration median={d.median():.1f}s mean={d.mean():.1f}s "
              f"| words per label median={w.median():.0f} max={w.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="download cache (keep outside git)")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    paths = fetch_meta(args.cache_dir)
    info = json.loads(paths["meta/info.json"].read_text(encoding="utf-8"))
    describe_info(info)
    tasks = pd.read_parquet(paths["meta/tasks.parquet"])
    print("\ntasks.parquet raw: columns", list(tasks.columns), "index name", tasks.index.name)
    if "task" not in tasks.columns:  # v3.0 writes the task string as the pandas index
        tasks = tasks.reset_index().rename(columns={tasks.index.name or "index": "task"})
    describe_tasks(tasks)
    task_by_idx = dict(zip(tasks["task_index"].astype(int), tasks["task"]))
    eps = pd.read_parquet(paths["meta/episodes/chunk-000/file-000.parquet"])
    describe_episodes(eps)
    df = read_first_episodes(args.episodes)
    describe_frames(df, task_by_idx)
    sample_labels_across_tasks(eps, ["laundry", "cable", "insert", "towel", "breakfast", "tool"])


if __name__ == "__main__":
    main()
