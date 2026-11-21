"""Make a deliberately corrupted copy of a small LeRobot v3 dataset (used on LIBERO-10) with a ground-truth manifest."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .load import load_meta

N_PER_FAULT = 8


def _read_all(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def make_corrupt_copy(src: Path, dst: Path, seed: int = 0, drop_frac: float = 0.10) -> dict:
    src, dst = Path(src), Path(dst)
    rng = np.random.default_rng(seed)
    meta = load_meta(src)
    fps = meta.fps
    df = _read_all(src)
    eps = sorted(df.episode_index.unique())
    pick = rng.permutation(eps)
    faults = {"drop_frames_naive": pick[0:8], "drop_frames_renumbered": pick[8:16], "flip_gripper_sign": pick[16:24],
              "mislabel_task": pick[24:32], "duplicate_exact": pick[32:40], "duplicate_near": pick[40:44]}
    tasks = meta.tasks.copy()
    stove = tasks.task[tasks.task.str.contains("moka pots on the stove")].iloc[0]
    stove_idx = int(tasks.task_index[tasks.task == stove].iloc[0])
    typo_src = [e for e in pick[44:] if int(df.task_index[df.episode_index == e].iloc[0]) == stove_idx][:5]
    faults["typo_task_string"] = np.array(typo_src)
    typo_task = stove.replace("stove", "stvoe")
    typo_idx = int(tasks.task_index.max()) + 1
    tasks = pd.concat([tasks, pd.DataFrame({"task": [typo_task], "task_index": [typo_idx]})], ignore_index=True)

    groups = {e: g.copy() for e, g in df.groupby("episode_index", sort=True)}
    meta_len = {e: len(g) for e, g in groups.items()}  # what meta/episodes will claim
    new_task = {}
    gstd = np.stack(df["observation.state"]).std(0)
    astd = np.stack(df["action"]).std(0)
    for e in faults["drop_frames_naive"]:
        g = groups[e]
        k = int(len(g) * drop_frac)
        drop = rng.choice(np.arange(1, len(g) - 1), size=k, replace=False)
        groups[e] = g.drop(g.index[drop])          # frame_index/timestamp keep their gaps; meta length left stale
    for e in faults["drop_frames_renumbered"]:
        g = groups[e]
        k = int(len(g) * drop_frac)
        drop = rng.choice(np.arange(1, len(g) - 1), size=k, replace=False)
        g = g.drop(g.index[drop]).copy()
        g["frame_index"] = np.arange(len(g))
        g["timestamp"] = (np.arange(len(g)) / fps).astype(np.float32)
        groups[e] = g
        meta_len[e] = len(g)                       # metadata made consistent: only the dynamics give it away
    for e in faults["flip_gripper_sign"]:
        g = groups[e]
        g["action"] = [np.r_[a[:-1], -a[-1]].astype(np.float32) for a in g["action"]]
    others = sorted(set(tasks.task_index) - {typo_idx})
    for e in faults["mislabel_task"]:
        g = groups[e]
        cur = int(g.task_index.iloc[0])
        t = int(rng.choice([x for x in others if x != cur]))
        g["task_index"] = t
        new_task[int(e)] = t
    for e in faults["typo_task_string"]:
        groups[e]["task_index"] = typo_idx
    next_id = max(eps) + 1
    copies = {}
    for kind in ("duplicate_exact", "duplicate_near"):
        for e in faults[kind]:
            g = groups[e].copy()
            if kind == "duplicate_near":
                g["observation.state"] = [(s + rng.normal(0, 0.005 * gstd)).astype(np.float32) for s in g["observation.state"]]
                g["action"] = [(a + rng.normal(0, 0.005 * astd)).astype(np.float32) for a in g["action"]]
            g["episode_index"] = next_id
            groups[next_id] = g
            meta_len[next_id] = len(g)
            copies[int(next_id)] = {"kind": kind, "source_episode": int(e)}
            next_id += 1
    _write_dataset(dst, groups, meta_len, tasks, meta.info)
    manifest = {"seed": seed, "drop_frac": drop_frac, "source": str(src),
                "faults": {k: [int(x) for x in v] for k, v in faults.items()},
                "duplicate_copies": copies, "mislabel_new_task": new_task, "typo_task": typo_task}
    # For duplicates the faulty episode is the copy, not the source
    manifest["faulty_episodes"] = {k: manifest["faults"][k] for k in faults if not k.startswith("duplicate")}
    for kind in ("duplicate_exact", "duplicate_near"):
        manifest["faulty_episodes"][kind] = [c for c, v in copies.items() if v["kind"] == kind]
    (dst / "corruption_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _write_dataset(dst: Path, groups: dict, meta_len: dict, tasks: pd.DataFrame, info: dict):
    out = pd.concat([groups[e] for e in sorted(groups)], ignore_index=True)
    out["index"] = np.arange(len(out))
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dst / "data" / "chunk-000" / "file-000.parquet")
    task_name = dict(zip(tasks.task_index, tasks.task))
    rows, start = [], 0
    for e in sorted(groups):
        ti = int(groups[e].task_index.iloc[0])
        n_rows = len(groups[e])
        rows.append({"episode_index": int(e), "tasks": [task_name[ti]], "length": int(meta_len[e]),
                     "data/chunk_index": 0, "data/file_index": 0,
                     "dataset_from_index": start, "dataset_to_index": start + n_rows})
        start += n_rows
    pd.DataFrame(rows).to_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    tasks.set_index("task").to_parquet(dst / "meta" / "tasks.parquet")
    info = dict(info)
    info.update(total_episodes=len(groups), total_frames=int(len(out)), total_tasks=len(tasks),
                splits={"train": f"0:{len(groups)}"})
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))


# LIBERO-10 tasks that share a scene or a motion pattern (harder mislabels)
CONFUSABLE = [("put both the alphabet soup and the tomato sauce in the basket",
               "put both the cream cheese box and the butter in the basket"),
              ("put both the alphabet soup and the tomato sauce in the basket",
               "put both the alphabet soup and the cream cheese box in the basket"),
              ("put the white mug on the left plate and put the yellow and white mug on the right plate",
               "put the white mug on the plate and put the chocolate pudding to the right of the plate")]


def make_single_fault_copy(src: Path, dst: Path, kind: str, param: float, n: int = 8, seed: int = 1) -> list[int]:
    """One fault type at one strength, for the sensitivity sweep. Returns the faulty episode ids."""
    src, dst = Path(src), Path(dst)
    rng = np.random.default_rng(seed)
    meta = load_meta(src)
    df = _read_all(src)
    groups = {e: g.copy() for e, g in df.groupby("episode_index", sort=True)}
    meta_len = {e: len(g) for e, g in groups.items()}
    tasks = meta.tasks.copy()
    eps = sorted(groups)
    if kind == "mislabel_confusable":
        t2i = dict(zip(tasks.task, tasks.task_index))
        a, b = CONFUSABLE[int(param)]
        pool = [e for e in eps if int(groups[e].task_index.iloc[0]) == t2i[a]]
        pick = rng.choice(pool, size=n, replace=False)
        for e in pick:
            groups[e]["task_index"] = t2i[b]
        faulty = [int(e) for e in pick]
    elif kind == "drop_frames_renumbered":
        pick = rng.choice(eps, size=n, replace=False)
        for e in pick:
            g = groups[e]
            k = max(1, int(round(len(g) * param)))
            drop = rng.choice(np.arange(1, len(g) - 1), size=k, replace=False)
            g = g.drop(g.index[drop]).copy()
            g["frame_index"] = np.arange(len(g))
            g["timestamp"] = (np.arange(len(g)) / meta.fps).astype(np.float32)
            groups[e] = g
            meta_len[e] = len(g)
        faulty = [int(e) for e in pick]
    elif kind == "duplicate_near":
        gstd = np.stack(df["observation.state"]).std(0)
        astd = np.stack(df["action"]).std(0)
        pick = rng.choice(eps, size=n, replace=False)
        nxt = max(eps) + 1
        faulty = []
        for e in pick:
            g = groups[e].copy()
            g["observation.state"] = [(s + rng.normal(0, param * gstd)).astype(np.float32) for s in g["observation.state"]]
            g["action"] = [(a + rng.normal(0, param * astd)).astype(np.float32) for a in g["action"]]
            g["episode_index"] = nxt
            groups[nxt] = g
            meta_len[nxt] = len(g)
            faulty.append(int(nxt))
            nxt += 1
    elif kind == "duplicate_time_shift":   # copy, drop the first `param` frames (a trimmed re-upload)
        pick = rng.choice(eps, size=n, replace=False)
        nxt = max(eps) + 1
        faulty = []
        for e in pick:
            g = groups[e].iloc[int(param):].copy()
            g["frame_index"] = np.arange(len(g))
            g["timestamp"] = (np.arange(len(g)) / meta.fps).astype(np.float32)
            g["episode_index"] = nxt
            groups[nxt] = g
            meta_len[nxt] = len(g)
            faulty.append(int(nxt))
            nxt += 1
    elif kind == "flip_gripper_partial":   # sign flipped on only the last `param` fraction of the episode
        pick = rng.choice(eps, size=n, replace=False)
        for e in pick:
            g = groups[e]
            cut = int(len(g) * (1 - param))
            acts = list(g["action"])
            for i in range(cut, len(acts)):
                acts[i] = np.r_[acts[i][:-1], -acts[i][-1]].astype(np.float32)
            g["action"] = acts
        faulty = [int(e) for e in pick]
    else:
        raise ValueError(kind)
    _write_dataset(dst, groups, meta_len, tasks, meta.info)
    return faulty
