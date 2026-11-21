"""Read a local LeRobot v3 dataset folder (meta/ + data/ parquet). No lerobot dependency, no videos."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .profiles import Profile


@dataclass
class Meta:
    info: dict
    tasks: pd.DataFrame        # columns: task (str), task_index (int)
    episodes: pd.DataFrame     # non-stats columns of meta/episodes, one row per episode

    @property
    def fps(self) -> float:
        return float(self.info["fps"])


@dataclass
class Episode:
    index: int
    task_index: int
    t: np.ndarray              # timestamps (s), as stored
    frame_index: np.ndarray
    global_index: np.ndarray
    action: np.ndarray         # (T, A) float64
    state: np.ndarray          # (T, S) float64
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)


def load_meta(root: Path) -> Meta:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    # LeRobot v3 stores the task string as the pandas index (named "task" or unnamed).
    tasks = tasks.reset_index()
    tasks = tasks.rename(columns={tasks.columns[0]: "task"})[["task", "task_index"]]
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    cols = [c for c in pq.read_schema(files[0]).names if not c.startswith(("stats/", "videos/"))]
    episodes = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    return Meta(info=info, tasks=tasks, episodes=episodes)


def _list_col(table: pa.Table, name: str) -> np.ndarray:
    col = table.column(name).combine_chunks()
    if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type) or pa.types.is_large_list(col.type):
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        return flat.reshape(len(col), -1)
    return col.to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(col), -1)


def _chunk_file(f: Path) -> tuple[int, int]:
    return int(f.parent.name.split("-")[-1]), int(f.stem.split("-")[-1])


def referenced_files(meta: Meta) -> set[tuple[int, int]] | None:
    e = meta.episodes
    if "data/file_index" not in e or "data/chunk_index" not in e:
        return None
    return set(zip(e["data/chunk_index"].astype(int), e["data/file_index"].astype(int)))


def data_files(profile: Profile, meta: Meta | None = None) -> tuple[list[Path], list[Path]]:
    """(files to audit, local files that meta/episodes never points to). Unreferenced files are not loaded:
    a LeRobot loader would never read them, and their episode ids can collide with real ones."""
    files = sorted((Path(profile.root) / "data").rglob("*.parquet"))
    if profile.data_files:
        files = [f for f in files if f.name in profile.data_files]
    ref = referenced_files(meta) if meta is not None else None
    if ref is None:
        return files, []
    return [f for f in files if _chunk_file(f) in ref], [f for f in files if _chunk_file(f) not in ref]


def load_episodes(profile: Profile, max_episodes: int | None = None, meta: Meta | None = None) -> list[Episode]:
    """Load every episode in the local (referenced) data files, keeping on-disk row order within each episode."""
    eps: list[Episode] = []
    meta = meta or load_meta(profile.root)
    want = ["timestamp", "frame_index", "episode_index", "index", "task_index", profile.action_key, profile.state_key]
    if profile.eef_pos and profile.eef_pos[0] not in want:
        want.append(profile.eef_pos[0])
    if profile.success_key:
        want.append(profile.success_key)
    want += profile.extra_columns
    for f in data_files(profile, meta)[0]:
        names = pq.read_schema(f).names
        cols = [c for c in want if c in names]
        tb = pq.read_table(f, columns=cols)
        epi = tb.column("episode_index").to_numpy()
        order = np.argsort(epi, kind="stable")
        epi = epi[order]
        act = _list_col(tb, profile.action_key)[order]
        st = _list_col(tb, profile.state_key)[order]
        ts = tb.column("timestamp").to_numpy().astype(np.float64)[order]
        fi = tb.column("frame_index").to_numpy()[order]
        gi = tb.column("index").to_numpy()[order]
        ti = tb.column("task_index").to_numpy()[order]
        eef = _list_col(tb, profile.eef_pos[0])[order] if profile.eef_pos else None
        succ = tb.column(profile.success_key).to_numpy(zero_copy_only=False)[order] if profile.success_key in cols else None
        extras = {c: tb.column(c).to_numpy(zero_copy_only=False)[order] for c in profile.extra_columns if c in cols}
        bounds = np.flatnonzero(np.diff(epi)) + 1
        starts = np.r_[0, bounds]
        ends = np.r_[bounds, len(epi)]
        for s, e in zip(starts, ends):
            ex = {}
            if eef is not None:
                ex["eef_pos"] = eef[s:e][:, profile.eef_pos[1]]
            if succ is not None:
                ex["success"] = bool(succ[s])
            for c, v in extras.items():
                ex[c] = v[s]
            eps.append(Episode(index=int(epi[s]), task_index=int(ti[s]), t=ts[s:e], frame_index=fi[s:e],
                               global_index=gi[s:e], action=act[s:e], state=st[s:e], extra=ex))
            if max_episodes and len(eps) >= max_episodes:
                return eps
    return eps
