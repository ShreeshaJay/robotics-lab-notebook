"""Run the full audit on one local LeRobot dataset folder and return a JSON-able summary + per-episode table."""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .hygiene import task_hygiene
from .load import data_files, load_episodes, load_meta
from .profiles import Profile

NEAR_DUP_D = 0.02        # RMS fingerprint distance (in arm-state std units) below which two episodes are near-duplicates
NEAR_DUP_LEN = 0.10      # ... and lengths within 10%
KNN_K = 5
MIN_TASK_EPISODES = 5


@dataclass
class AuditResult:
    summary: dict
    episodes: pd.DataFrame
    fingerprints: np.ndarray
    scales: M.Scales


def _gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _q(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {}


def pairwise_rms(F: np.ndarray) -> np.ndarray:
    sq = (F ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * F @ F.T, 0) / F.shape[1]
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2)


def meta_stats_duplicates(root: Path) -> dict:
    """Full-dataset exact-duplicate scan from per-episode stats in meta/episodes (no frame data needed):
    episodes with identical length and identical action/state mean+std are reported as likely copies."""
    import pyarrow.parquet as pq
    files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    want = ["episode_index", "length", "stats/action/mean", "stats/action/std", "stats/observation.state/mean"]
    if not all(c in pq.read_schema(files[0]).names for c in want):
        return {"available": False}
    e = pd.concat([pd.read_parquet(f, columns=want) for f in files], ignore_index=True)
    key = e.length.astype(str)
    for c in want[2:]:
        key = key + "|" + e[c].map(lambda v: ",".join(f"{x:.6g}" for x in np.asarray(v).ravel()))
    dup = key.duplicated(keep="first")
    groups = e[key.isin(key[dup])].groupby(key[key.isin(key[dup])]).episode_index.apply(list).tolist()
    return {"available": True, "n_episodes": int(len(e)), "duplicate_copies": int(dup.sum()),
            "duplicate_frac": float(dup.mean()), "duplicate_hours_frac": float(e.length[dup].sum() / e.length.sum()),
            "example_groups": groups[:10]}


def run_audit(p: Profile, scales: M.Scales | None = None, max_episodes: int | None = None, log=print) -> AuditResult:
    t0 = time.time()
    meta = load_meta(p.root)
    fps = meta.fps
    eps = load_episodes(p, max_episodes=max_episodes, meta=meta)
    used_files, orphan_files = data_files(p, meta)
    log(f"[{p.name}] loaded {len(eps)} episodes, {sum(e.n for e in eps):,} frames in {time.time() - t0:.1f}s")
    sc = scales or M.compute_scales(p, eps)
    task_of = dict(zip(meta.tasks.task_index.astype(int), meta.tasks.task.astype(str)))
    meta_len = dict(zip(meta.episodes.episode_index.astype(int), meta.episodes.length.astype(int)))
    rows = []
    for ep in eps:
        r = M.episode_metrics(p, ep, sc, fps)
        r["task"] = task_of.get(ep.task_index, None)
        r["meta_len_mismatch"] = int(meta_len.get(ep.index, -1) != ep.n)
        r["hash"] = M.episode_hash(ep)
        if "language_instruction" in ep.extra:
            r["language_empty"] = int(not str(ep.extra["language_instruction"]).strip())
        rows.append(r)
    df = pd.DataFrame(rows)
    log(f"[{p.name}] per-episode metrics done at {time.time() - t0:.1f}s")
    # --- duplicates ---
    df["exact_dup_copy"] = df.duplicated("hash", keep="first").astype(int)
    F = np.stack([M.fingerprint(p, ep, sc) for ep in eps])
    D = pairwise_rms(F)
    nn = D.min(1)
    df["nn_dist"] = nn
    lens = df.n_frames.values
    near_copy = np.zeros(len(df), int)
    for j in range(len(df)):
        if df.exact_dup_copy.iat[j]:
            continue
        i_cand = np.flatnonzero((D[j, :j] < NEAR_DUP_D) & (np.abs(lens[:j] - lens[j]) <= NEAR_DUP_LEN * lens[j]))
        if len(i_cand):
            near_copy[j] = 1
    df["near_dup_copy"] = near_copy
    # --- trajectory/task consistency (mislabel detector) ---
    tcount = df.task_index.value_counts()
    covered = df.task_index.map(tcount).ge(MIN_TASK_EPISODES).mean()
    knn_applicable = bool(len(tcount) >= 2 and covered >= 0.9)
    df["task_knn_flag"] = 0
    if knn_applicable:
        lab = df.task_index.values
        nbr = np.argsort(D, axis=1)[:, :KNN_K]
        for j in range(len(df)):
            labs = lab[nbr[j]]
            own = int((labs == lab[j]).sum())
            top_lab, top_n = Counter(labs).most_common(1)[0]
            if own <= 1 and top_lab != lab[j] and top_n >= 3:
                df.iat[j, df.columns.get_loc("task_knn_flag")] = 1
    # --- label hygiene + coverage over the full metadata (all episodes, not just the slice) ---
    if "tasks" in meta.episodes:
        ep_task = meta.episodes.tasks.map(lambda x: (list(x)[0] if len(x) else "") if x is not None else "")
    else:  # some v3 conversions omit meta/episodes.tasks; fall back to the data's task_index
        slice_task = dict(zip(df.episode_index, df.task.fillna("")))
        ep_task = meta.episodes.episode_index.map(lambda i: slice_task.get(int(i), ""))
        p.notes = p.notes + ["meta/episodes has no `tasks` column; per-episode tasks taken from data task_index."]
    ep_hours = meta.episodes.length / fps / 3600
    per_task_eps = Counter(ep_task)
    per_task_h = ep_hours.groupby(ep_task).sum()
    df["task_episodes_full"] = df.task.fillna("").map(lambda t: per_task_eps.get(t, 0))
    all_tasks = meta.tasks.task.astype(str).tolist()
    hyg = task_hygiene(all_tasks, {t: per_task_eps.get(t, 0) for t in all_tasks}, max_examples=100000)
    hyg["tasks_without_episodes"] = int(sum(1 for t in all_tasks if per_task_eps.get(t, 0) == 0))
    hyg["episode_task_not_in_table"] = int(sum(1 for t in per_task_eps if t and t not in set(all_tasks)))
    cnt = np.array([per_task_eps.get(t, 0) for t in all_tasks if t.strip()])
    cnt = cnt[cnt > 0]
    pr = cnt / cnt.sum()
    coverage = {
        "n_tasks_with_episodes": int(len(cnt)), "gini_episodes": _gini(cnt),
        "effective_n_tasks": float(np.exp(-(pr * np.log(pr)).sum())),
        "max_episodes_per_task": int(cnt.max()), "median_episodes_per_task": float(np.median(cnt)),
        "tasks_with_lt5_episodes": int((cnt < 5).sum()), "tasks_with_1_episode": int((cnt == 1).sum()),
        "top_task_share": float(cnt.max() / cnt.sum()),
        "top10pct_tasks_episode_share": float(np.sort(cnt)[::-1][: max(1, len(cnt) // 10)].sum() / cnt.sum()),
        "slice_tasks": int(df.task_index.nunique()),
    }
    dur_all = meta.episodes.length.values / fps
    lengths = {"n_episodes": int(len(dur_all)), "hours": float(dur_all.sum() / 3600), "mean_s": float(dur_all.mean()),
               "cv": float(dur_all.std() / dur_all.mean()), "lt_1s": int((dur_all < 1).sum()), **_q(dur_all)}
    # --- usable hours ---
    integ = (df.ts_gaps > 0) | (df.ts_nonmono > 0) | (df.fi_noncontig > 0) | (df.has_nan > 0) | (df.meta_len_mismatch > 0)
    failed = df.success.eq(False) if "success" in df and df.success.notna().any() else pd.Series(False, index=df.index)
    empty_task = df.task.fillna("").str.strip().eq("")
    dup = (df.exact_dup_copy > 0) | (df.near_dup_copy > 0)
    active_s = df.duration_s - df.idle_s
    too_short = active_s < 1.0
    retry = df.n_regrasp > 0
    raw_h = df.duration_s.sum() / 3600
    lost = {}
    keep = pd.Series(True, index=df.index)
    for name, mask in [("integrity", integ), ("failed_episode", failed), ("empty_task_label", empty_task),
                       ("duplicate", dup), ("too_short", too_short)]:
        m = keep & mask
        lost[name] = float(df.duration_s[m].sum() / 3600 / raw_h)
        keep &= ~mask
    lost["idle_time"] = float(df.idle_s[keep].sum() / 3600 / raw_h)
    usable_h = float(active_s[keep].sum() / 3600)
    strict_keep = keep & ~retry
    lost_retry = float(active_s[keep & retry].sum() / 3600 / raw_h)
    usable_strict_h = float(active_s[strict_keep].sum() / 3600)
    usable = {
        "definition": "usable = sum over episodes that pass integrity, success (if labelled), non-empty task, "
                      "not a duplicate copy and >=1 s of motion, of (duration - idle time in still runs >= 0.5 s). "
                      "Strict also drops episodes with a re-grasp.",
        "slice_raw_hours": float(raw_h), "usable_hours": usable_h, "usable_frac": usable_h / raw_h,
        "usable_strict_hours": usable_strict_h, "usable_strict_frac": usable_strict_h / raw_h,
        "lost_frac_by_reason": lost, "lost_frac_retry_strict": lost_retry,
        "full_dataset_hours": lengths["hours"],
        "full_dataset_usable_hours_extrapolated": lengths["hours"] * usable_h / raw_h,
    }
    meta_dups = meta_stats_duplicates(p.root)
    act = df[df.idle_s.notna()]
    tw_idle = float(act.idle_s.sum() / act.duration_s.sum())
    summary = {
        "dataset": p.name, "repo": p.repo, "revision": p.revision, "license": p.license, "kind": p.kind,
        "fps": fps, "codebase_version": meta.info.get("codebase_version"), "robot_type": meta.info.get("robot_type"),
        "slice": {"episodes": int(len(df)), "frames": int(df.n_frames.sum()), "hours": float(raw_h),
                  "data_files": [f.name for f in used_files],
                  "unreferenced_local_data_files": [f.name for f in orphan_files]},
        "lengths_full_meta": lengths,
        "lengths_slice": {"cv": float(df.duration_s.std() / df.duration_s.mean()), **_q(df.duration_s)},
        "idle": {"time_weighted_frac": tw_idle, "episode_median_frac": float(df.idle_frac.median()),
                 "lead_idle_s_median": float(df.lead_idle_s.median()), "trail_idle_s_median": float(df.trail_idle_s.median()),
                 "pauses_per_min": float(df.n_pauses.sum() / (df.duration_s.sum() / 60)),
                 "eef_idle_frac_1cm": float((df.eef_idle_frac_1cm * df.duration_s).sum() / df.duration_s.sum())
                 if "eef_idle_frac_1cm" in df else None},
        "smoothness": {"sparc_median": float(df.sparc.median()), "state_jerk_median": float(df.state_jerk.median()),
                       "action_jerk_median": float(df.action_jerk.median()), "jump_rate_median": float(df.jump_rate.median()),
                       "jump_rate_p99": float(df.jump_rate.quantile(0.99))},
        "gripper": {"closes_per_episode_median": float(df.n_close.median()), "closes_per_min": float(df.n_close.sum() / (df.duration_s.sum() / 60)),
                    "regrasp_episode_frac": float((df.n_regrasp > 0).mean()), "regrasps_per_episode": float(df.n_regrasp.mean()),
                    "empty_close_episode_frac": float((df.n_empty_close > 0).mean()),
                    "active_episode_frac": float(df.gripper_active.mean()),
                    "static_episode_frac": float((df.idle_frac >= 0.95).mean()),
                    "sign_corr_median": float(df.gripper_sign_corr.median()),
                    "sign_negative_frac": float((df.gripper_sign_corr < 0).sum() / max(df.gripper_active.sum(), 1))},
        "integrity": {k: int((df[k] > 0).sum()) for k in ["ts_gaps", "ts_nonmono", "fi_noncontig", "has_nan", "meta_len_mismatch"]},
        "duplicates": {"exact_copies": int(df.exact_dup_copy.sum()), "near_copies": int(df.near_dup_copy.sum()),
                       "near_dup_threshold": NEAR_DUP_D, "nn_dist": _q(nn, (1, 5, 50, 95))},
        "task_consistency": {"applicable": knn_applicable, "flagged": int(df.task_knn_flag.sum()),
                             "flagged_frac": float(df.task_knn_flag.mean()) if knn_applicable else None},
        "success": {"labelled": bool(failed.any() or df.success.notna().any()),
                    "failed_frac": float(failed.mean())},
        "language": {"empty_instruction_frac": float(df.language_empty.mean())} if "language_empty" in df else {},
        "hygiene": hyg, "coverage": coverage, "usable": usable, "meta_stats_duplicates": meta_dups,
        "notes": p.notes, "scales": sc.to_json(), "audit_seconds": round(time.time() - t0, 1),
    }
    log(f"[{p.name}] audit done in {summary['audit_seconds']}s")
    return AuditResult(summary, df, F, sc)


def save(res: AuditResult, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(res.summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    res.episodes.drop(columns=["hash"]).to_csv(out_dir / "episodes.csv", index=False)
