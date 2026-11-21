"""Small statistics helpers: Wilson score interval and per-run result loading."""

import json
import math
from pathlib import Path

RAW = Path(__file__).parent / "results" / "raw"


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def episodes(ev):
    """Per-episode rows of one eval entry (lerobot 0.6.1 eval_info.json, single- or multi-task layout)."""
    info = ev.get("info") or {}
    if "per_episode" in info:
        return info["per_episode"]
    # lerobot 0.6.1 multi-task layout: per_task[i].metrics.{successes, max_rewards, sum_rewards}. Episode j of a
    # run with --seed=S, n_envs=B was reset with seed S + j (batches are seeded consecutively, see lerobot_eval.py).
    seed0 = 1000
    for c in ev.get("cmd", []):
        if c.startswith("--seed="):
            seed0 = int(c.split("=")[1])
    rows = []
    for v in (info.get("per_task") or []):
        m = v.get("metrics", {})
        for j, (s, mr, sr) in enumerate(zip(m.get("successes", []), m.get("max_rewards", []), m.get("sum_rewards", []))):
            rows.append({"success": bool(s), "max_reward": mr, "sum_reward": sr, "seed": seed0 + len(rows)})
    return rows


def load(name):
    p = RAW / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def eval_of(res, tag):
    for e in res.get("evals", []):
        if e["tag"] == tag:
            return e
    return None


def summarize_eval(ev):
    rows = episodes(ev)
    k = sum(bool(r["success"]) for r in rows)
    n = len(rows)
    p, lo, hi = wilson(k, n)
    cov = [r["max_reward"] for r in rows]
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi,
            "mean_max_reward": sum(cov) / n if n else float("nan"),
            "seeds": sorted({r["seed"] for r in rows})[:1] + sorted({r["seed"] for r in rows})[-1:]}
