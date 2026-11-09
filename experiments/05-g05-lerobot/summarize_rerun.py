#!/usr/bin/env python3
"""Summarize the 50-episode rerun: per-task and overall success with Wilson 95% CIs, wall time, latency, VRAM.

Usage: python summarize_rerun.py results/rerun50
Reads, per model (g05, smolvla): eval_info_<m>.json, timing_<m>.json, timing_<m>.episodes.jsonl,
logs/rollout_<m>.nvsmi.csv. Writes summary_rerun50.json and per_task_rerun50.csv into the same folder.
Standard library only.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

MODELS = ("g05", "smolvla")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(x: list[float], q: float) -> float:
    s = sorted(x)
    return s[int(q * (len(s) - 1))]


def main() -> None:
    root = Path(sys.argv[1])
    out: dict = {}
    rows = []
    for m in MODELS:
        info = json.loads((root / f"eval_info_{m}.json").read_text())
        timing = json.loads((root / f"timing_{m}.json").read_text())
        eps = [json.loads(l) for l in (root / f"timing_{m}.episodes.jsonl").read_text().splitlines() if l.strip()]
        per_task = {}
        for t in info["per_task"]:
            succ = [bool(s) for s in t["metrics"]["successes"]]
            k, n = sum(succ), len(succ)
            per_task[t["task_id"]] = {"n": n, "k": k, "successes": succ}
        k = sum(v["k"] for v in per_task.values())
        n = sum(v["n"] for v in per_task.values())
        lo, hi = wilson(k, n)
        # Episodes run in task order (task 0 episodes 0..N-1, then task 1, ...), so map the timing log onto tasks.
        order = [(tid, i) for tid in sorted(per_task) for i in range(per_task[tid]["n"])]
        for rec, (tid, i) in zip(eps, order):
            rec["task_id"], rec["episode"] = tid, i
            if "success" in rec and rec["success"] != per_task[tid]["successes"][i]:
                rec["success_mismatch_vs_eval_info"] = True
        chunk = [c for e in eps for c in e["chunk_ms"]][1:]  # drop the first (compile / warm-up) call
        walls = [e["wall_s"] for e in eps]
        nv = list(csv.DictReader((root / "logs" / f"rollout_{m}.nvsmi.csv").open()))
        mem = [int(r["memory_used_mib"]) for r in nv]
        for tid, v in sorted(per_task.items()):
            tw = [e["wall_s"] for e in eps if e.get("task_id") == tid]
            rows.append({"model": m, "task_id": tid, "n": v["n"], "success": v["k"],
                         "ci95_lo": round(100 * wilson(v["k"], v["n"])[0], 1),
                         "ci95_hi": round(100 * wilson(v["k"], v["n"])[1], 1),
                         "mean_wall_s": round(statistics.mean(tw), 1) if tw else ""})
        out[m] = {
            "n_episodes": n, "n_success": k, "pc_success": round(100 * k / n, 1),
            "wilson_ci95_pc": [round(100 * lo, 1), round(100 * hi, 1)],
            "lerobot_overall": {x: info["overall"].get(x) for x in ("pc_success", "pc_success_ci95", "eval_s", "eval_ep_s")},
            "per_task": {tid: {"k": v["k"], "n": v["n"]} for tid, v in sorted(per_task.items())},
            "episode_wall_s": {"mean": round(statistics.mean(walls), 1), "median": round(statistics.median(walls), 1),
                               "min": min(walls), "max": max(walls),
                               "success_mean": round(statistics.mean([e["wall_s"] for e in eps if e.get("success")] or [0]), 1),
                               "failure_mean": round(statistics.mean([e["wall_s"] for e in eps if e.get("success") is False] or [0]), 1)},
            "chunk_latency_ms": {"n": len(chunk), "median": round(statistics.median(chunk), 1),
                                 "mean": round(statistics.mean(chunk), 1), "p90": pct(chunk, 0.9),
                                 "p99": pct(chunk, 0.99), "max": max(chunk), "first_call_s": timing.get("first_call_s")},
            "vram": {"torch_max_allocated_mib": timing.get("torch_max_allocated_mib"),
                     "torch_max_reserved_mib": timing.get("torch_max_reserved_mib"),
                     "nvidia_smi_peak_mib": max(mem) if mem else None, "nvidia_smi_samples": len(mem)},
            "episodes": [{k2: e.get(k2) for k2 in ("task_id", "episode", "success", "steps", "wall_s", "chunk_calls")}
                         for e in eps],
            "timing_mismatches": sum(1 for e in eps if e.get("success_mismatch_vs_eval_info")),
        }
    (root / "summary_rerun50.json").write_text(json.dumps(out, indent=2))
    with (root / "per_task_rerun50.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in MODELS:
        o = out[m]
        print(m, f"{o['n_success']}/{o['n_episodes']}", o["wilson_ci95_pc"], o["per_task"], o["episode_wall_s"],
              o["chunk_latency_ms"], o["vram"], "mismatches", o["timing_mismatches"])


if __name__ == "__main__":
    main()
