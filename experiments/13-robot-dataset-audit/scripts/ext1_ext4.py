"""Ext 1 (cross-dataset scorecard) and Ext 4 (cost per usable hour) from results/<name>/audit.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["libero_10", "fineart", "droid"]
LABEL = {"libero_10": "LIBERO-10 (sim)", "fineart": "FineART (lab teleop)", "droid": "DROID 1.0.1 (in-the-wild teleop)"}
A = {n: json.loads((ROOT / "results" / n / "audit.json").read_text()) for n in NAMES}
cams = {n: sum(1 for v in json.loads((ROOT / "data" / n / "meta" / "info.json").read_text())["features"].values()
               if v["dtype"] == "video") for n in NAMES}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


rows = [
    ("Source / robot / fps", lambda s: f"{s['kind'].split(' (')[0]} / {s['robot_type']} / {s['fps']:g}"),
    ("License (HF tag)", lambda s: s["license"].split(" ")[0]),
    ("Full dataset: episodes / hours", lambda s: f"{s['lengths_full_meta']['n_episodes']:,} / {s['lengths_full_meta']['hours']:,.1f}"),
    ("Audited slice: episodes / hours", lambda s: f"{s['slice']['episodes']:,} / {s['slice']['hours']:.1f}"),
    ("Episode length median (CV)", lambda s: f"{s['lengths_full_meta']['p50']:.1f} s ({s['lengths_full_meta']['cv']:.2f})"),
    ("Idle time, time-weighted", lambda s: pct(s["idle"]["time_weighted_frac"])),
    ("Lead / trail idle, median", lambda s: f"{s['idle']['lead_idle_s_median']:.1f} s / {s['idle']['trail_idle_s_median']:.1f} s"),
    ("Mid-episode pauses per min", lambda s: f"{s['idle']['pauses_per_min']:.2f}"),
    ("SPARC median (higher = smoother)", lambda s: f"{s['smoothness']['sparc_median']:.2f}"),
    ("State jerk / action jerk (median, normalised)", lambda s: f"{s['smoothness']['state_jerk_median']:.3f} / {s['smoothness']['action_jerk_median']:.3f}"),
    ("Grasps per minute", lambda s: f"{s['gripper']['closes_per_min']:.2f}"),
    ("Episodes with a re-grasp (retry proxy)", lambda s: pct(s["gripper"]["regrasp_episode_frac"])),
    ("Failed episodes (dataset flag)", lambda s: pct(s["success"]["failed_frac"]) if s["success"]["labelled"] else "not labelled"),
    ("Episodes with empty task string (full)", lambda s: pct(s["hygiene"]["episodes_with_empty_task_frac"])),
    ("Exact duplicate copies (full, meta stats)", lambda s: pct(s["meta_stats_duplicates"]["duplicate_frac"]) if s["meta_stats_duplicates"].get("available") else "n/a"),
    ("Task strings / effective tasks / Gini", lambda s: f"{s['hygiene']['n_tasks']:,} / {s['coverage']['effective_n_tasks']:,.0f} / {s['coverage']['gini_episodes']:.2f}"),
    ("Typo-candidate words / near-dup task groups", lambda s: f"{s['hygiene']['n_typo_words']} / {s['hygiene']['n_near_duplicate_groups']}"),
    ("**Usable share of raw time**", lambda s: f"**{pct(s['usable']['usable_frac'])}**"),
    ("Usable share, strict (drop re-grasp episodes)", lambda s: pct(s["usable"]["usable_strict_frac"])),
    ("Usable hours, full dataset (extrapolated)", lambda s: f"{s['usable']['full_dataset_usable_hours_extrapolated']:,.1f} of {s['usable']['full_dataset_hours']:,.1f}"),
]
md = ["| Metric | " + " | ".join(LABEL[n] for n in NAMES) + " |", "|-|" + "-|" * len(NAMES)]
md += [f"| {name} | " + " | ".join(f(A[n]) for n in NAMES) + " |" for name, f in rows]
score = {n: {"usable_frac": A[n]["usable"]["usable_frac"], "usable_strict_frac": A[n]["usable"]["usable_strict_frac"],
             "idle_frac": A[n]["idle"]["time_weighted_frac"], "regrasp_episode_frac": A[n]["gripper"]["regrasp_episode_frac"],
             "lost_frac_by_reason": A[n]["usable"]["lost_frac_by_reason"], "full_hours": A[n]["usable"]["full_dataset_hours"],
             "full_usable_hours": A[n]["usable"]["full_dataset_usable_hours_extrapolated"], "video_streams": cams[n]} for n in NAMES}
(ROOT / "results" / "scorecard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(ROOT / "results" / "scorecard.json").write_text(json.dumps(score, indent=2))
print("\n".join(md))

# figure: where the raw hours go (stacked bar, 2 px surface gaps between segments)
parts = [("usable", "#2a78d6"), ("idle_time", "#9ec5f4"), ("failed_episode", "#eb6834"), ("empty_task_label", "#eda100"),
         ("duplicate", "#e87ba4"), ("integrity", "#6b6a64"), ("too_short", "#c3c2b7")]
fig, ax = plt.subplots(figsize=(10, 2.9), facecolor="#fcfcfb")
for i, n in enumerate(NAMES):
    left = 0.0
    vals = {"usable": A[n]["usable"]["usable_frac"], **A[n]["usable"]["lost_frac_by_reason"]}
    for k, c in parts:
        v = vals.get(k, 0) * 100
        if v <= 0:
            continue
        ax.barh(i, v, left=left, color=c, height=0.55, edgecolor="#fcfcfb", linewidth=2, label=k.replace("_", " "))
        if v >= 4:
            ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="#1f1f1e")
        left += v
ax.set_yticks(range(len(NAMES)))
ax.set_yticklabels([LABEL[n] for n in NAMES], fontsize=8, color="#1f1f1e")
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% of raw recorded time (audited slice)", fontsize=8, color="#6b6a64")
h, l = ax.get_legend_handles_labels()
seen = {}
for hh, ll in zip(h, l):
    seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), fontsize=7, frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.3))
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=7, colors="#6b6a64")
ax.set_facecolor("#fcfcfb")
ax.set_title("Where the raw hours go: usable time vs time lost, by first failing reason", fontsize=10, loc="left", color="#1f1f1e")
fig.tight_layout()
fig.savefig(ROOT / "results" / "scorecard.png", dpi=130, facecolor="#fcfcfb")

# ---- Ext 4: cost per usable hour ----
PRICES = [("Perceptron subtasks", 9.0), ("Perceptron full stack", 29.0)]
SRC = ("departments/tech-scout/research/2026-11-19_wave2/D_egodata.md, citing "
       "perceptron.inc/blog/introducing-perceptron-egocentric-1-0 (2026-11-04): list price per video-hour")
cost = {"source": SRC, "assumptions": [
    "A1: the egocentric-video price applies unchanged to robot camera video (Perceptron prices human egocentric video; robot footage is untested).",
    "A2: one camera stream is labelled per robot-hour; the 'all streams' column multiplies by the video streams in info.json.",
    "A3: the audited slice is representative of the full dataset (usable share extrapolated).",
    "A4: auditing runs on a laptop CPU at $0 marginal cost; measured CPU wall-seconds per raw hour are reported."], "rows": []}
md4 = ["| Dataset | Usable share | Price (per video-hour) | $ per usable hour, label everything | Same, all camera streams | Full dataset: label all vs audit first and label usable only |",
       "|-|-|-|-|-|-|"]
for n in NAMES:
    r = A[n]["usable"]["usable_frac"]
    H = A[n]["usable"]["full_dataset_hours"]
    for pname, P in PRICES:
        row = {"dataset": n, "price": pname, "usd_per_video_hour": P, "usable_frac": r, "usd_per_usable_hour": P / r,
               "usd_per_usable_hour_all_streams": P * cams[n] / r, "streams": cams[n],
               "full_label_all_usd": P * H, "full_label_usable_only_usd": P * H * r,
               "audit_cpu_seconds_per_raw_hour": A[n]["audit_seconds"] / A[n]["slice"]["hours"]}
        cost["rows"].append(row)
        md4.append(f"| {LABEL[n]} | {pct(r)} | {pname} ${P:g} | ${P / r:.2f} | ${P * cams[n] / r:.2f} ({cams[n]} streams) | "
                   f"${P * H:,.0f} vs ${P * H * r:,.0f} |")
(ROOT / "results" / "cost_per_usable_hour.json").write_text(json.dumps(cost, indent=2))
(ROOT / "results" / "cost_per_usable_hour.md").write_text("\n".join(md4) + "\n", encoding="utf-8")
print("\n".join(md4))
print({n: round(A[n]["audit_seconds"] / A[n]["slice"]["hours"], 1) for n in NAMES}, "audit wall-seconds per raw hour")
