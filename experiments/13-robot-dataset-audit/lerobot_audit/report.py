"""One-page quality report: a Markdown page plus one 2x3 PNG figure per dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit import AuditResult, NEAR_DUP_D
from .load import load_meta

COLORS = {"libero_10": "#2a78d6", "fineart": "#eb6834", "droid": "#1baf7a"}   # validated categorical slots 1-3
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def _style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def _hist(ax, x, color, bins=40, log=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if log:
        x = x[x > 0]
        bins = np.geomspace(max(x.min(), 1e-3), x.max(), bins)
        ax.set_xscale("log")
    ax.hist(x, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.8)


def make_figure(res: AuditResult, root: Path, out_png: Path):
    s, df = res.summary, res.episodes
    c = COLORS.get(s["dataset"].replace("_corrupt", ""), "#2a78d6")
    meta = load_meta(root)
    dur = meta.episodes.length.values / meta.fps
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), facecolor=SURFACE)
    ax = axs[0, 0]
    _hist(ax, dur, c, log=True)
    ax.axvline(np.median(dur), color=INK, lw=1, ls="--")
    ax.text(np.median(dur), ax.get_ylim()[1] * 0.92, f" median {np.median(dur):.1f} s", fontsize=7, color=INK)
    _style(ax, f"Episode length (all {len(dur):,} episodes)", "seconds (log)", "episodes")
    ax = axs[0, 1]
    _hist(ax, df.idle_frac * 100, c, bins=30)
    _style(ax, f"Idle share (time-weighted {s['idle']['time_weighted_frac']*100:.1f}%)", "% of episode idle")
    ax = axs[0, 2]
    _hist(ax, df.sparc, c, bins=40)
    _style(ax, f"Smoothness (median SPARC {s['smoothness']['sparc_median']:.2f})", "SPARC (closer to 0 = smoother)")
    ax = axs[1, 0]
    mx = int(np.nanpercentile(df.n_close, 99)) + 1
    bins = np.arange(-0.5, mx + 1.5)
    ax.hist(df.n_close.clip(upper=mx), bins=bins, color=c, edgecolor=SURFACE, linewidth=0.8, label="grasps (close events)")
    ax.hist(df.n_regrasp.clip(upper=mx), bins=bins, color=INK, alpha=0.55, edgecolor=SURFACE, linewidth=0.8, label="re-grasps <= 2.5 s")
    ax.legend(fontsize=7, frameon=False)
    _style(ax, f"Gripper events (re-grasp in {s['gripper']['regrasp_episode_frac']*100:.0f}% of episodes)", "count per episode", "episodes")
    ax = axs[1, 1]
    _hist(ax, df.nn_dist, c, bins=40, log=True)
    ax.axvline(NEAR_DUP_D, color=INK, lw=1, ls="--")
    ax.text(NEAR_DUP_D, ax.get_ylim()[1] * 0.92, " near-dup threshold", fontsize=7, color=INK)
    _style(ax, f"Nearest-neighbour distance ({s['duplicates']['exact_copies']} exact copies)", "RMS distance, arm-state std units (log)")
    ax = axs[1, 2]
    ep_task = meta.episodes.tasks.map(lambda x: list(x)[0] if len(x) else "") if "tasks" in meta.episodes else df.task
    cnt = np.sort(ep_task.value_counts().values)[::-1]
    ax.plot(np.arange(1, len(cnt) + 1), cnt, color=c, lw=2)
    ax.set_yscale("log")
    if len(cnt) > 30:
        ax.set_xscale("log")
    _style(ax, f"Per-task coverage ({len(cnt):,} tasks, Gini {s['coverage']['gini_episodes']:.2f})", "task rank", "episodes (log)")
    fig.suptitle(f"{s['repo']} @ {s['revision'][:8]}  |  audited slice: {s['slice']['episodes']:,} episodes, "
                 f"{s['slice']['hours']:.1f} h", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_report(res: AuditResult, root: Path, out_dir: Path, gate: dict | None = None, extra_findings: list[str] | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(res, root, out_dir / "report.png")
    s = res.summary
    u, h, g, cov = s["usable"], s["hygiene"], s["gripper"], s["coverage"]
    L, Ls = s["lengths_full_meta"], s["slice"]
    md = [f"# Quality report: `{s['repo']}`", "",
          f"Revision `{s['revision']}` · license {s['license']} · {s['kind']} · {s['robot_type']} · {s['fps']:g} fps · "
          f"LeRobot {s['codebase_version']}. Full dataset {L['n_episodes']:,} episodes / {L['hours']:,.1f} h; audited slice "
          f"{Ls['episodes']:,} episodes / {Ls['hours']:.1f} h from {len(Ls['data_files'])} data file(s). Lengths, labels and "
          f"coverage use the full metadata; motion metrics use the slice.", "",
          "![report](report.png)", "",
          "| Metric | Value |", "|-|-|",
          f"| Episode length, median (p5-p95) | {L['p50']:.1f} s ({L['p5']:.1f}-{L['p95']:.1f}); CV {L['cv']:.2f}; {L['lt_1s']} under 1 s |",
          f"| Idle time (still >= 0.5 s), time-weighted | {_pct(s['idle']['time_weighted_frac'])} (lead {s['idle']['lead_idle_s_median']:.1f} s, trail {s['idle']['trail_idle_s_median']:.1f} s median; {s['idle']['pauses_per_min']:.2f} mid-episode pauses/min) |",
          f"| End-effector < 1 cm/s (physical cross-check) | {_pct(s['idle']['eef_idle_frac_1cm'])} |",
          f"| Smoothness: SPARC / state jerk / action jerk (median) | {s['smoothness']['sparc_median']:.2f} / {s['smoothness']['state_jerk_median']:.3f} / {s['smoothness']['action_jerk_median']:.3f} |",
          f"| Grasps per episode (median), per minute | {g['closes_per_episode_median']:.0f}, {g['closes_per_min']:.2f} |",
          f"| Episodes with a re-grasp (retry) | {_pct(g['regrasp_episode_frac'])} ({g['regrasps_per_episode']:.2f} per episode) |",
          f"| Gripper command/state sign agreement (median corr) | {g['sign_corr_median']:.2f}; negative in {_pct(g['sign_negative_frac'])} |",
          f"| Exact / near-duplicate copies in slice | {s['duplicates']['exact_copies']} / {s['duplicates']['near_copies']} |",
          f"| Exact copies, full dataset (meta stats scan) | " + (f"{s['meta_stats_duplicates']['duplicate_copies']} ({_pct(s['meta_stats_duplicates']['duplicate_frac'])} of episodes)" if s['meta_stats_duplicates'].get('available') else "n/a (no per-episode stats in meta)") + " |",
          f"| Integrity faults (gaps / non-monotonic / frame_index / NaN / meta length) | {' / '.join(str(v) for v in s['integrity'].values())} |",
          f"| Failed episodes (dataset success flag) | " + (_pct(s['success']['failed_frac']) if s['success']['labelled'] else "not labelled") + " |",
          f"| Task strings: total / empty / whitespace / dup groups / near-dup groups / typo words | {h['n_tasks']:,} / {h['n_empty']} / {h['n_whitespace_issues']} / {h['n_duplicate_groups_after_normalisation']} / {h['n_near_duplicate_groups']} / {h['n_typo_words']} |",
          f"| Episodes with an empty task | {_pct(h['episodes_with_empty_task_frac'])} |",
          f"| Coverage: tasks, effective tasks, Gini, tasks < 5 episodes | {cov['n_tasks_with_episodes']:,}, {cov['effective_n_tasks']:,.1f}, {cov['gini_episodes']:.2f}, {cov['tasks_with_lt5_episodes']:,} |",
          f"| **Usable hours (slice)** | **{u['usable_hours']:.2f} of {u['slice_raw_hours']:.2f} h = {_pct(u['usable_frac'])}** (strict, no re-grasp episodes: {_pct(u['usable_strict_frac'])}) |",
          f"| Usable hours, full dataset (extrapolated) | {u['full_dataset_usable_hours_extrapolated']:,.1f} of {u['full_dataset_hours']:,.1f} h |",
          "", "Raw time lost, by first failing reason: " + ", ".join(f"{k} {_pct(v)}" for k, v in u["lost_frac_by_reason"].items()) + ".", ""]
    ex = []
    if h["typo_examples"]:
        ex.append("typo candidates " + ", ".join(f"`{k}`->`{v}`" for k, v in list(h["typo_examples"].items())[:6]))
    if h["near_duplicate_examples"]:
        ex.append("near-duplicate tasks " + "; ".join(" / ".join(f"`{x}`" for x in grp) for grp in h["near_duplicate_examples"][:3]))
    if h["whitespace_examples"]:
        ex.append(f"whitespace e.g. `{h['whitespace_examples'][0]!r}`")
    if ex:
        md += ["Label hygiene examples: " + "; ".join(ex) + ".", ""]
    if gate:
        failed = [c for c in gate.values() if not c.passed]
        md += [f"Acceptance gate (shared thresholds, `specs/{s['dataset'].replace('_corrupt', '')}.json`): "
               f"{len(gate) - len(failed)}/{len(gate)} checks pass." + (" Failing: " + "; ".join(f"`{c.name}` ({len(c.flagged)} episodes)" for c in failed) + "." if failed else ""), ""]
    findings = list(s.get("notes", [])) + list(extra_findings or [])
    if findings:
        md += ["Notes and findings:"] + [f"- {x}" for x in findings] + [""]
    md += ["Definitions: idle = every arm dimension below 5% of its p95 speed and gripper still, for >= 0.5 s, on a common 10 Hz grid. "
           "Re-grasp = gripper re-closes within 2.5 s of opening. SPARC on sub-movements split at idle runs and gripper events. "
           "Usable = episodes passing integrity, success (if labelled), non-empty task, not a duplicate copy, minus idle time."]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
