"""Ext 2: per-task data quality (LIBERO-10 demos) vs per-task policy success (50-episode rerun, 5 episodes per task).
n = 10 tasks: suggestive only. Spearman rho with an exact-enough permutation p (200k shuffles)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
SUCC = ROOT.parent / "05-g05-lerobot" / "results" / "rerun50" / "per_task_rerun50.csv"
# lerobot libero_10 task_id -> instruction, from experiments/05-g05-lerobot/README.md (50-episode rerun table)
EVAL_TASKS = ["put both the alphabet soup and the tomato sauce in the basket",
              "put both the cream cheese box and the butter in the basket",
              "turn on the stove and put the moka pot on it",
              "put the black bowl in the bottom drawer of the cabinet and close it",
              "put the white mug on the left plate and put the yellow and white mug on the right plate",
              "pick up the book and place it in the back compartment of the caddy",
              "put the white mug on the plate and put the chocolate pudding to the right of the plate",
              "put both the alphabet soup and the cream cheese box in the basket",
              "put both moka pots on the stove",
              "put the yellow and white mug in the microwave and close it"]
ep = pd.read_csv(ROOT / "results" / "libero_10" / "episodes.csv")
assert set(ep.task) == set(EVAL_TASKS), "task strings do not match the eval suite"
g = ep.groupby("task")
q = pd.DataFrame({
    "sparc_median": g.sparc.median(),                       # smoothness (higher = smoother)
    "state_jerk_median": g.state_jerk.median(),             # smoothness (lower = smoother)
    "action_jerk_median": g.action_jerk.median(),
    "length_cv": g.duration_s.std() / g.duration_s.mean(),  # demo-length consistency (lower = more consistent)
    "length_mean_s": g.duration_s.mean(),
    "n_episodes": g.size(),                                 # simple count
    "grasps_per_episode": g.n_close.mean(),                 # simple count
    "regrasp_frac": g.n_regrasp.apply(lambda x: (x > 0).mean()),
    "idle_frac": g.idle_frac.mean(),
    "nn_dist_median": g.nn_dist.median(),                   # demo diversity
})
q["task_id"] = [EVAL_TASKS.index(t) for t in q.index]
s = pd.read_csv(SUCC)
for m in ("smolvla", "g05"):
    q[f"{m}_success"] = q.task_id.map(s[s.model == m].set_index("task_id").success / 5)
q = q.sort_values("task_id")
HYP = {"sparc_median": ("quality", +1), "state_jerk_median": ("quality", -1), "action_jerk_median": ("quality", -1),
       "length_cv": ("quality", -1), "n_episodes": ("count", 0), "grasps_per_episode": ("count", 0),
       "length_mean_s": ("other", 0), "regrasp_frac": ("other", -1), "idle_frac": ("other", 0), "nn_dist_median": ("other", 0)}
rng = np.random.default_rng(0)
rows = []
y = q.smolvla_success.values
for k, (grp, sign) in HYP.items():
    x = q[k].values
    rho = stats.spearmanr(x, y).statistic
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(20000)])
    p = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
    rows.append({"metric": k, "group": grp, "expected_sign": sign, "spearman_rho": float(rho), "perm_p_two_sided": p,
                 "direction_as_expected": (sign == 0) or (np.sign(rho) == sign)})
r = pd.DataFrame(rows)
g05_var = float(q.g05_success.var())
out = {"n_tasks": 10, "success_source": str(SUCC), "policy_used": "smolvla (G0.5 is 5/5 on 9 of 10 tasks, no variance to correlate)",
       "g05_success_variance": g05_var, "per_task": q.reset_index().to_dict(orient="records"), "correlations": rows}
(ROOT / "results" / "ext2").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "ext2" / "quality_vs_success.json").write_text(json.dumps(out, indent=2, default=float))
print(q[["task_id", "smolvla_success", "sparc_median", "state_jerk_median", "length_cv", "n_episodes", "grasps_per_episode", "regrasp_frac"]].round(3).to_string())
print(r.round(3).to_string())
# figure: four panels, the two hypothesis-quality metrics and the two counts
fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), facecolor="#fcfcfb")
for ax, k in zip(axs, ["sparc_median", "length_cv", "n_episodes", "grasps_per_episode"]):
    ax.scatter(q[k], q.smolvla_success * 5, s=36, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    for _, row in q.iterrows():
        ax.annotate(str(int(row.task_id)), (row[k], row.smolvla_success * 5), fontsize=7, color="#6b6a64", xytext=(4, 2), textcoords="offset points")
    rr = r.set_index("metric").loc[k]
    ax.set_title(f"{k}\nrho={rr.spearman_rho:+.2f}, perm p={rr.perm_p_two_sided:.2f}", fontsize=9, loc="left", color="#1f1f1e")
    ax.set_ylabel("SmolVLA successes / 5" if k == "sparc_median" else "", fontsize=8, color="#6b6a64")
    ax.tick_params(labelsize=7, colors="#6b6a64")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True); ax.set_facecolor("#fcfcfb")
fig.suptitle("LIBERO-10: per-task demo quality vs SmolVLA per-task success (n = 10 tasks, 5 rollouts each; suggestive only)", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(ROOT / "results" / "ext2" / "quality_vs_success.png", dpi=130, facecolor="#fcfcfb")
