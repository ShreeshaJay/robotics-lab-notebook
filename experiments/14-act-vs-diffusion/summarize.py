"""Build results/summary.json, results/per_run/*.json and the figures from results/raw/*.json."""

import base64
import io
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from t3_stats import RAW, eval_of, summarize_eval, wilson  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "results" / "figures"
PER = HERE / "results" / "per_run"
FIG.mkdir(parents=True, exist_ok=True)
PER.mkdir(parents=True, exist_ok=True)

# Categorical slots (validated: dataviz validate_palette.js, light surface #fcfcfb)
C = {"act": "#2a78d6", "diffusion": "#eb6834", "flow": "#1baf7a", "mse": "#eda100", "act_prior": "#e87ba4"}
LABEL = {"act": "ACT", "diffusion": "Diffusion (DDPM)", "flow": "Flow matching", "mse": "MSE BC",
         "act_prior": "ACT, z ~ prior"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

SEEDS = [1, 2, 3]
TASKS = {"pusht": "PushT", "aloha": "ALOHA insertion (scripted)"}


def all_results():
    out = {}
    for p in sorted(RAW.glob("*.json")):
        if p.stem.startswith("pilot"):
            continue
        out[p.stem] = json.loads(p.read_text())
    return out


def pooled(evs):
    k = sum(e["k"] for e in evs)
    n = sum(e["n"] for e in evs)
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def cost_of(r):
    return r.get("job_cost_usd", 0.0)


def main():
    R = all_results()
    S = {"main": {}, "ext2": {}, "ext3": {}, "ext1": {}, "ext4": None, "ext5": {}, "runs": {}}

    # per-run compact JSON
    for name, r in R.items():
        if "states" in r:
            continue
        row = {"name": name, "gpu": r.get("gpu"), "lerobot": r.get("lerobot"),
               "train_wall_s": r.get("train", {}).get("wall_s"), "job_wall_s": r.get("job_wall_s"),
               "job_cost_usd": r.get("job_cost_usd"), "train_args": (r.get("spec", {}).get("train") or {}).get("args"),
               "final_loss": ((r.get("train", {}).get("log_rows") or [{}])[-1]).get("loss"), "evals": {}}
        for e in r.get("evals", []):
            if e.get("info"):
                s = summarize_eval(e)
                s["wall_s"] = e["wall_s"]
                s["overrides"] = [c for c in e["cmd"] if c.startswith("--policy.") and "path" not in c and "device" not in c]
                s["per_episode_success"] = [int(bool(x["success"])) for x in __import__("t3_stats").episodes(e)]
                row["evals"][e["tag"]] = s
            else:
                row["evals"][e["tag"]] = {"error": (e.get("tail") or [""])[-1]}
        S["runs"][name] = row
        (PER / f"{name}.json").write_text(json.dumps(row, indent=1))

    def ev(name, tag="main"):
        return S["runs"].get(name, {}).get("evals", {}).get(tag)

    # ---- main table + ext5 ----
    for task in TASKS:
        for m in ["act", "diffusion", "flow"]:
            per = [ev(f"{m}-{task}-s{s}") for s in SEEDS]
            per = [x for x in per if x and "k" in x]
            if not per:
                continue
            S["main"][f"{m}-{task}"] = {
                "per_seed": [{"seed": s, "p": x["p"], "k": x["k"], "n": x["n"], "coverage": x["mean_max_reward"]}
                             for s, x in zip(SEEDS, per)],
                "pooled": pooled(per),
                "mean_coverage": st.mean(x["mean_max_reward"] for x in per),
                "train_wall_s": [S["runs"][f"{m}-{task}-s{s}"]["train_wall_s"] for s in SEEDS[: len(per)]],
                "cost_usd": sum(S["runs"][f"{m}-{task}-s{s}"]["job_cost_usd"] for s in SEEDS[: len(per)]),
            }
        a, d = S["main"].get(f"act-{task}"), S["main"].get(f"diffusion-{task}")
        if a and d:
            pa = [x["p"] for x in a["per_seed"]]
            pd = [x["p"] for x in d["per_seed"]]
            S["ext5"][task] = {
                "act_seed_range": max(pa) - min(pa), "diff_seed_range": max(pd) - min(pd),
                "act_seed_sd": st.stdev(pa) if len(pa) > 1 else None,
                "diff_seed_sd": st.stdev(pd) if len(pd) > 1 else None,
                "gap_diff_minus_act": st.mean(pd) - st.mean(pa),
                "welch_t": _welch(pd, pa),
            }

    # ---- ext2: ACT chunk length x temporal ensembling on ALOHA ----
    for k in [1, 10, 50, 100]:
        names = [f"act-aloha-s{s}" for s in SEEDS] if k == 100 else [f"act-aloha-c{k}-s{s}" for s in SEEDS]
        for ens in [False, True]:
            tag = "main" if (k == 100 and not ens) else ("ens" if ens else "main")
            if k == 1 and ens:
                continue  # identical to no ensembling at chunk length 1
            per = [ev(n, tag) for n in names]
            per = [x for x in per if x and "k" in x]
            if per:
                S["ext2"][f"c{k}-{'ens' if ens else 'noens'}"] = {"chunk": k, "ens": ens, "n_ckpt": len(per),
                                                                  **pooled(per)}

    for ens in [False, True]:  # seed-matched point for k = 100 (the k = 1/10/50 runs use training seed 1 only)
        x = ev("act-aloha-s1", "ens" if ens else "main")
        if x and "k" in x:
            S["ext2"][f"c100s1-{'ens' if ens else 'noens'}"] = {"chunk": 100, "ens": ens, "n_ckpt": 1,
                                                               **pooled([x])}

    # ---- ext3 / ext1: inference steps ----
    for m, tags in [("diffusion", ["main", "ddpm10", "ddpm1", "ddim10", "ddim1"]),
                    ("flow", ["main", "euler10", "euler1"])]:
        for task in TASKS:
            for tag in tags:
                per = [ev(f"{m}-{task}-s{s}", tag) for s in SEEDS]
                per = [x for x in per if x and "k" in x]
                if per:
                    S["ext3"][f"{m}-{task}-{tag}"] = {**pooled(per), "n_ckpt": len(per),
                                                      "coverage": st.mean(x["mean_max_reward"] for x in per),
                                                      "eval_wall_s": st.mean(x["wall_s"] for x in per)}
    mse = ev("mse-pusht-s1")
    if mse and "k" in mse:
        S["ext1"]["mse-pusht"] = {**pooled([mse]), "coverage": mse["mean_max_reward"]}

    probe = R.get("probe-pusht")
    if probe:
        S["ext4"] = {"chosen": probe["chosen"], "states": [
            {"seed": s["seed"], "dp_score": s["dp_score"], "dp_minority_frac": s["dp_minority_frac"],
             "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "traj"} for k, v in s["samples"].items()}}
            for s in probe["states"]], "job_cost_usd": probe.get("job_cost_usd")}
        fig_probe(probe)

    S["total_cost_usd"] = sum(cost_of(r) for r in R.values())
    (HERE / "results" / "summary.json").write_text(json.dumps(S, indent=1))
    fig_main(S)
    fig_chunk(S)
    fig_steps(S)
    print(json.dumps({k: S[k] for k in ["main", "ext5", "ext2", "ext3", "ext1"]}, indent=1)[:6000])
    print("total cost", S["total_cost_usd"])


def _welch(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    mx, my, vx, vy = st.mean(x), st.mean(y), st.variance(x), st.variance(y)
    se = (vx / len(x) + vy / len(y)) ** 0.5
    return (mx - my) / se if se > 0 else float("inf")


def _err(ax, x, d, color, label=None, marker="o"):
    ax.errorbar([x], [d["p"] * 100], yerr=_yerr(d), fmt=marker,
                color=color, ms=8, capsize=4, lw=2, label=label, markeredgecolor=SURF, markeredgewidth=2, zorder=3)


def fig_main(S):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, task in zip(axes, TASKS):
        methods = [m for m in ["act", "diffusion", "flow"] if f"{m}-{task}" in S["main"]]
        for i, m in enumerate(methods):
            d = S["main"][f"{m}-{task}"]
            _err(ax, i, d["pooled"], C[m])
            for j, ps in enumerate(d["per_seed"]):
                ax.plot(i + 0.22, ps["p"] * 100, marker="o", ms=5, color=C[m], alpha=0.55, ls="none")
            ax.text(i, min(100, d["pooled"]["hi"] * 100) + 3, f"{d['pooled']['p']*100:.0f}%\n({d['pooled']['k']}/{d['pooled']['n']})",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(methods)), [LABEL[m] for m in methods])
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylim(0, 110)
        ax.set_title(TASKS[task], fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Success rate (%)")
    fig.suptitle("Success, 3 training seeds pooled (Wilson 95% CI); small dots = single seeds", fontsize=10,
                 color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "success_main.png", dpi=150)
    plt.close(fig)


def fig_chunk(S):
    if not S["ext2"]:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ks = [1, 10, 50, 100]
    for ens, col, mk, off in [(False, C["act"], "o", 0.93), (True, C["diffusion"], "s", 1.07)]:
        xs, ds = [], []
        for k in ks:
            key = "c1-noens" if k == 1 else (f"c100s1-{'ens' if ens else 'noens'}" if k == 100 else f"c{k}-{'ens' if ens else 'noens'}")
            if key in S["ext2"]:
                xs.append(k * (off if k > 1 else 1.0))
                ds.append(S["ext2"][key])
        if not xs:
            continue
        ax.plot(xs, [d["p"] * 100 for d in ds], color=col, lw=2, zorder=2,
                label="temporal ensembling on (n_action_steps=1, m=0.01)" if ens else "ensembling off (execute whole chunk)")
        for x, d in zip(xs, ds):
            ax.errorbar([x], [d["p"] * 100], yerr=_yerr(d), fmt=mk, color=col, ms=8, capsize=4, lw=2,
                        markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        pk = f"c100-{'ens' if ens else 'noens'}"
        if pk in S["ext2"]:
            d = S["ext2"][pk]
            ax.errorbar([100 * off * 1.25], [d["p"] * 100], yerr=_yerr(d), fmt=mk, mfc="none", color=col, ms=8,
                        capsize=4, lw=1.5, zorder=3)
    ax.text(100 * 1.3, 8, "hollow: k=100, 3 seeds", fontsize=7.5, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(0.7, 200)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("ACT chunk length k (actions predicted per query)")
    ax.set_ylabel("Success rate (%), 50 episodes")
    ax.set_title("ACT, ALOHA insertion: chunk length x temporal ensembling (training seed 1)", fontsize=9.5,
                 loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "act_chunk_length.png", dpi=150)
    plt.close(fig)


def _yerr(d):
    return [[max(0.0, d["p"] - d["lo"]) * 100], [max(0.0, d["hi"] - d["p"]) * 100]]


def fig_steps(S):
    series = {("diffusion", "DDPM"): [("main", 100), ("ddpm10", 10), ("ddpm1", 1)],
              ("diffusion", "DDIM"): [("ddim10", 10), ("ddim1", 1)],
              ("flow", "Euler"): [("main", 100), ("euler10", 10), ("euler1", 1)]}
    styles = {("diffusion", "DDPM"): (C["diffusion"], "o", "-", "Diffusion, DDPM sampler"),
              ("diffusion", "DDIM"): (C["diffusion"], "s", "--", "Diffusion, DDIM sampler"),
              ("flow", "Euler"): (C["flow"], "^", "-", "Flow matching, Euler")}
    tasks = [t for t in TASKS if f"diffusion-{t}-ddpm10" in S["ext3"]]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for key, pts_def in series.items():
            col, mk, ls, lab = styles[key]
            pts = [(n, S["ext3"][f"{key[0]}-{task}-{tag}"]) for tag, n in pts_def if f"{key[0]}-{task}-{tag}" in S["ext3"]]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1]["p"] * 100 for p in pts], color=col, ls=ls, lw=2, label=lab)
            for n, d in pts:
                ax.errorbar([n], [d["p"] * 100], yerr=_yerr(d), fmt=mk, color=col, ms=8, capsize=4, lw=2,
                            markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        ax.set_xlim(0.7, 150)
        ax.set_ylim(-3, 100)
        ax.set_xlabel("Denoising / integration steps at inference")
        ax.set_ylabel("Success rate (%)")
        ax.set_title(f"{TASKS[task]}: success vs inference steps (3 ckpts pooled)", fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "inference_steps.png", dpi=150)
    plt.close(fig)


def fig_probe(probe):
    states = probe["states"]
    order = [k for k in ["diffusion", "flow", "act", "act_prior", "mse"] if k in states[0]["samples"]]
    fig, axes = plt.subplots(len(states), len(order), figsize=(2.6 * len(order), 2.75 * len(states)), squeeze=False)
    for r, s in enumerate(states):
        img = plt.imread(io.BytesIO(base64.b64decode(s["frame_png_b64"])), format="png")
        h = img.shape[0]
        sc = h / 512.0
        for c, k in enumerate(order):
            ax = axes[r][c]
            ax.imshow(img, alpha=0.55)
            tr = np.array(s["samples"][k]["traj"]) * sc
            for t in tr:
                ax.plot(t[:, 0], t[:, 1], color=C[k], lw=0.9, alpha=0.35)
            ax.scatter(tr[:, -1, 0], tr[:, -1, 1], s=10, color=C[k], edgecolors=SURF, linewidths=0.5, zorder=3)
            ap = np.array(s["agent_pos"]) * sc
            ax.scatter([ap[0]], [ap[1]], marker="x", color=INK, s=30, zorder=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(LABEL[k], fontsize=9.5, color=INK)
            m = s["samples"][k]
            ax.text(4, h - 6, f"mode A {m['frac_mode0']*100:.0f}% | spread {m['spread_px']:.0f}px", fontsize=7,
                    color=INK, va="bottom", bbox=dict(fc=SURF, ec="none", alpha=0.8, pad=1))
        axes[r][0].set_ylabel(f"reset seed {s['seed']}", fontsize=8.5, color=INK2)
    fig.suptitle(f"PushT: {probe['n_samples']} sampled action chunks (first {probe['n_exec']} actions) from one observation; x = agent",
                 fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "pusht_multimodality.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
