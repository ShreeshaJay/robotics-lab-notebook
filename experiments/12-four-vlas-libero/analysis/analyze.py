"""Tutorial 1 analysis: main table, extensions 1-6, figures. Standard library + matplotlib.

  python analysis/analyze.py            (from the tutorial folder)

Inputs: results/raw/** (this tutorial's runs) and the baseline's saved per-episode results
(experiments/05-g05-lerobot/results/rerun50/summary_rerun50.json). Outputs: results/*.json, results/*.csv,
results/figures/*.png.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "harness"))
import stats as st  # noqa: E402
import libero_common as lc  # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"
FIG = OUT / "figures"
BASE = ROOT.parent / "05-g05-lerobot" / "results" / "rerun50"
MODELS = ["g05", "groot", "smolvla", "pear"]
LABEL = {"g05": "G0.5", "groot": "GR00T N1.7", "smolvla": "SmolVLA", "pear": "PEAR"}
# $/s on Modal (COSTS.md)
PRICE = {"L4": 0.000222, "A10": 0.000306, "B200": 0.001736, "H100": 0.001097, "H200": 0.001261}
GPU_OF = {"g05": "L4", "smolvla": "L4", "groot": "A10", "pear": "B200"}


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def key(r):
    return (int(r["task_id"]), int(r["episode"]))


# ------------------------------------------------------------------ loading

def load_main() -> dict[str, dict]:
    """Per model: {(task, ep): record} for the 50 main episodes."""
    base = json.loads((BASE / "summary_rerun50.json").read_text())
    out = {}
    for m in ("g05", "smolvla"):
        out[m] = {key(e): {**e, "source": "baseline rerun50 (saved)"} for e in base[m]["episodes"]}
        timing = [json.loads(x) for x in (BASE / f"timing_{m}.episodes.jsonl").read_text().splitlines() if x.strip()]
        for (k, rec), t in zip(sorted(out[m].items()), timing):
            rec["chunk_ms"] = t["chunk_ms"]
    out["groot"] = {key(r): r for r in jl(RAW / "groot" / "main.jsonl") if "success" in r}
    pear = {}
    for tag in ("pilot", "main_a", "main_b"):
        for r in jl(RAW / "pear" / f"{tag}.jsonl"):
            if "success" in r:
                pear[key(r)] = r
    out["pear"] = pear
    return out


def vram_latency(main: dict) -> dict:
    base = json.loads((BASE / "summary_rerun50.json").read_text())
    res = {}
    for m in ("g05", "smolvla"):
        b = base[m]
        res[m] = {"chunk_ms_median": b["chunk_latency_ms"]["median"], "chunk_ms_p90": b["chunk_latency_ms"]["p90"],
                  "n_chunks": b["chunk_latency_ms"]["n"], "torch_peak_mib": b["vram"]["torch_max_allocated_mib"],
                  "gpu_peak_mib": b["vram"]["nvidia_smi_peak_mib"], "gpu": "L4 (GCP g2-standard-8)",
                  "pred_exec": "32 / 10" if m == "g05" else "50 / 50"}
    gm = json.loads((RAW / "groot" / "main.meta.json").read_text()) if (RAW / "groot" / "main.meta.json").exists() else []
    lat = [c for r in main["groot"].values() for c in r.get("chunk_ms", [])]
    if lat:
        res["groot"] = {"chunk_ms_median": round(statistics.median(lat), 1), "chunk_ms_p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 1),
                        "n_chunks": len(lat), "torch_peak_mib": max(x["torch_max_allocated_mib"] for x in gm) if gm else None,
                        "gpu_peak_mib": max(x["nvidia_smi_peak_mib"] or 0 for x in gm) if gm else None,
                        "gpu": ", ".join(sorted({x["gpu"] for x in gm})) + " (Modal)" if gm else None,
                        "pred_exec": "16 / 8"}
    turns = [t for r in main["pear"].values() for t in r.get("turn_log", [])]
    if turns:
        metas = [json.loads(p.read_text()) for p in (RAW / "pear").glob("*.meta.json") if p.stem.split(".")[0] in ("pilot", "main_a", "resume")]
        gs = [t["gen_s"] for t in turns]
        res["pear"] = {"chunk_ms_median": round(1000 * statistics.median(gs), 1), "chunk_ms_p90": round(1000 * sorted(gs)[int(0.9 * (len(gs) - 1))], 1),
                       "n_chunks": len(gs), "torch_peak_mib": None,
                       "gpu_peak_mib": max((x.get("nvidia_smi_peak_mib") or 0) for x in metas) if metas else None,
                       "gpu": "B200 (Modal), vLLM, up to %s episodes sharing the GPU" % max((x.get("conc") or 0) for x in metas) if metas else "B200",
                       "pred_exec": "1 waypoint / 1 (closed-loop controller, up to 100 native steps)",
                       "completion_tokens_per_turn_median": statistics.median(t["completion_tokens"] for t in turns),
                       "turns_per_episode_median": statistics.median(r["turns"] for r in main["pear"].values())}
    # same-GPU latency (median ms per policy call, first call excluded)
    same = {"L4": {"g05": res["g05"]["chunk_ms_median"], "smolvla": res["smolvla"]["chunk_ms_median"]}, "A10": {}}
    lat_l4 = [c for r in jl(RAW / "groot" / "latency_l4.jsonl") for c in r.get("chunk_ms", [])]
    if lat_l4:
        same["L4"]["groot"] = round(statistics.median(lat_l4), 1)
    if gm and all("A10" in x["gpu"] for x in gm) and lat:
        same["A10"]["groot"] = round(statistics.median(lat), 1)
    for m in ("g05", "smolvla"):
        p = RAW / "lerobot" / f"{m}_states.json"
        if p.exists():
            t = json.loads(p.read_text())["meta"]["timing"]
            same.setdefault(t["gpu"].replace("NVIDIA ", ""), {})[m] = t["chunk_latency_ms_median"]
    res["same_gpu_chunk_ms_median"] = same
    return res


# ------------------------------------------------------------------ main table

def main_table(main: dict) -> dict:
    tab = {}
    for m in MODELS:
        recs = main[m]
        if not recs:
            continue
        k, n = sum(r["success"] for r in recs.values()), len(recs)
        lo, hi = st.wilson(k, n)
        per_task = {t: sum(recs[(t, e)]["success"] for e in range(5) if (t, e) in recs) for t in range(10)}
        tab[m] = {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)], "per_task": per_task,
                  "source": next(iter(recs.values())).get("source", "this tutorial")}
    return tab


# ------------------------------------------------------------------ extension 3: failure step + partial credit

def load_trackers(main: dict) -> dict[str, dict]:
    """Tracker summaries per model on the 50 main episodes. G0.5/SmolVLA come from this tutorial's state-recording
    re-run (the saved baseline has no sim states)."""
    out = {"groot": {k: r["tracker"] for k, r in main["groot"].items()},
           "pear": {k: r["tracker"] for k, r in main["pear"].items()}}
    for m in ("g05", "smolvla"):
        p = RAW / "lerobot" / f"{m}_states.json"
        if p.exists():
            d = json.loads(p.read_text())
            out[m] = {key(r): {**r["tracker"], "_success": r["success"]} for r in d["track"]}
    return out


def steps_done(task_id: int, end: list[bool], initial: list[bool]) -> int:
    """Instruction steps completed at the episode's end: goal predicates that hold and did not hold at the start; for
    the physically sequential tasks (3, 9) only the prefix in instruction order counts (an empty, closed drawer is
    not progress)."""
    labels = [(l, idx) for l, idx in lc.STEP_LABELS[task_id] if not initial[idx]]
    counted = [end[idx] for _, idx in labels]
    if task_id in lc.ORDERED_TASKS:
        n = 0
        for c in counted:
            if not c:
                break
            n += 1
        return n
    return sum(counted)


def ext3(main: dict, trackers: dict) -> dict:
    res = {}
    for m in MODELS:
        tr = trackers.get(m)
        if not tr:
            continue
        rows = []
        for (t, e), s in sorted(tr.items()):
            succ = s.get("_success", main[m].get((t, e), {}).get("success")) if m in ("g05", "smolvla") else main[m][(t, e)]["success"]
            # steps = goal predicates that are false at the start (task 8's "stove on" already holds: 2 steps)
            n_steps = sum(1 for _, idx in lc.STEP_LABELS[t] if not s["predicates_initial"][idx])
            done = steps_done(t, s["predicates_end"], s["predicates_initial"])
            rows.append({"task_id": t, "episode": e, "success": bool(succ), "n_steps": n_steps,
                         "steps_done_end": done, "partial": round(done / n_steps, 4),
                         "ever": sum(s["predicates_ever"]), "predicates_end": s["predicates_end"]})
        fails = [r for r in rows if not r["success"]]
        # failed step = first unfinished step (index in instruction order, 1-based), for multi-step tasks
        dist = {}
        for r in fails:
            step = r["steps_done_end"] + 1
            label = f"step {step}" if r["n_steps"] > 1 else "single-step task"
            dist[label] = dist.get(label, 0) + 1
        multi = [r for r in fails if r["n_steps"] > 1]
        res[m] = {"n": len(rows), "n_fail": len(fails), "failed_step_counts": dist,
                  "fail_multistep": len(multi),
                  "fail_at_step1": sum(1 for r in multi if r["steps_done_end"] == 0),
                  "fail_at_step2plus": sum(1 for r in multi if r["steps_done_end"] >= 1),
                  "mean_partial_credit_all": round(statistics.mean(r["partial"] for r in rows), 4),
                  "mean_partial_credit_failures": round(statistics.mean(r["partial"] for r in fails), 4) if fails else None,
                  "success_rate_this_run": round(sum(r["success"] for r in rows) / len(rows), 4),
                  "rows": rows}
        if m in ("g05", "smolvla"):
            agree = sum(1 for r in rows if (r["task_id"], r["episode"]) in main[m] and main[m][(r["task_id"], r["episode"])]["success"] == r["success"])
            res[m]["rerun_vs_saved_agreement"] = f"{agree}/{len(rows)}"
            res[m]["rerun_k"] = sum(r["success"] for r in rows)
    return res


# ------------------------------------------------------------------ extension 1: horizon sweep

def ext1(main: dict) -> dict:
    res = {}
    sub = [(t, e) for t in range(10) for e in range(2)]
    # GR00T: horizon runs + main (8) on the 20-episode subset
    g = {}
    for r in jl(RAW / "groot" / "horizon.jsonl"):
        if "success" in r:
            g.setdefault(r["n_action_steps_used"], {})[key(r)] = r
    g[8] = {k: main["groot"][k] for k in sub if k in main["groot"]}
    res["groot"] = g
    for m in ("g05", "smolvla"):
        d = {}
        for p in sorted((RAW / "lerobot").glob(f"{m}_horizon_*.json")):
            h = int(p.stem.split("_")[-1])
            x = json.loads(p.read_text())
            recs = {}
            for tr, ep in zip(x["track"], x["episodes"]):
                recs[key(tr)] = {"success": tr["success"], "action_jerk": ep.get("action_jerk"), "steps": tr["steps"],
                                 "chunk_calls": ep.get("chunk_calls")}
            d[h] = recs
        # the default horizon comes from this tutorial's state re-run (same GPU and harness as the other points)
        default = 10 if m == "g05" else 50
        st_path = RAW / "lerobot" / f"{m}_states.json"
        if st_path.exists():
            x = json.loads(st_path.read_text())
            d.setdefault(default, {key(tr): {"success": tr["success"], "action_jerk": ep.get("action_jerk"), "steps": tr["steps"]}
                                   for tr, ep in zip(x["track"], x["episodes"]) if key(tr) in sub})
        res[m] = d
    summary = {}
    for m, d in res.items():
        summary[m] = {}
        for h, recs in sorted(d.items()):
            if not recs:
                continue
            k, n = sum(r["success"] for r in recs.values()), len(recs)
            jerks = [r.get("action_jerk") for r in recs.values() if r.get("action_jerk") is not None]
            summary[m][h] = {"k": k, "n": n, "rate": round(k / n, 3), "wilson95": [round(x, 3) for x in st.wilson(k, n)],
                             "action_jerk_mean": round(statistics.mean(jerks), 4) if jerks else None,
                             "source": "this tutorial"}
    return {"summary": summary, "_recs": res}


# ------------------------------------------------------------------ extension 2: counterfactual instructions

def ext2() -> dict:
    res = {}
    src = {"groot": jl(RAW / "groot" / "cf.jsonl"), "pear": jl(RAW / "pear" / "cf.jsonl") + jl(RAW / "pear" / "cf_b.jsonl")}
    for m in ("g05", "smolvla"):
        p = RAW / "lerobot" / f"{m}_cf_{1}.json"
        cands = sorted((RAW / "lerobot").glob(f"{m}_cf_*.json"))
        src[m] = json.loads(cands[0].read_text())["track"] if cands else []
    stopped = RAW / "pear" / "ext_stopped.meta.json"
    unfinished_cf = [u for u in json.loads(stopped.read_text())["unfinished_at_stop"] if u["condition"] == "counterfactual"]         if stopped.exists() else []
    for m, recs in src.items():
        recs = [r for r in recs if r.get("outcome")]
        if not recs:
            continue
        if m == "pear":  # episodes stopped unfinished and not completed in the resume run
            have = {key(r) for r in recs}
            recs = recs + [{**u, "outcome": "unfinished", "tracker": None} for u in unfinished_cf if key(u) not in have]
        counts = {}
        for r in recs:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        n = len(recs)
        inst = counts.get("instructed", 0) + counts.get("both", 0)
        default = counts.get("default", 0) + counts.get("both", 0)
        alt_partial = [sum(a and not i for a, i in zip(r["tracker"]["alt_end"], r["tracker"]["alt_initial"])) / len(r["tracker"]["alt_end"])
                       for r in recs if r.get("tracker")]
        res[m] = {"n": n, "n_finished": sum(1 for r in recs if r["outcome"] != "unfinished"), "counts": counts, "instructed_rate": round(inst / n, 3), "instructed_wilson95": [round(x, 3) for x in st.wilson(inst, n)],
                  "default_rate": round(default / n, 3), "default_wilson95": [round(x, 3) for x in st.wilson(default, n)],
                  "alt_partial_mean": round(statistics.mean(alt_partial), 3),
                  "per_task": {r["task_id"]: r["outcome"] for r in recs if r["episode"] == 0}}
    return res


# ------------------------------------------------------------------ extension 4: perturbation

def ext4(main: dict) -> dict:
    res = {}
    src = {"groot": jl(RAW / "groot" / "perturb.jsonl"), "pear": jl(RAW / "pear" / "perturb.jsonl") + jl(RAW / "pear" / "perturb_b.jsonl")}
    for m in ("g05", "smolvla"):
        p = RAW / "lerobot" / f"{m}_perturb.json"
        src[m] = json.loads(p.read_text())["track"] if p.exists() else []
    for m, recs in src.items():
        recs = [r for r in recs if "success" in r]
        if not recs:
            continue
        keys = [key(r) for r in recs]
        pert = [bool(r["success"]) for r in recs]
        ref = main[m]
        if m in ("g05", "smolvla") and (RAW / "lerobot" / f"{m}_states.json").exists():  # same GPU/harness re-run
            ref = {key(tr): tr for tr in json.loads((RAW / "lerobot" / f"{m}_states.json").read_text())["track"]}
        clean = [bool(ref[k]["success"]) for k in keys if k in ref]
        if len(clean) != len(pert):
            continue
        res[m] = {"n": len(recs), "clean_k": sum(clean), "perturbed_k": sum(pert),
                  "clean_wilson95": [round(x, 3) for x in st.wilson(sum(clean), len(clean))],
                  "perturbed_wilson95": [round(x, 3) for x in st.wilson(sum(pert), len(pert))],
                  "drop_pts": round(100 * (sum(clean) - sum(pert)) / len(pert), 1), "mcnemar": st.mcnemar_exact(clean, pert)}
    return res


# ------------------------------------------------------------------ extension 5: paired statistics

def ext5(main: dict) -> dict:
    keys = sorted(set.intersection(*[set(main[m]) for m in MODELS if main[m]]))
    pairs = {}
    present = [m for m in MODELS if main[m]]
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            keys = sorted(set(main[a]) & set(main[b]))  # pairwise: the episodes both models completed
            xa = [bool(main[a][k]["success"]) for k in keys]
            xb = [bool(main[b][k]["success"]) for k in keys]
            ka, kb, n = sum(xa), sum(xb), len(keys)
            la, ha = st.wilson(ka, n)
            lb, hb = st.wilson(kb, n)
            overlap = not (ha < lb or hb < la)
            mc = st.mcnemar_exact(xa, xb)
            bs = st.paired_bootstrap([int(x) for x in xa], [int(x) for x in xb], seed=1)
            bst = st.paired_bootstrap([int(x) for x in xa], [int(x) for x in xb], groups=[k[0] for k in keys], seed=2)
            pairs[f"{a}_vs_{b}"] = {"n": n, "k_a": ka, "k_b": kb, "wilson_overlap": overlap, "mcnemar": mc,
                                    "bootstrap_episode": bs, "bootstrap_task_cluster": bst,
                                    "separated_paired": mc["p"] < 0.05, "separated_by_intervals": not overlap}
    return {"n_common": len(keys), "pairs": pairs}


# ------------------------------------------------------------------ extension 6: power + cost per success

def ext6(main: dict, lat: dict) -> dict:
    out = {"power_unpaired": {}, "power_paired_examples": {}, "cost_per_success": {}}
    for p0 in (0.5, 0.7, 0.9):
        out["power_unpaired"][f"{int(100*p0)}->{int(100*p0)+10}"] = st.n_unpaired(p0, p0 + 0.1)
    # paired: discordance rates observed between model pairs, scaled to a 10-point gap
    for p_disc in (0.1, 0.2, 0.3):
        a_only = (p_disc + 0.1) / 2
        b_only = (p_disc - 0.1) / 2
        out["power_paired_examples"][f"discordant {int(100*p_disc)}%"] = st.n_paired(a_only, b_only)
    for m in MODELS:
        recs = main[m]
        if not recs:
            continue
        wall = sum(r["wall_s"] for r in recs.values())
        k = sum(r["success"] for r in recs.values())
        gpu = GPU_OF[m]
        if m == "pear":
            # episodes share one B200: GPU-seconds = container episode time, not the sum of episode walls
            # billed container time of every container that ran main episodes (pilot, main_a, resume), including the
            # work lost when two of them were stopped with episodes in flight
            metas = [json.loads(p.read_text()) for p in (RAW / "pear").glob("*.meta.json") if p.stem.split(".")[0] in ("pilot", "main_a", "resume")]
            gpu_s = sum(x["container_s"] for x in metas)
            startup = sum(x["startup_s"] for x in metas)
        else:
            gpu_s, startup = wall, None
        cost = gpu_s * PRICE[gpu]
        out["cost_per_success"][m] = {"gpu": gpu, "gpu_s_episodes": round(gpu_s), "startup_s": startup, "dollars": round(cost, 3),
                                      "successes": k, "dollars_per_success": round(cost / k, 4) if k else None,
                                      "gpu_s_per_success": round(gpu_s / k, 1) if k else None}
    return out


# ------------------------------------------------------------------ figures

def figures(tab, ext1r, ext3r, ext2r, ext4r) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    col = {"g05": "#4C72B0", "groot": "#55A868", "smolvla": "#C44E52", "pear": "#8172B2"}
    ms = [m for m in MODELS if m in tab]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, m in enumerate(ms):
        t = tab[m]
        ax.bar(i, 100 * t["rate"], color=col[m], width=0.6)
        ax.errorbar(i, 100 * t["rate"], yerr=[[100 * (t["rate"] - t["wilson95"][0])], [100 * (t["wilson95"][1] - t["rate"])]],
                    color="black", capsize=6, lw=1.2)
        ax.text(i, 100 * t["wilson95"][1] + 2, f"{t['k']}/{t['n']}", ha="center", fontsize=9)
        if t["n"] < 50:  # unfinished episodes: the range if all of them failed / succeeded
            lo_b, hi_b = 100 * t["k"] / 50, 100 * (t["k"] + 50 - t["n"]) / 50
            ax.plot([i + 0.36, i + 0.36], [lo_b, hi_b], color="#666666", lw=3, alpha=0.6)
            ax.text(i + 0.42, hi_b, "all 50:" + chr(10) + "%d-%d%%" % (lo_b, hi_b), fontsize=7, va="top", color="#444444")
    ax.set_xticks(range(len(ms)), [LABEL[m] + ("" if tab[m]["n"] == 50 else chr(10) + f"({tab[m]['n']} of 50 finished)") for m in ms])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Success on LIBERO-10 (%)")
    ax.set_title("Same 50 episodes (10 tasks x 5 fixed init states), Wilson 95% CI", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "success_ci.png", dpi=150)
    plt.close(fig)

    s = ext1r["summary"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for m, d in s.items():
        hs = sorted(d)
        axs[0].errorbar(hs, [100 * d[h]["rate"] for h in hs],
                        yerr=[[100 * (d[h]["rate"] - d[h]["wilson95"][0]) for h in hs], [100 * (d[h]["wilson95"][1] - d[h]["rate"]) for h in hs]],
                        marker="o", capsize=3, color=col[m], label=LABEL[m])
        jh = [h for h in hs if d[h]["action_jerk_mean"] is not None]
        axs[1].plot(jh, [d[h]["action_jerk_mean"] for h in jh], marker="o", color=col[m], label=LABEL[m])
    for a in axs:
        a.set_xscale("log", base=2)
        a.set_xlabel("Actions executed per policy call (log2)")
        a.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Success (%), n=20 per point")
    axs[0].set_ylim(-5, 108)
    axs[1].set_ylabel("Mean |a(t) - a(t-1)| (arm dims)")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Success vs execution horizon", fontsize=10)
    axs[1].set_title("Action smoothness (lower = smoother)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "horizon_sweep.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cats = [c for c in ["step 1", "step 2", "step 3", "single-step task"]
            if any(ext3r[m]["failed_step_counts"].get(c) for m in ext3r)]
    ccol = {"step 1": "#DD8452", "step 2": "#937860", "step 3": "#8C8C8C", "single-step task": "#DA8BC3"}
    ms3 = [m for m in MODELS if m in ext3r]
    bottom = [0] * len(ms3)
    for c in cats:
        vals = [ext3r[m]["failed_step_counts"].get(c, 0) for m in ms3]
        ax.bar(range(len(ms3)), vals, bottom=bottom, color=ccol[c], label=f"failed at {c}" if c != "single-step task" else "failed (task 5, one step)")
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, m in enumerate(ms3):
        ax.text(i, bottom[i] + 0.4, f"PC={ext3r[m]['mean_partial_credit_all']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ms3)), [LABEL[m] + chr(10) + f"{ext3r[m]['n_fail']} of {ext3r[m]['n']} failed" for m in ms3])
    ax.set_ylabel("Failed episodes")
    ax.set_title("Where failures stop (BDDL goal predicates); PC = mean partial credit", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "failure_steps.png", dpi=150)
    plt.close(fig)

    if ext2r:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        oc = [o for o in ["instructed", "both", "default", "neither", "unfinished"] if any(ext2r[m]["counts"].get(o) for m in ext2r)]
        ocol = {"instructed": "#55A868", "both": "#64B5CD", "default": "#C44E52", "neither": "#BBBBBB", "unfinished": "#EEEEEE"}
        ms2 = [m for m in MODELS if m in ext2r]
        left = [0] * len(ms2)
        for o in oc:
            vals = [ext2r[m]["counts"].get(o, 0) / ext2r[m]["n"] for m in ms2]
            ax.barh(range(len(ms2)), vals, left=left, color=ocol[o], label=o if o != "unfinished" else "unfinished (PEAR, stopped)",
                    edgecolor="#999999" if o == "unfinished" else None)
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(range(len(ms2)), [f"{LABEL[m]} (n={ext2r[m]['n']})" for m in ms2])
        ax.set_xlabel("Share of counterfactual episodes")
        ax.set_title("Swapped instruction, same scene: which task got done?", fontsize=10)
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        fig.savefig(FIG / "counterfactual.png", dpi=150)
        plt.close(fig)


def main() -> None:
    main_eps = load_main()
    tab = main_table(main_eps)
    lat = vram_latency(main_eps)
    trackers = load_trackers(main_eps)
    e3 = ext3(main_eps, trackers)
    e1 = ext1(main_eps)
    e2 = ext2()
    e4 = ext4(main_eps)
    e5 = ext5(main_eps)
    e6 = ext6(main_eps, lat)
    summary = {"main": tab, "latency_vram": lat, "ext1_horizon": e1["summary"],
               "ext2_counterfactual": e2, "ext3_failure_step": {m: {k: v for k, v in d.items() if k != "rows"} for m, d in e3.items()},
               "ext4_perturbation": e4, "ext5_paired": e5, "ext6_power_cost": e6}
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with open(OUT / "per_episode_main.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "episode", *[f"{m}_success" for m in MODELS], *[f"{m}_partial_credit" for m in MODELS]])
        for t in range(10):
            for e in range(5):
                row = [t, e]
                row += [int(main_eps[m][(t, e)]["success"]) if (t, e) in main_eps[m] else "" for m in MODELS]
                row += [next((r["partial"] for r in e3.get(m, {}).get("rows", []) if (r["task_id"], r["episode"]) == (t, e)), "") for m in MODELS]
                w.writerow(row)
    with open(OUT / "main_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "k", "n", "rate", "wilson_lo", "wilson_hi", "chunk_ms_median", "gpu_peak_mib", "torch_peak_mib", "gpu"])
        for m in MODELS:
            if m in tab:
                l = lat.get(m, {})
                w.writerow([m, tab[m]["k"], tab[m]["n"], tab[m]["rate"], *tab[m]["wilson95"], l.get("chunk_ms_median"),
                            l.get("gpu_peak_mib"), l.get("torch_peak_mib"), l.get("gpu")])
    figures(tab, e1, e3, e2, e4)
    print(json.dumps({k: v for k, v in summary.items() if k in ("main", "latency_vram")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
