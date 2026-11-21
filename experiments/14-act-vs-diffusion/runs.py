"""Run specifications for every phase of tutorial 3. Step counts are stated here and in README.md."""

PUSHT_REV = "7628202a2180972f291ba1bc6723834921e72c19"
ALOHA_REV = "8ab660912970111cbb26738b11458e6fc4a4aed1"

DS = {
    "pusht": ["--dataset.repo_id=lerobot/pusht", f"--dataset.revision={PUSHT_REV}"],
    "aloha": ["--dataset.repo_id=lerobot/aloha_sim_insertion_scripted", f"--dataset.revision={ALOHA_REV}"],
}

# Policy settings. PushT diffusion = the published lerobot/diffusion_pusht config. ACT on ALOHA = the
# published lerobot/act_aloha_sim_insertion_human config. The other two cells have no published LeRobot
# recipe, so they use stated, minimal adaptations (see README.md).
POLICY = {
    ("diffusion", "pusht"): [
        "--policy.type=diffusion", "--policy.n_obs_steps=2", "--policy.horizon=16", "--policy.n_action_steps=8",
        "--policy.drop_n_last_frames=7", "--policy.crop_shape=[84,84]", "--policy.use_group_norm=true",
        "--policy.pretrained_backbone_weights=null", "--policy.use_separate_rgb_encoder_per_camera=false",
    ],
    ("act", "pusht"): ["--policy.type=act", "--policy.chunk_size=20", "--policy.n_action_steps=10"],
    ("act", "aloha"): ["--policy.type=act", "--policy.chunk_size=100", "--policy.n_action_steps=100"],
    ("diffusion", "aloha"): [
        "--policy.type=diffusion", "--policy.n_obs_steps=2", "--policy.horizon=64", "--policy.n_action_steps=32",
        "--policy.drop_n_last_frames=31", "--policy.resize_shape=[240,320]", "--policy.crop_ratio=0.95",
    ],
}

BATCH = {"pusht": 64, "aloha": 8}
STEPS = {"pusht": 40_000, "aloha": 40_000}  # set from the pilot projection, see COSTS.md

EVAL = {
    "pusht": {"env": "pusht", "n_episodes": 100, "batch_size": 50, "seed": 1000, "overrides": []},
    "aloha": {"env": "aloha", "n_episodes": 50, "batch_size": 10, "seed": 1000,
              "overrides": ["--env.task=AlohaInsertion-v0"]},
}
SEEDS = [1, 2, 3]


def train_args(method, task, steps, seed, batch=None, extra=()):
    b = batch or BATCH[task]
    keys = {a.split("=")[0] for a in extra}
    pol = [a for a in POLICY[(method, task)] if a.split("=")[0] not in keys]
    return [*DS[task], *pol, *extra, f"--batch_size={b}", f"--steps={steps}",
            f"--save_freq={min(10_000, steps)}", f"--log_freq={max(50, steps // 100)}", f"--seed={seed}", "--num_workers=6"]


def ev(task, tag="main", n=None, bs=None, overrides=()):
    e = dict(EVAL[task])
    e["overrides"] = list(e["overrides"]) + list(overrides)
    e["tag"] = tag
    if n:
        e["n_episodes"] = n
    if bs:
        e["batch_size"] = bs
    return e


def pilot():
    chain = []
    for method, task, b in [("act", "aloha", 8), ("diffusion", "aloha", 8), ("diffusion", "pusht", 64),
                            ("act", "pusht", 64)]:
        chain.append({
            "name": f"pilot-{method}-{task}",
            "train": {"args": train_args(method, task, 300, 0, batch=b)},
            "evals": [ev(task, "pilot", n=10 if task == "pusht" else 4, bs=10 if task == "pusht" else 4)],
        })
    return [{"name": "pilot", "chain": chain}]


TRAIL = {"T3_SPACING": "trailing"}


def ext3_evals(task):
    """Extension 3: same checkpoint, fewer denoising steps. 'trailing' spacing so that a 1-step sampler
    starts from the last (pure-noise) timestep; at 100 steps it equals the default 'leading' grid."""
    out = []
    for tag, ov in [("ddpm10", ["--policy.num_inference_steps=10"]), ("ddpm1", ["--policy.num_inference_steps=1"]),
                    ("ddim10", ["--policy.noise_scheduler_type=DDIM", "--policy.num_inference_steps=10"]),
                    ("ddim1", ["--policy.noise_scheduler_type=DDIM", "--policy.num_inference_steps=1"])]:
        e = ev(task, tag, overrides=ov)
        e["env_extra"] = TRAIL
        out.append(e)
    return out


def flow_evals(task):
    return [ev(task, "main"), ev(task, "euler10", overrides=["--policy.num_inference_steps=10"]),
            ev(task, "euler1", overrides=["--policy.num_inference_steps=1"])]


ENS = ["--policy.n_action_steps=1", "--policy.temporal_ensemble_coeff=0.01"]


def main_runs():
    out = []
    for task in ["pusht", "aloha"]:
        for method in ["act", "diffusion"]:
            for s in SEEDS:
                evals = [ev(task)]
                if method == "diffusion":
                    evals += ext3_evals(task)
                if method == "act" and task == "aloha":
                    evals.append(ev(task, "ens", overrides=ENS))
                out.append({"name": f"{method}-{task}-s{s}",
                            "train": {"args": train_args(method, task, STEPS[task], s)},
                            "evals": evals})
    return out


CHUNK_SEEDS = [1]


def ext2_runs():
    out = []
    for k in [1, 10, 50]:
        for s in CHUNK_SEEDS:
            evals = [ev("aloha", "main")] + ([] if k == 1 else [ev("aloha", "ens", overrides=ENS)])
            out.append({"name": f"act-aloha-c{k}-s{s}",
                        "train": {"args": train_args("act", "aloha", STEPS["aloha"], s,
                                                     extra=[f"--policy.chunk_size={k}", f"--policy.n_action_steps={k}"])},
                        "evals": evals})
    return out


def ext1_runs():
    out = [{"name": f"flow-pusht-s{s}", "train": {"head": "flow", "args": train_args("diffusion", "pusht", STEPS["pusht"], s)},
            "evals": flow_evals("pusht")} for s in SEEDS]
    out.append({"name": "mse-pusht-s1", "train": {"head": "mse", "args": train_args("diffusion", "pusht", STEPS["pusht"], 1)},
                "evals": [ev("pusht", "main")]})
    return out


def probe_runs():
    ck = "/ckpts/runs/{}/checkpoints/last/pretrained_model"
    return [{"name": "probe-pusht", "n_samples": 64, "top_k": 3, "candidate_seeds": list(range(1000, 1040)),
             "policies": {"diffusion": {"path": ck.format("diffusion-pusht-s1")},
                          "flow": {"path": ck.format("flow-pusht-s1"), "head": "flow"},
                          "act": {"path": ck.format("act-pusht-s1")},
                          "mse": {"path": ck.format("mse-pusht-s1"), "head": "mse"}}}]


PHASES = {"pilot": pilot, "main": main_runs, "ext2": ext2_runs, "ext1": ext1_runs, "probe": probe_runs}


def pilot2():
    """Re-run the pilot evals after the libopengl0 fix, plus a bf16-autocast throughput check."""
    chain = []
    for method, task in [("act", "aloha"), ("diffusion", "aloha"), ("diffusion", "pusht"), ("act", "pusht")]:
        chain.append({"name": f"pilot-{method}-{task}",
                      "evals": [ev(task, "pilot", n=10, bs=10)]})
    chain.append({"name": "pilot-diffusion-pusht-amp",
                  "train": {"args": train_args("diffusion", "pusht", 300, 0, extra=["--policy.use_amp=true"])},
                  "evals": []})
    chain.append({"name": "pilot-diffusion-aloha-amp",
                  "train": {"args": train_args("diffusion", "aloha", 300, 0, extra=["--policy.use_amp=true"])},
                  "evals": []})
    return [{"name": "pilot2", "chain": chain}]


PHASES["pilot2"] = pilot2


def pilot3():
    """Evals after the forkserver fix + fp32 vs bf16-autocast throughput on the same GPU."""
    chain = [{"name": f"pilot-{m}-{t}", "evals": [ev(t, "pilot", n=10, bs=10)]}
             for m, t in [("act", "aloha"), ("diffusion", "aloha"), ("diffusion", "pusht"), ("act", "pusht")]]
    for tag, extra in [("fp32", []), ("amp", ["--policy.use_amp=true"])]:
        chain.append({"name": f"pilot3-diffusion-pusht-{tag}",
                      "train": {"args": train_args("diffusion", "pusht", 200, 0, extra=extra)}, "evals": []})
    return [{"name": "pilot3", "chain": chain}]


PHASES["pilot3"] = pilot3


def pilot4():
    """Eval timing after the t3_envfix workaround (10 episodes per pilot checkpoint)."""
    return [{"name": "pilot4", "chain": [{"name": f"pilot-{m}-{t}", "evals": [ev(t, "pilot", n=10, bs=10)]}
             for m, t in [("diffusion", "pusht"), ("act", "pusht")]]}]


PHASES["pilot4"] = pilot4


def all_training():
    return main_runs() + ext2_runs() + ext1_runs()


PHASES["all"] = all_training


def pilot5():
    """Smoke test of the flow / MSE heads, the DDIM-trailing overrides and ACT temporal ensembling."""
    small = lambda e: {**e, "n_episodes": 10, "batch_size": 10}  # noqa: E731
    chain = [
        {"name": "pilot-flow-pusht", "train": {"head": "flow", "args": train_args("diffusion", "pusht", 100, 0)},
         "evals": [small(e) for e in flow_evals("pusht")]},
        {"name": "pilot-mse-pusht", "train": {"head": "mse", "args": train_args("diffusion", "pusht", 100, 0)},
         "evals": [small(ev("pusht", "main"))]},
        {"name": "pilot-diffusion-pusht", "evals": [small(e) for e in ext3_evals("pusht")]},
        {"name": "pilot-act-aloha", "evals": [small(ev("aloha", "ens", overrides=ENS))]},
    ]
    return [{"name": "pilot5", "chain": chain}]


PHASES["pilot5"] = pilot5
