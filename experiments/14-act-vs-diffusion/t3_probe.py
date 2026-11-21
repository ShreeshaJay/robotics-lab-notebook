"""Extension 4: PushT multimodality probe.

From fixed PushT start states, draw many action chunks from each trained policy for the *same*
observation and compare where they go. Diffusion and flow heads sample fresh noise per draw. ACT is run
two ways: as deployed (latent z = 0, deterministic) and with z drawn from its N(0, I) prior. The MSE
head is deterministic by construction.

State selection (stated, to avoid cherry-picking one picture): among `candidate_seeds` reset states,
score each by how clearly the diffusion policy's samples split into two clusters (k-means, k=2, on the
first 8 executed actions), require the smaller cluster to hold >= 20% of samples, and keep the top_k.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/t3")
import t3_heads  # noqa: E402

N_EXEC = 8


def _load(path, head, env_cfg):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = Path(path)
    cfg.device = "cuda"
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    if cfg.type == "diffusion":
        t3_heads.apply_to_instance(policy.diffusion, head)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    return policy, pre, post


def _sample(policy, pre, post, obs, n, act_prior=False):
    from lerobot.envs import preprocess_observation

    batch = {"pixels": np.repeat(obs["pixels"][None], n, 0), "agent_pos": np.repeat(obs["agent_pos"][None], n, 0)}
    o = preprocess_observation(batch)
    o["task"] = [""] * n
    o = pre(o)
    policy.reset()
    restore = None
    if act_prior:
        proj = policy.model.encoder_latent_input_proj
        orig = proj.forward
        proj.forward = lambda z: orig(z + torch.randn_like(z))
        restore = (proj, orig)
    try:
        with torch.inference_mode():
            first = policy.select_action(o)
            queue = policy._queues["action"] if hasattr(policy, "_queues") else policy._action_queue
            acts = [first] + list(queue)
    finally:
        if restore:
            restore[0].forward = restore[1]
    acts = [post(a).detach().float().cpu().numpy() for a in acts]
    return np.stack(acts, 1)[:, :N_EXEC]  # (n, N_EXEC, 2), pixel coordinates in [0, 512]


def _two_cluster_score(traj):
    from sklearn.cluster import KMeans

    x = traj.reshape(len(traj), -1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x)
    lab = km.labels_
    frac = min(lab.mean(), 1 - lab.mean())
    c = km.cluster_centers_
    within = np.sqrt(((x - c[lab]) ** 2).sum(1)).mean() + 1e-6
    sep = np.linalg.norm(c[0] - c[1])
    return float(sep / within), float(frac), c, lab


def _png(arr):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def run(spec, vol_root):
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    from lerobot.envs.configs import PushtEnv

    torch.manual_seed(0)
    np.random.seed(0)
    env_cfg = PushtEnv()
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                   visualization_width=384, visualization_height=384)
    pols = {k: _load(v["path"], v.get("head", "diffusion"), env_cfg) for k, v in spec["policies"].items()}
    n = spec.get("n_samples", 64)

    scored = []
    for s in spec["candidate_seeds"]:
        obs, _ = env.reset(seed=int(s))
        tr = _sample(*pols["diffusion"], obs, n)
        score, frac, _, _ = _two_cluster_score(tr)
        scored.append((s, score, frac))
    eligible = [x for x in scored if x[2] >= 0.2]
    eligible.sort(key=lambda x: -x[1])
    chosen = [x[0] for x in eligible[: spec.get("top_k", 3)]]

    states = []
    for s in chosen:
        obs, _ = env.reset(seed=int(s))
        frame = env.render()
        st = {"seed": int(s), "agent_pos": obs["agent_pos"].tolist(), "frame_png_b64": _png(frame), "samples": {}}
        dp = _sample(*pols["diffusion"], obs, n)
        score, frac, centers, lab = _two_cluster_score(dp)
        st["dp_score"], st["dp_minority_frac"] = score, frac
        mode_dist = float(np.linalg.norm(centers[0] - centers[1]))
        variants = [(k, k, False) for k in pols]
        if "act" in pols:
            variants.append(("act_prior", "act", True))
        for label, key, prior in variants:
            tr = dp if label == "diffusion" else _sample(*pols[key], obs, n, act_prior=prior)
            x = tr.reshape(n, -1)
            d = np.linalg.norm(x[:, None, :] - centers[None], axis=-1)  # (n, 2)
            near = d.argmin(1)
            st["samples"][label] = {
                "traj": np.round(tr, 2).tolist(),
                "frac_mode0": float((near == 0).mean()),
                "mean_dist_to_nearest_mode_over_mode_gap": float(d.min(1).mean() / mode_dist),
                "spread_px": float(np.sqrt(((x - x.mean(0)) ** 2).sum(1)).mean() / np.sqrt(N_EXEC)),
            }
        states.append(st)
    return {"name": spec["name"], "scored_candidates": scored, "chosen": chosen, "states": states,
            "n_samples": n, "n_exec": N_EXEC}
