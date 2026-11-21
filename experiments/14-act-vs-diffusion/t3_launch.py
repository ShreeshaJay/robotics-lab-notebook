"""Entry point used inside the Modal container: `python t3_launch.py {train|eval} <lerobot CLI args>`.

Applies the action-head patch from t3_heads.py (env var T3_HEAD, or the `t3_head.txt` marker next to a
checkpoint given with --policy.path) and then hands over to the stock lerobot 0.6.1 CLI `main()`.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t3_heads  # noqa: E402


def _head_from_policy_path(argv):
    for a in argv:
        if a.startswith("--policy.path="):
            p = Path(a.split("=", 1)[1])
            for cand in (p / "t3_head.txt", p.parent / "t3_head.txt", p.parent.parent / "t3_head.txt",
                         p.parent.parent.parent / "t3_head.txt"):
                if cand.exists():
                    return cand.read_text().strip()
    return None


def main():
    mode = sys.argv[1]
    argv = sys.argv[2:]
    head = os.environ.get("T3_HEAD") or _head_from_policy_path(argv) or "diffusion"
    t3_heads.apply(head)
    spacing = os.environ.get("T3_SPACING")  # e.g. "trailing": lets a 1-step sampler start from t=T-1
    if spacing:
        from lerobot.policies.diffusion import modeling_diffusion as md

        orig = md._make_noise_scheduler
        md._make_noise_scheduler = lambda name, **kw: orig(name, timestep_spacing=spacing, **kw)
        print(f"[t3_launch] noise scheduler timestep_spacing={spacing}", flush=True)
    sys.argv = [f"lerobot-{mode}"] + argv
    if mode == "train":
        from lerobot.scripts.lerobot_train import main as m
    elif mode == "eval":
        # lerobot 0.6.1 builds AsyncVectorEnv with context="forkserver" and imports gym_pusht / gym_aloha only in
        # the parent, so the workers fail with "Namespace gym_pusht not found". Preloading the env packages in the
        # forkserver registers the gym ids there, and every worker forked from it inherits them.
        import multiprocessing as mp

        import t3_envfix  # noqa: F401  (patch in the parent too, for SyncVectorEnv)

        mp.set_forkserver_preload(["t3_envfix"])
        from lerobot.scripts.lerobot_eval import main as m
    else:
        raise SystemExit(f"unknown mode {mode}")
    m()


if __name__ == "__main__":
    main()
