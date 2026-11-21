"""Workarounds for lerobot 0.6.1 + gymnasium 1.4 AsyncVectorEnv evaluation of PushT / ALOHA.

1. lerobot builds AsyncVectorEnv with context="forkserver" but imports gym_pusht / gym_aloha only in the
   parent, so the workers cannot resolve the gym ids. This module is preloaded into the forkserver
   (see t3_launch.py), which registers the ids there; every worker forked from it inherits them.
2. lerobot's rollout calls env.call("task_description") (then "task") and catches AttributeError in the
   parent. Under AsyncVectorEnv the AttributeError is raised inside each worker, and gymnasium shuts the
   worker down, so the next step crashes. PushT and ALOHA are single-task envs, so we give their env
   classes empty `task_description` / `task` strings, which is what lerobot falls back to anyway ([""] * n_envs).
"""

import importlib

import gymnasium as gym
import gym_aloha  # noqa: F401
import gym_pusht  # noqa: F401


def _patch():
    for env_id, spec in list(gym.registry.items()):
        if not (env_id.startswith("gym_pusht/") or env_id.startswith("gym_aloha/")):
            continue
        ep = spec.entry_point
        if isinstance(ep, str):
            mod, _, attr = ep.partition(":")
            cls = getattr(importlib.import_module(mod), attr)
        else:
            cls = ep
        if isinstance(cls, type):
            # set both names: lerobot tries "task_description", then "task" (gym_pusht's base raises for the first)
            for name in ("task_description", "task"):
                if not isinstance(getattr(cls, name, None), str):
                    setattr(cls, name, "")


_patch()
