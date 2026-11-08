"""Smallest-possible LLM-as-policy eval with Inspect Robots on the CPU-only CubePick mock world.

1. Runs the free scripted baseline (no API calls) on the same scenes.
2. Runs an LLM agent policy (default: Claude Haiku 5.5 via the native Messages wire),
   one scene per eval() call, so a cumulative spend guard can stop between trials.
3. Prints scores, token usage and estimated cost; checks that no log file contains the key.

The API key is read from a dotenv file (path passed on the command line) into this
process only. It is never printed or written anywhere by this script.

Usage:
    python run_cubepick_agent.py --dotenv PATH/TO/.env --scenes 3 --log-dir logs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from inspect_robots import eval as ir_eval
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length, min_distance_to_goal, success_at_end
from inspect_robots.task import Task
from inspect_robots_agent.policy import LLMAgentPolicy

# USD per million tokens. Haiku 5.5 list price (<=100K-token prompts): $0.10 in / $0.50 out.
# Cache read/write rates are ASSUMED at the usual 0.1x / 1.25x of input.
PRICE = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}


def load_key(dotenv_path: str, name: str) -> str:
    values = dotenv_values(dotenv_path)
    value = values.get(name)
    if not value:
        raise SystemExit(f"{name} not found in dotenv file")
    os.environ[name] = value
    print(f"loaded {name} into process env (value not shown)")
    return value


def make_task(scene_seed: int, max_steps: int) -> Task:
    return Task(
        name="cubepick-reach-mini",
        scenes=[Scene(id=f"layout-{scene_seed}", instruction="reach the cube", init_seed=scene_seed)],
        scorer=[success_at_end(), episode_length(), min_distance_to_goal()],
        max_steps=max_steps,
    )


def cost_of(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) * PRICE["input"]
        + usage.get("output_tokens", 0) * PRICE["output"]
        + usage.get("cache_read_input_tokens", 0) * PRICE["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]
    ) / 1e6


def summarize(log) -> dict:
    s = log.samples[0]
    meta = s.trial_metadata[0] if s.trial_metadata else {}
    return {
        "scene": s.scene_id,
        "status": s.status,
        "termination": s.termination_reasons[0] if s.termination_reasons else None,
        "scores": s.epochs[0] if s.epochs else {},
        "llm_usage": meta.get("llm_usage", {}),
        "duration_s": round(log.stats.duration_s, 2),
        "steps": log.stats.total_steps,
        "error": s.error or log.error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dotenv", required=True)
    ap.add_argument("--model", default="anthropic/claude-haiku-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--max-llm-calls", type=int, default=25)
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--image-horizon", default="none",
                    help="'none' = send full image history (default here; see README), or an int")
    args = ap.parse_args()

    key = load_key(args.dotenv, "ANTHROPIC_API_KEY")
    results: dict[str, list] = {"scripted": [], "agent": []}

    for i in range(args.scenes):
        (log,) = ir_eval(make_task(i, args.max_steps), ScriptedPolicy(), CubePickEmbodiment(),
                         log_dir=f"{args.log_dir}/scripted")
        results["scripted"].append(summarize(log))

    spent = 0.0
    for i in range(args.scenes):
        if spent >= args.budget_usd:
            print(f"budget guard: stopping before scene {i} (spent ${spent:.4f})")
            break
        policy = LLMAgentPolicy(model=args.model, wire="messages", effort=args.effort,
                                images="always", max_llm_calls=args.max_llm_calls,
                                max_output_tokens=4000,
                                image_horizon=None if args.image_horizon == "none" else int(args.image_horizon))
        (log,) = ir_eval(make_task(i, args.max_steps), policy, CubePickEmbodiment(),
                         log_dir=f"{args.log_dir}/agent")
        row = summarize(log)
        row["cost_usd"] = round(cost_of(row["llm_usage"]), 6)
        spent += row["cost_usd"]
        results["agent"].append(row)
        print(json.dumps(row, default=str))

    results["agent_total_cost_usd"] = round(spent, 6)
    print(json.dumps(results, indent=2, default=str))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.log_dir) / "summary.json").write_text(json.dumps(results, indent=2, default=str))

    leaked = any(key in p.read_text(errors="ignore")
                 for p in Path(args.log_dir).rglob("*") if p.is_file() and p.stat().st_size < 50_000_000)
    print(f"api key present in any log file: {leaked}")


if __name__ == "__main__":
    main()
