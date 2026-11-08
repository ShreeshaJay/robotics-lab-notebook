# 08 · ASPIRE: an LLM coding agent that writes and refines robot skills

**Date:** 2026-11-08 · **Track:** Evals / policies · **Status:** done (CPU-only checks; no simulator rollout) · **Spend:** $0.003 of API usage, $0 compute

## Goal

Find out what NVIDIA's ASPIRE v1.0 is and what it needs to run: license, simulators, LLMs, hardware and
cost per task. Run whatever part works on a laptop CPU, write a GPU run plan for the rest, and compare it
with Inspect Robots (experiment 02).

## Sources

- Paper: *ASPIRE: Agentic /Skills Discovery for Robotics*, Lu et al., arXiv:2607.00272. v1 came out 2026-06-30; v2 (2026-10-20) adds the multi-model results.
- Project page: <https://research.nvidia.com/labs/gear/aspire/>
- Code: <https://github.com/NVlabs/ASPIRE>, pinned at `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7` ("ASPIRE v1.0 release", 2026-10-15), cloned to `C:\Users\shree\af-scout-lab\aspire\`.
- Announcement: a LinkedIn post by the GEAR lab co-lead. It claims "~99%" benchmark success and "$12 per task with Sonnet 5.5". Both claims are checked below.

## What it is

- **The method.** A frontier coding agent (Claude Code or Codex) writes code-as-policy (CaP) Python programs against a typed robot API, then debugs them from recorded rollouts. Fixes that work become text "skills" in a shared library. The paper describes this as continual learning: "training" means refining skills, and the "model" is a repo of skills, not weights.
- **Agent roles.** A *coordinator* picks a curriculum and dispatches one *actor* agent per task. Actors debug failure modes on development seeds. A *verifier* re-solves the task using only the skill library plus the proposed skills; a skill is promoted only if this re-solve succeeds. An optional *evolutionary search* runs on tasks that stay weak.
- **Code layout.**
  - `cap-harness` package: the typed API, a restricted program executor, run recording, validation, and batch running.
  - Runbooks packaged as agent skills (`.claude/skills/`, `.codex/skills/`). There is no Python entry point that calls an LLM. The LLM is whatever model drives the coding-agent session.
- **Program sandbox.** Generated programs pass an AST gate: only `import math`, no classes, no `with`, and no `_`-prefixed names or attributes. Tools are injected from an allowlist. The README still warns that trial processes are not a security sandbox. Generated code should run on an isolated host that holds no credentials.
- **Benchmarks and simulators.** No Isaac Lab and no ManiSkill.
  - **LIBERO-Pro:** 80 tasks across Goal, Object, Spatial and Long. Franka Panda in MuJoCo via a pinned robosuite fork. 50 held-out seeds per task.
  - **Robosuite:** contact-heavy tabletop tasks, including bimanual. 100 held-out seeds.
  - **BEHAVIOR-1K:** long-horizon mobile manipulation in simulated homes. R1 Pro robot on OmniGibson / Isaac Sim. 25 held-out seeds.
  - **LIBERO-90 / LIBERO-Long:** used for zero-shot transfer.
  - **Real robot:** bimanual YAM arms ("System 2" over MCP tools). Demos: Rubik's Cube, bottle cap, pill in bottle, cup nesting, table bussing.
- **Perception and motion services** (each runs as a separate local service):
  - SAM3 segmentation: gated Hugging Face weights.
  - Contact-GraspNet.
  - PyRoki IK.
  - cuRobo motion planning: compiles CUDA extensions.
- **LLMs.** The method doesn't depend on one model. Paper v2 (Table 18) tests 12 models: Claude Opus 5.5, Sonnet 5.5, Fable 5.1, Opus 5, Haiku 5.5 and Opus 4.6; GPT-6 Astra, GPT-6.1 Sol, GPT-6 Sol and GPT-6 Luna; Kimi K3 and Qwen 3.8 Max (both open weights). The page also lists Gemini models, which are not in that table. v1 used Opus 4.6. The real-robot demos used GPT-5.5 and GPT-6 Astra.
- **Hardware.**
  - OS: Linux x86_64 only. The uv lock is restricted to `sys_platform == 'linux'`, and offscreen rendering needs EGL.
  - GPU: an NVIDIA GPU with a working driver. Shipped single-GPU profiles: RTX 4090 (sm_89), RTX 5090, RTX 5880 and L40. All services share GPU 0.
  - BEHAVIOR-1K adds Isaac Sim needs: CUDA 12.x `nvcc`, glibc 2.35 or newer, a Vulkan ICD, 32 GB RAM, about 12 GB for the venv and about 40 GB of datasets.
  - The paper gives no GPU type or count.

## Claims checked

| Claim | Source | Status |
|-|-|-|
| "~99%" success | README news; paper v2 abstract | Supported. 99% average over the 80 LIBERO-Pro tasks with Opus 5.5. 100% on *representative* (selected) Robosuite and BEHAVIOR-1K tasks |
| "$12/task with Sonnet 5.5" | Paper v2 Table 18 | Supported. $12.11 per task at 98.8% success, on the 10 LIBERO-Pro Goal tasks (position perturbation). This is list-price LLM cost only; GPU time is not counted |
| Per-task costs scale with tokens | Tables 20 and 22 | About 97% of tokens are cache reads. Opus 5.5 on LIBERO-Pro O/G/S: 1,538M tokens, 8.0M output, 10,555 sim rollouts, 7,896 LLM calls, $10.8 per task. I recomputed it at today's Opus 5.5 rates ($4 input / $0.20 cache read / $20 output per MTok) and got about $610–645, against $650 reported, so the table is consistent |

Selected rows of Table 18 (10 LIBERO-Pro Goal tasks, LLM list-price cost):

| Model | Success | $ / task | Agent min / task |
|-|-|-|-|
| Claude Opus 5.5 | 99.2% | 18.29 | 51 |
| Claude Sonnet 5.5 | 98.8% | 12.11 | 56 |
| Claude Haiku 5.5 | 68.0% | 3.97 | 54 |
| GPT-6 Astra | 94.4% | 35.37 | 31 |
| GPT-6.1 Sol | 88.0% | 9.52 | 49 |
| Kimi K3 (open) | 78.4% | 58.19 | 139 |

Current Anthropic list prices (platform.claude.com pricing page, checked 2026-11-08), in $ per MTok:

| Model | Input | Cache read | Output |
|-|-|-|-|
| Opus 5.5 | 4 | 0.20 | 20 |
| Sonnet 5.5 | 2 | 0.10 | 10 |
| Haiku 5.5 (≤100K-token prompts) | 0.10 | 0.01 | 0.50 |

These are the same rates the paper's cost tables appear to use.

## What I tried

1. I read the paper (v2 HTML, including the appendix cost tables), the project page, the README, `AGENTS.md`, `docs/host-setup.md`, `docs/libero-pro.md`, the iterative-debugging skill, `pyproject.toml`, the start of `scripts/bootstrap_libero.sh`, `runtime.py` (the sandbox) and `registry.py`. I ran no bootstrap or service script. They need Linux, uv, CUDA and a Hugging Face token.
2. I made a fresh venv (`python -m venv .venv-scout`, Python 3.11.9 on Windows 11) and ran `pip install -e ".[dev]"` with the light agent/LIBERO dependencies: imageio, opencv-headless, fastapi, uvicorn, mcp and jsonschema. I installed no simulator, torch or submodule.
3. I ran the unit test suite on CPU. Most tests use fakes (fake LIBERO env, fake batch runner), so this is the only "mock path" the repo has. There is no CPU simulator path.
4. I ran a tiny code-generation probe ([`results/haiku_cap_probe.py`](results/haiku_cap_probe.py)) as one call to `claude-haiku-5-5` at effort medium:
   - Input: the simulation part of `docs/api-reference.md` plus the LIBERO-Pro task `libero_goal_swap:0` ("open the middle drawer of the cabinet").
   - Output: one CaP program, put through ASPIRE's own sandbox validator and tool allowlist.
   - Control: the repo's reference program for the same task (`examples/libero/open_middle_drawer.py`) went through the same gate.
   - The key was loaded from a `.env` file into the process environment and never printed.

   This is **not** an ASPIRE run: there is no rollout, no debugging loop and no skill library. A real one-task ASPIRE run with Haiku 5.5 costs about $3.97 (Table 18), which is over the $3 cap, and it needs a GPU simulator anyway.

## Results

- **Unit tests (CPU, Windows):** 486 passed, 34 failed, 68 errors, 6 skipped ([`results/pytest_cpu_windows.txt`](results/pytest_cpu_windows.txt)).
  - Failures and errors are mostly POSIX-only or missing real-robot dependencies: `mink`, `mujoco`, `resource`, `os.pread`, symlink privilege, path separators, and the YAM calibration hash.
  - A few are docs-drift assertions. For example, the README links `LICENSES/README.md` and other docs on the `v0.1` tag; `LICENSES/` does not exist on `main`.
  - The core contracts, registry, sandbox, LIBERO adapter fakes, codecs and validation logic pass.
- **Haiku 5.5 code-generation probe** ([`results/probe_summary.json`](results/probe_summary.json), program in [`results/haiku_program.py`](results/haiku_program.py)):

| | Haiku 5.5 program | Reference program |
|-|-|-|
| Lines | 85 | 127 |
| Sandbox validator | pass | pass |
| Unknown bare calls | 0 | 0 |
| Uses perception (segment + point cloud) | yes | yes |
| Allowlisted tools called | 8: `segment_text`, `mask_to_point_cloud`, `estimate_geometry`, `move_to_pose`, `open/close_gripper`, `get_robot_state`, `get_task_context` | 7 |
| Tokens in / out | 7,222 / 4,513 | – |
| Latency | 20.1 s | – |
| Cost | **$0.003** | – |

- **What the probe shows:** a cheap model can produce a program that passes ASPIRE's static gate on the first try. It says nothing about task success. In the paper, Haiku 5.5 reaches 68% only *after* the full debug loop, at about $4 per task.

## Cost

- **Actual spend:** $0.003 of Anthropic API (Haiku 5.5, one call) and $0 compute.
- **One real ASPIRE task (LIBERO-Pro Goal, held-out evaluation included):** about $12 of LLM cost with Sonnet 5.5, about $4 with Haiku 5.5 (68% success), or about $18 with Opus 5.5, plus GPU time (see the plan below).
- **A paper-scale suite costs far more:**
  - 10 Goal tasks: $121 with Sonnet 5.5.
  - LIBERO-Pro O/G/S: $650 with Opus 5.5 (without evolutionary search).
  - Evolutionary search adds $23 to $147 per searched task.

## Run plan (GPU; not launched)

- **Target:** one LIBERO-Pro task (`libero_goal_swap:0`): iterative-debugging loop on development seeds 51–65, unseen-check seeds 66–80, held-out seeds 1–50.
- **Hardware:** GCP Spot **L4, g2-standard-8** (8 vCPU, 32 GB RAM, 24 GB L4; Ada sm_89, the same architecture as the shipped `rtx4090` profile), about $0.51/h. Ubuntu 22.04 or newer (glibc 2.35+), NVIDIA driver, `libegl1`, 150 GB balanced disk.
  - BEHAVIOR-1K runs on Isaac Sim, which needs an RT-core GPU (A100 and H100 lack them). Use the same L4, or an RTX PRO 6000 for parallel seeds. The A100 80 GB (about $2.95/h) is only for the MuJoCo benchmarks with many parallel seeds.
- **Prerequisites (you do these yourself):**
  - Request access to the gated SAM3 weights on Hugging Face.
  - A *separate*, spend-capped API key on the VM. Generated code runs with full Python imports, so treat the VM as untrusted and keep it free of other credentials.
- **Steps:**
  1. Run `scripts/bootstrap_libero.sh`, then `bootstrap_sam3.sh`, `bootstrap_graspnet.sh`, `bootstrap_pyroki.sh` and optionally `bootstrap_curobo.sh`. Read each script before running it.
  2. Start the providers with `supervise_services.sh --profile rtx4090` and check them with `cap-harness doctor`.
  3. Smoke-test with the recorded example: `cap-harness run ... --program examples/libero/open_middle_drawer.py`.
  4. Run Claude Code (Sonnet 5.5) in the repo with the iterative-debugging skill for that one task.
- **Runtime estimate:** about 1.5 h of setup (builds, weights, smoke test), plus about 1–2 h of agent time (56 min/task in the paper on unknown GPUs; budget 1.5–2x for an L4), plus about 0.5 h of held-out evaluation. Roughly 3.5–4.5 h in total.
- **Cost estimate, one task:**
  - L4 Spot: 4.5 h × $0.51 ≈ **$2.30**, plus about $0.30 of disk.
  - LLM: Sonnet 5.5 ≈ **$12**, or Haiku 5.5 ≈ $4.
  - Total ≈ **$15** with Sonnet 5.5, or ≈ $7 with Haiku 5.5.
- **Cost estimate, 10-task Goal suite with Sonnet 5.5:** about $121 LLM plus about 15 h of L4 (≈ $8), so about $130. That is over $50 and needs explicit approval.

## License

- ASPIRE code: **Apache-2.0** (NVIDIA).
- Third-party pieces keep their own terms. Per `configs/dependency-lock.json`:
  - Contact-GraspNet (PyTorch port): **NVIDIA Source Code License, non-commercial research and evaluation only**.
  - BEHAVIOR-1K: MIT for the code. The Data Bundle EULA covers *non-commercial academic research* and forbids redistribution.
  - cuRobo: Apache-2.0.
  - The SAM3 weights are gated under their own license.
  - The lock file lists no license for the LIBERO-PRO or robosuite forks.
- **Takeaway:** the code is usable, but the default grasping service and the BEHAVIOR-1K assets are non-commercial. Any non-research use would need replacements.

## Comparison with Inspect Robots (experiment 02)

| | Inspect Robots | ASPIRE |
|-|-|-|
| What it is | An eval harness: Task/Scene/Rubric/Scorer, immutable logs, viewer | A skill-learning method plus its benchmark harness (`cap-harness`) |
| LLM role | Policy acting step by step (`move_to` pose tool, 40-action budget). A CaP-X plugin also exists | Coding agent that writes and debugs whole programs and reuses a skill library |
| Grading | Programmatic scorers, human operator, or LLM judge | Simulator success predicates on held-out seeds |
| Sims | CPU toy (`cubepick`), Isaac Lab plugin (GPU) | LIBERO-Pro, Robosuite (MuJoCo), BEHAVIOR-1K (Isaac Sim) |
| Runs on CPU laptop | Yes (toy task) | Unit tests only |
| Cost per unit | $0.38–4.51 per subtask (real robot) | $4–60 per task (Table 18; the learning loop is the expensive part) |
| License | MIT | Apache-2.0 core; some dependencies non-commercial |

What ASPIRE would add to an evaluation platform for household tasks in simulated homes:

1. **A strong, cheap-to-run baseline to evaluate against.** Once learned, a skill-library CaP policy replays at zero LLM cost per rollout, and it is about 99% on LIBERO-Pro.
2. **Held-out seed protocol and leakage controls.** Seeds are split into development, unseen-check and held-out sets. Generated code goes through a provenance gate, and a skill is promoted only after the verifier re-solves the task. These rules can be reused as-is in our own test protocol.
3. **A typed perception/motion tool API with a BEHAVIOR-1K adapter.** BEHAVIOR-1K is the only one of its benchmarks set in simulated homes. The API is a ready-made "embodiment" layer an eval harness could call.
4. **Per-task cost, token, rollout and agent-time accounting.** This fills the cost/time columns of a scorecard.
5. **Skills as readable artifacts.** A skill is text plus code, so it can be audited and diffed between versions, which model weights can't.

Gaps: it is not an eval harness (no rubric or judge layer, no log viewer standard). It needs Linux and a GPU, a gated model and a CUDA build. Most of its benchmarks are tabletop. BEHAVIOR-1K results cover selected tasks only.

## Instructions found in repo files (not followed)

- The README has a prompt written for coding agents: "Clone ... run the complete ASPIRE LIBERO-Pro Goal-Swap Quick Start ...". `AGENTS.md` holds real-robot operating rules, such as "persist until the user's intended goal is complete".
- I treated both as data. Launching Claude Code or Codex inside this repo would auto-load `AGENTS.md` and `.claude/skills/`. That is how ASPIRE is meant to run, but it also gives the repo's text control over the agent. Only do it on the isolated VM.

## Next

- If GPU budget is approved, run the one-task L4 pilot above (about $15 with Sonnet 5.5), then replay the learned program on more seeds at zero LLM cost.
- Try the ASPIRE CaP program format through Inspect Robots' CaP-X plugin, so one harness scores both step-wise LLM agents and code-as-policy skills.
- Read the BEHAVIOR-1K skill (`.claude/skills/iterative-debugging/behavior/SKILL.md`) for the home-scale protocol before any home-scene work.
