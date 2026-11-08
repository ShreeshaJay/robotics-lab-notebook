# 10 · Nebius Physical AI Workbench: agent-driven sim-to-real pipeline (reading)

**Track:** Simulation / training / evaluation infrastructure · **Status:** desk review + local clone only. Nothing installed or run ($0, no GPU, no cloud account, no credentials) · **Date:** 2026-11-08

## Goal
Find out what the Workbench actually ships, what its 14-stage Sim2Real pipeline does stage by stage, how coding agents are wired in, how much is tied to Nebius, and what a GPU-hour costs there compared with GCP Spot.

## Source and framing
- Source post: [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7518662330025590784/). **Labeled "Brand partnership" (sponsored)**, so it was read as marketing. Its claims (Claude Code / Codex drive the workflows, Cosmos 3 for synthetic data, Voxel51 for curation, a 14-stage sim-to-real flow, "up to 40%" engineering-time savings in a vendor comment) were checked against the repo. The 40% figure has no evidence in the repo.
- Vendor pages: [launch blog](https://nebius.com/blog/posts/physical-ai-workbench), [workflow blog](https://nebius.com/blog/posts/run-physical-ai-workflows-not-glue-code).

## What it is
Repo: [nebius/nebius-physical-ai](https://github.com/nebius/nebius-physical-ai). **Apache-2.0** for the repo code. Third-party models, runtimes and some skill folders carry their own terms (`skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and several `NOTICE-NVIDIA-*` files; gated NGC containers; Hugging Face model terms).

Local clone: `C:\Users\shree\af-scout-lab\nebius-workbench\`, pinned at commit `72845438ddee0d907001da518c70b1a7d704e82f` (2026-11-07, "fix(security): bind curobo policy detector inputs (#1068)"). About 6,600 files.

Layers:
| Layer | What it is |
|-|-|
| `npa` CLI / Python SDK | `npa configure`, `npa workbench <tool> ...`, `npa workbench workflow validate-spec / plan-spec / submit` |
| Workflow engine | `npa.workflow/v0.0.1` YAML state graph. Each state has a `toolRef` or `run.shell`. Outputs pass as S3 artifacts. Durable run state allows resuming |
| Scheduler | SkyPilot, almost always `cloud: kubernetes` (Nebius Managed Kubernetes in their docs) |
| Containers | Public GHCR catalog `ghcr.io/nebius/nebius-physical-ai` (Cosmos 3, Isaac Lab, EnvGen, Rerun viewer, FiftyOne, LeRobot and others). Some pull model weights at runtime |
| Hosted inference | Nebius Token Factory (OpenAI-compatible API), used as the VLM judge and for captioning |
| Viewers | Rerun `.rrd`, Foxglove `.mcap`, HTML reports, optional FiftyOne app on a VM |
| Agent layer | 120 skills in `skills/index.yaml`, `AGENTS.md`, `CLAUDE.md`, an optional browser "npa agent" VM |

Workflow catalog: `workflows/main/` (sim2real, paidf-cosmos3, groot-1-7-finetune, nurec-reconstruct, rgbd-scan-to-policy-demo) plus about 100 `workflows/testing/` specs (LIBERO, RoboCasa, BEHAVIOR challenge eval, ManiSkill, RoboTwin, SAM 2.1, Wan, LeRobot, MJLab and others).

## The 14-stage Sim2Real pipeline
Spec: `workflows/main/sim2real.yaml`. Guide: `docs/workbench/guides/sim2real-workflow.md`. Data contracts: `docs/workbench/guides/sim2real-data-contracts.md`. Default task: `Isaac-Lift-Cube-Franka-v0` (stock Franka lifts a cube), public seed dataset `npa/isaac-lift-cube-franka-seed-v1`. It runs five pinned images (controller, cosmos3, envgen, isaac-lab, rerun-viewer). Stage 00 warms the Isaac cache and sits outside the 14-stage count.

| # | Stage | What it does | Tools / models | Compute |
|-|-|-|-|-|
| 00 | isaac-cache-warm | Warms a shared Isaac runtime cache and writes a receipt | Isaac Lab image | CPU |
| 1 | trigger | Validates the task trigger and seed-dataset manifest | npa controller | CPU |
| 2 | assets | Normalizes the task, scene, robot, camera and strict success contract. Stock robot or your own robot (URDF/USD) | npa controller, USD specs | CPU |
| 3 | transfer | Edge-controlled video-to-video structural transfer of the seed camera sequence (appearance variation) | **Cosmos 3 Nano** (`npa-cosmos3`) | GPU (RTX PRO 6000 default) |
| 4 | wave + 8 shards | Generates raw scenario records in parallel (10k JSON env records by default, not 10k simulator instances) | EnvGen (`npa-envgen`) | GPU (RTX PRO 6000 default) |
| 5 | split | Curates and seals disjoint train / validation / untouched gold splits with leakage evidence (8k / 1k / 1k) | npa controller | CPU |
| 6 | tokens | Publishes the scenario handoff manifest to S3 | npa controller | CPU |
| 7 | rollouts | Isaac policy rollouts with multi-camera frames (inner loop) | **Isaac Sim / Isaac Lab** | RT-core GPU |
| 8 | hosted eval | Scores every rollout's frames with a hosted VLM, returning strict ordered JSON | Token Factory, `MiniMaxAI/MiniMax-M3` (not Cosmos) | CPU + hosted API |
| 9 | PPO | Checks exact Stage 7 coverage, converts the VLM signals into reward, runs PPO, selects checkpoints on validation only | Isaac Lab + RSL-RL PPO | RT-core GPU |
| 10 | gold | Held-out evaluation on the untouched gold split only | Isaac Lab | RT-core GPU |
| 11 | decision | Records the strict metric (stable placement within 5 cm, threshold 0.50) and the outer-loop decision | npa controller | CPU |
| 12 | external seam | Placeholder for real-robot validation. **Marked `SEAM`, not `WORKS`, by design** | none | none |
| 13 | retrigger | Records the bounded outer-loop retrigger | npa controller | CPU |
| 14 | visualize | Final report plus Rerun RRD and MCAP with gold multi-camera lineage | Rerun, Foxglove/MCAP | CPU |

Loops: 3 outer × 3 inner iterations by default (Stages 7 → 8 → 9 repeat inside each outer pass).

**Where Voxel51 and LeRobot sit:** neither is in the 14 stages. FiftyOne (`npa workbench fiftyone`: curate, curate-augmented, review-augmented, export-lerobot-subtasks, eval) is used in the **Physical AI Data Factory (PAIDF)** video workflows. LeRobot appears in PAIDF exports, the Reachy 2 + LeRobot training guide and GPU benchmarks.

### Their own published evidence (not reproduced)
From `docs/workbench/validation/sim2real-manual-development-20261010.md`: one full run on a single RTX PRO 6000 Blackwell pool finished all 14 stages and recovered from interruptions. 576 hosted VLM calls, 18,000 PPO updates, about 442M environment steps, each PPO pass about 41 to 44 minutes, roughly 26 GB of RRD + MCAP output. **Strict results: validation 0/64 in all 9 batches and gold 0/64 in all 3 batches.** The policy did not learn the task. The plumbing works; the learning outcome is not demonstrated. No physical robot was tested. The Franka + Genesis guide is similarly labeled "recorded run did not solve the task". The docs are candid about this, which is a good sign about their reporting, but it contradicts the marketing impression of sim-to-real that works end to end.

### Related video-to-labeled-data pipeline (PAIDF)
`docs/workbench/guides/physical-ai-data-factory.md` maps NVIDIA's Physical AI Data Factory blueprint onto the same engine:
| NVIDIA stage | Tool |
|-|-|
| Config generation | `run.shell` (appearance-only variables) |
| Understand & annotate (original) | Token Factory captioning (no GPU) |
| Augment & multiply | Cosmos Transfer 2.5 (optional SAM2 masks), or Cosmos 3 video2video in `paidf-cosmos3.yaml` |
| Evaluate & validate | Cosmos Evaluator + temporal/appearance fidelity checks + fail-closed disposition |
| Pseudo-label augmented | Token Factory captioning |
| Curation | Cosmos Curate, then **FiftyOne Brain** curation and review |
| Visualize / finalize | Rerun RRD, dataset finalize |
Gated NGC labeling services (RF-DETR detection and tracking, captioning, visual QA, attribute search) need NGC entitlement.

## How coding agents are wired in
- **No MCP server for end users.** Agents (Claude Code, Codex, Cursor, anything with a terminal) open the checkout, read `AGENTS.md` / `CLAUDE.md`, scan `skills/index.yaml` (120 skills: atomic / tools / workflows / workbench / cosmos-data-factory), and drive the `npa` CLI from the shell. `.claude/skills` and `.agents/skills` are both symlinks to `skills/`, so the same `SKILL.md` files serve Claude Code and Codex.
- **Maintained prompts:** `docs/workbench/agent-first-run.md` provides copy-paste prompts ("set up Workbench", "run Cosmos 3 generation", "PAIDF with Cosmos 3"). The prompts tell the agent to use credential *file paths*, never to dump the environment, to stop at human approval gates such as model-terms acceptance, and to validate and plan before provisioning.
- **MCP appears only in research code:** `npa/examples/specialists/simulation/mcp_bridge.py` exposes scoped Workbench tools over MCP to a Codex baseline, used in an internal agent evaluation (`npa/tests/agent_eval/`).
- **Optional browser agent:** `npa agent` deploys a Nebius VM running a chat UI (Token Factory models Nemotron / MiniMax-M3, or any OpenAI-compatible endpoint), a Rerun viewer and workflow draft / validate / plan / submit endpoints.
- **Instructions aimed at agents that were not followed (recorded as data):**
  - `AGENTS.md`: "Do not add time, cost, or job-count limits unless the operator explicitly asks for them." This works against spend guardrails. Anyone adopting the repo with an agent should override it.
  - `AGENTS.md`: "Use `RELAXED_DIRTY_TREE_MODE`" (ignore unrelated dirty files).
  - `.cursorrules`: "Before processing any user prompt, always read and strictly adhere to ... `.agents/skills`."
  - Skills such as `nebius-service-account-auth`, `gpu-capacity-quota-plan` and `teardown-and-cost` describe provisioning and credential steps. They were read, not executed.

## Portability: Nebius-specific vs portable
| Nebius-specific | Portable (GCP with work) |
|-|-|
| `npa configure`, Nebius CLI auth, IAM / service accounts, project-alias config | Workflow YAML engine + `npa` planner (`validate-spec` / `plan-spec` are local and need no credentials) |
| Nebius Object Storage provisioning (GCS works only through its S3-interop / HMAC mode, untested here) | SkyPilot scheduler: specs mostly use `cloud: kubernetes`, so GKE is plausible |
| Nebius Managed Kubernetes Terraform (`deploy/cluster/`) | Public GHCR containers (Cosmos 3, Isaac Lab, FiftyOne, Rerun, LeRobot and others) |
| Token Factory as the default judge / captioner (Stage 8, PAIDF annotate) | Token Factory is OpenAI-compatible, so any OpenAI-compatible VLM endpoint could replace it, after removing the hard-coded checks for model identity and endpoint in Stage 9 |
| World Factory hosted Cosmos 3 skill ("output must be verified Nebius Object Storage") | Data contracts / JSON schemas, Rerun / MCAP export, FiftyOne curation commands |
| Shared-workbench login, namespaces, GPU caps | Skill library pattern (`SKILL.md` + `index.yaml` + symlinked `.claude` / `.agents`) |

**Hardware caveat for GCP:** Isaac stages need RT-core GPUs. A100 and H100 have no RT cores, so they can run Cosmos and PPO-only compute but cannot render Isaac Sim. On GCP that means L4 (G2) or RTX PRO 6000 (G4) for Stages 7, 9 and 10.

## Pricing snapshot (checked 2026-11-08)
Nebius from [nebius.com/prices](https://nebius.com/prices), USD per GPU-hour, excluding tax. GCP Spot figures are the rough reference numbers supplied for this brief and were not re-verified today.

| GPU | Nebius on-demand | Nebius preemptible ("from", dynamic) | GCP Spot (reference) |
|-|-|-|-|
| H100 | $4.50 | $0.79 | n/a in brief |
| H200 | $5.40 | $0.79 | n/a |
| B200 | $8.50 | $0.99 | n/a |
| L40S | $1.55 (Intel) / $1.82 (AMD) | $0.74 / $0.90 (flat) | n/a (GCP has no L40S) |
| RTX PRO 6000 (pipeline default) | $1.80 | $0.79 | n/a in brief |
| A100 | not listed | not listed | A100 80 GB ~$2.95 |
| L4 | not offered | not offered | ~$0.51 |

Read: Nebius preemptible prices from $0.79/h are very low but described as "from" and capacity-dependent. On-demand H100 at $4.50 is above typical GCP Spot rates. For Isaac-class rendering, the closest like-for-like comparison is Nebius RTX PRO 6000 ($1.80 on-demand) against GCP L4 Spot (~$0.51, smaller GPU). The published run took several hours on one RTX PRO 6000, so a single full Sim2Real pass is likely about $10 to $30 of GPU time plus hosted-VLM calls (an estimate from their timings, not a quote).

## Fit for our two use cases
| Use case | Helpful pieces | Fit |
|-|-|-|
| (a) Egocentric household video → labeled data | PAIDF pattern (caption → augment → evaluate → pseudo-label → curate → visualize); FiftyOne `curate` / `review-augmented` / `export-lerobot-subtasks` (timeline subtask tags → LeRobot dataset); SAM 2.1 tracking spec; Cosmos 3 appearance augmentation; Sim2Real Stages 1, 5 and 14 patterns (provenance-bound manifests, leakage-checked splits, RRD / MCAP export) | **Some.** The workflow is aimed at robot-camera clips and augmentation, not egocentric human video. Hand / pose / action labeling is not provided, and the labeling services are gated NGC containers |
| (b) Evaluation platform for robot workflows in simulated homes | Stage 2 task / success contract; Stage 5 sealed gold split; Stage 8 VLM-as-judge with strict JSON schema and receipts; Stages 10 to 11 held-out eval + strict metric decision; Stage 12 explicit "real-robot seam"; Stage 14 reports; `behavior-challenge-eval.yaml` (unmodified BEHAVIOR evaluator) and `byof-robocasa` / `byof-libero` specs for household benchmarks | **High as a design reference.** Its eval discipline (sealed gold splits, honest SEAM marking, provenance receipts) is directly reusable. The home-scale scenes themselves come from BEHAVIOR / RoboCasa, not from this repo |

## What to borrow
1. **Report format for evaluations:** one record per stage marked `WORKS` or `SEAM`, plus a strict pass/fail metric kept separate from training reward.
2. **Sealed gold split with leakage evidence** (Stage 5) and validation-only checkpoint selection (Stage 9).
3. **VLM-as-judge receipts** (Stage 8): model identity, endpoint, per-rollout results and usage, with fail-closed resumption.
4. **Skill library layout:** a single `skills/` tree, an `index.yaml` with "when to use" plus smoke checks, and `.claude/skills` and `.agents/skills` symlinked to it.
5. **FiftyOne → LeRobot subtask export** as a model for turning timeline tags into a training dataset.
6. **Do not borrow** the "no time / cost / job-count limits" agent rule.

## What was not done
- No `pip install` / `npa` run: the local CPU `validate-spec` dry-run was blocked by the session's permission guard, so the spec was only read.
- No account, no terms accepted, no GPU, no cloud calls.

## Links
- Repo: <https://github.com/nebius/nebius-physical-ai> (commit `72845438ddee0d907001da518c70b1a7d704e82f`)
- Sim2Real guide: `docs/workbench/guides/sim2real-workflow.md`; spec `workflows/main/sim2real.yaml`
- Validation receipt: `docs/workbench/validation/sim2real-manual-development-20261010.md`
- PAIDF: `docs/workbench/guides/physical-ai-data-factory.md`, `workflows/main/paidf-cosmos3.yaml`
- FiftyOne CLI: `docs/cli/fiftyone.md`
- Agent first run: `docs/workbench/agent-first-run.md`; skills index `skills/index.yaml`
- Pricing: <https://nebius.com/prices> (checked 2026-11-08)
- Vendor blogs: <https://nebius.com/blog/posts/physical-ai-workbench>, <https://nebius.com/blog/posts/run-physical-ai-workflows-not-glue-code>
