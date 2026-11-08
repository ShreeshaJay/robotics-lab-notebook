# Robotics lab notebook

Hands-on notes from learning robotics and physical AI: open models, datasets, simulators and evaluation tools, tried one at a time and written up honestly, including what failed and what it cost.

Each experiment folder has a README with the goal, the exact steps, pinned versions, results, cost, and what I'd try next.

## Experiments
| # | Topic | Track | Status | Result |
|-|-|-|-|-|
| 01 | Inspecting the FineART step-labeled robot dataset (LeRobot format) | Robot data | done (CPU, $0) | 40.5k episodes, 534k timestamped step labels (~9 s each); no success/outcome fields |
| 02 | Inspect Robots: running an open robot-eval harness with frontier LLMs | Evals | done (CPU mock) | Haiku 5.5 agent 3/3 on toy reach task, ~$0.007 |
| 03 | SimReady Central assets for household scenes | Simulation | done (L4, ~$0.46) | 23-asset room renders headless in Isaac Sim 6.1 at 7.2 GB VRAM; re-runs ~2.5 min with a warm shader cache; articulated drawers and some colliders need fixes |
| 04 | Flora: open models for labeling egocentric video | Data labeling | done (A100 80 GB, ~$3.30) | Captions track human narrations closely; outcome score separates success (5) from a negative control (1); dense reward noisy on human video |
| 05 | G0.5 VLA vs SmolVLA in LeRobot | Robot learning | done (L4, ~$0.40) | LIBERO-10 smoke test: G0.5 6/6, SmolVLA 2/6; SmolVLA ~10x faster and 0.9 GB VRAM; small open labeler hallucinates steps |
| 06 | Driving a Colab GPU from a coding agent (colab-mcp) | Agentic tooling | done (Colab Pro+) | 7 notebook tools; A100 40 GB at 239 TFLOPS bf16; GPU must be picked by hand in the UI |
| 07 | GATOR: room video to posed sim objects | Simulation | read (paper only, $0) | Code "coming soon"; outputs posed GLB meshes with no physics or articulation |
| 08 | ASPIRE: LLM coding agents write robot skills | Agentic tooling | partly (CPU + one API call, $0.003) | $12.11/task with Sonnet 5.5 confirmed; Haiku-written drawer program passes the static check; sim run needs a GPU |
| 09 | RoboColosseum: blind real-robot evals | Evals | read ($0) | Reusable protocol (blind A/B, milestones, Rasch ranking); 20 LeRobot v3.0 demo sets, eval rollouts not found |
| 10 | Nebius Physical AI Workbench: agent-run sim-to-real loop | Agentic tooling | read ($0) | Solid 14-stage plumbing but their own run scored 0/64; repo tells agents not to add cost limits |

## Ground rules
- Public sample data only; each experiment notes its data source and license.
- No datasets, weights or videos in git. Scripts download them.
- Third-party code is linked or forked, not copied in.

## License
Code: MIT (see `LICENSE`). Write-ups: CC BY 4.0.
