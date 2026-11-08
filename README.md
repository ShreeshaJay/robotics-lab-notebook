# Robotics lab notebook

Hands-on notes from learning robotics and physical AI: open models, datasets, simulators and evaluation tools, tried one at a time and written up honestly, including what failed and what it cost.

Each experiment folder has a README with the goal, the exact steps, pinned versions, results, cost, and what I'd try next.

## Experiments
| # | Topic | Track | Status | Result |
|-|-|-|-|-|
| 01 | Inspecting the FineART step-labeled robot dataset (LeRobot format) | Robot data | done (CPU, $0) | 40.5k episodes, 534k timestamped step labels (~9 s each); no success/outcome fields |
| 02 | Inspect Robots: running an open robot-eval harness with frontier LLMs | Evals | done (CPU mock) | Haiku 5.5 agent 3/3 on toy reach task, ~$0.007 |
| 03 | SimReady Central assets for household scenes | Simulation | survey done (CPU, $0); sim run planned | 3,769 packages; kitchen/cleaning strong, bedroom/toiletries missing; L4 run ≈ $0.55–0.80 on Spot |
| 04 | Flora: open models for labeling egocentric video | Data labeling | planned (GPU) | Non-commercial license; 27B models need an 80 GB GPU |
| 05 | G0.5 VLA in LeRobot: subtasks and boxes before acting | Robot learning | planned (GPU) | Inference fits a 24 GB L4; non-commercial license |

## Ground rules
- Public sample data only; each experiment notes its data source and license.
- No datasets, weights or videos in git. Scripts download them.
- Third-party code is linked or forked, not copied in.

## License
Code: MIT (see `LICENSE`). Write-ups: CC BY 4.0.
