| Metric | LIBERO-10 (sim) | FineART (lab teleop) | DROID 1.0.1 (in-the-wild teleop) |
|-|-|-|-|
| Source / robot / fps | sim / panda / 10 | lab teleop / aloha / 30 | in-the-wild teleop / Franka / 15 |
| License (HF tag) | apache-2.0 | cc-by-4.0 | apache-2.0 |
| Full dataset: episodes / hours | 379 / 2.8 | 40,543 / 1,717.9 | 95,658 / 511.7 |
| Audited slice: episodes / hours | 379 / 2.8 | 791 / 38.4 | 3,970 / 21.7 |
| Episode length median (CV) | 25.9 s (0.21) | 123.7 s (0.75) | 14.8 s (0.78) |
| Idle time, time-weighted | 0.2% | 4.3% | 7.2% |
| Lead / trail idle, median | 0.0 s / 0.0 s | 2.3 s / 3.0 s | 0.5 s / 0.0 s |
| Mid-episode pauses per min | 0.05 | 0.64 | 1.59 |
| SPARC median (higher = smoother) | -1.91 | -2.99 | -1.99 |
| State jerk / action jerk (median, normalised) | 0.047 / 0.049 | 0.279 / 0.253 | 0.108 / 0.344 |
| Grasps per minute | 4.07 | 1.76 | 4.30 |
| Episodes with a re-grasp (retry proxy) | 7.9% | 47.4% | 12.8% |
| Failed episodes (dataset flag) | not labelled | not labelled | 17.5% |
| Episodes with empty task string (full) | 0.0% | 0.0% | 21.9% |
| Exact duplicate copies (full, meta stats) | n/a | 1.3% | 0.1% |
| Task strings / effective tasks / Gini | 10 / 10 / 0.08 | 151 / 90 / 0.54 | 49,630 / 28,885 / 0.32 |
| Typo-candidate words / near-dup task groups | 0 / 0 | 6 / 1 | 49 / 216 |
| **Usable share of raw time** | **99.8%** | **95.6%** | **66.6%** |
| Usable share, strict (drop re-grasp episodes) | 90.7% | 43.6% | 54.4% |
| Usable hours, full dataset (extrapolated) | 2.8 of 2.8 | 1,641.9 of 1,717.9 | 340.6 of 511.7 |
