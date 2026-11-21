# Tutorial 2: Audit a robot dataset

A reusable quality audit for LeRobot-format (v3) datasets, an executable acceptance gate (pytest), and a one-page
report per dataset. It was run on three public datasets: LIBERO-10 (sim), FineART (lab teleop) and a DROID slice
(in-the-wild teleop). Date: 2026-11-20. CPU only, on a laptop. Results are in [RESULTS.md](RESULTS.md).

## What it measures

| Family | Metrics | Where |
|-|-|-|
| Episode length | Distribution over the full metadata, CV, episodes under 1 s | `audit.py` |
| Idle time | Still runs of at least 0.5 s: every arm dimension below 5% of its p95 speed and the gripper still, on a common 10 Hz grid. Split into lead, trail and mid-episode pauses. End-effector < 1 cm/s cross-check where a Cartesian pose exists | `metrics.py` |
| Smoothness | SPARC on sub-movements, RMS state jerk and action jerk in normalised units at 10 Hz, raw-rate speed spikes (dropped-frame signature) | `metrics.py` |
| Gripper | Close/open events with hysteresis, re-grasps (re-close within 2.5 s of opening), closes onto nothing, command/state sign agreement | `metrics.py` |
| Duplicates | Exact copies (hash of action + state), near-duplicates (RMS trajectory fingerprint distance < 0.02 std), plus a full-dataset exact-copy scan from per-episode stats in `meta/episodes` | `audit.py` |
| Label hygiene | Empty strings, whitespace, duplicates after normalisation, near-duplicates (stemming plus typo correction), typo candidates (word-list check), tasks with no episodes | `hygiene.py` |
| Label consistency | k-NN on trajectory fingerprints: does an episode's neighbourhood share its task label? (only when tasks have >= 5 episodes) | `audit.py` |
| Coverage | Episodes and hours per task, Gini, effective number of tasks, long tail | `audit.py` |
| Usable hours | Defined in RESULTS.md and in `audit.py` | `audit.py` |
| Integrity | frame_index contiguity, timestamp gaps, meta length vs data rows, NaN/inf, data files that `meta/episodes` never points to | `load.py`, `audit.py` |

All motion metrics run on a common 10 Hz grid with per-dimension scales. So the same threshold means the same thing at
10, 15 and 30 fps, and in joint space or end-effector space. A small `Profile` per dataset (`profiles.py`) says which
columns are arm, gripper and success. Everything else is dataset-agnostic.

## Layout

| Path | What |
|-|-|
| `lerobot_audit/` | The module: `profiles`, `load`, `metrics`, `hygiene`, `audit`, `acceptance`, `corrupt`, `report` |
| `notebooks/audit_dataset.ipynb` | Walk-through notebook (executed copy included) |
| `tests/test_acceptance.py` | The pytest gate: one test per check, options `--dataset`, `--root`, `--spec` |
| `specs/<dataset>.json` | Gate spec: shared thresholds plus per-dataset schema, bounds and frozen scales |
| `scripts/` | `download.py`, `run_all.py`, `run_corruption_eval.py`, `ext1_ext4.py`, `ext2_quality_vs_success.py`, `ext3_sweep.py`, `check_orphans.py`, `make_specs.py`, `build_notebook.py` |
| `results/<dataset>/` | `report.md` + `report.png` (the one-pager), `audit.json`, `episodes.csv`, `gate.json` |
| `results/` | `scorecard.*` (Ext 1), `ext2/` (Ext 2), `corruption/` (Ext 3), `cost_per_usable_hour.*` (Ext 4), `hub_file_consistency.json`, pytest logs |

## Datasets, revisions and licenses

| Dataset | HF repo @ commit | License (HF tag) | Downloaded |
|-|-|-|-|
| LIBERO-10 | `lerobot/libero_10` @ `551d7d86f25edd0ffeda8b60053c15438cfd1d6a` | apache-2.0 | all metadata and all 147 data parquet files (8 MB) |
| FineART | `lerobot/FineART` @ `15746b76ddc21ffd38d73888fefe2e7a1496385b` | cc-by-4.0 | metadata + data files 000, 052, 105, 157, 210 (0.6 GB) |
| DROID 1.0.1 | `lerobot/droid_1.0.1` @ `71da9dcba338f53889907ac5762a2ebc6c2e55ff` | apache-2.0 | metadata + data files 000, 042, 077, 085 (audited) and 154 (unreferenced, evidence only), plus 072 to verify one duplicate pair (1.3 GB) |

No videos were downloaded: no metric here needs frames. Total download: about 2.0 GB. All three repos are public
and not gated, so no token was used. The licenses are the Hub tags. I didn't read the upstream LIBERO or DROID license
texts beyond those tags. Code dependencies: `pyspellchecker` 0.9.1 (MIT) supplies the English word list for typo
checks. Everything else is in `requirements.txt`.

LIBERO-10 is the 10-task suite that lerobot's LIBERO eval (`--env.task=libero_10`) runs. Its 10 task strings match
the eval instructions in `experiments/05-g05-lerobot` exactly.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -I scripts/download.py                 # libero_10, fineart, droid -> data/<name>/
.venv/Scripts/python -I scripts/check_orphans.py            # Hub file listing vs meta/episodes
.venv/Scripts/python -I scripts/run_all.py                  # audit + spec + gate + one-page report per dataset
.venv/Scripts/python -I scripts/run_corruption_eval.py      # Ext 3: corrupted LIBERO copy
.venv/Scripts/python -I scripts/ext3_sweep.py               # Ext 3: harder single-fault variants
.venv/Scripts/python -I scripts/ext2_quality_vs_success.py  # Ext 2
.venv/Scripts/python -I scripts/ext1_ext4.py                # Ext 1 scorecard + Ext 4 cost table
.venv/Scripts/python -m pytest tests/test_acceptance.py --dataset libero_10
.venv/Scripts/python -m pytest tests/test_acceptance.py --dataset libero_10 --root data/libero_10_corrupt --spec libero_10
```

`-I` keeps the scripts' folder and the current folder off `sys.path`. The downloaded data is untrusted, and the
scripts put the repo root on the path themselves.

Versions: Python 3.11.9, numpy 2.4.6, pandas 3.1.1, pyarrow 26.0.1, scipy 1.17.1, matplotlib 3.11.2, pytest 9.2.0,
huggingface_hub 2.4.0, nbclient 0.11.0. The full freeze is in `requirements.txt`. lerobot itself is not installed:
the loader reads the v3 parquet layout directly.

## Papers read alongside

- DROID, arXiv 2403.12945: how the data was collected and what the success flag means.
- Open X-Embodiment, arXiv 2310.08864: why schema conventions (gripper sign, state layout) need checking.
- Data Quality in Imitation Learning, arXiv 2306.02437: action divergence and transition diversity as quality measures.
- robomimic, arXiv 2108.03298: operator quality and its effect on learning.
- SPARC: Balasubramanian et al., "On the analysis of movement smoothness", J. NeuroEng. Rehabil. 2015.
