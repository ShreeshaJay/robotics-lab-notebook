# Results: Tutorial 2, audit a robot dataset

Run on 2026-11-20 on a laptop CPU. Cost $0. Three LeRobot v3 datasets were audited: `lerobot/libero_10` (all 379
episodes), `lerobot/FineART` (5 of 211 data files: 791 episodes, 38.4 h) and `lerobot/droid_1.0.1` (4 of the 86
referenced data files: 3,970 episodes, 21.7 h). Length, label and coverage metrics use each dataset's full metadata.
Motion metrics use the slice. The one-page reports are in `results/<dataset>/report.md`.

## Headline numbers (Ext 1 scorecard: identical metrics)

| Metric | LIBERO-10 (sim) | FineART (lab teleop) | DROID 1.0.1 (in-the-wild teleop) |
|-|-|-|-|
| Full dataset: episodes / hours | 379 / 2.8 | 40,543 / 1,717.9 | 95,658 / 511.7 |
| Episode length median (CV) | 25.9 s (0.21) | 123.7 s (0.75) | 14.8 s (0.78) |
| Idle time, time-weighted | 0.2% | 4.3% | 7.2% |
| Mid-episode pauses per min | 0.05 | 0.64 | 1.59 |
| SPARC median (higher = smoother) | -1.91 | -2.99 | -1.99 |
| Normalised state jerk / action jerk (median) | 0.047 / 0.049 | 0.279 / 0.253 | 0.108 / 0.344 |
| Episodes with a re-grasp (retry proxy) | 7.9% | 47.4% | 12.8% |
| Failed episodes (dataset flag) | not labelled | not labelled | 17.5% |
| Episodes with an empty task string | 0% | 0% | 21.9% |
| Exact duplicate copies (full dataset) | none in slice (meta has no stats) | 1.3% (519 episodes) | 0.1% (96 episodes) |
| Task strings / effective tasks / Gini | 10 / 10 / 0.08 | 151 / 90 / 0.54 | 49,630 / 28,885 / 0.32 |
| Typo-candidate words / near-duplicate task groups | 0 / 0 | 6 / 1 | 49 / 216 |
| **Usable share of raw time** | **99.8%** | **95.6%** | **66.6%** |
| Usable share, strict (drop re-grasp episodes) | 90.7% | 43.6% | 54.4% |
| Usable hours, full dataset (extrapolated from slice) | 2.8 of 2.8 | 1,641.9 of 1,717.9 | 340.6 of 511.7 |
| Acceptance gate on the clean data (shared thresholds) | 15/15 pass | 9/15 pass | 9/14 pass (k-NN label check n/a) |

Full table: `results/scorecard.md`. Figure: `results/scorecard.png`.

**Usable hours, defined.** These are hours that a language-conditioned imitation learner can use as delivered. An
episode counts only if it passes all of these:
- integrity: no timestamp gaps, contiguous frame_index, meta length equal to the data rows, no NaN;
- not flagged failed, where the dataset has a success flag;
- a non-empty task string;
- not an exact or near-duplicate copy of an earlier episode;
- at least 1 s of motion.

From each passing episode, subtract its idle time. Idle time is runs of at least 0.5 s in which every arm dimension
moves less than 5% of its p95 speed and the gripper is still, measured on a common 10 Hz grid. "Strict" also drops
every episode with a re-grasp (a re-close within 2.5 s of opening).

Where DROID's raw time goes: 15.4% is in episodes flagged failed, 13.8% has an empty instruction and 4.2% is idle.
FineART loses 4.3% to idle and 0.2% to a duplicate.

## Findings the audit surfaced (verified by hand)

1. **DROID has unreferenced data files.** `meta/episodes` points to data files 000-085, but the repo holds 156 data
   files. The other 70 (file-086 to file-155, 5.4 GB) are referenced nowhere. A sampled orphan (file-154) reuses
   episode ids 94870+ with lengths that disagree with the metadata in 606 of 607 episodes. The first audit pass read it
   and reported 22% "integrity failures". The loader now follows `meta/episodes` and lists unreferenced files
   separately (`results/hub_file_consistency.json`). I didn't test whether lerobot's own loader would pick these up.
2. **Exact duplicate episodes.**
   - FineART: episodes 30424 and 30449 are byte-identical in action and state. The metadata-stats scan finds 519
     copies (1.3% of episodes, 1.2% of hours), mostly within about 100 episode ids of their twin.
   - DROID: episodes 80912 and 95021 are byte-identical, with the same instruction, building and collector. The scan
     finds 96 copies.
   - I checked one pair per dataset frame by frame. The rest are inferred from identical length plus identical
     action/state mean and std.
3. **Mislabelled DROID metadata columns.** `date` equals `collector_id` in 100% of sampled episodes, and
   `task_category` holds the lab name (Gates, TRI, BWW).
4. **Feature names that don't match the data.**
   - LIBERO's `observation.state` is named `x,y,z,rx,ry,rz,rw,gripper`, but it holds xyz, axis-angle and two finger
     joints.
   - FineART's dims 5 and 12, named `*_gripper`, behave like rotary wrist joints. The finger opening is in dims 6
     and 13.
   - LIBERO's `meta/episodes` has no `tasks` column.
5. **Label hygiene.**
   - FineART has typos `relplace`, `vace`, `prefit` and `unhang`, plus the near-duplicate pair `Coil cords` /
     `Coiling cords`. `stevia` and `truvia` are false positives (brand names).
   - DROID has 293 task strings with double or trailing spaces, and 216 near-duplicate groups, mostly plural variants
     ("close the open drawer" / "drawers"). Its typo list mixes real typos (`bule`, `botte`, `carboard`, `caan`) with
     British spellings (`colour`, `centre`).
   - 21.9% of DROID episodes have an empty task.
   - DROID's long tail: 43,164 of 49,630 instructions are used once.
6. **48 of 3,970 DROID episodes (1.2%) are at least 95% idle (the robot never really moves).** 46 of them are flagged
   failed, which is consistent with the dataset's own flag.
7. **The publisher's figure agrees with ours.** FineART's own `meta/sampling/sampling_report.json` strips 3.1% of
   frames as idle. That is the same order as our 4.3%, by a different method.

## Extensions

| # | Hypothesis (written before the run) | Result | Verdict |
|-|-|-|-|
| 1 | Real teleop (DROID, FineART) has far more idle time and retries than LIBERO, so usable hours could be 50-70% of raw | Idle is 20-35x LIBERO's but small in absolute terms (4.3%, 7.2% vs 0.2%). Usable share: DROID 66.6% (in range), FineART 95.6% (not). DROID's loss comes from failed episodes and empty instructions, not idle time or retries. Only the strict definition, which drops whole re-grasp episodes, gives 44-54% | **Refuted** as stated: the mechanism is wrong and FineART is outside the range. The DROID number lands in range for different reasons |
| 2 | Action smoothness and demo-length consistency predict per-task success; simple counts don't | SmolVLA per-task success (k/5) vs LIBERO-10 per-task metrics, Spearman, n = 10: SPARC +0.47 (p = 0.18, expected sign); state jerk +0.21 and action jerk +0.23 (wrong sign); length CV +0.12 (wrong sign); episode count -0.26; grasps per episode -0.19. Strongest: idle share +0.62 (p = 0.06) and demo diversity (NN distance) +0.60 (p = 0.07). Nothing survives a 10-test correction. G0.5 is 5/5 on 9 of 10 tasks, so it has no variance to correlate | **Inconclusive** (suggestive only, n = 10 tasks × 5 rollouts) |
| 3 | The pytest gate catches every injected fault on a corrupted copy | All 7 injected fault types (the 4 required plus a typo string and two kinds of duplicate) are caught: 49/49 faulty episodes, 0 false positives, and the clean copy passes 15/15. The harder variants below find the limits | **Confirmed** for the specified faults; misses at lower fault strength, listed below |
| 4 | Quality changes the effective price per usable hour | At Perceptron's $9 per video-hour for subtasks, the price per usable hour is $9.02 (LIBERO), $9.42 (FineART) and $13.52 (DROID). At $29 (full stack): $29.06, $30.34 and $43.56. Auditing first and labelling only usable time saves about $1,540 of $4,605 on all of DROID at $9 per hour | **Confirmed** (table below) |

### Ext 2 detail
| Metric | Expected | Spearman rho vs SmolVLA | Permutation p |
|-|-|-|-|
| SPARC median | + | +0.47 | 0.18 |
| State jerk median | - | +0.21 | 0.56 |
| Action jerk median | - | +0.23 | 0.52 |
| Length CV | - | +0.12 | 0.74 |
| Episodes per task (count) | 0 | -0.26 | 0.46 |
| Grasps per episode (count) | 0 | -0.19 | 0.59 |
| Idle share | (none) | +0.62 | 0.06 |
| NN trajectory distance (diversity) | (none) | +0.60 | 0.07 |

Success comes from `experiments/05-g05-lerobot/results/rerun50/per_task_rerun50.csv`: SmolVLA, 5 rollouts per task,
seed 0, the same initial states for every model. The LIBERO-10 task strings match the eval instructions exactly.
Figure: `results/ext2/quality_vs_success.png`. Per-task success at n = 5 has Wilson intervals about 60 points wide,
so the outcome variable is mostly noise.

### Ext 3 detail: what the gate catches and misses (LIBERO-10 copy, seed 0, 8 episodes per fault)

Main corrupted copy (`data/libero_10_corrupt`, manifest in `results/corruption/corruption_eval.json`):

| Injected fault | Episodes | Caught by intended check | Other checks that fired |
|-|-|-|-|
| Dropped frames (10%), timestamps left with gaps | 8 | 8 (`timestamps_regular`) | frame_index, meta length, continuity |
| Dropped frames (10%), renumbered so the metadata looks clean | 8 | 8 (`dynamic_continuity`) | none |
| Gripper sign flipped | 8 | 8 (`gripper_sign_convention`) | none |
| Task mislabelled (random other task) | 8 | 8 (`task_trajectory_consistency`) | none |
| Typo task string ("...on the stvoe") | 5 | 5 (`task_labels_clean`) | k-NN consistency |
| Exact duplicate episodes | 8 | 8 (`no_exact_duplicates`) | none |
| Near duplicates (0.5% std noise) | 4 | 4 (`no_near_duplicates`) | action bounds |

`pytest tests/test_acceptance.py --dataset libero_10` passes 15/15. On the corrupted copy, 10 of 15 tests fail
(`results/pytest_libero_10_*.txt`).

Harder variants (`results/corruption/sweep.json`):

| Variant | Caught (of 8) | Note |
|-|-|-|
| Dropped frames, renumbered, 1% / 2% / 5% | 0 / 6 / 8 | **Miss below about 2%:** 2-3 dropped frames per episode stay under the 3-spike minimum |
| Mislabel to a confusable task (same scene or same motion), 3 pairs | 7 / 6 / 7 | 1-2 missed per pair; 1 false positive in 2 of 3 runs |
| Near duplicate, noise 1% / 2% std | 8 / 8 | |
| Near duplicate, noise 5% / 10% std | 0 / 0 by the duplicate check | **Miss as duplicates.** They are caught only incidentally: the noise breaks action bounds and continuity. Their NN distance (0.04-0.08) is far below clean LIBERO's minimum (0.15), so a data-calibrated threshold would catch them |
| Duplicate trimmed by 5 or 20 frames (re-upload) | 0 / 0 | **Miss.** The fingerprint is time-normalised, so a trimmed copy no longer aligns. It would need alignment (DTW or a sliding window) |
| Gripper sign flipped on the last 25% / 50% only | 0 / 1 | **Miss.** The episode-level correlation stays positive. It would need a windowed check |

The same gate on the clean real datasets fails checks that are true findings: FineART's duplicate, typos and long-tail
tasks; DROID's whitespace, typos, empty instructions and long tail. It also fails checks that are calibration limits:
- k-NN label consistency flags 161 of 791 FineART episodes. Joint-space trajectories don't separate FineART's tasks,
  so this check only works where tasks have distinctive motions, as in LIBERO.
- `idle_fraction` and `episode_length` fire on real data by design.
- `dynamic_continuity` flags 13 DROID episodes. Most are almost-static episodes where 3 spikes among few moving frames
  give a high rate. I haven't verified whether any are real dropped frames.

### Ext 4 detail: cost per usable hour

| Dataset | Usable share | Price per video-hour | $ per usable hour, label everything | Same, all camera streams | Full dataset: label all vs audit first and label usable only |
|-|-|-|-|-|-|
| LIBERO-10 | 99.8% | $9 subtasks | $9.02 | $18.04 (2 streams) | $25 vs $25 |
| LIBERO-10 | 99.8% | $29 full stack | $29.06 | $58.12 | $82 vs $82 |
| FineART | 95.6% | $9 subtasks | $9.42 | $37.67 (4 streams) | $15,461 vs $14,777 |
| FineART | 95.6% | $29 full stack | $30.34 | $121.37 | $49,819 vs $47,614 |
| DROID | 66.6% | $9 subtasks | $13.52 | $40.56 (3 streams) | $4,605 vs $3,066 |
| DROID | 66.6% | $29 full stack | $43.56 | $130.69 | $14,839 vs $9,878 |

Price source: Perceptron Egocentric 1.0 list prices per video-hour (subtasks $9; full stack $29), from
`research/2026-11-19_wave2/D_egodata.md`, which cites perceptron.inc's launch post of 2026-11-04. No other price was
used. The auditing itself cost $0. It took 0.4-7 s of laptop CPU per raw hour (DROID is slowest, mostly
spell-checking 49,630 task strings).

Assumptions, each labelled:
- **A1:** an egocentric human-video price applies unchanged to robot camera video.
- **A2:** one stream is labelled per robot-hour. The "all streams" column multiplies by the camera count in
  info.json.
- **A3:** the audited slice represents the full dataset.
- **A4:** CPU auditing has $0 marginal cost.

## Caveats

- **Slices, not full datasets, for motion metrics.** FineART: 791 of 40,543 episodes (1.9%), from 5 files spread
  across the index range. DROID: 3,970 of 95,658 (4.2%), from 4 of the 86 referenced files. The full-dataset usable
  hours are extrapolations.
- **The thresholds are choices.** These include:
  - idle below 5% of p95 speed for at least 0.5 s;
  - re-grasp within 2.5 s;
  - near-duplicate below 0.02 std;
  - at least 3 spikes and a rate above 2% for continuity.

  The 3-spike minimum was added after seeing DROID's almost-static episodes, and before the final Ext 3 run. The
  sweep shows its cost: 1% frame drops go undetected.
- **The re-grasp proxy is confounded.** Folding and bimanual handover in FineART re-close the gripper quickly on
  purpose. FineART's 47% is not a retry rate, and it drives the strict usable figure (43.6%). "Closes onto nothing"
  (LIBERO 27%, FineART 74%, DROID 37%) is also confounded: knob turns, pushes and an ALOHA gripper that closes when
  idle all count.
- **LIBERO-10 is already curated.** It has 379 of LIBERO's 500 original demos and near-zero idle, which suggests the
  conversion removed failed demos and no-op frames. I didn't verify this against the dataset card. Its 99.8% usable
  share says little about raw sim teleop.
- **Smoothness numbers mix control modes.** LIBERO actions are delta end-effector commands; FineART and DROID actions
  are absolute joint targets. Everything is normalised per dimension on a 10 Hz grid, but cross-dataset jerk values
  are indicative, not physical.
- **Licenses come from the Hub tags only:** LIBERO-10 apache-2.0, FineART cc-by-4.0, DROID 1.0.1 apache-2.0.
