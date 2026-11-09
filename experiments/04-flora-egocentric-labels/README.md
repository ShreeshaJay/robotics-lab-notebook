# 04 · Flora: open models for labeling egocentric video

**Date:** planned 2026-11-07, run 2026-11-08 · **Track:** Data labeling · **Status:** run at full precision on one A100 80 GB on P08_02, then on 4 more clips and a negative control (see "More clips") · **Spend:** about $3.30 (1.10 h of one Spot `a2-ultragpu-1g` plus its disk) for P08_02, plus about $7.40 for the more-clips extension ($1.26 for a first attempt stopped by Spot preemptions, about $6.15 for the retry)

## Goal

Learn what NVIDIA's Flora models actually produce on ordinary egocentric household video, and what it takes
to run them. On one public 107 s kitchen clip, get (1) temporal segments, (2) timestamped action captions and
(3) a dense 0.5 s process reward plus a 1 to 5 outcome score, with all three models at the precision the authors
validated (BF16 weights, FP32 Gated-DeltaNet parameters). Then check whether the labels match what happens in
the clip, and how their granularity compares with FineART-style step labels ([experiment 01](../01-fineart-dataset/README.md)).

The original plan was a 24 GB L4 with the two 27B models in 4-bit. I ran the faithful fallback instead: one
A100 80 GB with `--mode reference`, i.e. the unmodified upstream commands, models loaded one at a time.

## Results (run on 2026-11-08)

### Headline

| Model | What it produced on P08_02 (107 s) | Wall time | Peak VRAM (nvidia-smi) |
|-|-|-|-|
| Flora-TempSeg (9B, LoRA on Qwen3.5-9B) | 40 segments from 15 overlapping 16 s windows | 7.7 min (11 s load, 161 s of generation, the rest video decoding) | 19.2 GiB (18.5 GiB allocated by PyTorch) |
| Flora-Captioner (27B) | 40 captions, all `complete` | 21.1 min (6.7 min SHA-256 check of 109 GB of shards, 2.5 min load, 11.8 min captioning of which 175 s is generation) | 64.1 GiB (59.9 GiB allocated) |
| Flora-Rewarder (27B), dense | a score every 0.5 s, 11 clips, 0 invalid | 5.6 min (vLLM load 21 s, `torch.compile` 51 s, warm-up 99 s, inference about 40 s) | 71.9 GiB (51.0 GiB weights, the rest is vLLM's KV-cache reservation at `--gpu-memory-utilization 0.88`) |
| Flora-Rewarder (27B), outcome | score 5, P(5) = 0.9999, progress 1.0 | 1.9 min (compile cache warm) | 70.1 GiB |

- The caption stage (TempSeg then Captioner, one process) took 29.4 min end to end. NVIDIA reports 23 min for a 107 s
  video on one H100, including downloads. The difference is mostly I/O and CPU, not the GPU (see "Problems and fixes").
- Annotation and reward never ran at the same time. Flora's annotation session frees TempSeg before it loads the
  Captioner, and the rewarder is a separate vLLM process, so one 80 GB GPU is enough for everything except the
  rubric reward (which defaults to two GPUs and was not run).
- Flora's own load-time checks passed: TempSeg adapter values and GDN parameters bit-exact and FP32, Captioner GDN
  parameters bit-exact and FP32, all parameters on the GPU.

### Quality read

I compared the outputs with the EPIC-KITCHENS ground-truth narrations for this video
([`results/epic_ground_truth_P08_02.json`](results/epic_ground_truth_P08_02.json), 34 narrations) and with frames
I pulled at the flagged times.

**Segments and captions: good.** The clip has three phases, and Flora recovers all three in order with roughly the
right boundaries:

| Phase (ground truth) | Ground-truth narrations | Flora captions (examples) |
|-|-|-|
| 0 to 31 s: load the dishwasher | "put plates on counter", "open dishwasher", "place fork", "arrange mug", "arrange glass", "arrange bowls in drawer" | `0.0-5.2 s` place plate on counter with both hands · `7.0-9.2 s` pick up knife and fork with right hand, place knife and fork in dishwasher with right hand · `23.9-27.3 s` pick up glass with left hand, place glass in dishwasher rack with left hand |
| 31 to 58 s: start the dishwasher | "close dishwasher", "open cupboard", "unwrap", "start dishwasher", "throw away wrap" | `39.6-43.6 s` open dishwasher door with right hand, unwrap soap with both hands · `43.6-45.2 s` place detergent in dishwasher with right hand, close detergent compartment with right hand · `55.6-59.4 s` place debris in trash can with left hand, pick up cloth with left hand |
| 58 to 107 s: clean counter and sink | "get sponge", "clean counter top" (71 to 82 s), "clean sink" (84 to 90 s), "squeeze sponge", "place sponge back", "wipe off hands" | `70.6-82.4 s` wipe counter with sponge in right hand · `84.4-89.8 s` wipe sink with sponge in right hand · `91.4-92.8 s` fold cloth with both hands, squeeze cloth with right hand |

- Timing is close: "clean counter top" 71.3 to 82.4 s in the ground truth vs Flora 70.6 to 82.4 s; "clean sink"
  84.4 to 89.9 s vs 84.4 to 89.8 s.
- Errors I could see:
  - **Object naming drifts.** The same blue sponge cloth is called "cloth" in some segments and "sponge" in others.
    At 69.8 to 82.4 s Flora emits two overlapping segments for the same action, one with "cloth" and one with "sponge".
  - **A missed action.** "Start dishwasher" (pressing the control panel, 48.5 to 54 s) becomes "hold debris with left
    hand, close dishwasher door with both hands" over a 10.4 s segment.
  - **Reasonable guesses at hidden objects.** The dishwasher tablet is "soap" and then "detergent". The towel hung on the
    oven handle after drying hands (ground truth "wipe off hands") is "place cloth on oven handle".
- 5 of 40 segments were marked `fine_repaired_by_fine_retry`, Flora's own re-segmentation of windows it was unsure about.
  4 of those are where the overlaps are.
- Frames with the caption burned in: [`results/frames/`](results/frames/).

**Outcome reward: right, and it discriminates.** For the task "Clean the kitchen counter top and the sink." the full
clip scored 5 with P(5) = 0.9999 and a correct rationale ("wiped the countertop and sink with a blue cloth, then rinsed
the cloth"). As a negative control I scored only the first 45 s (dishwasher loading, no cleaning) against the same task:
score 1, P(1) = 0.9994, progress 0.0002, and a rationale saying the counter and sink were not cleaned and the dishes
were loaded into the dishwasher instead. Both rationales say "the robot" about a human, because the prompts are
written for robots.

**Dense (process) reward: plausible-sounding, not reliable on this clip.** 92.4% of the 0.5 s steps are "normal",
7.1% "failure" and 0.5% "suboptimal" (mean 2.85 of 3). I checked the flagged spans against the video:

| Span | Rewarder says | What the frame shows |
|-|-|-|
| 29.4 to 29.9 s (2) | "left hand hit the white bowl in the dishwasher rack" | a plausible minor bump while arranging bowls |
| 47.9 to 56.4 s (1) | "the rack is misaligned with the cabinet opening" | the person is pressing the dishwasher's control panel with the door slightly open, i.e. starting it. A false positive |
| 95.6 to 97.4 s (1) | "the blue cloth is placed on the top of the cabinet instead of the sink drain" | the person is wringing the cloth over the sink. A false positive that alternates 1/3 every 0.25 s where overlapping clips are merged |

![dense reward](results/reward_curve.png)

- One clip-level flag did not survive the merge: clip 1 said the plate "falls onto the bottom of the dishwasher" at
  12.6 to 13.6 s. The overlapping clip scored that span normal, and the merged output kept "normal".
- The first 30 s (loading the dishwasher, not part of the stated task) is scored "normal". The dense reward judges
  *how well* each motion is executed, not whether it serves the task. Task progress belongs to the outcome reward.
- My read: the dense reward's failure vocabulary ("misaligned", "placed instead of") comes from robot pick-and-place,
  and human household motions that look unusual to it get flagged. Treat it as a candidate list for human review,
  not as a label.

### Granularity: Flora captions vs FineART step labels

| | Flora on P08_02 | EPIC narrations (P08_02) | FineART subtask labels |
|-|-|-|-|
| Segment length | mean 3.2 s, median 2.1 s (1.0 to 12.6 s) | mean 2.8 s | about 9 s (5 to 15 s typical) |
| Segments per 107 s | 40 | 34 | about 12 at 9 s each |
| Text | imperative, median 11 words, 26 of 40 are 2 or 3 clauses joined by commas, every caption names the hand | 2 to 3 word verb-object | imperative, one action per label, median 8 words, attribute-rich object plus destination |
| Timeline | start and end per segment; segments may overlap (4 overlaps here) | start and end; overlaps allowed | start-only, contiguous, non-overlapping |

- Flora labels at the level of the atomic action (grasp, pass between hands, place), which is EPIC's level, not
  FineART's subtask level. A FineART-style step such as "Load the plates and cutlery into the dishwasher" would
  correspond to about 10 Flora segments.
- Getting FineART-style labels from Flora means post-processing: merge consecutive segments that share an object or
  goal into one step, write one imperative per merged span, drop the hand detail, resolve overlaps into one timeline,
  and fix object names. The per-hand detail Flora adds is something FineART does not have, and it is useful for
  bimanual robot data.

### What this teaches about labeling egocentric video

1. **Open models can now do a credible first pass on segments and captions.** On this clip the phase structure, the
   order and most boundaries matched human narrations to within about a second, at full precision on one 80 GB GPU.
   The errors are the kind a reviewer fixes quickly: inconsistent object names, overlapping duplicates, one merged action.
2. **Pick the label grain on purpose.** Flora's grain (about 3 s, per hand) is finer than the 5 to 15 s step labels
   used to train step-following policies. Both are useful, but they are different products, and converting between
   them is a deliberate merge step.
3. **Outcome scoring is more trustworthy than dense scoring.** The episode-level score separated success from a
   negative control with near-certain probabilities and correct rationales. (A later control in "More clips", where
   half the task was done, still scored 5, so this holds for "nothing done", not for "partly done".) The 0.5 s failure flags on human video
   were mostly false positives. Use dense scores to prioritize review, not to filter data automatically.
4. **Always run a negative control.** A score of 5 alone does not tell you the scorer can say no. Truncating the clip
   to a part where the task was not done took two minutes and showed that it can.
5. **The bottleneck is I/O and CPU, not the GPU.** About 60% of the annotation wall time was hashing 109 GB of FP32
   shards, reading them from a pd-balanced disk, and single-threaded PNG encoding of frames. For volume work, the
   disk type, a warm page cache and a long-running process that keeps the model loaded matter more than the GPU.
6. **License limits use.** Everything here is research/evaluation only under the NVIDIA License (below). Labels made
   with Flora cannot go into a commercial product or a dataset sold to others without a separate license from NVIDIA.

## More clips (2026-11-08 plan, run 2026-11-09)

**Status: run.** I ran Flora (TempSeg, Captioner, Rewarder) at full precision with `--mode reference` on 4 more public
EPIC-KITCHENS clips plus one negative control, on one Spot A100 80 GB. My first attempt on 2026-11-08 was stopped by
three Spot preemptions during setup. For the retry I used a VM that stops instead of being deleted on preemption. It was
preempted once, I restarted it, and everything finished in one 2 h run.

### Tries and preemptions

All times UTC, 2026-11-09. Tries 1 to 3 were the first attempt; tries 4 and 5 are the retry.

| Try | Zone | VM running | Reached | End |
|-|-|-|-|-|
| 1 | us-central1-a | 02:00:12 to 02:08:43 (8.5 min) | clip downloads 70% done, weights 37 GB of 185 GB | preempted, VM and disk deleted |
| 2 | us-central1-c | 02:11:39 to 02:25:35 (13.9 min) | setup, downloads and weight prefetch in progress | preempted, VM and disk deleted |
| 3 | us-central1-a | 02:28:28 to 02:31:13 (2.8 min) | setup just started | preempted, VM and disk deleted |
| 4 | us-central1-c | not created | | `STOCKOUT` (no Spot A100 capacity in the zone) |
| 5 | us-central1-a | 04:34:24 to 04:35:47 (1.4 min) | OS packages being installed | preempted, **VM stopped, disk kept** |
| 5, restarted | us-central1-a | 04:37:28 to 06:38:10 (120.7 min) | everything | I deleted it |

- `a2-ultragpu-1g` exists only in us-central1-a and us-central1-c (I checked again with `gcloud compute machine-types list`),
  so zones b and f were not an option.
- For the retry I created the VM with `ON_END=STOP` (`--instance-termination-action=STOP`). The preemption in try 5
  stopped the VM and kept its disk, and `gcloud compute instances start` brought it back about 1.5 min later. That
  preemption cost 1.4 min of VM time and about 4 min of wall time, not a 20 min setup.
- Totals for the retry: 1 preemption, 1 restart, 2.04 VM-hours.

Problems on the retry and what I did:

| Problem | Cause | Fix |
|-|-|-|
| The first `instances create` with `ON_END=STOP` failed: "VM has a Local SSD attached but an undefined value for `--discard-local-ssds-at-termination-timestamp`" | `a2-ultragpu-1g` bundles a 375 GB local SSD, and GCE requires saying what happens to it when the VM stops at its time limit | `create_vm.sh` now adds `--discard-local-ssds-at-termination-timestamp=true` when `ON_END=STOP` on `a2-ultragpu` machines (nothing is stored on the local SSD) |
| After the restart, `apt-get install` failed with "dpkg was interrupted" | the preemption hit in the middle of the package install | `sudo dpkg --configure -a`, then I re-ran the driver. Any restart should run this first |
| `run.py` downloads would have restarted from zero after a preemption | plain `urlopen` | clip downloads now resume with an HTTP `Range` request, and encodes write to `*.part.mp4` and are renamed when done. The weight downloads resume through the Hugging Face cache. Not exercised: the one preemption came before any download started |
| `run.py --clips` logs a failed Flora stage but exits 0 | by design, so one failed stage does not stop the next | my driver script checked the `exit_code` in each `*.meta.json` before going on (all were 0) |

### Clips

I picked them from the EPIC-100 narration annotations: short videos (60 to 120 s, because Flora's annotation input
limit is 120 s) or 110 s windows with the highest share of the target verbs. Each clip is from a different
participant, none from P08. Manifest: [`more_clips.json`](more_clips.json). Ground-truth narrations (CC BY-NC 4.0)
are in `results/more_clips/epic_ground_truth_<clip>.json`, with times on the clip's own clock.

| Clip | Kind | Length | Source | EPIC narrations (mean length) | Outcome task |
|-|-|-|-|-|-|
| P14_07 | dishwashing | 90.3 s, whole video | `train/P14`, 340 MB | 12 (5.2 s): take sponge, open tap, wash spoon (16 s), wash bowl (30 s), turn off water | Wash the spoon and the bowl in the sink. |
| P01_19, 301 to 416 s | wiping | 115 s window | `train/P01`, 1.8 GB video; ffmpeg reads only the window over HTTPS (the server accepts byte ranges) | 31 (4.5 s): rinse and squeeze cloth, wipe sink, wipe counter, wipe cooker (30 s), wash sink | Wipe the sink, the counter and the cooker with the cloth. |
| P28_24 | food prep | 68.3 s, whole video | `test/P28`, 258 MB | 16 (5.1 s): cut tomato, throw piece in bin, chop tomato, take dish from microwave, chop tomato | Chop the tomato. |
| P11_24 | put-away | 83.6 s, whole video | `test/P11`, 315 MB | 24 (2.4 s): put away wine glass, bowls, chopping block, mug and scissors in cupboards, put bag in bin | Put the dishes away in the cupboards. |
| P14_07, first 46 s | negative control | 46.0 s | cut from the P14_07 re-encode | spoon washed, bowl not yet picked up (EPIC: pick up bowl at 46.5 s) | same task as P14_07; expected well below 5 |

All URLs are under `https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/`, the same direct route as
P08_02. Nothing asked for a sign-up or a token.

### How I ran it

```bash
# laptop (Git Bash)
PROJECT=<project> NAME=scout-flora3-a100 MACHINE=a2-ultragpu-1g SPOT=1 ON_END=STOP HOURS=3 DISK_GB=300 \
  ZONE=us-central1-a bash ../../tools/gpu-vm/create_vm.sh
# VM: one idempotent driver, re-run after a restart (finished phases are skipped by marker files):
#   apt packages; then in parallel `run.py fetch --clips`, `setup.sh` (WITH_BNB=0), and `run.py prefetch` once venv 1 exists;
#   then caption and reward with --validate-only, then `run.py caption --clips more_clips.json`, `run.py reward --clips more_clips.json`
# after a preemption: gcloud compute instances start ...; sudo dpkg --configure -a; re-run the driver
# laptop: copy results back, merge_steps.py --batch, delete the VM, check that no instance or disk is left
```

`caption --clips` is one annotation process over all 4 clips (`--input-jsonl`); `reward --clips` is one dense process
(4 clips) and one outcome process (4 clips plus the negative control), both with `--jobs`.

### Per-clip results

| Clip | Segments (all `complete`) | Mean segment | Overlapping pairs | Fine re-segmentations | Dense reward mean, normal / suboptimal / failure | Outcome score, P(score), progress |
|-|-|-|-|-|-|-|
| P14_07 (90 s) | 35 | 2.8 s | 5 | 4 | 2.73: 74% / 24% / 2% | 5, P(5) = 1.000, 1.000 |
| P01_19_w301 (115 s) | 35 | 4.4 s | 10 | 4 | 2.52: 65% / 23% / 12% | 5, P(5) = 0.994, 0.998 |
| P28_24 (68 s) | 23 | 3.9 s | 5 | 5 | 2.65: 65% / 34% / 0.3% | 5, P(5) = 1.000, 1.000 |
| P11_24 (84 s) | 31 | 3.5 s | 10 | 7 | 2.72: 73% / 26% / 1% | 5, P(5) = 1.000, 1.000 |
| P14_07 first 46 s (negative control) | not run | | | | not run | **5, P(5) = 0.9995, 0.9999** |

All dense runs had 0 invalid clips. Files: `results/more_clips/captions/<clip>.json`, `dense.jsonl`, `terminal.jsonl`.

**Segments and captions: the phase structure holds up in all four kitchens.**

| Clip | Ground truth (EPIC) | Flora captions (examples) |
|-|-|-|
| P14_07 | take sponge 6 to 8 s, wash spoon 29 to 45 s, pick up bowl 47 s, wash bowl 48 to 78 s, turn off water 78 to 80 s | `5.7-7.5 s` pick up sponge with right hand · `29.6-34.6 s` hold spoon with left hand, scrub spoon with sponge in right hand, flip spoon… · `46.2-47.4 s` hold sponge with right hand, pick up bowl with left hand · `78.0-79.2 s` hold bowl with left hand, turn off faucet with right hand |
| P01_19_w301 | wipe sink and counter 9 to 38 s, bin 40 to 42 s, wipe cooker 57 to 89 s, wipe counter 90 to 97 s, wash sink 104 to 110 s | `13.3-22.8 s` wipe counter with cloth in right hand · `36.6-42.0 s` hold cloth with right hand, open trash can with right hand, place debris in trash can… · `56.4-73.2 s` wipe stove with cloth in right hand · `89.0-99.2 s` wipe counter with cloth in right hand |
| P28_24 | cut tomato, throw piece in bin 10 to 12 s, chop 14 to 36 s, take dish from microwave 38 to 46 s, chop 47 to 63 s, put down knife 64 to 67 s | `9.4-11.6 s` hold knife with left hand, open trash can with right hand, place tomato in trash can… · `36.7-41.7 s` open microwave door with left hand, pick up plate with left hand · `65.0-68.3 s` place knife on counter with left hand |
| P11_24 | put away glass, bowls, chopping block, mug, scissors 5 to 46 s; open bag 57 to 72 s; put bag in bin 73 to 81 s | `5.3-9.8 s` open cabinet with left hand, pass glass to left hand, place glass in cabinet… · `28.1-31.6 s` pick up mug with left hand, place mug on shelf… · `73.2-80.8 s` place plastic bag in bin with both hands |

- Timing is close where the ground truth is specific: "pick up bowl" 46.5 s vs Flora 46.2 s, "turn off water" 78 s vs
  78.0 s, cooker 57 to 89 s vs stove 56.4 to 89.0 s, bin 10 to 12 s vs 9.4 to 11.6 s.
- Errors I saw:
  - **Object identity.** In P11_24, "open bag" (57 to 72 s) becomes "straighten paper towel", "fold tissue" and
    "separate wipes", and the chopping block is a "cutting board" and then a "mat" put "in drawer". In P01_19 the cloth
    and the sponge swap names again, as in P08_02.
  - **Overlapping duplicates.** 5 to 10 overlapping pairs per clip, often the same action twice with one extra clause
    (P01_19: `13.3-22.8 s` and `13.3-25.6 s`, both "wipe counter").
  - **Small actions folded in.** EPIC's "open tap" in P14_07 is folded into "pass sponge…, turn on faucet".

**Outcome reward: it gave 5 to all four successful clips, and also to the negative control.** See the next section.

**Dense reward: 26 to 35% of steps are flagged on four successful human demos** (P08_02: 8%). I checked four
flagged spans against frames (`results/more_clips/frames/`):

| Span | Rewarder says | What the frame shows |
|-|-|-|
| P01_19_w301 39.9 to 46.9 s (1) | "the right hand loses grip on the rag" | at 42 s the person is at the bin, dropping rubbish in on purpose (EPIC: open bin, throw rubbish, close bin at 40 to 42 s). A false positive |
| P01_19_w301 103 to 110 s (1, alternating with 3 every 0.25 s) | "the cloth was dropped from the right hand due to poor grip" | at 106 s the hand is wiping the sink rim under the running tap (EPIC: wash sink). No drop visible |
| P28_24 19.9 to 36.4 s (2) | "the right hand is jittering… the left hand is taking more time to chop" | at 30 s, ordinary chopping on the board. A style judgement, not an error |
| P14_07 86.4 to 89.7 s (1) | "the right hand was not able to grasp the towel due to misalignment" | at 88 s the camera points at the drawers and the hand reaches for a drawer, after EPIC's last narration (83 s). I can't confirm it from one frame |

This matches what I saw on P08_02: the dense reward's failure vocabulary comes from robot grasping, and on human
video it gives a review queue, not a label.

### Negative control: the outcome score did not drop

| Video | Task | Score | P(5) | Progress | Rationale (first sentence) |
|-|-|-|-|-|-|
| P14_07, full 90 s | Wash the spoon and the bowl in the sink. | 5 | 1.000 | 1.000 | "The robot picked up the spoon and bowl, scrubbed them with a sponge, and rinsed them under the faucet." |
| P14_07, first 46 s | same | **5** | 0.9995 | 0.9999 | "The robot picked up the bowl and spoon, placed them in the sink, and used a sponge and dish soap to scrub and rinse both items." |

- The rationale for the cut clip is wrong. The frame at 45.5 s (`results/more_clips/frames/P14_07_first46s_45.5s.jpg`)
  shows the spoon going into the rack and the bowl still in the sink, unwashed. Flora's own captions for the full clip
  put "pick up bowl" at 46.2 s, after the cut.
- The P08_02 control (first 45 s against a cleaning task) dropped to 1. The difference: there, nothing in the cut clip
  served the task. Here half the task is done, and the bowl is in view, in the sink, from the first second.
- So on this one example the outcome reward told "nothing done" from "done", but not "half done" from "done".
  It gave 5 to all 5 videos, so this run gives no evidence that it can say no to a partial episode.

### Merge to steps (`merge_steps.py`, CPU)

The rule, with thresholds fixed before looking at results:

1. **One timeline.** A segment that starts inside the previous one is trimmed to start where the previous one
   ends. A segment almost fully inside the previous one is folded into it.
2. **Verb and object per clause.** Each Flora caption is split at commas, the hand phrases are dropped, and the
   verb and object are read from each clause. For example, "pick up glass with left hand" gives (pick up, glass).
3. **Greedy merge.** The next segment joins the current step when three things hold:
   - it is adjacent (gap of 1 s or less);
   - the step stays 15 s or shorter;
   - the two share a verb-object pair, or the object, or enough caption words (Jaccard 0.3 or more).
4. **Short steps.** A step shorter than 4 s joins the neighbour it shares more words with.
5. **Output.** Each step gets one imperative, taken from the clause about the step's main object. Steps are written
   as FineART-style rows: `style: subtask`, a start `timestamp`, and `content`; contiguous, non-overlapping and start-only.

Scoring against the EPIC narrations:

- **Boundary error:** the distance from each internal step boundary to the nearest EPIC narration start or end.
  EPIC boundaries are dense, so I also report evenly spaced boundaries as a chance baseline.
- **Matched human steps:** an EPIC narration counts as matched when one step covers at least half of it and mentions
  its verb or noun (a small synonym table maps tap to faucet, cupboard to cabinet, and so on). "Strict" counts only
  the step's one-line text. "Lenient" also counts the Flora captions merged into the step.

Results, from `results/more_clips/steps_<clip>.json` (`python merge_steps.py --batch results/more_clips/captions results/more_clips`):

| Clip | Units | Mean length | Boundary error, median (even-grid baseline) | Within 1 s of a human boundary | EPIC narrations matched, strict / lenient |
|-|-|-|-|-|-|
| P08_02, raw segments | 40 | 3.2 s | 0.27 s (0.56 s) | 82% | 26 / 26 of 34 |
| P08_02, merged steps | 10 | 10.7 s | 0.22 s (0.71 s) | 100% | 11 / 30 of 34 |
| P14_07, raw | 35 | 2.8 s | 0.85 s (2.79 s) | 53% | 11 / 11 of 12 |
| P14_07, merged | 9 | 10.0 s | 1.84 s (3.65 s) | 38% | 10 / 12 of 12 |
| P01_19_w301, raw | 35 | 4.4 s | 0.38 s (1.07 s) | 79% | 27 / 27 of 31 |
| P01_19_w301, merged | 11 | 10.5 s | 0.58 s (1.74 s) | 70% | 22 / 29 of 31 |
| P28_24, raw | 23 | 3.9 s | 0.47 s (1.13 s) | 73% | 16 / 16 of 16 |
| P28_24, merged | 6 | 11.3 s | 0.34 s (2.02 s) | 80% | 11 / 16 of 16 |
| P11_24, raw | 31 | 3.5 s | 0.35 s (0.34 s) | 77% | 20 / 20 of 24 |
| P11_24, merged | 9 | 9.3 s | 0.54 s (0.67 s) | 62% | 14 / 22 of 24 |

- **Length is on target on every clip.** Merged steps average 9.3 to 11.3 s, inside FineART's 5 to 15 s range.
- **Boundaries beat chance on all four new clips, but P08_02 was the best case.** The merged-boundary median error is
  0.34 to 1.84 s against a baseline of 0.67 to 3.65 s, and 38 to 80% of boundaries land within 1 s of a human one
  (P08_02: 100%).
  - On P11_24 the margin over chance is small, for raw segments (0.35 vs 0.34 s) and for steps (0.54 vs 0.67 s). Its
    24 narrations are short and dense, so any boundary lands near one.
  - P14_07 has sparse narrations (12, one of them 30 s long), so its errors in seconds are larger, but its margin over
    chance is the largest.
- **Step text is still the weak part.** Strict matches are 10 of 12, 22 of 31, 11 of 16 and 14 of 24. Lenient matches
  are 12, 29, 16 and 22, so the merged captions hold the information but the one-line text loses some. Typical misses
  name a sub-action, not the goal:
  - "Hold tomato." for 0 to 23 s of cutting (P28_24);
  - "Hold bowl." for 49 to 62 s of scrubbing the bowl (P14_07);
  - "Straighten paper towel." for opening a bag (P11_24, an object error carried through).
- These are five clips with one annotator each, so they are a sanity check, not a measured accuracy.

### Wall time per phase

These are for the restarted VM (try 5), from my driver's phase log (`results/more_clips/logs/phases.log`) and Flora's
time-stamped events.

| Phase | Wall time | Notes |
|-|-|-|
| Restart after the preemption | 1.4 min stopped, 0.3 min for `instances start` | plus about 2 min to find and fix the interrupted `dpkg` |
| OS packages | 0.2 min | |
| Clip downloads and encodes (in parallel with the next two) | 14.4 min | reading the P01_19 window over HTTPS took 6.5 min of it |
| `setup.sh` (two venvs) | 22.3 min | the critical path, mostly the vLLM venv |
| Weight prefetch, 185 GB | 14.7 min | started once venv 1 existed |
| `--validate-only` and `--dry-run` | 1.7 min | |
| Caption, 4 clips in one process | 73.4 min | per clip below |
| Dense reward, 4 clips | 9.9 min | loading the weights from a cold page cache took 3.7 min (21 s in the first run, warm cache) |
| Outcome reward, 5 videos | 2.6 min | compile cache warm |
| Copy results back, 6 check frames, delete | 8.6 min | |

| Caption stage per clip | TempSeg | Captioner load | Captioning | Total |
|-|-|-|-|-|
| P14_07 (90 s) | 5.6 min | 8.5 min (7.4 min SHA-256 check of 109 GB, once) | 10.3 min | 24.4 min |
| P01_19_w301 (115 s) | 9.0 min | 0.3 min | 10.9 min | 20.1 min |
| P28_24 (68 s) | 4.3 min | 0.3 min | 7.8 min | 12.4 min |
| P11_24 (84 s) | 6.0 min | 0.3 min | 9.8 min | 16.1 min |

- **The re-hash is skipped, as I expected from the code.** Clips 2 to 4 logged `manifest_verified_reuse` with 0 bytes
  read for both models. The batch mode does reload TempSeg and the Captioner for every clip, but from a warm page cache
  that takes about 20 s.
- After the first clip, the caption stage costs about 10.5 min per minute of video. GPU generation was 19 min of the
  73 min (180 generations). The rest is frame decoding and PNG encoding on the CPU, as in the first run.
- Peak VRAM (nvidia-smi): caption 64.8 GiB, dense 71.9 GiB, outcome 71.9 GiB, the same as the first run.

### Cost

| Item | Used | Cost |
|-|-|-|
| `a2-ultragpu-1g` Spot at $2.952/h | 122.1 min running (1.4 + 120.7) = 2.04 VM-hours | $6.01 |
| 300 GB pd-balanced boot disk at $0.10/GB-month | 123.8 min (billed while stopped too) | $0.08 |
| Bundled 375 GB local SSD | same | not in the $2.952/h rate; about $0.08 at $0.08/GB-month (I did not look up its Spot rate) |
| **Retry total** | | **about $6.10 to $6.20** |
| First attempt (tries 1 to 3) | 0.42 VM-hours | about $1.26 |
| **More clips, all tries** | 2.46 VM-hours | **about $7.40** |

After copying the results I ran `gcloud compute instances delete scout-flora3-a100 --zone us-central1-a --quiet`.
Then `gcloud compute instances list` and `gcloud compute disks list` both returned "Listed 0 items".

### What this adds

1. **Segments and captions held up in four more kitchens.** Phase order and the main boundaries matched human
   narrations on all four new clips. The recurring errors are object names and overlapping duplicates.
2. **The outcome reward is not a completion check.** It scored all five videos 5, including one where half the task
   was not done, with a rationale describing the missing half. Checking partial episodes needs something else, for
   example the per-step rubric reward (two 80 GB GPUs by default) or a human.
3. **Merging to steps is cheap and lands at the right grain,** but the boundary agreement on P08_02 was the best of five clips.
4. **For Spot runs, keep the disk.** One preemption cost 1.4 min of VM time instead of a 20 min setup.

## Steps actually run

```bash
# laptop (Git Bash): one Spot A100 80 GB, hard-deleted after 3 h by GCE, 300 GB pd-balanced boot disk
PROJECT=<project> NAME=scout-flora-a100 MACHINE=a2-ultragpu-1g SPOT=1 HOURS=3 DISK_GB=300 ZONE=us-central1-a \
  bash ../../tools/gpu-vm/create_vm.sh            # us-central1-a had Spot capacity on the first try
gcloud compute scp setup.sh run.py scout-flora-a100:flora-lab/ --zone us-central1-a

# VM (Ubuntu 24.04.5, Python 3.12.3, driver 580.178.04, ready about 30 s after create)
WITH_BNB=0 bash setup.sh                          # no bitsandbytes on 80 GB; about 18 min, mostly the vLLM venv
export FLORA_LICENSE_ACK=noncommercial-research-eval
python3 run.py prefetch --mode reference          # new: all weights up front, same revisions (15 min, 185 GB)
python3 run.py fetch    --mode reference          # in parallel: clip download (about 6 min) + 720p re-encode (98 s)
python3 run.py validate --mode reference          # --validate-only and --dry-run both passed
python3 run.py caption  --mode reference          # TempSeg -> Captioner, unmodified upstream CLI
python3 run.py reward   --mode reference          # dense + outcome, unmodified upstream CLI
python3 run.py summary  --mode reference
# extra: outcome reward on the first 45 s only (negative control), same model, same task text

# laptop: copy back small results, then
gcloud compute instances delete scout-flora-a100 --zone us-central1-a --quiet   # and confirm no instances or disks remain
```

Setup, prefetch and fetch ran in parallel and were all done 21 min after the VM was created.

Changes to the scripts for this run:

- `setup.sh`: `WITH_BNB=0` skips bitsandbytes (only needed for the 4-bit L4 path).
- `run.py`: sets `VLLM_USE_FLASHINFER_SAMPLER=0` (lesson from experiment 05); adds a `prefetch` stage so model
  timings exclude downloads; writes a time-stamped GPU-memory trace (`*.gpu.csv`) next to each log, so peak VRAM
  can be split per model.

### Pinned versions

- Flora `d8b755ec9a990418b1f31c7f48c75341c7229ea6`.
- Models: TempSeg `114a1076…`, Captioner `fef0bbfe…`, Qwen3.5-9B base `c2022362…` (all from Flora's `models.json`);
  Rewarder `b64cee90…`.
- Annotation venv: torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 5.22.1.
- Rewarder venv: vLLM 0.33.0, which pulled torch 2.14.0+cu130, transformers 5.20.2.
- Full lists: [`results/freeze-annotation.txt`](results/freeze-annotation.txt), [`results/freeze-rewarder.txt`](results/freeze-rewarder.txt).
- Image family `common-cu129-ubuntu-2404-nvidia-580` (Deep Learning VM). I did not record the dated image name.

### Problems and fixes

| Problem | Cause | Fix / note |
|-|-|-|
| `gcloud compute scp` failed with `~/flora-lab/: not a directory`, then "Multiple remote sources not supported" | Windows gcloud uses PuTTY's `pscp`, which does not expand `~` or accept several remote files | Use paths relative to the home directory, and tar the results on the VM so one file is copied back |
| Captioner stage spent 6.7 min before loading anything | Flora SHA-256-verifies all 109 GB of Captioner shards (stored in FP32) on every load; pd-balanced read about 270 MB/s | Expected behaviour. A pd-ssd disk (the original plan) or a local SSD would cut this; a long-running process would pay it once |
| 11.8 min to caption 40 segments with only 175 s of GPU generation | `pipeline.py` writes every input frame to PNG on one CPU core (seen with `py-spy dump`) | None for this run. It is upstream behaviour |
| Logs looked stalled while the model was working | `run.py` tees child output into block-buffered files | Watched the GPU trace instead; harmless |
| vLLM logged 96 "Attempted to load weight … float32 into parameter with dtype bfloat16" warnings for `linear_attn.dt_bias` / `norm.weight` | vLLM 0.33.0 keeps these Qwen3.5 GDN parameters in BF16, while the annotation path keeps them in FP32 | Upstream behaviour, not changed. A small precision difference between the two paths to keep in mind |
| "FlashInfer kernels are disabled" warning | no `nvcc`/`ninja` for JIT | Harmless with `VLLM_USE_FLASHINFER_SAMPLER=0` |

Nothing asked for a Hugging Face token: all four repos reported `gated: false` and downloaded anonymously.

## Cost

| Item | Rate (Google Cloud Billing Catalog, us-central1, effective 2026-11-08) | Used | Cost |
|-|-|-|-|
| `a2-ultragpu-1g` Spot: 12 vCPU × $0.01712 + 170 GB RAM × $0.002294 + A100 80GB GPU $2.3568 | **$2.952/h** (on-demand would be about $5.03/h) | 1.10 h (created 19:58:57Z, deleted 21:04:45Z) | $3.24 |
| 300 GB pd-balanced boot disk | $0.10/GB-month | 1.10 h | $0.05 |
| Ingress (185 GB from Hugging Face, 405 MB from data.bris) | free | | $0 |
| **Total** | | | **about $3.30** |

The pricing pages on cloud.google.com render their tables client-side and could not be read as text, so the rates
come from Google's Cloud Billing Catalog API (Compute Engine SKUs), the source behind those pages. The VM and its
disk were deleted at the end; `gcloud compute instances list` and `gcloud compute disks list` both returned nothing.

Time split of the 1.10 VM-hours: setup and downloads 21 min, annotation 29 min, rewards 8 min, negative control
2 min, the rest checks and copying.

## What Flora is

- A three-model suite from NVIDIA GEAR that turns robot and human manipulation video into labels: when
  each action happens, what it is, and how well it went. Announced 2026-10-20; NVIDIA says it is the data
  engine used to train GR00T-2.
- Project page: <https://research.nvidia.com/labs/gear/flora/> · paper: arXiv:2610.31779 ·
  code: <https://github.com/NVlabs/Flora> (the README's clone line says `NVIDIA/Flora`; both resolve to the same
  commit) · weights: <https://huggingface.co/collections/nvidia/flora>
- The repo is inference code only ("sample vanilla inference code"). No training code.

### The three models

| Model (HF repo) | Params | Base | Download | Measured peak VRAM on A100 (this run) | Fits one 24 GB L4? |
|-|-|-|-|-|-|
| Flora-TempSeg / "Flora-segmenter-9B" (`nvidia/Flora-TempSeg`) | 9.0 B (LoRA r=32 on the base) | Qwen3.5-9B (Apache-2.0) | 0.4 GB adapter + 19.3 GB base | 19.2 GiB | **Probably** at release precision (not tested on an L4) |
| Flora-Captioner / "Flora-captioner-27B" (`nvidia/Flora-Captioner`) | 27.4 B (full fine-tune) | Qwen3.5-27B | 109 GB (shards stored in FP32) | 64.1 GiB | **No.** Only as an unvalidated 4-bit experiment |
| Flora-Rewarder / "Flora-rewarder-27B" (`nvidia/Flora-Rewarder`) | 27.8 B (full fine-tune incl. vision) | Qwen3.5-27B | 55.6 GB (BF16) | 51.0 GiB weights, 72 GiB with vLLM's KV-cache reservation | **No.** Only as an unvalidated 4-bit experiment |

- The validated setup is Linux x86_64, Python 3.12 and one **H100 80 GB**. This run shows one **A100 80 GB** also
  works for annotation, dense and outcome rewards. The rubric reward needs **two** 80 GB GPUs by default
  (`--tensor-parallel-size 2`).
- The Captioner loader caps each GPU at (VRAM minus 12 GiB) and then refuses to run if any module would land
  off-GPU. On an A100 80 GB that cap is 68 GiB, enough for the 51 GiB of BF16 weights.
- Precision matters to the authors: BF16 weights with FP32 Gated-DeltaNet (`A_log`, `dt_bias`) parameters,
  verified bit-exact at load time. Their `AGENTS.md` says not to change prompts, frame sampling or
  precision, because other values "give worse results without any error".
- Reported throughput: a 107 s video took 23 minutes on one H100, first run including downloads. At scale they
  quote about $0.72 per hour of video (eight GB300s at $10 per GPU-hour).

### Inputs

- **Annotation (TempSeg then Captioner):** one MP4 per job: 120 s or less, 256 MiB or less, 4096 px or less per side,
  at least 4 FPS, one video stream. Profiles: `human` (one egocentric camera), `yam` (parallel grippers),
  `dex`, `tianji`. Robot profiles take up to three synchronized cameras. TempSeg reads overlapping 16 s windows
  at 4 FPS (64 frames; 32 s with `--yam32`). The Captioner sees a target interval plus BEFORE/AFTER context
  (8 FPS, at most 40 frames). Pass native video. Do not resample.
- **Rewarder:** outcome uses at most 64 frames of the episode. Dense uses 16 s clips of 64 frames at 4 FPS, one
  every 10 s. Rubric uses each camera at 2 FPS, at most 166 frames. Up to 4 camera views are tiled into a collage.
  A `--task` instruction in plain English is required.

### Outputs

- **Annotation:** one JSON per video (`schema_version: flora_demo_annotation_v1`). `segments[]` holds
  `{start_s, end_s, caption, status, segmentation_provenance}` in seconds. Segments may overlap.
  `caption` is one English imperative instruction, filled only when `status` is `complete`.
- **Outcome reward:** `score` 1 to 5, `probabilities` P(1..5), `expected_score`, `progress` 0 to 1 (for ranking,
  not a success probability) and an optional one-line `rationale`.
- **Dense (process) reward:** a score every 0.5 s (3 normal, 2 suboptimal, 1 failure). Also `segments` with
  rationales for non-normal spans, `per_second`, `fraction`, `worst_score`, `mean_score` and `invalid_clips`.
  It is retrospective (each clip sees 4 s ahead), so it is meant for offline labeling. The dense system
  prompt is phrased for robot manipulation (see the quality read for what that does on human video).
- **Rubric:** one score per user-defined step (the medoid of 9 samples) plus a thinking trace. Not run.

### License and access

- **License:** the code and all three sets of weights are under the **NVIDIA License**, the same text in
  `LICENSE` and in `LICENSE.pdf` in each HF repo. §3.3 says the Work and derivatives "only may be used or
  intended for use non-commercially", defined as **"for research or evaluation purposes only"**. NVIDIA and
  its affiliates are exempt. The model cards add: "not intended or supported for commercial or production use."
- **Outputs:** the license has no separate clause on outputs. But running the model to produce labels is itself
  "use" of the Work, so labeling data for a commercial product, or for a dataset sold to third parties, is
  outside the grant. The Apache-2.0 license of the Qwen base models does not change that. A commercial use
  would need a separate license from NVIDIA. Internal research or evaluation is the permitted lane, and this run
  was personal learning under it.
- **Gating:** none. All repos showed `gated: false` and downloaded without a token. A dead helper in
  `captioner/batch_utils/models.py` still mentions "private weights", but the CLI never calls it.
- **Test clip:** EPIC-KITCHENS is non-commercial (CC BY-NC 4.0 on the project site; data.bris lists a
  "Non-Commercial Government Licence"). That is fine for personal learning notes. No sign-up is needed.
  The ground-truth narrations copied into `results/` come from the EPIC-KITCHENS-100 annotations (CC BY-NC 4.0).

## Test input

- **EPIC-KITCHENS-55 `P08_02`:** 107 s, 1920x1080 at 59.94 fps, head-mounted GoPro in a home kitchen.
- Source: `https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P08/P08_02.MP4`
  (405,011,907 bytes, direct download). DOI 10.5523/bris.3h91syskeag572hl6tvuovwv4d. I picked it by scanning
  the EPIC-100 video list for 60 to 120 s videos with the highest share of wash/clean/wipe actions. It turned out
  to be three activities: about 31 s loading a dishwasher, 27 s starting it, then 49 s cleaning the counter and sink.
  The outcome-reward task "Clean the kitchen counter top and the sink." covers only the last part.
- The raw file is over Flora's 256 MiB limit, so `run.py` re-encodes it **once**: 1280x720, H.264 CRF 20, audio
  dropped, every native frame timestamp kept (no trim, no fps change). Result: 107.4 s, 74 MiB, 6,439 frames.
- Not used: Flora's own examples (4 s GIF excerpts, too short), Ego4D and Ego-Exo4D (signed license agreement needed).

## Results files

| File | What it is |
|-|-|
| `results/captions_reference.json` | Flora annotation output (40 segments) |
| `results/dense_reference.jsonl` | dense reward: per-0.5 s scores, non-normal segments with rationales, per-clip outputs |
| `results/terminal_reference.jsonl` | outcome reward on the full clip |
| `results/terminal_negcontrol_first45s.jsonl` | outcome reward on the first 45 s (negative control) |
| `results/summary_reference.json` | `run.py summary` output |
| `results/epic_ground_truth_P08_02.json` | EPIC-KITCHENS-100 narrations for P08_02, for comparison |
| `results/frames/*.png`, `results/reward_curve.png` | 4 frames with captions burned in; dense reward over time |
| `results/logs/` | stage logs, `*.meta.json` (exit code, wall time, peak VRAM), `*.gpu.csv` (GPU memory every 2 s) |
| `more_clips.json` | the 4 extra clips and the negative control: URL, window, outcome task |
| `merge_steps.py` | CPU script: Flora segments to FineART-style steps, scored against EPIC narrations |
| `results/more_clips/epic_ground_truth_*.json` | EPIC-KITCHENS-100 narrations for the 4 extra clips, on the clip clock |
| `results/more_clips/captions/<clip>.json` | Flora annotation output for the 4 extra clips |
| `results/more_clips/dense.jsonl`, `terminal.jsonl` | dense reward (4 clips) and outcome reward (4 clips plus the negative control) |
| `results/more_clips/steps_<clip>.json` | merged steps, FineART-style rows and scores, for P08_02 and the 4 extra clips |
| `results/more_clips/*_jobs.jsonl`, `clip_probe.json` | the batch job files and the ffprobe of each input |
| `results/more_clips/logs/` | stage logs with UTC stamps, `*.meta.json`, `*.gpu.csv`, and the driver's `phases.log` |
| `results/more_clips/frames/*.jpg` | 6 small frames used to check the negative control and dense-reward flags |

No weights and no video are in this repo.

## Notes on sources

- The repo includes an `AGENTS.md` written for coding agents. I read it as documentation only. Its rules (run
  `--validate-only` first, ask before downloading weights or re-encoding, never commit videos or weights)
  match this plan. Nothing in it was unusual, and nothing else I read during the run asked for anything unusual.
- The project page's comparison table names several proprietary models I can't verify. Those comparisons are
  not used here.

## Next steps

- Test partial episodes properly: more truncated controls (a quarter, half, three quarters of the task), and the
  per-step rubric reward, which needs two 80 GB GPUs by default. The outcome reward gave a half-done episode 5.
- Replace the rule-based step text in `merge_steps.py` with a language model over the merged captions, and compare
  it with the strict match counts above.
- Measure the per-video cost with a warm, long-running process on a pd-ssd or local SSD, where the 109 GB hash and
  load are paid once.
