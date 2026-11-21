# Results: Four VLAs on the same 50 LIBERO-10 episodes

Setup, versions, licenses and deviations: [README.md](README.md). Spend: [COSTS.md](COSTS.md) (about $52 of $60).
All numbers come from `results/summary.json` (`python analysis/analyze.py`). n is stated everywhere; intervals are
Wilson 95%. Episodes = 10 tasks x LIBERO init states 0-4, seed = init-state index, 520-step budget.

## Main comparison

| Model | Success | Wilson 95% | Latency per call, L4 (median) | Latency per call, A10 (median) | Peak VRAM: torch / whole GPU |
|-|-|-|-|-|-|
| G0.5 | **49/50 (98%)** | 89.5-99.6% | 620 ms (32 predicted, 10 executed) | 395 ms | 11.0 GiB / 12.6 GiB (L4) |
| GR00T N1.7 | **45/50 (90%)** | 78.6-95.7% | 211 ms (16 predicted, 8 executed) | 127 ms | 6.0 GiB / 9.1 GiB (L4) |
| SmolVLA | **26/50 (52%)** | 38.5-65.2% | 324 ms (50 predicted, 50 executed) | 215 ms | 0.9 GiB / 2.0 GiB (L4) |
| PEAR | **26/35 finished (74%)**; all 50: 52-82% | 57.9-85.8% (n=35) | one turn: median 24.9 s, p90 170 s (B200, 5-16 episodes batched) | n/a | weights 51.1 GiB; vLLM reserves 165-173 GB of a B200 for KV |

- G0.5 and SmolVLA are the baseline's saved results (2026-11-08, GCP L4). GR00T and PEAR were run here on the same
  init states. The whole-GPU peaks include EGL rendering of the simulator.
- **PEAR finished 35 of 50 episodes.** The other 15 were still running (10) or never started (5, all of task 9)
  when I stopped the last container to stay under the $60 ceiling (COSTS.md). By task, finished/successes:
  task 0 4/5, 1 4/4, 2 5/5, 3 0/1, 4 4/5, 5 2/5, 6 3/5, 7 4/4, 8 0/1, 9 none. The 35 finished episodes are not a
  random sample: unfinished ones are the long ones, and long PEAR episodes are almost all failures (every failure
  used the full 520 steps, 123k generated tokens on average vs 34k for successes). So 74% is an upper-biased
  estimate. Counting all 15 unfinished as failures gives 26/50 = 52%; as successes, 41/50 = 82%.
- **PEAR's 39.0% claim.** The paper's 39.0% "zero-shot on LIBERO-Long" is the stage-1 checkpoint (step 148,
  trained on Goal/Spatial/Object only), which is not released. The released `Qwen3.8-27B-PEAR` continued RL on all
  four LIBERO suites (including these 50 init states of LIBERO-Long) and on RoboDojo, so our run is **not
  zero-shot** and cannot confirm or refute 39.0%. The closest published number is 53.2% on LIBERO-Long for the
  LIBERO-only stage-2 model; our all-50 range (52-82%) contains it, and the 35-episode estimate is above it.
- PEAR per episode: median 20 turns, 857 generated tokens per turn, 969 s wall with up to 16 episodes sharing the
  GPU; no episode ended on a parse error or the 256k-token budget.
- GR00T scored 90% against NVIDIA's 94.35% (189/200) on LIBERO-10; the difference is within our interval. We used
  NVIDIA's own observation and action code, fixed init states and a 520-step budget (NVIDIA: 720), and a Qwen3-VL
  stand-in for the gated backbone config (README).
- **Run-to-run variance (unplanned finding).** Re-running SmolVLA on the same 50 init states (A10, same code and
  seed) gave 20/50, not 26/50: 10 of 50 episodes flipped (8 success to failure, 2 the other way; McNemar p = 0.11).
  G0.5's re-run reproduced all 50 outcomes. Fixed init states do not make a stochastic policy's episodes
  repeatable, so one run of 50 carries about +-6 episodes of noise for SmolVLA.

Figure: `results/figures/success_ci.png`.

## Extensions (in the order run)

| Ext. | Question | n | Verdict on the hypothesis written before the run |
|-|-|-|-|
| 3 | Where do failures stop? | 50 per VLA, 35 PEAR | **Partly confirmed**: 25 of 38 multi-step failures (66%) stopped at step 2; no evidence it is "more so for weaker models", and progress does not reorder the table |
| 1 | Execution horizon | 20 per point, 3 VLAs | **Refuted** (no intermediate peak); **confirmed** that the best horizon differs by model |
| 2 | Language vs scene | 18 per model (PEAR: 11 finished) | **Confirmed** for GR00T (scene default) and PEAR (follows language); SmolVLA mostly does neither |
| 5 | Paired statistics | 50 (VLA pairs), 35 (PEAR pairs) | **Confirmed**: McNemar separates GR00T vs PEAR and SmolVLA vs PEAR, whose intervals overlap; it does not separate G0.5 vs GR00T |
| 6 | Power and cost per success | analytic + measured | **Confirmed**: 388 episodes per model to detect 50->60% (155 paired at 20% discordance) |
| 4 | Camera + lighting perturbation | 10 per VLA, 4 PEAR | **Refuted at this strength** (no model dropped by more than 1 episode); n=10 cannot see drops below ~30 points |

### Extension 3: which step failed, partial credit (from BDDL goal predicates after every step)

Steps are the task's goal predicates that are false at the start (task 8's "stove on" already holds, so it has 2
steps; task 5 has one). For tasks 3 and 9 (object into a container, then close it) only the prefix in instruction
order counts, so closing an empty drawer is not progress. G0.5 and SmolVLA come from this tutorial's state-recording
re-run (G0.5 49/50, identical to the saved run; SmolVLA 20/50).

| Model | Failures | Stopped at step 1 | Stopped at step 2 | Single-step task (5) | Mean partial credit, all / failures |
|-|-|-|-|-|-|
| G0.5 | 1/50 | 1 | 0 | 0 | 0.98 / 0.00 |
| GR00T N1.7 | 5/50 | 2 | 3 | 0 | 0.93 / 0.30 |
| SmolVLA (re-run) | 30/50 | 8 | 18 | 4 | 0.58 / 0.30 |
| PEAR (35 finished) | 9/35 | 2 | 4 | 3 | 0.80 / 0.22 |

- Most multi-step failures complete step 1 and then fail (25 of 38 pooled; SmolVLA 18 of 26). That points at the
  second half of long-horizon tasks, consistent with compounding error, rather than at initial perception.
- "More so for weaker models": SmolVLA 69% vs GR00T 60% at step 2, with GR00T n=5: inconclusive.
- Ranking by partial credit (0.98, 0.93, 0.80, 0.58) gives the same order as success rate: the middle does not
  reorder here.
- GR00T's weak task is 8 (both moka pots, 2/5): in 2 of its 3 failures both pots were on the stove at some point
  but not at the same time. PEAR's weak tasks are 3 (bowl into drawer) and 5 (book into the caddy's back
  compartment).

Figure: `results/figures/failure_steps.png`.

### Extension 1: actions executed per policy call (20-episode subset: tasks 0-9 x init states 0-1)

| Model (chunk) | 1 | 4 | 8 | default | full chunk |
|-|-|-|-|-|-|
| GR00T N1.7 (16) | 19/20 | 19/20 | 17/20 (default) | 17/20 | 18/20 (16) |
| G0.5 (32) | **14/20** | 18/20 | 20/20 | 19/20 (10) | 20/20 (32) |
| SmolVLA (50) | 13/20 | 14/20 | 14/20 | 8/20 (50 = full) | **8/20** |

- G0.5 is worst when it replans every step (14/20 vs 20/20 at 8 and 32; McNemar 0 vs 6, p = 0.03) and flat from 8
  up. SmolVLA is worst executing its whole 50-step chunk (8/20 vs 13-14/20 at 1-8; p = 0.07-0.18, suggestive only).
  GR00T is flat (17-19/20).
- So no model peaks at an intermediate horizon, but the best setting differs by model (G0.5 bad at 1, SmolVLA bad at
  full). A leaderboard that fixes SmolVLA's default of 50 understates it by about 30 points on this subset relative
  to the same model run at 4-8.
- Smoothness (mean |a(t) - a(t-1)| over the arm dimensions) falls as the horizon grows for GR00T and SmolVLA
  (0.069 -> 0.061, 0.074 -> 0.055): re-planning more often makes the commands jumpier, as the hypothesis says. G0.5
  at horizon 1 is the exception (smoothest, 0.042) and also the least successful.
- PEAR has no action chunk (one end-effector target per turn, executed by a closed-loop controller), so the sweep
  does not apply to it.
- Caveats: the default points for G0.5 and SmolVLA come from this tutorial's re-run (same GPU and harness); the
  sweep ran on A10/A10G (and one GR00T shard on an L4).

Figure: `results/figures/horizon_sweep.png`.

### Extension 2: a different valid instruction in the same scene

For each of 9 tasks (task 8 has no distinct alternative), the instruction was replaced by another task in the same
LIBERO scene (tasks 0 and 1 swap with each other; others use LIBERO-90 tasks of that scene, e.g. "put the frying pan
on the stove" in task 2's kitchen). Both goals were checked after every step; the episode ended when either held.
2 init states per task, n = 18 per model.

| Model | Did the instructed task | Did the scene's original task | Neither | Unfinished |
|-|-|-|-|-|
| G0.5 | 6 | 4 | 8 | 0 |
| GR00T N1.7 | 4 | **11** | 3 | 0 |
| SmolVLA | 3 | 2 | 13 | 0 |
| PEAR | **11** | 0 | 0 | 7 (container stopped) |

- GR00T mostly does the scene's trained task whatever it is told (11/18), and follows the new instruction only when
  it is the other trained LIBERO-10 task of the same scene (tasks 0 and 1). SmolVLA mostly does nothing useful.
- PEAR did the instructed task in all 11 episodes it finished and never the default one. On those same 11
  episodes: G0.5 5, GR00T 4, SmolVLA 3 (instructed). It also did the LIBERO-90 tasks it was never trained on (frying
  pan on the stove, wine bottle on the rack, mug in front of the other mug), which none of the VLAs did.
- Hypothesis: confirmed for GR00T and PEAR. "PEAR executes worse" is only partly true: it is below G0.5 and GR00T on
  the main task but above SmolVLA.
- Caveat: PEAR's 7 unfinished episodes (tasks 4, 5, 6, 7) had done neither task when stopped, so PEAR's rate over
  all 18 is between 11/18 and 18/18.

Figure: `results/figures/counterfactual.png`.

### Extension 5: paired McNemar and paired bootstrap

| Pair | n | Successes | Separate intervals overlap? | Discordant (A only / B only) | McNemar p | Bootstrap 95% CI of A-B: episodes / clustered by task |
|-|-|-|-|-|-|-|
| G0.5 vs GR00T | 50 | 49 / 45 | yes | 4 / 0 | 0.125 | +0.02 to +0.16 / 0.00 to +0.20 |
| G0.5 vs SmolVLA | 50 | 49 / 26 | no | 24 / 1 | <0.0001 | +0.30 to +0.60 / +0.28 to +0.62 |
| GR00T vs SmolVLA | 50 | 45 / 26 | no | 23 / 4 | 0.0003 | +0.20 to +0.56 / +0.16 to +0.60 |
| G0.5 vs PEAR | 35 | 35 / 26 | no | 9 / 0 | 0.004 | +0.11 to +0.40 / +0.11 to +0.44 |
| GR00T vs PEAR | 35 | 34 / 26 | **yes** | 8 / 0 | **0.008** | +0.09 to +0.37 / +0.08 to +0.40 |
| SmolVLA vs PEAR | 35 | 16 / 26 | **yes** | 5 / 15 | **0.041** | -0.51 to -0.06 / -0.47 to -0.06 |

- Pairing separates two pairs whose intervals overlap (GR00T vs PEAR, SmolVLA vs PEAR): confirmed.
- G0.5 vs GR00T is the cautionary case: an episode-level bootstrap excludes zero, McNemar (4 vs 0, p = 0.125) and
  the task-clustered bootstrap do not. With so few discordant pairs, resampling episodes as independent is too
  optimistic; the 5 episodes per task are not independent draws.
- The PEAR rows use the 35 episodes PEAR finished, which favour PEAR (see above).

### Extension 6: power and cost per success

Episodes per model for 80% power at alpha 0.05 (two-sided), `python analysis/stats.py p1 p2 [alpha power disc]`:

| Gap | Unpaired | Paired, 10% discordant | 20% | 30% |
|-|-|-|-|-|
| 50% -> 60% | 388 | 77 | 155 | 234 |
| 70% -> 80% | 294 | | | |
| 90% -> 100% | 74 | | | |

Observed discordance between our model pairs ranged from 8% (G0.5 vs GR00T) to 57% (SmolVLA vs PEAR). Detecting a
10-point gain near 50% takes several hundred episodes unpaired, as expected (confirmed), and pairing on fixed init
states cuts that two- to five-fold when the models mostly agree.

| Model | GPU | GPU-seconds for the run | $ for the run (GPU only) | Successes | $ per success |
|-|-|-|-|-|-|
| GR00T N1.7 | A10 | 570 | 0.17 | 45 | **0.004** |
| G0.5 | L4 | 1,551 | 0.34 | 49 | 0.007 |
| SmolVLA | L4 | 981 | 0.22 | 26 | 0.008 |
| PEAR | B200 | 15,807 | 27.44 | 26 (of 35 finished) | **1.06** |

PEAR's figure is the billed time of every container that ran its main episodes, including work lost when two were
stopped; with ideal batching it would be roughly half. Either way a PEAR success costs two orders of magnitude more
than a VLA success, before any real-robot latency concern (a turn takes 25 s at the median).

### Extension 4: camera moved 3 cm sideways and 2 cm up, lights dimmed and warmed (init state 0 of each task)

| Model | Clean | Perturbed | McNemar |
|-|-|-|-|
| G0.5 | 10/10 | 9/10 | p = 1.0 |
| GR00T N1.7 | 9/10 | 9/10 | p = 1.0 |
| SmolVLA | 3/10 | 6/10 | p = 0.45 |
| PEAR | 3/4 | 4/4 | n = 4 finished; 6 stopped unfinished |

- No model lost more than one episode, so "every model drops" is refuted at this strength. The perturbation was
  mild compared with LIBERO-Plus's viewpoint shifts, and n = 10 cannot detect drops smaller than about 30 points.
  SmolVLA's "gain" is within its run-to-run noise (see the main section). Clean references for G0.5 and SmolVLA
  are from the same-GPU re-run.

## Caveats

- PEAR: 35/50 main episodes, 11/18 counterfactual and 4/10 perturbed episodes finished; the unfinished ones are
  listed in `results/raw/pear/*stopped*.meta.json` and `resume.meta.json`. Its episodes ran 5-16 at a time on one
  GPU, so turn latency includes batching. Its simulator frames were rendered with the NVIDIA EGL driver in the pilot
  and with OSMesa afterwards (to free GPU memory); both are MuJoCo's own renderer.
- GPU types differ between runs (baseline GCP L4; this tutorial mostly Modal A10). Latency is compared only within
  one GPU type.
- GR00T's backbone config and processor come from Qwen3-VL-2B-Instruct instead of the gated Cosmos-Reason2-2B.
- The counterfactual set and the perturbation are single, hand-picked conditions; the horizon sweep uses 20 episodes
  per point. All extension results are smoke-test sized and suggestive, except where a p-value is given.
