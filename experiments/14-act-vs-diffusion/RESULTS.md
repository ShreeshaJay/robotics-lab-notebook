# Results: ACT vs Diffusion Policy from scratch (LeRobot 0.6.1)

Every number below is a success rate with a Wilson 95% interval.
- **PushT:** 100 evaluation episodes per checkpoint, seeds 1000-1099. Success means final coverage > 95%.
- **ALOHA insertion:** 50 episodes per checkpoint, seeds 1000-1049.

"3 seeds pooled" means 3 independently trained checkpoints (training seeds 1, 2, 3), each evaluated on the same evaluation seeds. All runs are from scratch and use reduced step counts: **40k steps** for every cell, against the 200k (PushT) and 100k (ALOHA) of the reference recipes. Treat them as reduced-budget results, not converged ones.

## Main comparison

| Task | Method | Steps × batch | Per seed (s1, s2, s3) | Pooled | 95% CI | Mean max reward | Train wall per seed (A10G) |
|-|-|-|-|-|-|-|-|
| PushT | ACT | 40k × 64 | 34, 28, 28 / 100 | **30.0%** (90/300) | 25.1-35.4 | 0.70 | 32-38 min |
| PushT | Diffusion (DDPM, 100 steps) | 40k × 64 | 66, 48, 54 / 100 | **56.0%** (168/300) | 50.3-61.5 | 0.94 | 81 min |
| PushT | Flow matching (ext. 1, 100 Euler steps) | 40k × 64 | 59, 67, 58 / 100 | **61.3%** (184/300) | 55.7-66.7 | 0.95 | 81 min |
| ALOHA insertion | ACT (chunk 100) | 40k × 8 | 38, 40, 30 / 50 | **72.0%** (108/150) | 64.3-78.6 | 3.61 / 4 | 60-65 min |
| ALOHA insertion | Diffusion (DDPM, 100 steps) | 40k × 8 | 19, 16, 19 / 50 | **36.0%** (54/150) | 28.8-43.9 | 2.79 / 4 | 75 min |

How to read the last two columns:
- PushT "mean max reward" is `avg_max_reward`, which is coverage normalized to the 95% success threshold and capped at 1. ALOHA's is the highest stage reached (4 = inserted).
- Training wall time is training only. Evaluation added 1.5-13 min per job (see COSTS.md).

**Headline: the ranking flips between tasks.** Diffusion beats ACT on PushT by 26 points, and ACT beats diffusion on ALOHA insertion by 36 points. Both gaps are far outside the intervals.
- The ALOHA diffusion cell has no published LeRobot recipe. It uses lerobot defaults plus a 240x320 resize (README.md), and batch 8 to match ACT. That cell is the least tuned in the table. The flip should be read as "at equal reduced budget with default-ish settings", not as a property of the methods.

Figure: `results/figures/success_main.png`. Videos (successful episodes): `results/videos/pusht_diffusion_s1_seed1002_success.mp4` and `results/videos/aloha_insertion_act_s2_seed1000_success.mp4`.

## Extensions (in priority order)

### Ext 2: ACT chunk length × temporal ensembling (ALOHA insertion)

Training seed 1 for k = 1, 10 and 50. At k = 100 the table shows seed 1, with the 3-seed pool in brackets.

| Chunk k | Ensembling off (execute k actions) | Ensembling on (query every step, m = 0.01) |
|-|-|-|
| 1 | 0/50 = 0% (0-7) | same as off (k = 1) |
| 10 | 3/50 = 6% (2-16) | 0/50 = 0% (0-7) |
| 50 | 24/50 = 48% (35-61) | 23/50 = 46% (33-60) |
| 100 | 38/50 = 76% (63-86); 3 seeds 108/150 = 72% (64-79) | 37/50 = 74% (60-84); 3 seeds 104/150 = 69% (62-76) |

**Verdict: partly confirmed, partly refuted.**
- Confirmed: chunk length 1 fails completely on insertion (0/50), and success rises steadily with k.
- Refuted: "ensembling matters most at long chunks". Ensembling made no measurable difference at any k, with a slightly negative point estimate everywhere. At k = 100 the change was −2 points for seed 1 and −3 points pooled.

Caveats: 1 training seed for k < 100, one ensembling coefficient, and 50 episodes. Figure: `results/figures/act_chunk_length.png`.

### Ext 5: seed spread vs method gap (main runs)

| Task | Gap (pooled) | Seed range, ACT | Seed range, diffusion | Welch t over seed means (n = 3 v 3) |
|-|-|-|-|-|
| PushT | diffusion − ACT = +26 pts | 6 pts (28-34) | 18 pts (48-66) | 4.6 |
| ALOHA | diffusion − ACT = −36 pts | 20 pts (60-80) | 6 pts (32-38) | −5.6 |

**Verdict: refuted for PushT.** The seed spread (at most 18 points) is smaller than the 26-point gap. The worst diffusion seed (48%) beats the best ACT seed (34%).
- Seed noise is still large: 18 points is about 70% of the PushT gap, and 20 points is about 55% of the ALOHA gap.
- A single-seed comparison could easily have reported anything from a 14-point to a 38-point PushT gap.
- Flow vs diffusion on PushT (+5 points) is inside the seed spread, which is where the hypothesis's warning bites.

### Ext 3: inference budget (denoising / integration steps)

The same checkpoints were re-evaluated with fewer steps. Reduced-step runs use `trailing` timestep spacing, so a 1-step sampler starts from the pure-noise timestep. 3 checkpoints are pooled per cell.

| Sampler | PushT 100 | PushT 10 | PushT 1 | ALOHA 100 | ALOHA 10 | ALOHA 1 |
|-|-|-|-|-|-|-|
| Diffusion, DDPM | 56.0% (50-62) | 48.0% (42-54) | 0/300 (0-1.3) | 36.0% (29-44) | 43.3% (36-51) | 0/100 (0-3.7)\* |
| Diffusion, DDIM (deterministic) | not run | 44.7% (39-50) | 0/300 | not run | 8.0% (5-14) | 0/150 |
| Flow matching, Euler | 61.3% (56-67) | 62.7% (57-68) | 40.0% (35-46) | - | - | - |

\* One of the three ALOHA checkpoints crashed MuJoCo at 1 DDPM step (`PhysicsError: mjWARN_BADQACC`): its 1-step actions were invalid enough to blow up the simulation. The cell pools the other two (0/100).

**Verdict: confirmed.**
- Diffusion collapses to 0% at 1 step on both tasks. Its 1-step coverage on PushT is 0.08.
- 10 DDPM steps cost little: −8 points on PushT and +7 on ALOHA, both inside the intervals. So "sharply below about 10" holds in the sense that 10 is fine and 1 is useless.
- Flow matching degrades gently. 10 Euler steps equal 100 (63% vs 61%), and 1 step still gives 40% with 0.91 coverage.
- Surprise: switching the same diffusion checkpoints to the deterministic DDIM sampler at 10 steps barely hurt on PushT (45%) but collapsed ALOHA to 8%.

Figure: `results/figures/inference_steps.png`. Per-eval wall time per checkpoint on PushT, 100 episodes: 226 s at 100 steps, 145 s at 10, 138 s at 1. Simulation dominates at low step counts.

### Ext 4: PushT multimodality probe (with MSE BC baseline)

- **Method:** from 3 PushT start states, 64 action chunks were drawn per policy for the same observation, keeping the first 8 actions.
- **State selection:** the 3 states were the 40 reset seeds 1000-1039 with the clearest two-cluster split in the diffusion samples (k-means, minority ≥ 20%). Selection is therefore conditioned on diffusion being bimodal there.
- **MSE BC:** the same diffusion network trained with a plain MSE regression objective (`t3_heads.py`). Its success is 37/100 = 37% (28-47), coverage 0.88, 1 seed. It sits between ACT (30%) and diffusion (56%).

| State (seed) | Diffusion | Flow | ACT (z = 0, as deployed) | ACT (z ~ N(0, I)) | MSE BC |
|-|-|-|-|-|-|
| 1023 | 2 modes, 58/42%, spread 44 px | 2 modes, 27/73% | 1 trajectory, 1.2 gaps from either mode | spread 1 px | 1 trajectory, 0.44 gaps from nearest mode |
| 1012 | 2 modes, 50/50%, spread 60 px | 1 mode | 1 trajectory, 0.27 gaps (near one mode) | spread 1 px | 0.39 gaps |
| 1038 | 2 modes, 72/28%, spread 41 px | 1 mode (spread 9 px) | 0.55 gaps | spread 1 px | 0.50 gaps (midpoint) |

"Gaps" means the distance of the predicted chunk to the nearest diffusion-mode centre, in units of the distance between the two centres. 0.5 means exactly between the modes, and diffusion's own samples sit at about 0.3.

**Verdict: partly confirmed.**
- Confirmed: diffusion samples split into two distinct strategies at every probed state. MSE BC outputs one chunk that lies between the two modes (0.39-0.50 gaps).
- Refuted: "ACT picks one mode with its latent variable". ACT's CVAE latent is effectively unused at inference. Sampling z from the prior moves the chunk by about 1 px, against 40-60 px of diffusion spread. So ACT behaves as a deterministic regressor; its L1 loss lets it commit to something other than the mean, but not reliably to a data mode.
- Flow matching was bimodal at only 1 of the 3 states.

Figure: `results/figures/pusht_multimodality.png`. Raw samples: `results/raw/probe-pusht.json`.

### Ext 1: flow-matching head at equal compute

- **Setup:** LeRobot 0.6.1 has no flow-matching variant of Diffusion Policy (its `multi_task_dit` uses a different, CLIP-conditioned DiT network). So `t3_heads.py` swaps only the objective and sampler of the diffusion U-Net:
  - Objective: rectified-flow / conditional flow matching, with a linear path and velocity regression.
  - Sampler: explicit Euler.
- **Equal compute:** same network and parameters, same 40k steps × batch 64, same seeds. Training wall time per seed was 81 min for both.
- **Result:** flow 61.3% (55.7-66.7) vs diffusion 56.0% (50.3-61.5) on PushT, mean coverage 0.95 vs 0.94. At 10 Euler steps flow reached 62.7%.

**Verdict: confirmed.** Flow matching matches diffusion's success (+5 points, inside the intervals and inside the seed spread) with 10x fewer integration steps (10 vs 100). It keeps 40% even at 1 step.

## Caveats

- Reduced budgets: 40k steps for every cell, 20% of the PushT and 40% of the ALOHA reference recipe. Absolute numbers would rise with full training, and the gap between methods might change.
- The ACT/PushT and Diffusion/ALOHA cells use stated adaptations, not published recipes (README.md). ALOHA uses the **scripted** insertion demos (50 episodes), which are easier than the human demos.
- 50 ALOHA episodes per checkpoint give wide per-seed intervals (about ±12 points). The pooled 150 give about ±7.
- Ext 2 used 1 training seed for k = 1/10/50 because the budget did not allow 3. Ext 4 and the MSE baseline used 1 seed.
- PushT and ALOHA success both come from the simulator's own success flag (`is_success` / max reward = 4). No learned judge was used.
- LeRobot 0.6.1 needed two evaluation workarounds, both in `t3_envfix.py` and neither changing scoring:
  - Its `AsyncVectorEnv` uses a `forkserver` that never imports `gym_pusht` / `gym_aloha`.
  - Its `env.call("task_description")` kills gymnasium 1.4 workers on these single-task envs.
- No run was preempted (attempt 1 for all jobs), so no training was resumed.
