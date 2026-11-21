# Costs: tutorial 3 (ceiling $40)

## Prices (modal.com/pricing, checked 2026-11-20)

| Resource | $/s | $/h |
|-|-|-|
| NVIDIA L4 | 0.000222 | 0.80 |
| NVIDIA A10 (A10G) | 0.000306 | 1.10 |
| NVIDIA T4 | 0.000164 | 0.59 |
| CPU, per physical core (2 vCPU) | 0.0000131 | 0.047 |
| Memory, per GiB | 0.00000222 | 0.008 |

Every job requests 1 GPU, 4 physical cores and 16 GiB. The container rate is therefore:
- **L4:** $1.116/h ($0.000310/s).
- **A10:** $1.418/h ($0.000394/s).

The per-job costs below are measured container wall time multiplied by that rate. The authoritative total is Modal's own per-app billing (`modal billing report`), which also counts image builds, container start-up and lost attempts.

## Pilot and projection

The pilot (`pilot`, one container, L4) ran 300 training steps of each of the 4 method × task cells, then short evals. `pilot2`-`pilot5` were short fix-and-smoke-test containers on the A10G:
- `pilot2`, `pilot3`: evals failed on lerobot 0.6.1 env bugs, which were fixed in `t3_envfix.py`.
- `pilot4`: eval timing.
- `pilot5`: flow and MSE heads, the DDIM/trailing overrides and ACT ensembling. It was preempted, then stopped once the parts that mattered had been checked.

| Cell | s/step, L4 | s/step, A10G |
|-|-|-|
| ACT, ALOHA, batch 8 | 0.112 | 0.088 (measured in the main run) |
| Diffusion, ALOHA, batch 8 | 0.174 | 0.107 |
| Diffusion, PushT, batch 64 | 0.181 | 0.114 |
| ACT, PushT, batch 64 | 0.063 | 0.044 (measured in the main run) |

- **The A10G is cheaper per step than the L4.** For diffusion on PushT it costs 0.114 s × $0.000394 = $4.5e-5 per step, against $5.6e-5 on the L4. Jobs therefore requested `["A10", "L4"]`, A10 first, and every main job got an A10/A10G.
- **bf16 autocast (`--policy.use_amp=true`) gave no speed-up** on diffusion/PushT on the A10G: 0.115 against 0.114 s/step in fp32. All runs use fp32.

Projection at 40k steps for every cell (A10G), made before scaling:

| Block | Jobs | Projected $ |
|-|-|-|
| PushT diffusion, flow (3 seeds each), MSE (1 seed) | 7 | 12.6 |
| PushT ACT, 3 seeds | 3 | 2.0 |
| ALOHA ACT, chunk 100, 3 seeds | 3 | 3.5 |
| ALOHA ACT, chunk 1/10/50, 1 seed each | 3 | 3.5 |
| ALOHA diffusion, 3 seeds | 3 | 5.1 |
| All evals (main + extensions) | - | 3.3 |
| Probe, start-up overhead | - | 0.8 |
| **Projected total** | | **≈ 31** (pilots ≈ 1 more) |

That was under the $40 ceiling with about 20% headroom. There was not room for 3 seeds of the chunk-length ablation, which would add about $7, so it used 1 seed.

## Per-run log (all GPU jobs)

| Run | GPU | Train s | Eval s | Job s (container) | Est. $ |
|-|-|-|-|-|-|
| act-aloha-c1-s1 | A10G | 3494 | 182 | 3681 | 1.45 |
| act-aloha-c10-s1 | A10G | 3543 | 331 | 3881 | 1.53 |
| act-aloha-c50-s1 | A10G | 3452 | 227 | 3696 | 1.46 |
| act-aloha-s1 | A10G | 3637 | 230 | 3891 | 1.53 |
| act-aloha-s2 | A10G | 3905 | 318 | 4243 | 1.67 |
| act-aloha-s3 | A10G | 3593 | 229 | 3846 | 1.52 |
| act-pusht-s1 | A10G | 1918 | 82 | 2015 | 0.79 |
| act-pusht-s2 | A10G | 1928 | 82 | 2026 | 0.80 |
| act-pusht-s3 | A10G | 2279 | 110 | 2403 | 0.95 |
| diffusion-aloha-s1 | A10G | 4486 | 575 | 5102 | 2.01 |
| diffusion-aloha-s2 | A10G | 4466 | 721 | 5233 | 2.06 |
| diffusion-aloha-s3 | A10G | 4553 | 769 | 5335 | 2.10 |
| diffusion-pusht-s1 | A10G | 4849 | 784 | 5652 | 2.23 |
| diffusion-pusht-s2 | A10G | 4851 | 803 | 5670 | 2.23 |
| diffusion-pusht-s3 | A10G | 4838 | 794 | 5646 | 2.22 |
| flow-pusht-s1 | A10G | 4865 | 391 | 5288 | 2.08 |
| flow-pusht-s2 | A10G | 4837 | 503 | 5368 | 2.11 |
| flow-pusht-s3 | A10G | 4823 | 499 | 5341 | 2.10 |
| mse-pusht-s1 | A10G | 4829 | 93 | 4954 | 1.95 |
| pilot | L4 | 386 | 98 | 560 | 0.17 |
| pilot2 | A10G | 190 | 101 | 358 | 0.14 |
| pilot3 | A10G | 170 | 204 | 423 | 0.17 |
| pilot4 | A10G | 0 | 126 | 134 | 0.05 |
| probe-pusht | A10G | 0 | 0 | 97 | 0.04 |
| **Sum of job estimates** | | | | | **33.37** |

Diffusion eval time includes the extension 3 re-evaluations (4 extra passes per checkpoint). ACT/ALOHA eval time includes the ensembling pass.

Costs not in the table:
- the first `pilot4` attempt, overwritten by its re-run (about $0.08);
- the first `pilot5` attempt, preempted after about 15 min and then stopped (about $0.35);
- `smoke` and `prefetch` (under $0.05);
- image builds.

No main job was preempted: every one finished on its first attempt (`attempts/<run>.json` in the volume).

## Total

- **Modal per-app billing** (`modal billing report --start 2026-11-20 -r h --tz local`, app `t3-tut03-actdiff`): **$33.84 for the hourly intervals through 18:00-19:00 PST**. The last jobs (flow-s3, mse, probe) ended around 19:10, and that interval was still incomplete when this was written.
- **Job-level estimate:** $33.37, plus about $0.5 of untracked attempts, about **$33.9**. Expect the final bill around $34-35.
- **Ceiling $40: not exceeded.** About $5-6 was left unspent.
- **Cost per block:**
  - PushT: ACT $2.54, diffusion $6.68, flow $6.30, MSE $1.95.
  - ALOHA: ACT $4.72, diffusion $6.17, chunk ablation $4.44.
  - Pilots: about $1.0. Probe: $0.04.
- **Storage:** the `t3-tut03-ckpts` volume keeps the checkpoints (every 10k steps, all runs) and eval videos. It costs $0.09/GB-month, which is cents for the rest of November.

## Queueing and the workspace GPU limit

- The Modal workspace allows 10 GPUs at once, and tutorial 1 shares it. Tutorial 3 jobs queued several times because of this limit.
  - One pilot (`pilot2`, first attempt) sat with no container for more than 10 minutes, and was stopped and relaunched.
  - The main launch (19 jobs, one `.map`) started with only 4 containers, then grew to 8.
- At 8 GPUs, tutorial 3 blocked tutorial 1's PEAR run. On the coordinator's instruction (2026-11-20), the running app's `job` function was capped at **6 containers** with `FunctionUpdateSchedulingParams(max_containers=6)`. The 8 tasks already running were not stopped. `modal_app.py` now sets `max_containers=6` for later launches.
- Caveat: Modal does not shrink containers that are already up. A container that finished one input picked the next queued one, so for about an hour the app still held 8 GPUs. Cancelling the map would also have killed the running inputs. I told the coordinator, who chose to let the map drain rather than stop the app. The count fell to 6 as containers scaled down, and later waves (the probe) were capped at 6.
- Queueing costs nothing, but it stretched wall-clock time. The 19 jobs ran from 15:05 to 19:10 PST.
- A10/L4 containers can be preempted. Modal restarts the same input, and `_job` then resumes training from the last 10k-step checkpoint (`--resume=true`) and skips evals that already finished. Every container attempt is logged in the volume under `attempts/<run>.json`. Only `pilot5` was preempted.
