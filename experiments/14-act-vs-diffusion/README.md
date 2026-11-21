# Tutorial 3: ACT vs Diffusion Policy from scratch

Date: 2026-11-20. Compute: Modal, one NVIDIA L4 per job. Spend: see [COSTS.md](COSTS.md). Results: [RESULTS.md](RESULTS.md).

## What was run

ACT and Diffusion Policy were trained from scratch with LeRobot on two simulated tasks, with 3 training seeds per method per task:

- PushT: `lerobot/pusht`, 206 episodes, 10 Hz.
- ALOHA bimanual insertion: `lerobot/aloha_sim_insertion_scripted`, 50 scripted episodes, 50 Hz.

Every checkpoint was evaluated in simulation with the same evaluation seeds (`lerobot-eval --seed=1000`). PushT used episodes seeded 1000-1099 (100 per checkpoint) and ALOHA used 1000-1049 (50 per checkpoint). So every compared model saw the same initial states. Success rates come with Wilson 95% intervals.

The extensions from the tutorial plan, in the order they were prioritised:

| Ext | What | Runs |
|-|-|-|
| 2 | ACT chunk length 1/10/50/100, temporal ensembling on/off, ALOHA insertion | 3 extra ACT trainings (1 seed each for k = 1, 10, 50); k = 100 reuses the 3 main ACT runs |
| 5 | Seed spread vs method gap | main runs only |
| 3 | Diffusion success vs denoising steps (100/10/1), DDPM and DDIM samplers | inference only, on the 6 main diffusion checkpoints |
| 4 | PushT multimodality probe, with an MSE behaviour-cloning baseline | 1 MSE training + 1 probe job |
| 1 | Flow-matching action head at equal compute | 3 trainings on PushT (same network, steps and batch as diffusion) |

## Step counts (reduced from the reference recipes)

| Cell | Steps | Batch | Reference recipe |
|-|-|-|-|
| PushT: ACT, diffusion, flow, MSE BC | 40,000 | 64 | 200k steps (lerobot/diffusion_pusht) |
| ALOHA insertion: ACT (all chunk lengths), diffusion | 40,000 | 8 | 100k steps (lerobot/act_aloha_sim_insertion_human) |

Checkpoints were saved every 10k steps so that a preempted run could resume. Only the final (40k) checkpoint was evaluated. All weights stay on the Modal Volume `t3-tut03-ckpts` under `runs/<run>/`; none are in this folder.

## Policy settings

- **Diffusion, PushT:** the published `lerobot/diffusion_pusht` config. That is 2 observation steps, horizon 16, 8 executed actions, 84x84 random crop, ResNet-18 from scratch with GroupNorm, a U-Net with dims (512, 1024, 2048), DDPM with 100 steps and epsilon prediction, lr 1e-4 cosine.
- **ACT, ALOHA:** the published `lerobot/act_aloha_sim_insertion_human` config (LeRobot defaults). That is chunk 100 with all 100 executed, ImageNet ResNet-18, a CVAE with kl_weight 10, lr 1e-5.
- **ACT, PushT** (there is no published LeRobot recipe): ACT defaults, with chunk 20 and 10 executed actions. ACT's ALOHA chunk is 2 s at 50 Hz, and 20 steps is 2 s at PushT's 10 Hz. Re-planning halfway mirrors diffusion's 8 of 16.
- **Diffusion, ALOHA** (no published LeRobot recipe): lerobot 0.6.1 defaults, which are horizon 64 and 32 executed actions. On top of that: 2 observation steps, images resized to 240x320 with a 0.95 random crop, ImageNet ResNet-18, and `drop_n_last_frames=31`.
- **Flow matching** and **MSE BC** use the PushT diffusion network unchanged, with a swapped objective and sampler. See `t3_heads.py`.

## Versions

- lerobot **0.6.1** from PyPI (git tag v0.6.1, commit `7e241bd6`). The setup notes recommended 0.7.0, but this run deliberately did not use it. lerobot 0.7.0 ships the G0.5 policy code under the separate G0.5 Community License, which says that downloading counts as acceptance. Accepting terms is outside what this run may do, and 0.6.1 contains no `policies/g05`. So 0.6.1 is the last release that is Apache-2.0 throughout.
- torch 2.11.0+cu128, torchvision 0.26.0+cu128, diffusers 0.39.0, gymnasium 1.4.0, gym-aloha 0.1.5, gym-pusht 0.1.8, mujoco 3.8.1, dm_control 1.0.41, torchcodec 0.11.1. The full freeze is in `results/meta/freeze.txt`.
- Container image: Debian slim with Python 3.12. Apt packages: ffmpeg, libgl1, libegl1, libopengl0, libgles2, libglvnd0, libosmesa6. Rendering uses `MUJOCO_GL=egl`.
- Modal client 1.6.4.

## Licenses

| Item | License |
|-|-|
| lerobot 0.6.1 | Apache-2.0 |
| gym-aloha 0.1.5, gym-pusht 0.1.8 | Apache-2.0 |
| `lerobot/pusht` @ `7628202a` | MIT (dataset card) |
| `lerobot/aloha_sim_insertion_scripted` @ `8ab66091` | MIT (dataset card) |
| torchvision ResNet-18 ImageNet weights (ACT; diffusion on ALOHA) | BSD-3-Clause (torchvision) |
| Code in this folder | Apache-2.0 (same as LeRobot) |

No gated models and no weights from the Hub were used. Every policy was trained from scratch.

## Files

- `modal_app.py`: the Modal app `t3-tut03-actdiff`. It defines the image, the generic `job` (train, then any number of `lerobot-eval` passes) and the `probe` function. Checkpoints stay on the Modal Volume `t3-tut03-ckpts`.
- `runs.py`: every run specification (flags, steps, seeds, eval settings).
- `t3_launch.py`: the in-container entry point. It applies the head patch, then calls the stock `lerobot-train` / `lerobot-eval` `main()`.
- `t3_heads.py`: the flow-matching and MSE heads for LeRobot's diffusion network.
- `t3_probe.py`: the multimodality probe (extension 4).
- `t3_envfix.py`: two evaluation workarounds for lerobot 0.6.1 with gymnasium 1.4, preloaded into the AsyncVectorEnv forkserver (see RESULTS.md caveats).
- `t3_stats.py`, `summarize.py`: the Wilson intervals, summary JSON and figures.
- `results/raw/*.json`: the raw record of every job: commands, training log rows, per-episode eval results and wall time.
- `results/per_run/*.json`: one compact JSON per run. `results/summary.json` holds the aggregates.
- `results/figures/*.png`, `results/videos/*.mp4`.

## Reproduce

```bash
# Modal client 1.6.4, signed in
modal run modal_app.py::prefetch               # pinned datasets -> volume, pip freeze
modal run modal_app.py::main --phase pilot     # throughput pilot (pilot2-5: fixes and smoke tests)
modal run --detach modal_app.py::main --phase all   # main + ext2 + ext1 (19 jobs, max 6 GPUs at once)
modal run modal_app.py::main --phase probe
python summarize.py
```

With `--detach`, results also land in the volume under `results/`. Fetch them with `modal volume get t3-tut03-ckpts results/<name>.json results/raw/`.
