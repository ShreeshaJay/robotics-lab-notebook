# 05 · G0.5 VLA in LeRobot: subtasks and boxes before acting

**Date:** 2026-11-07 · **Track:** Robot learning · **Status:** planned, not yet run · **Spend so far:** $0 (research and scripts on a laptop CPU)

## Goal

Two things on one 24 GB NVIDIA L4:

1. Run Galaxea's G0.5 vision-language-action model in simulation through LeRobot. One rollout in LIBERO,
   video saved, with no training on my side.
2. Try LeRobot's auto-annotation pipeline (`lerobot-annotate`), which writes timestamped subtasks and
   bounding boxes into a LeRobot dataset. That is the data G0.5's "System 2" reasoning is trained on. Run it on
   2 to 3 episodes of a small public dataset.

Sources: [LeRobot G0.5 docs](https://huggingface.co/docs/lerobot/main/en/g05) ·
[G0.5 blog](https://opengalaxea.github.io/G05/) (renders client-side; I could not read its body, so no numbers here come from it) ·
[annotation pipeline docs](https://huggingface.co/docs/lerobot/main/en/annotation_pipeline) ·
[walkthrough Space "Prompting Robots with LeRobot"](https://huggingface.co/spaces/lerobot/language-conditioned-policies)
(I read its chapters `06-annotation-pipeline.mdx` and `g05-case-study.mdx` from the Space repo) ·
LeRobot source at `v0.7.0` / main `d490e472`.

## What G0.5 is

- A **Qwen3.5-2B-based VLA**. One inference call emits an optional chain of thought (current subtask,
  optionally boxes) and then an action chunk conditioned on it. LeRobot calls these System 2 and System 1, but
  they are one model.
- On top of the Qwen3.5 backbone: an action expert, a flow-matching head and an ActionCodec (discrete action
  tokens). It is natively implemented in LeRobot (`policy.type=g05`) since **lerobot 0.7.0** (PyPI, 2026-11-05).

### Checkpoints

| HF repo (converted for LeRobot) | Embodiment / contract | System 2 (subtask + boxes)? | Params | Download |
|-|-|-|-|-|
| `lerobot/g05_base` | Galaxea R1 Lite/Pro, 27-D whole body, 3 cams, 6-step history | yes | 2.86 B | 11.6 GB (FP32) |
| `lerobot/g05_libero` | LIBERO, 20-D grouped (EEF 6 + gripper), 2 cams | **no** (`predict_cot=false`) | 2.86 B | 11.6 GB |
| `lerobot/g05_robotwin20` | RoboTwin 2.0, two arms, 3 cams | **no** | 2.86 B | 11.6 GB |
| `lerobot/g05_so101` | SO-100/SO-101 arm, 1 to 3 cams | yes | 2.86 B | 11.6 GB |

The original release is [`OpenGalaxea/G05`](https://huggingface.co/OpenGalaxea/G05) (57.7 GB, Hydra
checkpoints, also a `g05-droid` variant). That repo is **gated** (`gated: auto`: you sign in and accept the
conditions). I did not open it. The `lerobot/g05_*` copies are **not gated**, but they carry the same license.

### Memory (estimates; nothing measured yet)

- **Inference:** about 6 GB of BF16 weights (`policy.dtype=bfloat16` by default) plus activations. That **fits an L4
  easily.** The docs strongly recommend `flash-linear-attention` (pure Triton). Without it, the Gated DeltaNet
  layers fall back to a slow PyTorch path.
- **Full fine-tune:** about 2.86 B × 16 bytes (FP32 weights + grads + AdamW states) ≈ 46 GB before activations,
  so **it does not fit an L4**. It needs an 80 GB A100/H100. The walkthrough's fine-tune used **LoRA, 10k steps,
  batch 32, on one A100**. LeRobot's PEFT path (`--policy.use_peft` / `--peft.*`) might fit an L4 at small batch
  sizes. That is unverified.

### Simulators

- **LIBERO:** runs in-process through `lerobot[libero]` (`hf-libero`, MuJoCo, Linux only, `MUJOCO_GL=egl`
  headless). There is a documented `lerobot-eval` command for `g05_libero`. **This is the one I use.**
- **RoboTwin 2.0:** supported (`--env.type=robotwin`, 50 dual-arm tasks, `g05_robotwin20`), but it needs SAPIEN
  with Vulkan, plus CuRobo and pytorch3d compiled against torch 2.7.1. It runs in its own environment behind
  `lerobot-env-server --docker=true`, and the first Docker build takes about 20 min. That is out of scope for a smoke test.
- **Caveat:** the LIBERO and RoboTwin checkpoints have **no System 2**, so the sim rollout shows actions only. The
  subtask and box outputs exist only on `g05_base` and `g05_so101`, which target real robots.
  "Zero-shot" here means no fine-tuning by me. `g05_libero` was itself trained on LIBERO.

### License and access

- **G0.5 Community License (non-commercial + limited patent license)**, from Galaxea (Xinghaitu (Beijing) AI
  Technology Co.), under Hong Kong law. It covers the weights **and** LeRobot's `src/lerobot/policies/g05` code.
  That code is excluded from LeRobot's Apache-2.0 and ships inside the `lerobot` wheel from 0.7.0 on.
- **Permitted:** academic research, personal use, education, and evaluation that is not commercial. A for-profit
  "Internal Evaluation PoC" is allowed only on "Authorized Hardware using an Authorized Image". **The license text
  does not define those terms.**
- **Prohibited:** any "Commercial Use", defined very broadly. It includes any product or service "derived from the
  Materials (**including Output**)", hosted services, and training or improving any model for a commercial purpose.
  So outputs such as predicted subtasks, boxes or actions **cannot** go into a dataset or product sold to others.
- **Acceptance by download:** the first paragraph says that downloading or using the materials counts as acceptance.
  Both scripts therefore refuse to install or download anything G0.5-related until a human sets `G05_LICENSE_ACK`.
- **Not covered:** the annotation pipeline, `lerobot-annotate`, is ordinary LeRobot Apache-2.0 code and uses the
  Apache-2.0 Qwen models. Its outputs carry the source dataset's license.

## The auto-annotation walkthrough (exact steps, from the Space)

1. **Start from a LeRobotDataset v3.0 with only a task string per episode.** Their example is
   `maximellerbach/omx_multicubes`: 171 episodes of an OMX arm placing cubes, annotated copy published as
   `maximellerbach/omx_multicubes_annotated`, Apache-2.0.
2. **Run `lerobot-annotate`.** The documented path runs on **Hugging Face Jobs** (paid HF compute, needs `hf auth login`):
   `HF_TOKEN=hf_... uv run python examples/annotations/run_hf_job.py`, or `lerobot-annotate ... --job.target=h200`.
   It starts from the `vllm/vllm-openai` image, boots `vllm serve Qwen/Qwen3.6-27B`, and drives it over the
   OpenAI-compatible API. Note: the linked `examples/annotations/run_hf_job.py` is **not in LeRobot main or
   v0.7.0**. The `--job.target` flag is the working route. The case study instead ran **locally with smaller
   models: about 1 h for 171 episodes on an RTX 3080 16 GB laptop.**
   - **`plan` module (subtasks):** frames sampled at 2 FPS are packed into **timestamped contact sheets**
     (JPEG grids with the time burned in, 60 frames per prompt, auto-windowed). It runs in two passes:
     1. **Describe:** the VLM narrates only what it sees.
     2. **Segment:** atomic subtasks, using a causal event-boundary rule (grasp, release, new location, lid
        change), stitched into a gap-free cover so each frame has exactly one subtask. Optional seeded relabel.
     - It also emits `plan`, `memory` and `task_aug` rows. All of these go to `language_persistent`.
   - **`interjections` module:** corrections and preferences plus `say` tool-call speech atoms, into `language_events`.
   - **`vqa` module (bounding boxes):** grounded Q/A at 1 Hz per camera. Boxes `[x_min, y_min, x_max, y_max]` are
     stored as [0, 1] fractions (the VLM answers on a 0 to 1000 grid). Also keypoints, counts, attributes and spatial
     relations. Output goes to `language_events`.
   - A **validator** checks timestamps, pairing and JSON shapes. A **writer** rewrites `data/chunk-*/file-*.parquet` in
     place and adds the `say` tool to `meta/info.json`. `--push_to_hub=true --new_repo_id=...` uploads the result.
3. **Inspect** in the LeRobot Dataset Visualizer Space.
4. **Fine-tune G0.5** on the annotated data:
   - Command: `lerobot-train --policy.path=lerobot/g05_base --policy.embodiment=<robot> --policy.use_language_recipe=true`.
   - The boxes come from `vqa` answers on `--policy.cot_bbox_camera`.
   - The recipe mixes 4/6 chain of thought, 1/6 subtask-as-task, and 1/6 plain.
   - The case study used LoRA, 10k steps, batch 32, one A100 (about 2 epochs).
5. **Roll out** with `lerobot-rollout --interactive=true`:
   - System 2 with `runtime_cot_fields=["bbox","subtask"]`.
   - `/subtask` and `/autosteer` steer it.
   - Result: subtasks and box labels copied the annotation wording. Success rate did not clearly change.
     The demonstration-order bias won over `/subtask` instructions.

**Which models the annotation calls:**

- One **open-weight VLM, self-hosted with vLLM**. The default is `Qwen/Qwen3.6-27B` (Apache-2.0, about 27.8 B params,
  BF16 about 56 GB). The config's only backend is `openai`, meaning an OpenAI-compatible client with
  `api_key="EMPTY"` against `http://localhost:8000/v1`.
- **No paid model API is required.** You *could* point `--vlm.api_base` at a paid OpenAI-compatible endpoint,
  but nothing defaults to one.
- **Paid parts that are optional:**
  - HF Jobs compute. Quoted: about $1.50 of H200 time per 100 single-camera 30 s episodes.
  - The Space's interactive *playground*, which can use OpenAI or Anthropic as an external planner. That is a
    separate feature from annotation.
- **Quality, per their WGO-Bench run** (Qwen3.6-27B, 100 episodes, 743 human segments):
  - Segment F1 0.197, matched-label accuracy 0.61, end-to-end semantic F1 0.121.
  - That is about 1.8x the open baseline.
  - Their own words: useful "for bootstrapping annotations at scale, not a replacement for human-quality labels".
  - WGO-Bench itself is CC BY-NC-SA 4.0.

## Plan: one g2-standard-8 (1x L4 24 GB) in us-central1

| Item | Choice |
|-|-|
| Machine | `g2-standard-8`: 8 vCPU, 32 GB RAM, 1x NVIDIA L4 24 GB (bundled) |
| Image | Deep Learning VM family `common-cu129-ubuntu-2404-nvidia-580`, project `deeplearning-platform-release` (CUDA 12.9, driver 580, Python 3.12). Ref: <https://docs.cloud.google.com/deep-learning-vm/docs/images> |
| Disk | **200 GB pd-balanced**. About 12 GB G0.5, about 10 GB Qwen3.5-4B, LIBERO assets, the dataset, and about 25 GB of venvs (torch twice, vLLM) |

Smoke test:

- **Rollout:**
  - Policy and task: `lerobot/g05_libero` on `libero_10`, task 0, 1 episode, seed 0, 512x512 cameras, relative control.
  - Command: exactly the documented `lerobot-eval` command, cut down.
  - Output: `lerobot-eval` renders the episode MP4 to `results/rollout/videos/`.
- **Annotate:**
  - Dataset: `lerobot/svla_so101_pickplace` (Apache-2.0, v3.0, 50 episodes of about 8 s, 30 fps, 86 MB), episodes 0 to 2,
    camera `observation.images.up`.
  - Modules: `plan` and `vqa` on, `interjections` off. No Hub push. It runs on a local copy, and a pristine copy is kept.
  - **VLM: `Qwen/Qwen3.5-4B`** (Apache-2.0, 9.3 GB), served by `vllm 0.33.0` from its own venv.
  - Why not the default: `Qwen3.6-27B` is about 56 GB in BF16, the FP8 build about 28 GB, and even
    `Qwen3.5-27B-GPTQ-Int4` is 30 GB on disk. None fits 24 GB. Expect lower quality than the 27B numbers above.

```bash
export G05_LICENSE_ACK=reviewed-g05-community-license   # only after a human has read and accepted the license
bash setup.sh         # OS pkgs, lerobot[g05,libero,annotations]==0.7.0 + flash-linear-attention, vLLM venv, ~/.libero/config.yaml
python3 run.py fetch      # pinned: g05_libero, LIBERO assets, svla_so101_pickplace
python3 run.py rollout    # one LIBERO episode, MP4 saved
python3 run.py annotate   # subtasks + boxes on episodes 0-2
python3 run.py summary    # -> $WORK/results/summary.json (copy back here; no MP4s in git)
# delete the VM and the disk afterwards
```

Pinned versions:

- Packages: lerobot 0.7.0 (transformers 5.22.x via the extra), torch 2.11.0/cu128, flash-linear-attention 0.6.0, vLLM 0.33.0,
  and robosuite 1.4.0, bddl 1.0.1, mujoco 3.3.1 (the same pins as lerobot's `docker/Dockerfile.sim`).
- Hub revisions: `g05_libero@9b7ee7d7`, `libero-assets@0b3ea86b`, `svla_so101_pickplace@f641879e`, `Qwen3.5-4B@851bf6e8`.

What I expect to learn:

1. Whether `g05_libero` completes the first `libero_10` task, and the time per policy call on an L4.
2. Peak GPU memory for inference (`run.py` samples `nvidia-smi`).
3. Whether a 4B VLM produces sane subtask boundaries and boxes on 8 s pick-and-place clips.
4. How many VLM calls and minutes the annotation takes per episode.

## Cost estimate (not yet spent)

Prices: us-central1 from gcloud-compute.com (built from Google SKU data, updated 2026-11-01). `g2-standard-8`
costs $0.8536/h on-demand or $0.5108/h spot. pd-balanced is about $0.12/GB-month at the regional-average list price.
Check the live [GCP pricing](https://cloud.google.com/compute/vm-instance-pricing) before running.

| Phase | L4 hours (est.) |
|-|-|
| setup.sh (torch, lerobot, vLLM wheels) | 0.4 |
| fetch (about 14 GB) | 0.1 |
| rollout (load, 1 episode of up to 520 steps at 20 Hz, render) | 0.25 |
| annotate (vLLM start + 9 GB download + 3 short episodes) | 0.3 |
| debugging buffer | 0.7 |
| **Total** | **about 1.75 (cap 2.5)** |

- **On-demand:** 1.75 h × $0.8536 ≈ **$1.5**. The disk adds about $0.10. **About $1.6 total, $2.5 worst case.** Spot is about $1.0.
- Ingress is free. Not included: any G0.5 fine-tune, which needs an 80 GB GPU for a full fine-tune; `a2-ultragpu-1g` costs
  about $5.07/h on-demand.

## Status

Planned, not yet run. Before it can run:

- GPU quota for `NVIDIA_L4_GPUS` in us-central1 must be granted, and the project's budget alerts and caps must be in place.
- A human must decide on the G0.5 Community License. Its download-equals-acceptance clause applies to both
  `pip install lerobot` and the checkpoints, and the for-profit evaluation lane depends on undefined
  "Authorized Hardware". Neither script proceeds without `G05_LICENSE_ACK`.
- To inspect the code, I shallow-cloned LeRobot at main `d490e472`. That clone includes `src/lerobot/policies/g05`.
  I also fetched only the `LICENSE` and `NOTICE` text files from `lerobot/g05_libero`, not the weights.
