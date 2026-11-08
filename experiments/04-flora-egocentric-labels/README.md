# 04 · Flora: open models for labeling egocentric video

**Date:** 2026-11-07 · **Track:** Data labeling · **Status:** planned, not yet run · **Spend so far:** $0 (research and scripts on a laptop CPU)

## Goal

Learn what NVIDIA's Flora models actually produce on ordinary egocentric household video, and what it
takes to run them on one modest GPU. The smallest meaningful test: on one public 1 to 2 minute
kitchen-cleaning clip, get (1) timestamped action captions and (2) a dense progress/quality reward plus an
episode outcome score, on one 24 GB NVIDIA L4.

## What Flora is

- A three-model suite from NVIDIA GEAR that turns robot and human manipulation video into labels: when
  each action happens, what it is, and how well it went. Announced 2026-10-20; NVIDIA says it is the data
  engine used to train GR00T-2.
- Project page: <https://research.nvidia.com/labs/gear/flora/> · paper: arXiv:2610.31779 ·
  code: <https://github.com/NVlabs/Flora> (the README's clone line says `NVIDIA/Flora`; both resolve to the same
  commit) · weights: <https://huggingface.co/collections/nvidia/flora>
- The repo is inference code only ("sample vanilla inference code"). No training code.

### The three models

| Model (HF repo) | Params | Base | Download | BF16 weights in VRAM | Fits one 24 GB L4? |
|-|-|-|-|-|-|
| Flora-TempSeg / "Flora-segmenter-9B" (`nvidia/Flora-TempSeg`) | 9.0 B (LoRA r=32 on the base) | Qwen3.5-9B (Apache-2.0) | 0.4 GB adapter + 19.3 GB base | about 18 to 19 GB, plus FP32 GDN and LoRA tensors | **Probably, tight** (not verified) |
| Flora-Captioner / "Flora-captioner-27B" (`nvidia/Flora-Captioner`) | 27.4 B (full fine-tune) | Qwen3.5-27B | 109 GB (shards stored in FP32) | about 55 GB | **No.** Only as an unvalidated 4-bit experiment (about 18 to 19 GB) |
| Flora-Rewarder / "Flora-rewarder-27B" (`nvidia/Flora-Rewarder`) | 27.8 B (full fine-tune incl. vision) | Qwen3.5-27B | 55.6 GB (BF16) | about 56 GB plus KV cache | **No.** Only as an unvalidated 4-bit experiment |

- The validated setup is Linux x86_64, Python 3.12 and one **H100 80 GB**. The rubric reward needs **two** 80 GB GPUs by
  default (`--tensor-parallel-size 2`). No minimum VRAM is published.
- The Captioner loader caps each GPU at (VRAM minus 12 GiB) and then refuses to run if any module would land
  off-GPU. On an L4 that cap is about 10 GiB, so the unmodified pipeline stops at the Captioner by design.
- Precision matters to the authors: BF16 weights with FP32 Gated-DeltaNet (`A_log`, `dt_bias`) parameters,
  verified bit-exact at load time. Their `AGENTS.md` says not to change prompts, frame sampling or
  precision, because other values "give worse results without any error". So a 4-bit run is a feasibility
  check, not a quality measurement.
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
  prompt is phrased for robot manipulation. Human egocentric video is in-distribution per the paper, but
  check this.
- **Rubric:** one score per user-defined step (the medoid of 9 samples) plus a thinking trace.

### License and access

- **License:** the code and all three sets of weights are under the **NVIDIA License**, the same text in
  `LICENSE` and in `LICENSE.pdf` in each HF repo. §3.3 says the Work and derivatives "only may be used or
  intended for use non-commercially", defined as **"for research or evaluation purposes only"**. NVIDIA and
  its affiliates are exempt. The model cards add: "not intended or supported for commercial or production use."
- **Outputs:** the license has no separate clause on outputs. But running the model to produce labels is itself
  "use" of the Work, so labeling data for a commercial product, or for a dataset sold to third parties, is
  outside the grant. The Apache-2.0 license of the Qwen base models does not change that. A commercial use
  would need a separate license from NVIDIA. Internal research or evaluation is the permitted lane.
- **Gating:** none. All three repos show `gated: false`, and `HF_TOKEN` is optional. A dead helper in
  `captioner/batch_utils/models.py` still mentions "private weights", but the CLI never calls it.
- **Test clip:** EPIC-KITCHENS is non-commercial (CC BY-NC 4.0 on the project site; data.bris lists a
  "Non-Commercial Government Licence"). That is fine for personal learning notes. No sign-up is needed.

## Test input

- **EPIC-KITCHENS-55 `P08_02`:** 107 s, 1920x1080 at 59.94 fps, head-mounted GoPro in a home kitchen. The
  ground-truth narrations include "clean counter top", "clean sink" and "clean sponge" (from `EPIC_100_train.csv`).
- Source: `https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P08/P08_02.MP4`
  (405,011,907 bytes, direct download). DOI 10.5523/bris.3h91syskeag572hl6tvuovwv4d. I picked it by scanning
  the EPIC-100 video list for 60 to 120 s videos with the highest share of wash/clean/wipe actions.
- The raw file is over Flora's 256 MiB limit, so `run.py` re-encodes it **once**: 720p, H.264 CRF 20, audio
  dropped, every native frame timestamp kept (no trim, no fps change).
- Why not the Flora repo's own examples: they are 4 s GIF excerpts (320x180, 10 fps), too short for a 1 to 2
  minute test.
- Not used: Ego4D and Ego-Exo4D (they need a signed license agreement).

## Plan: one g2-standard-8 (1x L4 24 GB) in us-central1

| Item | Choice |
|-|-|
| Machine | `g2-standard-8`: 8 vCPU, 32 GB RAM, 1x NVIDIA L4 24 GB (the GPU is bundled in the machine type) |
| Image | Deep Learning VM family `common-cu129-ubuntu-2404-nvidia-580`, project `deeplearning-platform-release` (CUDA 12.9, driver 580, Python 3.12). Pin a dated image name for repeatability. Ref: <https://docs.cloud.google.com/deep-learning-vm/docs/images> |
| Disk | **400 GB pd-ssd** boot disk. About 185 GB of weights, about 20 GB of venvs, and the clip. pd-ssd because the 109 GB Captioner shards are read on every video, and pd-balanced throughput at this size makes that slow |
| Zone | any us-central1 zone with L4 capacity (for example `us-central1-a`) |

Steps on the VM:

```bash
# 1. copy setup.sh and run.py to the VM, then
bash setup.sh                                  # OS packages, Flora at pinned commit, two venvs; no weights
# 2. read Flora/LICENSE yourself; the script will not run the models until you set:
export FLORA_LICENSE_ACK=noncommercial-research-eval
# 3. smoke test
python3 run.py fetch                           # public clip -> 720p MP4 within Flora limits
python3 run.py validate                        # CPU only: --validate-only and --dry-run
python3 run.py caption  --mode l4              # TempSeg (as released) + Captioner (4-bit NF4)
python3 run.py reward   --mode l4              # dense + outcome reward (vLLM bitsandbytes 4-bit)
python3 run.py summary  --mode l4              # -> $WORK/results/summary_l4.json
# 4. copy summary_l4.json and the *.meta.json files back here; delete the VM AND the disk
```

- Pinned: Flora `d8b755ec9a990418b1f31c7f48c75341c7229ea6`, torch 2.11.0/cu128, transformers 5.22.1
  (annotation) and 5.20.2 (rewarder), vLLM 0.33.0, bitsandbytes 0.50.2. Model revisions: TempSeg, Captioner and
  the Qwen3.5-9B base come from Flora's `models.json`. Rewarder `b64cee90…` (HF main on 2026-11-07).
- `--mode reference` runs the unmodified upstream commands, meant for an 80 GB GPU.

### What I expect to learn (and the likely failure points)

1. Whether TempSeg at release precision fits the L4. It is about 19 GB of BF16 weights plus FP32 parts, with
   CUDA context and activations on top. This is the first OOM risk.
2. Whether a 4-bit NF4 Captioner gives captions that read like the narrations ("clean counter top"…). Quality
   will be below the paper's numbers by construction.
3. Whether vLLM's in-flight bitsandbytes path supports the Qwen3.5 hybrid (Gated DeltaNet) layers at all. I
   could not verify this offline. If it fails, the honest result is: "Rewarder needs an 80 GB GPU."
4. Wall-clock per stage and peak GPU memory. `run.py` samples `nvidia-smi` and parses Flora's own
   `FLORA_RUNTIME` events.

If 3 fails, the cheapest faithful fallback is one `a2-ultragpu-1g` (A100 80 GB) for about 2 h with
`--mode reference` (see cost).

## Cost estimate (not yet spent)

Prices: us-central1, from gcloud-compute.com, built from Google's SKU data, last updated 2026-11-01:
`g2-standard-8` $0.8536/h on-demand, $0.5108/h spot. `a2-ultragpu-1g` $5.0688/h on-demand, $3.0081/h spot. Disk at
the regional-average list price of $0.20/GB-month for pd-ssd. Check the live
[GCP pricing](https://cloud.google.com/compute/vm-instance-pricing) before running. Spot prices move.

| Phase | L4 hours (est.) |
|-|-|
| setup.sh (wheels: torch, vLLM, transformers) | 0.4 |
| fetch + validate | 0.2 |
| weight downloads, about 185 GB | 0.4 |
| caption (TempSeg + 4-bit Captioner load and run on 107 s) | 1.0 |
| reward (vLLM load/quantize, kernel compile, dense + outcome) | 0.5 |
| debugging buffer | 1.0 |
| **Total** | **about 3.5 (cap 5)** |

- **L4 on-demand:** 3.5 h × $0.8536 ≈ **$3.0**. Add the 400 GB pd-ssd at about $0.11/h for about 5 h ≈ $0.55.
  **About $3.5 total, $5 worst case.** On spot it is about $2.3 (risk: preemption mid-download).
- **Faithful fallback (A100 80 GB, reference mode, about 2 h):** about $10 on-demand or about $6 on spot, plus disk.
- Ingress from Hugging Face and data.bris is free. Delete the disk afterwards. Keeping 400 GB pd-ssd for a month is about $80.

## Status

Planned, not yet run. Before it can run:

- GPU quota for `NVIDIA_L4_GPUS` in us-central1 must be granted, and the project's budget alerts and caps must be in place.
- A human must read and accept the NVIDIA License (non-commercial research and evaluation). `run.py` refuses to
  load weights until `FLORA_LICENSE_ACK` is set. No terms were accepted while preparing this.

## Notes on sources

- The repo includes an `AGENTS.md` written for coding agents. I read it as documentation only. Its rules (run
  `--validate-only` first, ask before downloading weights or re-encoding, never commit videos or weights)
  match this plan. Nothing in it was unusual.
- The project page's comparison table names several proprietary models I can't verify. Those comparisons are
  not used here.
