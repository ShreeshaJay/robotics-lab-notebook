# 11 · SimFoundry: one video in, a physics-ready OmniGibson scene out

**Track:** Simulation / reconstruction · **Status:** code read and CPU-only tests run ($0, no GPU, no credentials, no API calls). GPU run planned, not run · **Date:** 2026-11-08

## Goal
Find out exactly what NVIDIA's SimFoundry runs, under which licences, with which gated models and paid APIs, and what its output contains. Run whatever works on CPU without credentials. Write a low-risk plan for one GPU run on a public clip, with the acceptance tests from experiment 03.

## What it is
**SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation.** Ranawaka, Wong, Pai, Chu, Dai, Moghani, Yin, Jiang, Durbano, Huynh, Fang, Xu, Zhang, Fei-Fei, Fan, Wen, Mandlekar, Zhu (NVIDIA GEAR, Georgia Tech, Stanford, UT Austin). arXiv [2606.28276](https://arxiv.org/abs/2606.28276) (read v4 HTML). Repo [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry), pinned here at **`9e34ebefcd020583fbb755a8b57268dce78eca26`** (2026-08-26, "Refinement (#7)"). Clone: `C:\Users\shree\af-scout-lab\simfoundry\`.

It takes one short handheld video of a **single flat support surface** (desk, table, counter), segments every object on it, generates a textured mesh per object from one crop, poses each mesh back into the scene, adds colliders, mass and friction, settles the layout in PyBullet and writes an OmniGibson scene. Three sub-pipelines:

| Pipeline | What it does | Released? |
|-|-|-|
| A Reconstruction | Video to OmniGibson scene (stages 1b–14, optional 9 articulation, optional 2c background splat) | Yes |
| B Augmentation | "Digital cousins" (VLM-proposed object, scene and task variants) and task YAMLs | Yes |
| C Application | Smoke test, policy eval, teleop; demo generation and training code "not yet released" | Partly |

**Scope limit that matters most:** the reconstruction is built from **one canonical frame** and **one support plane** (`s3_ground.floor_categories: [desk, table, counter]`). The capture guide says to keep other surfaces out of frame and every object visible in every frame. The paper's limitations section says layouts are limited to tabletop-style scenes; multi-level or non-planar scenes are future work. A whole-room walkthrough is **out of scope**: one run gives one counter or table, not a room.

### Stages (pipeline A), models and licences
| Stage | Purpose | Model / tool | Licence (code / weights) | Gated? |
|-|-|-|-|-|
| 1b | Sample frames from the video (default 15) | ffmpeg, OpenCV | LGPL/GPL build / Apache-2.0 | No |
| 2 | Depth for the video | **Depth Anything 3**, `DA3NESTED-GIANT-LARGE-1.1` (FoundationStereo for stereo input) | Code Apache-2.0; **weights CC BY-NC 4.0** (HF card, checked 2026-11-08). FoundationStereo: NVIDIA, non-commercial | No |
| 3 | Pick the canonical frame, find the support plane | **Gemini 3.1 Pro preview** (frame choice and surface name) + **SAM 3** | Gemini: paid API. SAM 3: SAM License (Meta, source-available) | SAM 3: **yes** |
| 4 | World frame from the canonical camera | Geometry only | Apache-2.0 | No |
| 5 | Iteratively detect, segment and **inpaint away** each object | Gemini 3.1 Pro (detection, `box_2d`), SAM 3 (masks), **Gemini 3 Pro Image** (object removal; FLUX.1-Kontext-dev optional) | Paid API; FLUX.1-Kontext-dev non-commercial | FLUX: yes |
| 6 | Clean, upsampled RGBA crop per object | Gemini 3 Pro Image (or gpt-image-1, or FLUX), Gemini validity check, rembg | Paid APIs; rembg MIT | No |
| 7 | Image to textured mesh | **Hunyuan3D-2.1** (default; TRELLIS.2 or Pixal3D optional), Real-ESRGAN | Tencent Hunyuan 3D 2.1 Community License: **no grant in the EU, UK or South Korea**. TRELLIS.2 MIT. Pixal3D needs gated DINOv3 | Hunyuan: no |
| 8 | Pose and scale each mesh back into the scene | **FoundationPose** (Any6D in its own env), nvdiffrast, Gemini front-face pick | FoundationPose, Any6D, nvdiffrast: **non-commercial** | No (Google Drive weights, unversioned) |
| 9 (opt.) | Articulation: which objects move, part segmentation, joints, joint critic loop | Fork of **articulate-anything** + Hunyuan3D-Part (P3-SAM, X-Part) or PartField, CoTracker; Gemini 3.1 Pro via **Vertex only** | articulate-anything MIT; Hunyuan3D-Part community licence; PartField NVIDIA non-commercial; CoTracker CC BY-NC 4.0 | No |
| 10 | Compile object metadata | — | Apache-2.0 | No |
| 11 | Sim-ready URDF: **CoACD** colliders (32 hulls), VLM **mass and friction** per object | CoACD, Gemini 3.1 Pro | CoACD MIT | No |
| 12 | Depenetrate and settle all objects | **PyBullet** | Zlib | No |
| 13 | URDF to USD into an OmniGibson dataset | OmniGibson / **Isaac Sim 5.1.0** | OmniGibson MIT; Isaac Sim NVIDIA Omniverse License Agreement | EULA |
| 14 | OmniGibson scene JSON + preview PNG | OmniGibson / Isaac Sim | as above | EULA |
| 2c + auto-bg (opt.) | Background as a 3D Gaussian splat (VOID inpainting, nerfstudio, 3DGRUT) | VOID on CogVideoX-Fun, splatfacto | VOID weights CogVideoX License | VOID: **yes** |

SimFoundry's own code (`simfoundry/`, `scripts/`, `tests/`, config and docs) is Apache-2.0. Nothing third-party is vendored except three MIT files and patch fragments; everything else is fetched by the install scripts.

## Licences: what blocks what
| Restriction | Components | In the default video path? |
|-|-|-|
| Non-commercial / research only | FoundationPose, FoundationStereo, Any6D, nvdiffrast, **DA3 Giant weights (CC BY-NC 4.0)**, PartField, CoTracker | **Yes**: stages 2 and 8 are required |
| Territory limit (no EU, UK, South Korea) | Hunyuan3D-2.1 (stage 7 default) | Yes; TRELLIS.2 (MIT) is the swap |
| Source-available custom licence | SAM 3 (trade-control and field-of-use terms; share-alike on redistribution) | Yes |
| Terms accepted **by the installer on your behalf** | NVIDIA Omniverse License Agreement, BEHAVIOR-1K dataset ToS, Anaconda ToS (`--accept-nvidia-eula --accept-dataset-tos --accept-conda-tos`, `OMNI_KIT_ACCEPT_EULA=YES` in `install_simfoundry.sh`) | Yes |
| No licence at all | TeleMoMa (teleop only, not installed) | No |

Two corrections to the README-level summary in experiment 07: the non-commercial parts are **required**, not only optional, and the repo's licence table lists Depth Anything 3 as Apache-2.0, which covers its code only; the default Giant checkpoint is CC BY-NC 4.0. The mesh model is Hunyuan3D-**2.1**.

### Gated models
| Model | Needed for | Terms to get access |
|-|-|-|
| `facebook/sam3` | Stages 3 and 5 (always) | HF login; agree to share your HF contact info with Meta (Meta Privacy Policy) and accept the SAM License |
| `netflix/void-model` (+ ungated `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`, ~41 GB) | Auto-background only | HF login + accept; CogVideoX License on weights |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | Pixal3D backend only | Manual approval; DINOv3 License |
| `black-forest-labs/FLUX.1-Kontext-dev` | Only if a config selects `flux` | Click-through; non-commercial licence |
| `briaai/RMBG-2.0` | **Not used**: the code blocks it with a stub | CC BY-NC terms |

The README asks for DINOv3 and RMBG-2.0 access in its quick start; neither is needed for the default run.

## The Gemini dependency
**Which stages call it:** A3 (surface name, frame choice), A5 (detection; object removal by image editing), A6 (upsampling by image editing; validity check), A8 (front-face pick), A9 (articulation classifier, part tree, merge, joint actor/critic up to 5 rounds), A11 (mass and friction), and all of pipeline B. Defaults: `gemini-3.1-pro-preview` for text and vision, `gemini-3-pro-image` for image edits.

**Auth:** `simfoundry/models/vlm.py::resolve_gemini_auth` takes `GEMINI_API_KEY` if set, else Vertex AI with application-default credentials and `GCLOUD_PROJECT`. Stage 9 is hard-wired to Vertex (`s9_articulate_objects.vlm_backend: vertex`), so articulation needs a GCP project with Vertex AI, not just a key.

**Is it pluggable?** Partly.
- There is a provider base class, `VLM_API` in `simfoundry/models/vlm.py`, with `Gemini`, `GPT` (image edits only, `gpt-image-1`), `Imagen3` and `FLUX1` subclasses. Callers only use `vlm(prompt=..., image_paths=..., temperature=..., top_p=..., seed=...)` and `vlm.get_result_text(result)`.
- But there is **no factory**: `Gemini(...)` is constructed directly at about 14 call sites, and stage 5 checks model names against a Gemini-only allow-list (`DETECTION_MODELS`).
- **Text/vision stages (3, 5 detection, 6 validity, 8, 11, B):** a small change. Add an `AnthropicVLM` or `OpenAIChat` class to `vlm.py` implementing `__call__` and `get_result_text` (and the cache hooks), add a `make_vlm(model_name)` factory, and switch the call sites: `3_segment_ground_plane.py` (`main`), `simfoundry/pipeline/frame_selection.py::refine_with_vlm`, `simfoundry/pipeline/front_canonicalization.py` (`canonicalize_front` caller), `5_decompose_scene.py` (`vlm_pro`), `6_upsample_object_images.py` (`validity_model`), `11_make_objects_sim_ready.py::main`, `B_augmentation/stages/{1,4,7}`. Roughly 150–250 lines. Risk: stage 5 parses Gemini's `box_2d` convention (`[ymin, xmin, ymax, xmax]` scaled to 0–1000); other models need a prompt change and a check of `canonicalize_vlm_bbox_xyxy`.
- **Image-editing stages (5 removal, 6 upsampling):** Anthropic models do not generate images, so these cannot move to Claude. Stage 6 already accepts `gpt-image-1`; stage 5 removal accepts only Gemini or FLUX, so OpenAI there needs a new masked-edit adapter (moderate).
- **Stage 9:** the Vertex calls live in the articulate-anything fork, a separate repo. Swapping providers there is a separate change.

**Token use per video (estimate from the code, not measured).** Per object: about 6–8 text/vision calls with 1–6 images each, plus 2–4 generated images (removal and upsampling, with retries). Per articulated object: about 12 more calls (tree, merge, up to 5 actor/critic rounds). Prices checked 2026-11-08 (no free tier for either model): Gemini 3.1 Pro preview $2 / $12 per 1M input / output tokens (output includes thinking); Gemini 3 Pro Image $0.134 per output image.

| Scene | Text/vision calls | Tokens (in + out) | Generated images | Est. cost |
|-|-|-|-|-|
| 2 objects, 1 articulated (plan below) | ~30–35 | ~0.15–0.25 M in, ~0.07–0.16 M out | 5–11 | **~$2–4** |
| 5 objects, no articulation | ~40 | ~0.15–0.25 M in, ~0.1–0.2 M out | ~12–20 | ~$3–6 |

The code has a response cache (`--cache-mode` / `--test-mode`), so a re-run of the same scene can replay calls for $0.

## What the output contains
| Item | What SimFoundry writes |
|-|-|
| Format | OmniGibson scene JSON (`Data/<scene>/s14_og/reconstructed_og_scene.json`) referencing per-object **USD** files in an OmniGibson dataset (`objects/<category>/<model>/usd/<model>.usd`), built from URDFs (`s11_sim/`). Also GLB meshes (`s7_mesh/`), settled poses, preview PNG |
| Colliders | CoACD convex decomposition, up to 32 hulls per object (falls back to a single convex hull if CoACD fails). Concave objects (cups, bins, basins) keep some of their cavity |
| Mass | One VLM estimate per rigid object (stage 11), written into the URDF inertial; per-part masses for articulated objects from stage 9 |
| Friction | One VLM estimate per object (`link_physics_materials`, static and dynamic); per-joint damping and friction for articulated objects |
| Articulation | Stage 9 only: URDF joints (`mobility.urdf`) carried into USD; falls back to a rigid import if any link has no mesh. Capture guide: close all lids and doors before filming |
| Support surface | **An infinite floor plane at the table height** (`use_floor_plane: True`). The table, counter or sink itself is not an object, so a sink cut into a counter is not modelled |
| Background / room | None by default. Optional Gaussian splat is **visual only, with no collision** (~90 min per scene on one RTX 3090), or a mesh background |
| Semantics | Object category phrase from the VLM (e.g. `trash can`), task YAMLs from pipeline B |

## Sample data
- Six example capture videos in the repo, `docs/assets/example_videos/` (Fruits, Mailbox, PutCupInBowl, PutMarkerInCup, ThrowAwayTrash, ThrowAwayTrashCousin; 15–20 s, 720p–1080p). They are repo docs, so Apache-2.0.
- Example finished scenes and assets on HF: dataset `nadunRanawaka1/simfoundry-assets` (not downloaded here).
- No cached model responses ship with the repo, so `--test-mode` cannot replay a full run offline.

## Paper numbers (reported, not reproduced)
| Metric | Value |
|-|-|
| Runtime, pipeline A | ~20 min (2 objects) to ~67 min (11 objects) on one **RTX 3090 24 GB** (Table L.4); about 5 min per object amortized. The repo's agent guide reports ~20 min for a 3-object tabletop on a 24 GB card |
| Background splat | ~90 min per scene, one GPU |
| Geometry vs SAM3D, zero-shot (Table L.2) | F1 0.92 / 0.87 / 0.81 (easy / medium / hard) vs 0.71 / 0.66 / 0.68; Chamfer 0.0042 / 0.0047 / 0.0091 m vs 0.0081 / 0.0087 / 0.0088 m (hard is slightly worse); position error 6 / 7.6 / 18 mm vs 16 / 18 / 22 mm |
| With ~3 min manual tuning per object | F1 0.93–0.99 |
| Sim vs real policy ranking | Mean Pearson 0.911, mean max rank violation 0.018 (7 tasks, 5 policy architectures) |
| Cousin training, real-world gain | +17% object, +21% scene, +40% task cousins (averages) |
| Articulation, mass and friction accuracy | **Not evaluated quantitatively** in the paper |
| Stated failure modes | Non-deterministic remote VLM outputs; Gemini inpainting sometimes leaves duplicated or degenerate objects; monocular scale errors; articulation depends on mesh segmentation and fails on occluded internals |

## What I tried
1. Cloned the repo and pinned the commit above. Read the paper (v4 HTML), README, `AGENTS.md`, `docs/INSTALL.md`, `docs/AGENT_INSTALL.md`, `THIRD_PARTY_LICENSES.md`, the installers (`install.sh`, `install_everything.sh`, `install_simfoundry.sh` relevant parts, `install_articulate.sh`, `download_checkpoints.sh`), `login_services.sh`, `api_keys.template.txt`, the pipeline README and config, `vlm.py`, and stages 3, 5, 9, 11, 12, 13 and 14. Checked the Isaac Sim version in BEHAVIOR-1K's `setup.sh` at the pinned commit (5.1.0) and the DA3 and SAM 3 model cards.
2. **CPU tests, no credentials.** Fresh venv `C:\Users\shree\af-scout-lab\simfoundry-venv` (uv, Python 3.11.9). Pins: `pytest==8.3.4 numpy==1.26.4 omegaconf==2.3.0 hydra-core==1.3.2 pillow==10.4.0`, later `opencv-python-headless==4.10.0.84 trimesh==4.4.9 scipy==1.13.1`. Read each test and the modules it imports first. Ran with all API-key variables unset, `GOOGLE_APPLICATION_CREDENTIALS` pointed at a missing file, an empty `CLOUDSDK_CONFIG` and `HF_HUB_OFFLINE=1`, so no call could reach Google, OpenAI or Hugging Face. Did **not** install torch, open3d or openai. Log: [`results/cpu-tests.txt`](results/cpu-tests.txt).
3. **Pipeline dry run.** `run_reconstruction.py --dry-run --exec-mode direct --stream-5-8 --detect-articulation` on `ThrowAwayTrash.mp4`: prints the stage plan, runs nothing, writes nothing.
4. Extracted one frame each from three example videos with local ffmpeg to see what they show.
5. **Not done:** no install script, no login script, no `gcloud` or `hf` login, no key set, no terms accepted, no model download, no GPU, no cloud VM, no API call. No permission guard blocked anything.

### Content flagged while reading (treated as data, not acted on)
- The README says to build the environments yourself "or ask your agent to do with AGENT_INSTALL.md". `docs/AGENT_INSTALL.md` is written to coding agents: run the installer detached, write the user's HF token and project into `api_keys.txt`, run `login_services.sh --default`. Not followed.
- `AGENTS.md` asks agents to install a pre-commit hook symlink. Not done.
- The installers accept NVIDIA, BEHAVIOR-1K and Anaconda terms on the user's behalf (`docs/INSTALL.md` says so explicitly). `install_everything.sh` also adds `conda-forge` to the user's global conda config.
- `login_services.sh` passes the HF token and W&B key as command-line arguments (`hf auth login --token ...`, `wandb login ...`), which exposes them in the process list on a shared machine.
- `install_articulate.sh` clones a personal fork (`nadunRanawaka1/articulate-anything-sf`) at branch `main` with no pinned commit, runs that repo's own install scripts, and uses `sudo` for `apt-get` and to install Blender under a system directory.

## Results
| Check | Result |
|-|-|
| Documented 4-file test subset | **20 passed, 1 skipped** |
| 16 more test files with light deps | **132 passed**, 5 skipped, 23 failed, 2 collection errors. 19 failures and both errors are missing heavy deps I chose not to install (torch, open3d, openai, matplotlib). 4 failures are Windows-only: path-separator assertions and a case-insensitive filesystem (`Laptop` vs `laptop`). No failure points at a logic bug |
| Dry run | Plan: 1b, 2, 3, 4, streamed 5–8, 10, 11, 12, 13, 14. `--detect-articulation` is **silently dropped** with a warning when `deps/articulate-anything` and its conda envs are missing. The message says stage 9 "is not available in this release", which is misleading: it just is not installed |
| Example videos | `ThrowAwayTrash.mp4`: a lidded pedal bin and a cup on a white table, 1080p60, 17.9 s. `Mailbox.mp4`: a red lidded box and a small cube on a wood table. Both contain one articulated object |
| Platform | Linux only in practice: `simfoundry/__init__.py` builds `REPO_DIR` by splitting on `/`; installers need mamba, `/usr/local/cuda-12.8`, apt and sudo |

## GPU run plan (not run)
### Clip
**`docs/assets/example_videos/ThrowAwayTrash.mp4`** from the pinned repo. Licence: Apache-2.0 (the repo's own docs), so it is redistributable with the licence notice. Why this one: the pipeline is validated on this kind of capture; it has a lidded bin (articulation test) and a cup (concave container for the hollow test); and it needs no download beyond the clone.

I did not find a whole-room walkthrough clip with a verified CC BY licence within the time box, and the pipeline would not use one anyway (one support plane per run). If a second, more home-like clip is wanted later, the candidate is the mip-NeRF 360 **"counter"** scene (a kitchen counter, ~240 frames, can be turned into a video with ffmpeg). Its licence was **not verified**; check it before use.

### Machine and disk
| Item | Choice | Reason |
|-|-|-|
| VM | GCP **g2-standard-8**, 1× L4 24 GB, 8 vCPU, 32 GB RAM, **Spot ~$0.51/h**, `--instance-termination-action=DELETE`, shutdown timer | 24 GB is the documented supported profile with `s7_mesh.low_vram=true`; articulation needs ≥18 GB |
| OS | Ubuntu 22.04 or 24.04 x86_64 | Isaac Sim 5.1 wheels are `manylinux_2_35` (glibc ≥ 2.35) |
| Simulator | **Isaac Sim 5.1.0** (pulled by BEHAVIOR-1K's `setup.sh` at commit `d89aae4e`), not 6.1 | Needs RT cores: the L4 (Ada) has them. Experiment 03 ran Isaac Sim 6.1 headless on this L4 with driver 595.91.07 (6.1 needs ≥ 595.58.03). Use the same 595 driver path; check Isaac Sim 5.1's own minimum driver in its requirements page before the run |
| CUDA toolkit | 12.8 at `/usr/local/cuda-12.8` | `install_any6d.sh` and the articulation build hard-code it |
| Disk | **250 GB pd-balanced** (~$0.034/h) | README: ~250 GB full install. Skipping auto-background saves ~41 GB (VOID) plus three envs, so ~180 GB is the floor; 250 GB leaves room for Isaac Sim caches |

### Steps
1. Create the Spot VM in a zone with L4 Spot capacity. SSH through IAP. Install the 595 driver from Ubuntu's signed repo as in experiment 03, CUDA 12.8 toolkit, Miniforge, `git-lfs`, `ffmpeg`. Read every downloaded script before running it; never pipe a remote script into a shell.
2. Clone SimFoundry and `git checkout 9e34ebefcd020583fbb755a8b57268dce78eca26`.
3. Core envs only: `bash scripts/installation/install_everything.sh --only "simfoundry hunyuan any6d da3"`, detached with a log. **Before running it, you read and accept the three sets of terms it accepts for you** (see prerequisites). `flash-attn==2.8.3` builds from source; set `MAX_JOBS=6` so 32 GB RAM is not exhausted.
4. Articulation: pin the fork first (`ARTICULATE_ANYTHING_REPO` + a known commit), then `bash scripts/installation/install_articulate.sh --default`. Confirm with `mamba env list | grep articulate-anything`.
5. Checkpoints: run only the needed lines of `download_checkpoints.sh` (FoundationStereo, FoundationPose refiner and scorer, SAM 2.1, Depth Pro, Real-ESRGAN). Skip VOID and CogVideoX (~41 GB, background only). Hunyuan3D-2.1, DA3 and SAM 3 weights download on first use.
6. You place your credentials on the VM yourself (HF token, Gemini key or Vertex ADC). Verify with the read-only checks in `docs/AGENT_INSTALL.md` §3a and §5, plus `bash scripts/pipeline/A_reconstruction/run.sh --dry-run --detect-articulation` (stage 9 must now appear in the plan).
7. Run A with caching so a re-run costs $0 in API calls:
   ```bash
   bash scripts/pipeline/A_reconstruction/run.sh --scene-name throwaway_trash \
     --video-fpath docs/assets/example_videos/ThrowAwayTrash.mp4 \
     --detect-articulation --cache-mode -- s7_mesh.low_vram=true
   ```
   Log `nvidia-smi` every 2 s and per-stage times (the orchestrator writes a timing summary).
8. `bash scripts/pipeline/C_application/run.sh --scene-name throwaway_trash --mode smoke-random` (headless, writes MP4s), then the pass tests below as a small OmniGibson script.
9. Copy small outputs into `results/`: scene JSON, `stage_info.json` files, timing summary, GPU CSV, preview PNG, smoke-test MP4 and pass-test JSON. **Delete the VM and disk.** A disk snapshot after step 5 avoids a reinstall but costs about $7/month for ~150 GB; delete it if no second run is planned.

### Pass tests (from experiment 03)
| Test | How, in the OmniGibson scene | Pass if |
|-|-|-|
| Small items fall through surfaces (03: bar soap fell through the wash basin) | Spawn a 2 cm cube and a soap-sized box (8×5×2.5 cm, 0.1 kg) 5 cm above (a) the closed bin lid and (b) the cup rim; step 5 s | (a) rests on the lid, (b) ends inside the cup's AABB (`InsideAABB` predicate already exists in `simfoundry/tasks/predicates.py`); neither below the floor plane |
| Articulated part attached and opens (03: drawers rendered 1.5 m from the carcass) | Read the bin's joints; check every child link lies within the parent's AABB at rest; drive the lid joint to its upper limit, step 3 s, then back to zero | Links stay attached (child AABB inside or touching parent), joint reaches ≥80% of its range, lid returns to closed. If stage 9 fell back to rigid import, record a **fail** with the logged reason |
| Objects sink into the floor (03: vacuum sank 0.2 m with convex hulls) | Compare each object's min-z after 5 s of settling with the floor plane and with `s12_physics` PyBullet poses | min-z ≥ −5 mm; position drift from the PyBullet poses < 1 cm (PhysX and PyBullet settle differently, so drift here is a finding in itself) |
| Is the "sink" hollow (03: convex hulls filled the counter's sink cutout) | No sink in this clip and the support is a plane, so test the concave proxies: drop a 1.5 cm sphere into the cup and, with the lid opened, into the bin | Sphere ends below the rim (inside the cavity), not resting on top of a filled hull. Also record CoACD hull count per object |

### Runtime and cost (estimate)
| Phase | Time | Cost (Spot $0.51/h + disk $0.034/h) |
|-|-|-|
| VM, driver, toolchain | 0.3–0.5 h | $0.15–0.30 |
| Core envs (4) incl. Isaac Sim 5.1 and flash-attn build | 2–3 h | $1.10–1.65 |
| Articulation envs + Blender | 0.5–1 h | $0.30–0.55 |
| Checkpoints and first-use HF weights (~30 GB) | 0.3–0.5 h | $0.15–0.30 |
| Pipeline A, 2 objects + articulation on an L4 (paper: ~20 min on a 3090; the L4 is slower) | 0.75–1.5 h | $0.40–0.80 |
| C smoke test + pass tests (Isaac Sim boot ~90 s each) | 0.5 h | ~$0.30 |
| **GPU VM total** | **~4.5–7 h** | **~$2.40–3.90** |
| **Gemini API** (2 objects, 1 articulated) | — | **~$2–4**; set a hard cap of $10 |
| **All in** | | **~$4.50–8**, plus ~$1 if a Spot pre-emption forces a restart mid-install |

## Prerequisites: you do these yourself
1. **Hugging Face:** sign in, open `facebook/sam3`, review and accept its conditions (shares your HF contact info with Meta; SAM License). Create a read-only token. You put it on the VM yourself.
2. **Gemini, pick one route:**
   - Vertex AI (needed for stage 9 articulation as configured): a GCP project with the Vertex AI API and billing enabled; run `gcloud auth application-default login` on the VM yourself; export `GCLOUD_PROJECT`. Enable `gemini-3.1-pro-preview` and `gemini-3-pro-image` for that project and region.
   - API key (no articulation unless stage 9 is changed): create a paid-tier key in AI Studio (neither model has a free tier) and export `GEMINI_API_KEY` on the VM yourself.
   - Either way, set a budget alert or quota cap (suggest $10).
3. **Terms the installer would accept for you:** read and decide on the NVIDIA Omniverse License Agreement (Isaac Sim 5.1, Kit, `pxr`), the BEHAVIOR-1K dataset terms and the Anaconda ToS before running `install_simfoundry.sh`.
4. **Model licences:** confirm research-only use is acceptable for FoundationPose, FoundationStereo, Any6D, nvdiffrast, DA3 Giant (CC BY-NC 4.0), PartField and CoTracker, and that you are outside the EU, UK and South Korea for Hunyuan3D-2.1 (or switch stage 7 to TRELLIS.2).
5. **GCP capacity:** confirm L4 Spot quota in the chosen region.
6. **VM-only sudo:** approve `sudo apt-get` and the Blender system install that `install_articulate.sh` performs, and choose the fork commit to pin.
7. Not needed for this run (despite the README quick start): DINOv3, RMBG-2.0, FLUX.1-Kontext-dev and VOID access, W&B, Docker Hub or NGC logins.

## Comparison
| Property | SimFoundry (this repo) | GATOR (exp 07, paper only) | SimReady Central in Isaac Sim (exp 03) |
|-|-|-|-|
| Released | Code (Apache-2.0) + installers; data-gen and training code not yet | Paper and demo page only | Asset catalog, runs today |
| Input | One handheld video of one tabletop | Casual photos or room-tour frames + masks + pointmaps | None (catalog by category) |
| Scale | One support plane per run | Per object, recomposed to room scale | Any room, built by hand |
| Matches the real scene | Yes (actual objects and layout) | Yes | No (category cousins) |
| Output | OmniGibson scene JSON + USD assets | Posed GLB meshes | OpenUSD packages |
| Colliders | CoACD, ≤32 hulls per object | None authored | Authored (convex or SDF) |
| Mass / friction | VLM estimate per object, unvalidated | None | Authored |
| Articulation | Optional stage 9 (VLM + part segmentation + joint critic) | None | Some assets, brittle in 03 |
| Room shell | Infinite plane; optional visual-only splat | Not produced | Built by hand |
| Paid API | Gemini, required | GPT-6-Astra agent for refinement | None |
| Licence blockers | Several required parts non-commercial; Hunyuan territory limit | Unknown until release | Mostly CC BY 4.0 |
| First-run cost | ~$4.50–8 (GPU + Gemini) | Not runnable | ~$0.46 (run in 03) |

Net: SimFoundry is the only one of the three that turns a real capture into a scene with colliders, mass, friction and optional joints today. It covers a counter or a table, not a room, and its default stack is research-only. For a room, the plausible combination is several SimFoundry runs (one per surface) placed in a hand-built or catalog shell, with the 03 pass tests as the gate.

## Next
1. Do the prerequisites, then run the GPU plan once on `ThrowAwayTrash.mp4`, with all four pass tests.
2. If it passes, try a home counter clip with a verified licence to see how the single-plane assumption holds up with real clutter.
3. Prototype a `make_vlm` factory and a Claude or OpenAI text adapter for the text stages (CPU-testable with the existing fake-client tests), and keep the image-edit stages on Gemini or `gpt-image-1`.
4. For anything beyond research use, list commercial substitutes per stage: TRELLIS.2 for Hunyuan, DA3 Small/Base (Apache-2.0) for DA3 Giant, and alternatives to FoundationPose and nvdiffrast.

## Links
- Project page: <https://research.nvidia.com/labs/gear/simfoundry/>
- Repo: <https://github.com/NVlabs/SimFoundry> (commit `9e34ebefcd020583fbb755a8b57268dce78eca26`)
- Paper: <https://arxiv.org/abs/2606.28276> · HTML v4: <https://arxiv.org/html/2606.28276v4>
- BEHAVIOR-1K `setup.sh` at the pinned commit: <https://github.com/StanfordVL/BEHAVIOR-1K/blob/d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864/setup.sh>
- SAM 3: <https://huggingface.co/facebook/sam3> · DA3 Nested Giant: <https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1>
- Gemini pricing: <https://aistudio.google.com/docs/pricing>
- Experiment 03: [`../03-simready-household/README.md`](../03-simready-household/README.md) · Experiment 07: [`../07-gator-video-to-sim/README.md`](../07-gator-video-to-sim/README.md)
