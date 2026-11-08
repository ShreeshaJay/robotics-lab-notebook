# 07 · GATOR: casual photos or room video to posed, textured objects for simulation

**Track:** Simulation / reconstruction · **Status:** desk review only. No code is released, so nothing was installed or run ($0, no GPU, no downloads) · **Date:** 2026-11-08

## Goal
Find out what NVIDIA's GATOR actually releases, what it takes as input and produces as output, what a first run would cost, and how its output compares with the SimReady Central assets used in experiment 03.

## What it is
**GATOR: Generative and Agentic 3D Object Reconstruction From Casual Images.** Qirui Wu, Stan Birchfield, Hesam Rabeti, Angel X. Chang, Bowen Wen (NVIDIA Learning and Perception Research). arXiv [2610.11215](https://arxiv.org/abs/2610.11215), v1 2026-10-08, cs.CV / cs.RO. Paper license CC BY 4.0. Project page: <https://research.nvidia.com/labs/lpr/gator/>. Source post: [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7514799354826645504/) (its links are lnkd.in shorteners to the paper and project page).

It reconstructs **one object at a time** from one or more images and returns a textured mesh plus its pose in the scene. Composing all of a room's objects gives a room-scale scene. Three stages:

1. **Pose-aware modality mixing.** Local attention fuses patch-aligned RGB, target-mask and pointmap features, so the target is isolated in clutter and its scene-relative pose and scale come out with it.
2. **Text-guided generation.** A structure → geometry → appearance flow cascade adopted from **TRELLIS.2** (1.3B structure model as initialization; geometry and appearance flows initialized from Pixal3D), with stage-specific text adapters (category text for structure, descriptions for geometry and appearance). Frozen encoders: DINOv3, SigLIP2, T5Gemma2, NAF. Inference: 12 sampling steps per flow, guidance 3.
3. **Agentic refinement.** An LLM agent (the paper uses **GPT-6-Astra through the Codex CLI**, "extra-high" reasoning) edits the mesh in **Blender**, re-renders it against the input photos and keeps or reverts each edit. Budget: 10 minutes per object.

### Inputs and outputs
| Item | What the paper says |
|-|-|
| Input | Casual images (or frames of a room-tour video) + per-object target masks + pointmaps (from estimated cameras and depth) + category text / descriptions |
| Upstream estimators | Masks: SAM 3 (ScanNet++). Cameras + depth: Pi3X (Toys4K). For ScanNet++ the camera/depth model is not named. There is no end-to-end "video in" script described |
| Output mesh | **GLB**, with PBR textures (base color, normal, roughness/metalness) |
| Pose / scale | Scene-relative pose. Metric size only if the input depth is metric (paper A.1) |
| USD | Not mentioned. The project page shows scenes in Isaac Sim (a 46-object room settling, humanoid robots interacting with the room), so a GLB → USD step exists internally but is not described |
| Physics properties | **None authored.** No mass, friction, density or collider settings in the paper. Section D.6 runs a PyBullet test (15 objects from a ScanNet++ meeting room, convex collision approximations, "same assumed mass and physics settings" for all) |
| Articulation | **None.** Every object is a single rigid mesh. Doors and drawers are not discussed |
| Room shell | Walls, floor and fixed structure are not described as an output |
| Runtime | ~30 s generation per object; +up to 10 min agent refinement per object. No per-scene time given |
| Hardware | Not stated (training or inference) |

### Reported results (paper, not reproduced)
| Benchmark | GATOR | Best baseline | Note |
|-|-|-|-|
| Toys4K, 4 views | CD 0.009, F1 0.757 | Pixal3D-MV CD 0.011, F1 0.701 | 19.7% CD reduction |
| Tabletop (LM-O, HB, HANDAL) | CD 0.009, F1 0.904, ADD-SB@0.1 98.47% | Pixal3D CD 0.018 | |
| ScanNet++ (3 rooms) | CD 0.031, F1 0.603, IoU 0.733 | RecGen CD 0.041 | 23% CD reduction |
| PyBullet settle test | Refined chairs stay within 1.3° of upright | Unrefined chairs lean or topple 40–79° | 3 chairs; shows refinement matters for stable contact |

Paper caveats: agent edits are local and slightly *worsen* CD/F1/IoU on HANDAL; refinement depends on a proprietary model; gross mask or depth failures still break reconstruction (D.7). Training used ~980k procedurally cluttered assets plus 3D-FRONT, InternScenes, SceneSmith and SAGE scenes.

## What's released (checked 2026-11-08)
| Item | Status |
|-|-|
| Paper | Yes, arXiv, CC BY 4.0 |
| Project page | Yes: videos, interactive 3D viewers, point-cloud toggle |
| Code | **No.** Project page says "Code (Coming Soon)". `github.com/NVlabs/GATOR` and `github.com/NVlabs/gator` return 404; GitHub search for `gator` under NVlabs: 0 repos |
| Weights | **No.** Not mentioned anywhere |
| Code / weight license | Unknown. Base model TRELLIS.2 is MIT ([model card](https://huggingface.co/microsoft/TRELLIS.2-4B)); DINOv3, SAM 3 and T5Gemma2 carry their own terms, so the final license will depend on NVIDIA's release |
| Refinement agent | Needs a paid, proprietary LLM API (GPT-6-Astra via Codex) as published. "Open agents" are listed as future work |

## What I tried
- Read the arXiv abstract, the full arXiv HTML (including appendices A.1, A.4, A.5, D.6, D.7) and the project page.
- Checked GitHub (direct URLs + API search) for a GATOR repo: none.
- Checked the closest released NVIDIA alternative, **SimFoundry** ([project](https://research.nvidia.com/labs/gear/simfoundry/), [repo](https://github.com/NVlabs/SimFoundry), arXiv [2606.28276](https://arxiv.org/abs/2606.28276)) at README level only.
- **Not done:** no venv, no install, no downloads. With no code or weights, there is no CPU-only step to run. Nothing was created in `C:\Users\shree\af-scout-lab\gator\`.

### Content flagged while reading (treated as data, not acted on)
- arXiv HTML Appendix A.4 contains the refinement agent's prompt, including a **base64-encoded block of agent instructions**. It is addressed to the paper's refinement agent. Not decoded or followed.
- SimFoundry README asks readers to run `install_everything.sh`, `login_services.sh`, `gcloud auth application-default login`, `hf auth login`, set a Gemini API key, and request access to gated HF models. None of this was done.

## Results
- GATOR is a strong **per-object** reconstructor that also recovers pose, which is what makes room recomposition possible. As released, it is a paper plus a demo page.
- Its output is **visual geometry, not a sim asset**: GLB + PBR textures + pose. Colliders, mass, friction, articulation and the room shell all have to be added by someone else.
- The video-to-scene demo on the project page is not described as a single pipeline in the paper; cameras, depth and masks come from separate estimators (Pi3X, SAM 3).

## Comparison with SimReady Central assets (experiment 03)
Experiment 03 built a 23-asset SimReady room in Isaac Sim 6.1 headless. Weak points there: articulated drawers (ForgeCAD chest-of-drawers bodies rendered ~1.5 m from the carcass under a blanket kinematic override) and colliders (bar soap fell through the wash basin).

| Property | SimReady Central | GATOR (as published) |
|-|-|-|
| Matches the real room | No: catalog cousins chosen by category | **Yes**: the actual objects, textures and layout from the photos/video |
| Format | OpenUSD packages | GLB; USD conversion needed |
| Colliders | Authored (e.g. ForgeCAD chair: 35 convex hulls) | None; paper uses generic convex approximations in PyBullet |
| Mass / friction | Authored (sometimes measured, e.g. ALLSIDES) | None |
| Semantics | Class label + Wikidata Q-code | Category text used as input; no label schema in output |
| Articulation | Some (e.g. 90 Palatial appliances), but brittle in our run | **None**: drawers and doors become solid blocks |
| Room shell | Built by hand in 03 (floor + 3 walls) | Not produced |
| Licence | CC-BY-4.0 (most datasets) | Unknown until release |
| Cost per asset | Download (median 60 MB package) | ~30 s GPU + up to 10 min of LLM agent time |

Net: GATOR solves "does the scene look like *this* room", which SimReady cannot. It does not help with either 03 weak point. Articulation would be absent rather than broken, and colliders would need to be generated (convex decomposition or SDF) and checked with the same fall-through test used in 03. A realistic combination: GATOR for the room-specific geometry and layout, SimReady or a part-articulation model for anything that must open.

## Run plan (for when code and weights are released)
Not executable today. Kept here so the first run is cheap once the repo appears.

**Machine:** GCP `a2-ultragpu-1g` (A100 80 GB), Spot, ~$2.95/h. Reason: TRELLIS.2 alone needs ≥24 GB and is verified only on A100/H100; GATOR adds DINOv3, SigLIP2 and T5Gemma2 encoders and multi-view conditioning, so peak memory is unknown. Try `g2-standard-8` (L4 24 GB, Spot ~$0.51/h) only after measuring peak memory on the A100, and expect it to be several times slower or to need a low-VRAM mode.

**Clip (public, never private footage):**
- First choice: the project page's own sample, if the release ships one with its licence stated.
- Second: one room from **ScanNet++**, which matches the paper's evaluation. It needs an approved application and acceptance of the ScanNet++ Terms of Use (non-commercial research). A person must apply and accept; the agent must not.
- Fallback with no application: one living-room sequence from **ARKitScenes** ([repo](https://github.com/apple/ARKitScenes)), CC BY-NC-SA 4.0, downloaded to a scratch disk on the VM, not to this repo.

**Steps**
1. Read the released install scripts before running them; pin every version; no `curl | sh`.
2. Create the Spot VM with a 200 GB disk and a shutdown timer. Pull the code and weights to the VM disk only.
3. Run the upstream estimators (camera/depth + SAM 3 masks) on ~100–300 frames of the clip.
4. **Generation only** (no agent) on 10–15 objects. Log time and `nvidia-smi` peak memory every 2 s, as in 03.
5. **Refinement** on 3–5 of the same objects, to measure LLM time, token cost and whether it changes stability.
6. Convert GLB → USD, add convex-decomposition colliders and a default density, drop into the 03 Isaac Sim 6.1 headless scene and run the same settle / fall-through check.
7. Copy small outputs (metrics JSON, GPU CSV, a few PNG renders) into `results/`. Delete the VM and disk.

**Estimated runtime and cost**
| Phase | Time | Cost at A100 Spot $2.95/h |
|-|-|-|
| Setup, weight download, estimators | ~1.0–1.5 h | ~$3–4.50 |
| Generation, 15 objects × ~0.5 min | ~0.25 h (+ overhead) | ~$1 |
| Refinement, 5 objects × 10 min (sequential) | ~0.9 h | ~$2.70 GPU, **plus LLM API cost (unknown; GPT-6-Astra at extra-high effort)** |
| Isaac Sim import + settle test | ~0.5 h | ~$1.50 |
| **Total** | **~2.5–3.5 h** | **~$8–11 GPU + LLM API** |

A full 46-object room with refinement would add ~8 h of sequential agent time (~$23 GPU) unless refinement runs in parallel. Spot pre-emption can add a restart.

## Licence notes
- Paper: CC BY 4.0. Code and weights: not released, licence unknown.
- Base model TRELLIS.2: MIT. Other components (DINOv3, SAM 3, T5Gemma2, Pi3X) have their own licences; check each at release.
- Clips: ScanNet++ (Terms of Use, non-commercial research, application required); ARKitScenes (CC BY-NC-SA 4.0). Neither permits commercial use without separate permission.
- SimFoundry: code Apache-2.0; some optional third-party parts are non-commercial or research-only per its `THIRD_PARTY_LICENSES.md`.

## Next
1. **Watch** `github.com/NVlabs` and the project page for the GATOR code/weights release; re-check monthly.
2. When released: read the licence and install scripts first, then follow the run plan above.
3. In the meantime, the released option for "video → physics-ready scene" is **SimFoundry** (Apache-2.0 code, single video in, Hunyuan3D-2 / Depth Anything 3 / FoundationPose / SAM 3 inside, OmniGibson scene out, optional `--detect-articulation`, 24 GB GPU with `low_vram`, ~250 GB disk, Gemini API for VLM stages, gated HF models). A separate scout would be needed before running it.
4. Whatever reconstructor is used, keep the 03 checks (collider fall-through, articulation test) as the acceptance test for a reconstructed room.

## Sources
- arXiv abstract: <https://arxiv.org/abs/2610.11215> · HTML: <https://arxiv.org/html/2610.11215>
- Project page: <https://research.nvidia.com/labs/lpr/gator/>
- Source post: <https://www.linkedin.com/feed/update/urn:li:activity:7514799354826645504/>
- TRELLIS.2 model card: <https://huggingface.co/microsoft/TRELLIS.2-4B>
- SimFoundry: <https://research.nvidia.com/labs/gear/simfoundry/> · <https://github.com/NVlabs/SimFoundry>
- ARKitScenes: <https://github.com/apple/ARKitScenes>
- Experiment 03: `../03-simready-household/README.md`
