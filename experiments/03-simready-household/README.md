# 03 · SimReady Central assets for household scenes

**Track:** Simulation · **Status:** metadata survey done (CPU, $0); Isaac Sim 6.1 scene ran on one Spot L4 on 2026-11-08 (0.84 VM-hours, about $0.46), VM and disk deleted · **Date:** 2026-11-07, run 2026-11-08

## Goal
Find out whether NVIDIA's SimReady Central library (free OpenUSD assets that ship with collision, mass, friction and semantic labels) covers what a small home needs: bedroom, bathroom and kitchen, plus the things a cleaner touches during a short-term-rental changeover. Then plan the cheapest sensible first run of a small Isaac Sim scene on one cloud NVIDIA L4.

I downloaded metadata only. No USD geometry or textures were pulled in bulk. I inspected three sample assets' small text layers.

## Files
| File | What it is |
|-|-|
| `analyze_catalog.py` | Fetches the catalog metadata and prints counts by dataset and category plus the turnover checklist. Writes `coverage.csv` |
| `coverage.csv` | Checklist output: hits per item, by source, with example names |
| `inspect_physics.py` | Prints the colliders, mass, friction and semantic labels of one asset, with `usd-core` and no Isaac Sim |
| `scene.py` | Isaac Sim 6.1 headless scene: 23 assets, convex-hull colliders, RGB + depth + instance segmentation. Ran on 2026-11-08 (see Run results) |
| `requirements.txt` | Pinned local venv (Python 3.12.13, `huggingface-hub==0.36.0`, `usd-core==25.8`) |
| `results/` | Small outputs of the 2026-11-08 run: `metrics_*.json`, `gpu_*.csv` (nvidia-smi every 2 s), Isaac logs, RGB / depth / instance PNGs, and the VM scripts (`vm_setup_all.sh`, `vm_run.sh`) |

Reproduce: `uv venv --python 3.12`, `uv pip install -r requirements.txt`, then `python -I analyze_catalog.py --data-dir <somewhere-outside-repo> --fetch`.

## Where the data actually lives
- **Space** [`nvidia/simready-central`](https://huggingface.co/spaces/nvidia/simready-central) is a search UI (OpenSearch + SigLIP2 embeddings). It holds no assets. Its endpoint `https://nvidia-simready-central.hf.space/install_categories.json` maps every catalog package to a category path. That endpoint was the most useful single file here (catalog generated 2026-11-02).
- **Catalog of record:** [`NVIDIA-Omniverse/simready-central`](https://github.com/NVIDIA-Omniverse/simready-central), `submissions/<owner>/<dataset>`. Each line is a commit-pinned URL to one package's `com.nvidia.simready.packaging.json`.
- **Assets** are in 21 public HF datasets. The main ones:

| Dataset | Packages in catalog | License | Notes |
|-|-|-|-|
| [`nvidia/simready-assets`](https://huggingface.co/datasets/nvidia/simready-assets) | 2,573 | CC-BY-4.0 | NVIDIA's own assets. The catalog pins commit `653d91e1…` |
| [`PalatialSim/palatial-nv-asset-library`](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library) | 957 (1,013 in the dataset) | CC-BY-4.0 | Folders are opaque IDs. Names come from `data/catalog.jsonl` |
| [`ForgeCAD/simready-assets`](https://huggingface.co/datasets/ForgeCAD/simready-assets) | 76 | CC-BY-4.0 | Parametric CAD: furniture, appliances |
| [`ALLSIDES/simready-scanned-objects`](https://huggingface.co/datasets/ALLSIDES/simready-scanned-objects) | 69 | CC-BY-4.0 | 3D scans with weighed mass |
| `LGElectronics/simready-assets` | 21 | CC-BY-4.0 | Fridges, washers, TVs |
| 16 robot/OEM datasets | 73 | mostly CC-BY-4.0. FANUC and AgiBot use Apache-2.0, OnRobot BSD-3, UR "other" | Arms, grippers, data-center gear |

- **Access:** none of the 21 datasets is gated or private. No sign-in or terms were needed to read metadata or download files. Every NVIDIA package ships the full CC-BY-4.0 text in `.metadata/LICENSE.txt`. Attribution is required if renders or assets are redistributed.

## Findings

### 1. Size and shape of the catalog
- **3,769 packages** in the catalog: 3,732 categorized, plus 37 present on the dataset's `main` branch but not yet indexed. The blog's "2,000+" figure undercounts it.
- By top-level category: Industrial 1,214 · OEM (partner) 1,196 · **Residential 1,019** · Hospital 256 · Lighting 47.
- About half of Residential is **food** (~493: fruit 136, vegetables 86, beverages 63, breads 61, snacks 54 and others). The rest is dishware 109, utensils 81, cooking gear 47, cleaning supplies 34, office supplies 123, and only small furniture and bathroom groups (bathroom 18, sinks 5, furnishing 8).
- Palatial's 957 are mostly kitchen: kitchenware 189, containers 143, food 76, bowls 42, home appliances 42 (many articulated).
- **Packages are heavy.** Median 60 MB, p90 509 MB. All 2,597 packages that had been indexed by 2026-10-16 total 466 GB. Furniture runs 280–800 MB each because of high-resolution textures plus duplicated `source_assets/`.

### 2. Coverage for a rental turnover
Counts come from name and class-label token matching (`analyze_catalog.py`), excluding industrial and robot categories. Treat them as ±a few. I checked the examples by eye.

| Area | Item | Hits | What's actually there |
|-|-|-|-|
| Bedroom | bed frame | 2 | Hospital beds only |
| Bedroom | mattress, pillow/cushion | 0 | **Missing** |
| Bedroom | bed linens | 1 | `bedding_a01`: a hospital bedding block, 2.0 × 0.87 × 0.40 m, one rigid body. On `main` but not yet indexed |
| Bedroom | towels | ~12 cloth + 4 paper | Cleaning, microfiber and dish towels. No bath towel as such |
| Bedroom | nightstand, dresser | 2, 1 | `table_bedside_b01` (hospital set), ForgeCAD `chest_of_drawers` |
| Bedroom | lamp, blinds, rug/mat | 11, 10, 4 | Ceiling lamps, a bedside lamp, window blinds, kitchen/cloth mats |
| Bedroom | clothes hanger, laundry basket | 0 | **Missing** |
| Living | sofa | 4 | 3 soft sofas filed under hospital, 1 visitor sofa (not indexed) |
| Living | chair/stool | 14 | ForgeCAD `dining_chair` and bar stool, office chairs. The rest are hospital or surgical |
| Living | table | 3 | Low coffee table, bedside table, desk. **No dining table** |
| Living | TV remote | 0 | **Missing** (TVs exist: LG, NVIDIA `TV_a01`) |
| Kitchen | cups/mugs/glasses | 141 | Strong |
| Kitchen | cutlery | 136 | Strong (forks, 26 knives, spoons, ladles) |
| Kitchen | pots/pans | 98 | Strong (22 pans plus Palatial cookware) |
| Kitchen | bowls / plates | 94 / 28 | Good / OK |
| Kitchen | kettle | 38 | 25 NVIDIA kettles, ForgeCAD and Palatial electric kettles |
| Kitchen | dish rack, counter, fridge | 3, 3, 10+ | Counter and island fixtures exist |
| Bathroom | toilet | 2 | `toilet_a01`, wall-mounted toilet |
| Bathroom | sink/basin | 6 | 2 wash basins, wall sink, vanity sink (ForgeCAD), surgical sink |
| Bathroom | soap | 15 | Bar soaps, dispensers, dish soap |
| Bathroom | toilet paper | 0 rolls | Holder only (`fixture_toiletPaperHolder_double_a01`) |
| Bathroom | toiletries (shampoo, toothbrush, toothpaste, razor) | 0 | **Missing** |
| Bathroom | shower/bathtub | 0 | Only a ForgeCAD shower seat |
| Bathroom | mirror | 3 | 2 bathroom mirrors |
| Cleaning | spray bottle/disinfectant | 6 | `cleaning_windowSpray_a01`, 4 disinfectants, Palatial spray bottle |
| Cleaning | trash bin | 3 | ForgeCAD pedal can, wheelie bin, Palatial compost bin |
| Cleaning | vacuum | 2 | Two vacuum cleaners (`serviceTool_*`, 780 MB package) |
| Cleaning | mop/bucket, sponge/cloth/brush, paper towel | 13, 15, 4 | Good |

**What's missing:** mattresses, pillows, sheets and duvets as separate items, residential bed frames, bath towels, toiletries, toilet-paper rolls, showers and bathtubs, dining tables, wardrobes, hangers, laundry baskets, and remotes. Bedrooms are the weakest area, kitchens and cleaning supplies the strongest.

### 3. Physics and semantics that come with the assets
These come from composed stages of sample assets (`inspect_physics.py`) plus the dataset READMEs and catalogs.

| Source | Collider (default) | Mass | Friction / material | Semantics |
|-|-|-|-|-|
| NVIDIA (`kitchenware_mug_a01` inspected) | Variant sets `PhysX` / `MuJoCo` / `Newton`. **`PhysX=Enabled` is the default and uses `physics:approximation = "sdf"`.** The MuJoCo variant uses `convexHull`. Colliders sit on instanceable meshes | Authored `physics:mass`. Mug = **1.33 kg**, several times a real mug. Its own validation record fails the mass rule `RB.MB.001` | Physics material: density 2600, static 0.35, dynamic 0.30, restitution 0.05 | `SemanticsLabelsAPI:wikidata_class` ("mug") + `wikidata_qcode` on the root. Material labels on materials. No `class` taxonomy |
| ForgeCAD (`dining_chair` inspected) | `physics` variant. **Compound of 35 `convexHull` colliders** | 6.5 kg | static 0.5, dynamic 0.45 ("felt pads on a wood floor") | `class` ("chair") + `wikidata_qcode` + per-part material |
| Allsides (README) | CoACD convex decomposition by default. `PhysX` variant → SDF (default there) or convexHull | Weighed real object | From material class | Wikidata Q-code labels |
| Palatial (catalog) | Median 27 collider prims per asset | `mass_status: verified` means checked against the USD, **not** physically measured. Examples: a towel at 1.5 kg, some geometry-derived | In each asset's physics JSON | `class_label` + Wikidata Q-code. **90 articulated** appliances (microwave, stand mixer, cabinets) |

Other things I noticed:
- **Textiles are rigid bodies.** Towels and bedding are single rigid meshes. Nothing in the catalog is cloth or deformable, so bed-making or towel-folding tasks would need cloth simulation added separately.
- Masses need a sanity pass before any grasp or force evaluation. A quick filter: compare authored mass against bounding-box volume × a plausible density.
- Semantic taxonomies differ by vendor (`wikidata_class`, `class`, `class_label`). `scene.py` adds a uniform `class` label so instance segmentation stays consistent.
- Every package carries SimReady validation and benchmark JSONs. Palatial's benchmarks list `engine_version 6.1.0.0`, so the assets are being tested against Isaac Sim 6.1.

## Run plan: one L4 VM, headless

### Version and install route
- **Isaac Sim 6.1.0**: the version the assets' benchmarks use. It needs Python 3.12, Ubuntu 22.04/24.04 (glibc ≥ 2.35) and an NVIDIA driver ≥ **595.58.03** on Linux ([requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)).
- **Recommended: the container** `nvcr.io/nvidia/isaac-sim:6.1.0`. It isolates Isaac's pinned dependencies, needs no NGC login, and has documented cache mounts ([container docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html)). Run `scene.py` with `/isaac-sim/python.sh`.
- **Alternative: pip** in a fresh venv: `pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com` ([pip docs](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)).
- **EULA:** both routes require accepting the NVIDIA Omniverse/Isaac Sim license: `-e ACCEPT_EULA=Y` for the container, `OMNI_KIT_ACCEPT_EULA=YES` for pip. **I read and accepted it on 2026-11-08 and passed `-e ACCEPT_EULA=Y` to the container on that one VM only.** It is not in any committed default. `scene.py` refuses to start unless `OMNI_KIT_ACCEPT_EULA` or `ACCEPT_EULA` is already set by the runner, and never sets it. Leave `PRIVACY_CONSENT` unset (that flag opts in to data collection).

### VM
| Choice | Pick | Why |
|-|-|-|
| Machine | **g2-standard-8** (1× L4 24 GB, 8 vCPU, 32 GB RAM), zone `us-central1-a` | NVIDIA's GCP guide says "g2-standard-4 or better" ([GCP guide](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_advanced_cloud_setup_gcp.html)). But -4 has 16 GB RAM, below Isaac Sim's 32 GB minimum. The L4 isn't in the requirements table but is Ada with RT cores, so it is supported (A100/H100 are not, because they lack RT cores) |
| Provisioning | **Spot** for experiments, on-demand only if a run must not be pre-empted | Spot is about 40% cheaper here |
| Image | `ubuntu-2204-lts` family from `ubuntu-os-cloud` (current image, not the 2023 one in NVIDIA's example) | Matches NVIDIA's guide. glibc 2.35 |
| Boot disk | **200 GB pd-balanced** | Container plus caches (tens of GB), 3.7 GB of assets, outputs. NVIDIA's guide uses 100 GB pd-ssd, which is tight once shader and asset caches grow |
| Driver | NVIDIA ≥ 595.58.03 from a signed apt repository, checked with `nvidia-smi` | NVIDIA's GCP guide downloads and runs Google's `install_gpu_driver.py`. If you use it, download it, read it, then run it. Never pipe a remote script into a shell |
| Headless | No display. `SimulationApp({"headless": True})`. Optional WebRTC livestream (`runheadless.sh`, ports 49100/tcp and 47998/udp) only if I want to look. Otherwise SSH through IAP and copy PNGs back | Smaller attack surface, no open ports |
| Guardrails | Billing budget alert, `--max-run-duration` / auto-delete on the VM, delete the disk afterwards (or snapshot it) | A stopped VM still bills its disk |

### The scene (`scene.py`, ran 2026-11-08)
- **6 × 5 m room** (floor + 3 walls + lights) with 23 assets: bedding block (as the bed), nightstand, chest of drawers, sofa, coffee table, dining chair, towel on the bed, vacuum, kitchen counter with kettle, pan, mug, tumbler, plate, fork, knife, sponge, spray bottle and paper-towel roll, a pedal trash can, a toilet, a wash basin and bar soap. 20 NVIDIA + 3 ForgeCAD packages at pinned commits, 3.72 GB without thumbnails.
- **Colliders:** after referencing each asset, the script de-instances it and sets `physics:approximation = convexHull` on every collision mesh, removes `PhysxSDFMeshCollisionAPI`, and caps hulls at 64 vertices. I tested this override offline with `usd-core` on the sample mug: `sdf` became `convexHull`. Reason: SDF is NVIDIA's PhysX default, and a public LinkedIn stress test reported SDF colliders exhausting GPU memory at scale (I did not verify that report). `--collider sdf` is there for an A/B memory comparison.
- Furniture is kinematic. Small items are dynamic and are placed onto their supports from runtime bounding boxes, then settle for 240 updates.
- **Outputs:** two cameras, 1280×720, `rgb`, `distance_to_image_plane`, `instance_segmentation_fast` and `instance_id_segmentation_fast` (PNG + NPY), plus `metrics.json` with timings, `nvidia-smi` GPU memory at each stage, collider counts, authored masses and final heights. Items that fall through the floor are a quick collider check.
- **Caveat:** the Replicator and Kit API names follow the 6.1 docs (`rep.annotators.get`, `rep.functional.create.camera`, `rep.orchestrator.step`), with fallbacks to older names. Expect a fix or two on first contact.

### Runtime and cost estimate
Prices are from Google's [accelerator-optimized pricing page](https://cloud.google.com/products/compute/pricing/accelerator-optimized) and [disk pricing page](https://cloud.google.com/compute/disks-image-pricing), us-central1, read on 2026-11-07. The L4 is included in the G2 machine price.

| Item | Rate |
|-|-|
| g2-standard-8 on-demand | $0.8536/h |
| g2-standard-8 Spot (current) | $0.5108/h |
| (g2-standard-16, 64 GB RAM, if 32 GB proves tight) | $1.1472/h on-demand, $0.6865/h Spot |
| pd-balanced 200 GB | $0.000136986/GiB-h → $0.027/h (≈ $20/month if left lying around) |
| External IPv4, if used | small hourly charge (not checked). IAP-only SSH avoids it |

| Phase (first session, estimate) | Time |
|-|-|
| VM create, driver install, reboot | 15–20 min |
| Docker + NVIDIA Container Toolkit + pull Isaac Sim 6.1 (image size not stated in docs, likely tens of GB) | 15–25 min |
| `scene.py --download-only` (3.7 GB from HF) | 3–8 min |
| First Isaac launch, cold shader cache | 5–15 min |
| Load 23 assets, settle, render 2 cams × 10 frames | 3–6 min |
| **Total first session** | **~1–1.5 h** |
| Re-runs with warm caches | ~5–10 min each |

**Expected spend for the first session:** about **$0.55–0.80 on Spot** ($0.538/h all-in) or **$0.90–1.35 on-demand** ($0.881/h all-in), then about $0.05–0.15 per re-run. Add $0.027/h for any idle hours the disk is kept. VRAM estimate: 6–10 GB for the RTX renderer, two 720p render products and textures, so it should fit 24 GB with convex hulls. The SDF A/B run is where memory could spike. This is an estimate, not a measurement.

## Run results (2026-11-08)

**Verdict:** the scene runs headless on one L4. 23 SimReady assets (208 collision meshes), physics settle, and two 720p cameras with RGB, depth and instance segmentation. Peak GPU memory was **7.0–7.3 GB of 23 GB**. A re-run with a warm Kit cache takes **under 2.5 minutes**.

### Setup
- **VM:** `g2-standard-8` Spot (1× L4 24 GB, 8 vCPU, 32 GB RAM), 200 GB pd-balanced, created with `tools/gpu-vm/create_vm.sh` (3 h hard cap, auto-delete). Image `common-cu129-ubuntu-2404-nvidia-580` (Ubuntu 24.04, kernel 7.0.0-1014-gcp). Headless. No firewall ports opened, no livestream.
- **Driver:** the image ships 580.178.04. I purged it and installed Ubuntu's signed `nvidia-headless-no-dkms-595-server-open` plus `linux-modules-nvidia-595-server-open-$(uname -r)` (precompiled, no DKMS), `libnvidia-gl-595-server` and `libnvidia-extra-595-server` from `noble-updates`, then rebooted. `nvidia-smi` showed **595.91.07** (CUDA 13.2). The EGL, GLX and Vulkan ICD files were present.
- **Docker CE** 29.9.3 came from Docker's signed apt repo. The image already had NVIDIA Container Toolkit 1.19.1 from NVIDIA's signed repo (apt-held). It was wired in with `nvidia-ctk runtime configure`.
- **Container:** `nvcr.io/nvidia/isaac-sim:6.1.0`, pulled without an NGC login, run as uid 1234 with `-e ACCEPT_EULA=Y` (my own acceptance, this VM only) and no `PRIVACY_CONSENT`. Assets were mounted read-only. Exact commands: `results/vm_setup_all.sh` and `results/vm_run.sh`. The committed copy has no EULA default.
- **Assets:** 23 packages at pinned commits, **3.72 GB** (3,720,248,729 bytes), downloaded on the VM only.

### Timings
| Phase | Time |
|-|-|
| VM create (`gcloud`, until RUNNING) | 17–20 s |
| Driver swap 580 → 595 (apt) | 67 s |
| Reboot until `nvidia-smi` showed 595 | < 90 s |
| Docker CE + toolkit wiring | 55 s |
| `docker pull` Isaac Sim 6.1.0 | **248 s**. Image is **32.3 GB on disk, 10.6 GB compressed** |
| Asset download, 3.72 GB (after the fix below) | ~75 s |
| VM create to first scene launch | ~10 min |

| Scene phase (`metrics.json`) | Cold Kit cache | Warm Kit cache |
|-|-|-|
| `SimulationApp` start | 55 s | 38 s |
| Load 23 assets | 48 s | 5 s |
| Settle, 240 updates (the first rendered frames compile RTX shaders) | **328 s** (GPU idle, ~7 CPU cores busy) | **23 s** |
| Capture: 2 cams × 720p × 4 RT subframes, plus saving PNG/NPY | 1.8–1.9 s first, then ~1.35–1.5 s per frame | same |
| Whole script / container wall | 452 s / 503 s | 87 s / 145 s |

### GPU memory (nvidia-smi, sampled every 2 s, whole GPU)
| Run | Colliders | Peak | After settle | After render |
|-|-|-|-|-|
| convexHull, cold (`gpu_convexhull.csv`) | 208 meshes → convex hull, ≤64 verts | 7,212 MiB | 7,141 MiB | 7,180 MiB |
| convexHull, warm cache (`gpu_convexhull_warmcache.csv`) | same | 7,335 MiB | 7,264 MiB | 7,303 MiB |
| **SDF A/B** (`--collider sdf`, `gpu_sdf.csv`) | same 208 meshes → SDF | 7,232 MiB | 7,161 MiB | 7,200 MiB |

- **SDF vs convex hull, VRAM:** no measurable difference at this scale (+20 MiB). Physics runs on the CPU (`--gpu-dynamics` was off), so SDF data does not live on the GPU here. Renderer and textures take about 7 GB, and physics is noise next to that. The reported SDF memory blow-up remains unverified. It would need GPU dynamics and hundreds of SDF bodies to test.
- **SDF vs convex hull, behaviour:** this is the real difference. Convex hulls fill concave shapes: the counter's sink cutout became solid, so the mug, tumbler, plate and knife rested **over** the sink at counter height (`final_min_z` ≈ 0.92 m). With SDF they fell **into** the sink (0.12–0.17 m), and the spray bottle stayed upright. SDF settle took about 26 s longer (cooking; PhysX warned that several meshes are not watertight).

### Render outputs (`results/`)
- `convexhull_cam_{kitchen,living}_009_rgb.png`: final RGB frames. `sdf_cam_*_009_rgb.png`: the same views with SDF colliders.
- `convexhull_cam_*_000_depth_mm.png`: 16-bit depth in mm. `*_instance_vis.png` with `*_instance_info.json`: instance segmentation (hash-coloured) and the id-to-prim map.
- `run1_counter_bug_kitchen_rgb.png`: the first run, before the counter fix (below).
- Not kept: per-frame NPY (3.7 MB each) and the other 9 frames, which are near-identical.

### Problems and fixes
| # | Problem | Fix |
|-|-|-|
| 1 | First Spot VM (`us-central1-a`) was **preempted after ~10 min**, during setup. The VM and disk were auto-deleted (`--instance-termination-action=DELETE`) | Recreated in `us-central1-b`, which ran 41 min without preemption. Folded all setup into one script (`vm_setup_all.sh`) so a preemption costs less. Driver, Docker, pull and download all run before the single reboot |
| 2 | `snapshot_download(allow_patterns=…)` lists the **whole** `nvidia/simready-assets` tree for every package. Anonymous calls hit **HTTP 429** on the second package | `scene.py --download-only` now lists only the package folder (`list_repo_tree(path_in_repo=…)`) and fetches files with `hf_hub_download` (4 threads, back-off retry on 429) |
| 3 | `scene.py` checked only `OMNI_KIT_ACCEPT_EULA` (pip route). The container convention is `ACCEPT_EULA` | Accept either variable, still only if the runner set it. The script never sets it |
| 4 | `fixture_counter_a01` is 3.0 m long along the asset's **+y**, not x. At yaw 0 it ran through the back wall, and 8 of 10 counter items fell to the floor | Counter `yaw=90`, centre x 1.48. Trash can moved to x −0.25 to make room. After the fix all 10 items rest on the counter |
| 5 | `rep.functional.create.camera` fallback only caught `AttributeError` | Also catch `TypeError` (defensive; the 6.1 call itself worked) |
| 6 | The Kit shader cache was not persisted, so the "warm" re-run still spent 320 s compiling | The cache lives in `/isaac-sim/kit/cache` (4.4 GB after one run). That path is now mounted in `vm_run.sh` |
| 7 | (runner error) I replaced `run.sh` while a run was reading it. Bash reads scripts lazily, so that run's log was clobbered. Its `metrics.json` and images were intact | Do not edit a running script. The cold-run log (`run_convexhull_run1.log`) covers the same steps |

**Known issues left open** (not fixed, to keep changes minimal):
- **Articulated assets as kinematic.** The bedside table, counter, toilet and trash can have joints between their parts. Making every body kinematic triggers `CreateJoint - cannot create a joint between static bodies` (harmless). The ForgeCAD **chest of drawers' drawers render ~1.5 m away from its carcass**: the drawer bodies don't follow the parent transform. This needs a proper articulation path (fixed root plus jointed drawers), not a blanket kinematic override.
- **Vacuum** (13 dynamic bodies) sinks 0.2 m into the floor with convex hulls, but not with SDF.
- **Bar soap** falls through the wall-mounted wash basin to the floor in both collider modes. The basin's collider or its mount needs a look.
- The sofa (`sofa_soft_b01`) is armchair-sized (0.71 m wide).

### Spend
| Item | Amount |
|-|-|
| VM 1 (preempted) | 9.7 min |
| VM 2 | 40.8 min |
| Total | **0.84 VM-hours × $0.5108/h Spot = $0.43**, + 200 GB pd-balanced × 0.84 h = $0.02, + ephemeral IPv4 (a few cents at most) → **≈ $0.46** |

Both VMs and their disks are deleted. `gcloud compute instances list` and `gcloud compute disks list` for the project both returned 0 items, and no static addresses remain.

## Next steps
1. Persist `/isaac-sim/kit/cache` (and the image) on a small reusable disk or image. Cold shader compilation is about 5 of the 8.5 minutes per fresh VM.
2. Handle articulated SimReady assets properly: a fixed base and jointed drawers or doors instead of all-kinematic. Then the chest, nightstand and counter shelves become door and drawer tasks.
3. Sanity-filter masses across the household subset (authored mass vs. bounding-box volume × density). The mug is still 1.33 kg and the counter 357 kg.
4. An SDF memory test that could actually stress it: `--gpu-dynamics`, then 10× the small dynamic items, then compare against convex hull.
5. Fill the bedroom and bathroom gaps from outside the catalog: mattresses, pillows, toiletries, toilet paper, bath towels, a shower or tub. Candidates to scout: Objaverse/procedural generation, Infinigen-Indoors, or Isaac Sim's own residential sample assets. Check each one's license.
6. Decide how to handle cloth (towels, sheets). Rigid stand-ins are fine for navigation and pick-and-place but useless for folding or bed-making.
7. Try the 90 articulated Palatial appliances (microwave, cabinets) for door and drawer tasks.

## Notes on sources
- [SimReady Central Space](https://huggingface.co/spaces/nvidia/simready-central) · [HF blog post](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories) · [catalog repo](https://github.com/NVIDIA-Omniverse/simready-central)
- The catalog repo's README and `AGENTS.md` include "tell your agent: read and follow the skill…" prompts for submitting or mirroring packages. I treated them as documentation and did not run those skills. This survey only reads metadata.
