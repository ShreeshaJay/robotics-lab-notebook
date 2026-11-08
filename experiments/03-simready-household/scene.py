"""Small "rental turnover" room built from SimReady assets, rendered headless in Isaac Sim.

STATUS: ran end to end on 2026-11-08 (container, one L4; see README Run results). Targets Isaac Sim 6.1.0 (pip `isaacsim` or the
nvcr.io/nvidia/isaac-sim:6.1.0 container) on one NVIDIA L4 (24 GB). API names were
checked against the 6.1 docs; every version-sensitive call has a fallback, but expect
small fixes on the first real run.

What it does
  1. (--download-only, plain Python, no Isaac Sim) pulls 22 SimReady packages from
     Hugging Face at pinned commits, skipping thumbnails. About 3.8 GB.
  2. Builds a 6 x 5 m room (floor + 3 walls + lights), references the assets, selects their
     PhysX / physics variants, and FORCES convex-hull colliders on every collision mesh
     (SDF colliders are the default on NVIDIA's own assets and are reported to exhaust GPU
     memory at scale). `--collider sdf` keeps SDF for an A/B memory test.
  3. Furniture is kinematic (static); small items are dynamic and are dropped onto their
     support (counter, bed, basin) using runtime bounding boxes, then physics settles.
  4. Two cameras render RGB, depth (distance_to_image_plane) and instance segmentation
     (instance_segmentation_fast + instance_id_segmentation_fast) for --frames frames.
  5. Writes PNG/NPY outputs plus metrics.json (timings, GPU memory via nvidia-smi,
     collider counts, asset masses).

Run (on the VM, after you have read and accepted NVIDIA's EULA yourself):
  python scene.py --download-only --assets /data/simready
  OMNI_KIT_ACCEPT_EULA=YES python scene.py --assets /data/simready --out /data/out
  # container: /isaac-sim/python.sh scene.py --assets ... --out ...

Asset licenses: nvidia/simready-assets and ForgeCAD/simready-assets are CC-BY-4.0.
Attribute "NVIDIA SimReady Assets" / "ForgeCAD SimReady Assets" if you publish renders.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

NV = "nvidia/simready-assets"
NV_REV = "653d91e182011f357cf1644013cc94ddf194d349"          # commit pinned by SimReady Central's index
NV_MAIN = "7879e0e50423290300ac20ecd6e727325f680b60"         # dataset main on 2026-11-07 (bedding_a01 is newer)
FC = "ForgeCAD/simready-assets"
FC_REV = "0573053fa2c9174ca2a4678a462ec4dfd3035d59"


def nv(name, rev=NV_REV, **kw):
    return dict(name=name, repo=NV, rev=rev, usd=f"sm_{name}_01.usd", **kw)


def fc(name, **kw):
    return dict(name=name, repo=FC, rev=FC_REV, usd=f"{name}.usda", **kw)


# Room: x in [-3, 3], y in [-2.5, 2.5], floor top at z = 0. Back wall at y = +2.5.
# static=True -> kinematic rigid body. on=<name> -> dropped onto that asset's top.
# xy is the target centre (or offset from the support's centre when `on` is set).
ASSETS = [
    # bedroom / living (x < 0)
    nv("bedding_a01", rev=NV_MAIN, cls="bed", xy=(-2.0, 1.6), yaw=0, static=True),
    nv("table_bedside_b01", cls="nightstand", xy=(-0.7, 2.1), yaw=0, static=True),
    fc("chest_of_drawers", cls="dresser", xy=(-2.6, -0.4), yaw=90, static=True),
    nv("sofa_soft_b01", cls="sofa", xy=(-1.4, -1.8), yaw=0, static=True),
    nv("furniture_table_coffee_low_a01", cls="table", xy=(-1.4, -0.6), yaw=0, static=True),
    fc("dining_chair", cls="chair", xy=(0.4, -0.3), yaw=200, static=True),
    nv("cleaning_towel_a01", cls="towel", on="bedding_a01", xy=(0.3, 0.0), yaw=15),
    nv("serviceTool_vacuum_cleaner_a01", cls="vacuum", xy=(-0.3, 1.0), yaw=30),
    # kitchen (back-right)
    nv("fixture_counter_a01", cls="counter", xy=(1.48, 2.1), yaw=90, static=True),  # 3.0 m long axis is +y in the asset
    nv("kitchenware_kettle_a01", cls="kettle", on="fixture_counter_a01", xy=(-0.5, 0.0), yaw=0),
    nv("kitchenware_pan_a01", cls="pan", on="fixture_counter_a01", xy=(-0.15, 0.05), yaw=40),
    nv("kitchenware_mug_a01", cls="mug", on="fixture_counter_a01", xy=(0.15, -0.05), yaw=0),
    nv("dishware_glass_tumbler_a01", cls="glass", on="fixture_counter_a01", xy=(0.3, 0.05), yaw=0),
    nv("dishware_plastic_bigPlate_a01", cls="plate", on="fixture_counter_a01", xy=(0.5, -0.02), yaw=0),
    nv("kitchenware_fork_a01", cls="fork", on="fixture_counter_a01", xy=(0.5, 0.15), yaw=90),
    nv("kitchenware_knife_a01", cls="knife", on="fixture_counter_a01", xy=(0.58, 0.15), yaw=90),
    nv("cleaning_sponge_b01", cls="sponge", on="fixture_counter_a01", xy=(-0.35, -0.12), yaw=0),
    nv("cleaning_windowSpray_a01", cls="spray_bottle", on="fixture_counter_a01", xy=(0.7, -0.05), yaw=0),
    nv("consumable_paperTowel_roll_a01", cls="paper_towel", on="fixture_counter_a01", xy=(-0.75, 0.0), yaw=0),
    fc("pedal_trash_can", cls="trash_bin", xy=(-0.25, 2.1), yaw=0, static=True),
    # bathroom corner (front-right)
    nv("toilet_a01", cls="toilet", xy=(2.5, -1.2), yaw=-90, static=True),
    nv("washBasin_a01", cls="sink", xy=(2.65, -0.1), yaw=-90, static=True, mount_top=0.85),
    nv("cleaning_barSoap_a01", cls="soap", on="washBasin_a01", xy=(0.0, 0.0), yaw=0),
]

CAMERAS = [  # name, position, look_at
    ("cam_living", (2.6, -2.3, 1.7), (-1.2, 0.6, 0.4)),
    ("cam_kitchen", (0.2, 0.2, 1.6), (1.6, 2.2, 0.9)),
]


# ----------------------------------------------------------------------------- helpers
def local_usd(root: Path, a: dict) -> Path:
    return root / a["repo"].replace("/", "__") / a["name"] / "simready_usd" / a["usd"]


def download(root: Path) -> None:
    """Plain Python: needs only `huggingface_hub`. Run before starting Isaac Sim."""
    # snapshot_download(allow_patterns=...) lists the WHOLE dataset tree (hundreds of thousands of files)
    # for every package and hits HF's 429 rate limit; list only the package folder instead.
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    api = HfApi()
    for a in ASSETS:
        dst = root / a["repo"].replace("/", "__")
        t = time.time()
        files = [f.path for f in api.list_repo_tree(a["repo"], path_in_repo=a["name"], recursive=True,
                                                    revision=a["rev"], repo_type="dataset")
                 if isinstance(f, RepoFile) and "/.thumbs/" not in f.path]
        def get(fp):  # anonymous HF API calls are rate-limited (429): back off and retry
            for k in range(6):
                try:
                    return hf_hub_download(a["repo"], fp, repo_type="dataset", revision=a["rev"], local_dir=dst)
                except Exception as e:  # noqa: BLE001
                    if "429" not in str(e) or k == 5:
                        raise
                    time.sleep(15 * (k + 1))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(get, files))
        ok = local_usd(root, a).exists()
        print(f"{'ok ' if ok else 'MISSING'} {a['name']:<36} {time.time()-t:6.1f}s")


def gpu_mem() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        used, total, util = (int(x) for x in out.strip().splitlines()[0].split(","))
        return {"used_mib": used, "total_mib": total, "util_pct": util}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True, help="where packages are / will be downloaded")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--collider", choices=["convexHull", "convexDecomposition", "sdf", "asset-default"],
                    default="convexHull")
    ap.add_argument("--hull-vertex-limit", type=int, default=64)
    ap.add_argument("--gpu-dynamics", action="store_true", help="PhysX GPU pipeline (default: CPU, fine for 22 bodies)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--settle-steps", type=int, default=240, help="app updates with physics playing before frame 0")
    ap.add_argument("--steps-between", type=int, default=6)
    ap.add_argument("--rt-subframes", type=int, default=4)
    return ap.parse_args()


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = parse()
    if args.download_only:
        download(args.assets)
        return 0
    # pip route: OMNI_KIT_ACCEPT_EULA=YES; container route: -e ACCEPT_EULA=Y. Either must come from the runner.
    if not any(os.environ.get(v, "").strip().lower() in ("yes", "y", "1")
               for v in ("OMNI_KIT_ACCEPT_EULA", "ACCEPT_EULA")):
        sys.exit("Refusing to start: a person must read and accept the NVIDIA Isaac Sim EULA and then set "
                 "OMNI_KIT_ACCEPT_EULA=YES (pip) or pass -e ACCEPT_EULA=Y (container) themselves. "
                 "This script never accepts it.")

    metrics: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "gpu": {}, "t": {}, "assets": {}}
    t0 = time.time()
    metrics["gpu"]["before_app"] = gpu_mem()

    from isaacsim import SimulationApp  # must be created before any omni/pxr-dependent import
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    metrics["t"]["app_start_s"] = round(time.time() - t0, 1)

    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    try:
        from pxr import UsdSemantics
    except ImportError:
        UsdSemantics = None

    # ---------------------------------------------------------------- stage + room
    t = time.time()
    ctx = omni.usd.get_context()
    ctx.new_stage()
    app.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    if PhysxSchema:
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateEnableGPUDynamicsAttr(bool(args.gpu_dynamics))
        px.CreateBroadphaseTypeAttr("GPU" if args.gpu_dynamics else "MBP")

    def box(path, size, center, rgb):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        api = UsdGeom.XformCommonAPI(c)
        api.SetTranslate(Gf.Vec3d(*center))
        api.SetScale(Gf.Vec3f(*size))
        c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    box("/World/Room/floor", (6.0, 5.0, 0.1), (0, 0, -0.05), (0.55, 0.45, 0.35))
    box("/World/Room/wall_back", (6.0, 0.1, 2.6), (0, 2.55, 1.3), (0.85, 0.85, 0.82))
    box("/World/Room/wall_left", (0.1, 5.0, 2.6), (-3.05, 0, 1.3), (0.85, 0.85, 0.82))
    box("/World/Room/wall_right", (0.1, 5.0, 2.6), (3.05, 0, 1.3), (0.85, 0.85, 0.82))
    UsdLux.DomeLight.Define(stage, "/World/Lights/dome").CreateIntensityAttr(400.0)
    for i, (x, y) in enumerate([(-1.5, 0.5), (1.5, 0.5)]):
        s = UsdLux.SphereLight.Define(stage, f"/World/Lights/ceiling_{i}")
        s.CreateIntensityAttr(30000.0)
        s.CreateRadiusAttr(0.15)
        UsdGeom.XformCommonAPI(s).SetTranslate(Gf.Vec3d(x, y, 2.45))
    metrics["t"]["room_s"] = round(time.time() - t, 2)

    # ---------------------------------------------------------------- reference assets
    t = time.time()
    loaded: dict[str, dict] = {}
    for a in ASSETS:
        usd = local_usd(args.assets, a)
        if not usd.exists():
            print(f"[skip] {a['name']}: {usd} not found (run --download-only first)")
            continue
        meta_stage = Usd.Stage.Open(str(usd), Usd.Stage.LoadNone)
        up = UsdGeom.GetStageUpAxis(meta_stage)
        mpu = UsdGeom.GetStageMetersPerUnit(meta_stage) or 1.0
        sr = dict(meta_stage.GetRootLayer().customLayerData.get("SimReady_Metadata", {}))
        del meta_stage

        outer = f"/World/Assets/{safe(a['name'])}"
        UsdGeom.Xform.Define(stage, outer)
        inner = stage.DefinePrim(outer + "/asset", "Xform")
        inner.GetReferences().AddReference(str(usd))
        vsets = inner.GetVariantSets()
        wanted = [("PhysX", "Enabled"), ("physics", "physics")]
        if args.collider in ("convexHull", "sdf"):
            wanted.append(("collision_mode", args.collider))      # Allsides-style assets
        for vs_name, sel in wanted:
            if vs_name in vsets.GetNames() and sel in vsets.GetVariantSet(vs_name).GetVariantNames():
                vsets.GetVariantSet(vs_name).SetVariantSelection(sel)
        api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(outer))
        api.SetRotate(Gf.Vec3f(90.0 if up == UsdGeom.Tokens.y else 0.0, 0.0, float(a.get("yaw", 0))))
        if abs(mpu - 1.0) > 1e-6:
            api.SetScale(Gf.Vec3f(mpu, mpu, mpu))
        loaded[a["name"]] = {"cfg": a, "outer": outer, "inner": inner}
        metrics["assets"][a["name"]] = {"up": str(up), "metersPerUnit": mpu,
                                        "authored_mass_kg": sr.get("mass"), "qcode": sr.get("qcode")}

    # payload loading is asynchronous in Kit; pump until done
    for _ in range(2000):
        app.update()
        try:
            msg, done, total = ctx.get_stage_loading_status()
            if done >= total and not msg:
                break
        except Exception:  # noqa: BLE001
            break
    metrics["t"]["asset_load_s"] = round(time.time() - t, 1)
    metrics["gpu"]["after_load"] = gpu_mem()

    # ---------------------------------------------------------------- colliders, bodies, labels
    def deinstance(root):
        for _ in range(10):  # nested instances
            inst = [p for p in Usd.PrimRange(root) if p.IsInstance()]
            if not inst:
                return
            for p in inst:
                p.SetInstanceable(False)

    def fix_physics(root, static: bool) -> dict:
        info = {"colliders": 0, "rigid_bodies": 0}
        if args.collider != "asset-default":
            deinstance(root)
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                info["rigid_bodies"] += 1
                if static:
                    UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)
            if args.collider == "asset-default" or not p.HasAPI(UsdPhysics.CollisionAPI) or not p.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(args.collider)
            if PhysxSchema:
                if args.collider != "sdf" and p.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                    p.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
                if args.collider == "convexHull":
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(p).CreateHullVertexLimitAttr(args.hull_vertex_limit)
            info["colliders"] += 1
        if info["rigid_bodies"] == 0 and not static:  # asset shipped without a body: make it one
            UsdPhysics.RigidBodyAPI.Apply(root)
            info["rigid_bodies_added"] = 1
        return info

    def add_class_label(prim, cls: str):
        if UsdSemantics is not None:
            UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([cls])
            return
        try:
            from isaacsim.core.experimental.utils.semantics import add_labels
            add_labels(prim, labels=[cls], taxonomy="class")
        except Exception:  # noqa: BLE001
            rep.functional.modify.semantics(prim, {"class": cls}, mode="add")

    for name, rec in loaded.items():
        rec_info = fix_physics(rec["inner"], bool(rec["cfg"].get("static")))
        metrics["assets"][name].update(rec_info)
        add_class_label(stage.GetPrimAtPath(rec["outer"]), rec["cfg"]["cls"])

    # ---------------------------------------------------------------- placement from runtime bounds
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def bounds(path):
        bbc.Clear()
        r = bbc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return r.GetMin(), r.GetMax()

    def move_to(rec, x, y, z_bottom=None, z_top=None):
        lo, hi = bounds(rec["outer"])
        api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(rec["outer"]))
        tr = api.GetXformVectors(Usd.TimeCode.Default())[0]
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        dz = (z_bottom - lo[2]) if z_bottom is not None else (z_top - hi[2])
        api.SetTranslate(Gf.Vec3d(tr[0] + x - cx, tr[1] + y - cy, tr[2] + dz))

    supports = {}
    for name, rec in loaded.items():          # floor / wall items first
        cfg = rec["cfg"]
        if "on" in cfg:
            continue
        lo, hi = bounds(rec["outer"])
        if cfg.get("mount_top") and (hi[2] - lo[2]) < 0.6:   # looks wall-mounted
            move_to(rec, *cfg["xy"], z_top=cfg["mount_top"])
        else:
            move_to(rec, *cfg["xy"], z_bottom=0.002)
        supports[name] = bounds(rec["outer"])
        metrics["assets"][name]["size_m"] = [round(hi[i] - lo[i], 3) for i in range(3)]
    for name, rec in loaded.items():          # then items dropped onto supports
        cfg = rec["cfg"]
        if "on" not in cfg:
            continue
        if cfg["on"] not in supports:
            print(f"[skip] {name}: support {cfg['on']} not loaded")
            continue
        slo, shi = supports[cfg["on"]]
        sx, sy = (slo[0] + shi[0]) / 2 + cfg["xy"][0], (slo[1] + shi[1]) / 2 + cfg["xy"][1]
        move_to(rec, sx, sy, z_bottom=shi[2] + 0.03)   # small drop, physics settles it
    metrics["t"]["setup_s"] = round(time.time() - t0, 1)

    # ---------------------------------------------------------------- cameras + annotators
    def make_camera(name, pos, look):
        try:
            cam = rep.functional.create.camera(position=pos, look_at=look, parent="/World", name=name)
        except (AttributeError, TypeError):
            cam = rep.create.camera(position=pos, look_at=look, name=name)
        return str(cam.GetPath()) if hasattr(cam, "GetPath") else cam

    def annotator(name, **init):
        if hasattr(rep, "annotators") and hasattr(rep.annotators, "get"):
            try:
                return rep.annotators.get(name, init_params=init) if init else rep.annotators.get(name)
            except TypeError:
                return rep.annotators.get(name)
        return rep.AnnotatorRegistry.get_annotator(name, init_params=init or None)

    rigs = []
    for cname, pos, look in CAMERAS:
        rp = rep.create.render_product(make_camera(cname, pos, look), (args.width, args.height))
        ann = {"rgb": annotator("rgb"),
               "depth": annotator("distance_to_image_plane"),
               "instance": annotator("instance_segmentation_fast", colorize=False),
               "instance_id": annotator("instance_id_segmentation_fast", colorize=False)}
        for an in ann.values():
            an.attach(rp)
        rigs.append((cname, rp, ann))

    # ---------------------------------------------------------------- settle physics, capture
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None

    def save(cname, i, ann):
        stem = args.out / f"{cname}_{i:03d}"
        rgb = np.asarray(ann["rgb"].get_data())[..., :3]
        depth = np.asarray(ann["depth"].get_data(), dtype=np.float32)
        inst = ann["instance"].get_data()
        inst_id = ann["instance_id"].get_data()
        np.save(f"{stem}_depth.npy", depth)
        np.save(f"{stem}_instance.npy", np.asarray(inst["data"]))
        np.save(f"{stem}_instance_id.npy", np.asarray(inst_id["data"]))
        with open(f"{stem}_instance_info.json", "w") as f:
            json.dump({"instance": inst.get("info", {}), "instance_id": inst_id.get("info", {})}, f, default=str)
        if Image is not None:
            Image.fromarray(rgb.astype(np.uint8)).save(f"{stem}_rgb.png")
            d = np.nan_to_num(depth, posinf=0.0)
            Image.fromarray(np.clip(d * 1000.0, 0, 65535).astype(np.uint16)).save(f"{stem}_depth_mm.png")
            ids = np.asarray(inst["data"]).astype(np.uint32)
            pal = (np.stack([ids * 2654435761 >> 8, ids * 40503 >> 4, ids * 9973], -1) % 255).astype(np.uint8)
            Image.fromarray(pal).save(f"{stem}_instance_vis.png")
        else:
            np.save(f"{stem}_rgb.npy", rgb)

    def capture():
        try:
            rep.orchestrator.step(delta_time=0.0, rt_subframes=args.rt_subframes)
        except TypeError:
            rep.orchestrator.step(rt_subframes=args.rt_subframes)

    timeline = omni.timeline.get_timeline_interface()
    t = time.time()
    timeline.play()
    for _ in range(args.settle_steps):
        app.update()
    metrics["t"]["settle_s"] = round(time.time() - t, 1)
    metrics["gpu"]["after_settle"] = gpu_mem()

    frame_times = []
    for i in range(args.frames):
        for _ in range(args.steps_between):
            app.update()
        timeline.pause()
        tf = time.time()
        capture()
        for cname, _, ann in rigs:
            save(cname, i, ann)
        frame_times.append(round(time.time() - tf, 3))
        timeline.play()
    timeline.stop()
    metrics["t"]["frame_s"] = frame_times                      # frame 0 includes shader warm-up
    metrics["t"]["total_s"] = round(time.time() - t0, 1)
    metrics["gpu"]["after_render"] = gpu_mem()

    # where did dynamic items end up? (fell through / flew away = collider problem)
    for name, rec in loaded.items():
        lo, hi = bounds(rec["outer"])
        metrics["assets"][name]["final_min_z"] = round(lo[2], 3)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(json.dumps({k: metrics[k] for k in ("t", "gpu")}, indent=1))
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
