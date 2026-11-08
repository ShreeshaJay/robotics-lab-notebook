"""Print the physics + semantics an OpenUSD SimReady asset carries, without Isaac Sim.

Uses only `usd-core` (pip, pinned in requirements.txt). Opens the asset's entry .usd,
optionally selects variants (e.g. PhysX=Enabled), and lists for every prim:
  - applied API schemas (RigidBody, Collision, MeshCollision, Mass, Semantics, ...)
  - physics:approximation (convexHull / convexDecomposition / sdf / none ...)
  - mass / density / friction / restitution
  - semantic labels (Semantics API or UsdSemantics labels)
  - SimReady_Metadata on the root layer (license, Q-code, extents, mass)

Usage:
  python -I inspect_physics.py path/to/asset.usd [--variant PhysX=Enabled] [--variant collision_mode=convexHull]

Instanceable prims are walked through their prototypes so instanced colliders show up.
"""
from __future__ import annotations

import argparse
import json
import sys

from pxr import Usd

KEYS = (
    "physics:approximation", "physics:mass", "physics:density",
    "physics:staticFriction", "physics:dynamicFriction", "physics:restitution",
    "physics:collisionEnabled", "physics:rigidBodyEnabled",
    "physxCollision:contactOffset", "physxSDFMeshCollision:sdfResolution",
)


def walk(prim):
    yield prim
    if prim.IsInstance():
        proto = prim.GetPrototype()
        if proto:
            yield from walk(proto)
    for child in prim.GetChildren():
        yield from walk(child)


def describe(prim) -> dict | None:
    schemas = [s for s in prim.GetAppliedSchemas()
               if any(t in s for t in ("Physics", "Physx", "Mass", "Collision", "Semantic", "RigidBody", "Material"))]
    attrs = {}
    for a in prim.GetAttributes():
        n = a.GetName()
        if n in KEYS or n.startswith(("semantic:", "semantics:")) or "semantic" in n.lower():
            v = a.Get()
            if v is not None:
                attrs[n] = v if isinstance(v, (int, float, str, bool)) else str(v)
    if not schemas and not attrs:
        return None
    return {"path": str(prim.GetPath()), "type": prim.GetTypeName(), "schemas": schemas, "attrs": attrs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd")
    ap.add_argument("--variant", action="append", default=[], help="SetName=Selection on the default prim")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.usd)
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    vsets = root.GetVariantSets()
    print("variant sets on default prim:",
          {n: (vsets.GetVariantSet(n).GetVariantNames(), vsets.GetVariantSet(n).GetVariantSelection())
           for n in vsets.GetNames()})
    for v in args.variant:
        name, sel = v.split("=", 1)
        if vsets.HasVariantSet(name):
            vsets.GetVariantSet(name).SetVariantSelection(sel)
            print(f"selected {name}={sel}")
        else:
            print(f"(no variant set {name!r})")

    meta = stage.GetRootLayer().customLayerData.get("SimReady_Metadata", {})
    if meta:
        slim = {k: v for k, v in meta.items() if k not in ("validation", "Variants")}
        print("SimReady_Metadata:", json.dumps(slim, default=str, indent=1)[:2000])

    print(f"upAxis={stage.GetMetadata('upAxis')} metersPerUnit={stage.GetMetadata('metersPerUnit')}")
    for prim in walk(stage.GetPseudoRoot()):
        d = describe(prim)
        if d:
            print(json.dumps(d, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
