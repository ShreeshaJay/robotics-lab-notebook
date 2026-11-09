"""Survey SimReady Central's catalog for household coverage.

Metadata only. Nothing here downloads USD geometry or textures.

Inputs (fetched with --fetch into DATA_DIR, outside this notebook folder):
  install_categories.json  Space endpoint: every package in the catalog -> category path
                           https://nvidia-simready-central.hf.space/install_categories.json
  package_sizes.json       Space repo file: root-USD URL -> package bytes (snapshot 2026-10-16)
  palatial_catalog.jsonl   PalatialSim/palatial-nv-asset-library data/catalog.jsonl
                           (Palatial folders are opaque IDs like D00001; this maps them to names,
                            class labels, mass and collider counts)

Usage:
  python -I analyze_catalog.py --data-dir C:/path/to/data --fetch      # once
  python -I analyze_catalog.py --data-dir C:/path/to/data              # analysis -> stdout + coverage.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

SPACE_ID = "nvidia/simready-central"
SPACE_REV = "e355a49f6f5cc0dbd8e0637d702b0f78c996cb6d"        # Space commit used for these notes
PALATIAL_ID = "PalatialSim/palatial-nv-asset-library"
PALATIAL_REV = "7455e87ba9bdf264d3917d88a6bdc57c7e2028b4"     # dataset commit used for these notes
INSTALL_URL = "https://nvidia-simready-central.hf.space/install_categories.json"  # live; not pinnable

# household reset checklist -> name tokens that count as a hit.
# Tokens are matched against names split on _ - space and camelCase (so "toiletPaperHolder" ->
# toilet, paper, holder). "all" means every token in the tuple must appear.
CHECKLIST: dict[str, dict] = {
    # bedroom
    "bed (frame)":            {"any": ["bed"], "exclude": ["light", "unit", "head"]},
    "mattress":               {"any": ["mattress"]},
    "pillow / cushion":       {"any": ["pillow", "cushion"]},
    "bed linens / bedding":   {"any": ["bedding", "sheet", "duvet", "blanket", "linen", "comforter", "quilt"]},
    "towel":                  {"any": ["towel", "dishcloth"]},
    "nightstand / bedside":   {"any": ["bedside", "nightstand"]},
    "wardrobe / dresser":     {"any": ["wardrobe", "dresser", "drawers", "closet"]},
    "lamp":                   {"any": ["lamp"]},
    "clothes hanger":         {"any": ["hanger"], "exclude": ["banana", "fruit"]},
    "curtain / blinds":       {"any": ["curtain", "blinds"]},
    "rug / mat":              {"any": ["rug", "mat", "carpet"]},
    # living
    "sofa":                   {"any": ["sofa", "couch"]},
    "chair / stool":          {"any": ["chair", "stool"]},
    "table / desk":           {"any": ["table", "desk"], "exclude": ["lamp", "phone", "jug", "mat", "xray", "bucky", "tennis", "surgical"]},
    "tv remote":              {"any": ["remote"]},
    # kitchen
    "cup / mug / glass":      {"any": ["cup", "mug", "glass", "tumbler"]},
    "plate":                  {"any": ["plate", "bigplate", "smallplate"]},
    "bowl":                   {"any": ["bowl"]},
    "cutlery (fork/knife/spoon)": {"any": ["fork", "knife", "spoon", "cutlery"]},
    "pot / pan":              {"any": ["pot", "pan", "skillet", "saucepan", "cookware"]},
    "kettle":                 {"any": ["kettle"]},
    "dish rack":              {"any": ["rack"], "also": ["dish", "drying"]},
    # bathroom
    "toilet":                 {"any": ["toilet"], "exclude": ["holder", "paper"]},
    "sink / basin":           {"any": ["sink", "basin", "washbasin"]},
    "soap (bar/dispenser/dish)": {"any": ["soap", "barsoap", "dishsoap", "soapdispenser"]},
    "toilet paper (roll)":    {"all": ["toilet", "paper"], "exclude_all": ["holder"]},
    "toilet paper holder":    {"all": ["toilet", "paper", "holder"]},
    "toiletries (shampoo, toothbrush, ...)": {"any": ["shampoo", "conditioner", "toothbrush", "toothpaste", "lotion", "razor", "deodorant"]},
    "shower / bathtub":       {"any": ["shower", "bathtub"]},
    "mirror":                 {"any": ["mirror"]},
    # cleaning
    "spray bottle / disinfectant": {"any": ["spray", "windowspray", "disinfectant"]},
    "trash bin":              {"any": ["trash", "bin", "garbage", "wastebasket"], "exclude": ["library", "book", "potato"]},
    "vacuum":                 {"any": ["vacuum"], "exclude": ["wine", "stopper", "sealer"]},
    "mop / bucket":           {"any": ["mop", "bucket", "pail"], "exclude": ["champagne", "ice"]},
    "sponge / cloth / brush": {"any": ["sponge", "cloth", "brush", "wipes"]},
    "paper towel roll":       {"all": ["paper", "towel"]},
    "laundry basket":         {"any": ["laundry", "hamper"], "also": ["basket", "hamper"]},
}

# Categories whose items are clearly not residential (keeps "bin", "rack", "table" honest).
NON_HOME_PREFIXES = ("Industrial/", "OEM/Vertiv", "OEM/Trane", "OEM/FANUC", "OEM/Comau",
                     "OEM/Franka", "OEM/OnRobot", "OEM/Robotiq", "OEM/Standard Bots",
                     "OEM/Universal Robots", "OEM/AgiBot", "OEM/Menlo", "OEM/NVIDIA")


def tokens(name: str) -> set[str]:
    name = re.sub(r"_[A-Z0-9]{10}$", "", name)                   # Allsides item IDs
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)             # camelCase
    toks = {t.lower() for t in re.split(r"[\s_\-()]+", name) if t}
    toks |= {t[:-1] for t in toks if t.endswith("s") and len(t) > 3}  # crude plural
    return toks


def matches(rule: dict, toks: set[str]) -> bool:
    if "exclude" in rule and toks & set(rule["exclude"]):
        return False
    if "exclude_all" in rule and set(rule["exclude_all"]) <= toks:
        return False
    if "all" in rule:
        return set(rule["all"]) <= toks
    if not toks & set(rule["any"]):
        return False
    return not ("also" in rule and not toks & set(rule["also"]))


def fetch(data: Path) -> None:
    from huggingface_hub import hf_hub_download
    data.mkdir(parents=True, exist_ok=True)
    print("GET", INSTALL_URL)
    with urllib.request.urlopen(INSTALL_URL, timeout=120) as r, open(data / "install_categories.json", "wb") as f:
        shutil.copyfileobj(r, f)
    p = hf_hub_download(SPACE_ID, "services/explorer/dist/package_sizes.json", repo_type="space",
                        revision=SPACE_REV)
    shutil.copy(p, data / "package_sizes.json")
    p = hf_hub_download(PALATIAL_ID, "data/catalog.jsonl", repo_type="dataset", revision=PALATIAL_REV)
    shutil.copy(p, data / "palatial_catalog.jsonl")
    print("fetched into", data)


def load(data: Path):
    inst = json.loads((data / "install_categories.json").read_text(encoding="utf-8"))
    pal = {}
    with open(data / "palatial_catalog.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pal[r["asset_id"]] = r
    sizes = json.loads((data / "package_sizes.json").read_text(encoding="utf-8"))["sizes"]
    return inst, pal, sizes


def build_rows(inst: dict, pal: dict) -> list[dict]:
    rows = []
    for key, v in inst["assets"].items():
        name, label, extra = v["asset_path"], "", {}
        if v["repository"].startswith("PalatialSim/") and v["asset_path"] in pal:
            p = pal[v["asset_path"]]
            name, label = p["name"], p.get("class_label") or ""
            extra = {"mass_kg": p.get("total_mass_kg"), "collider_prims": p.get("collider_prims"),
                     "physics_kind": p.get("physics_kind")}
        rows.append({"key": key, "repo": v["repository"], "asset": v["asset_path"], "name": name,
                     "class_label": label, "category": v["category_path"] or "(not indexed yet)",
                     "status": v["status"], **extra})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--csv", type=Path, default=HERE / "coverage.csv")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.data_dir)
    inst, pal, sizes = load(args.data_dir)
    rows = build_rows(inst, pal)

    print(f"# catalog generated_at {inst['generated_at']}  packages={len(rows)}")
    print("coverage block:", json.dumps({k: inst['coverage'][k] for k in
          ('assets', 'mapped_assets', 'needs_review_assets', 'assets_with_qcode')}))
    print("\n## packages by dataset")
    for repo, n in collections.Counter(r["repo"] for r in rows).most_common():
        print(f"| {repo} | {n} |")

    print("\n## packages by top-level category")
    top = collections.Counter(r["category"].split("/")[0] for r in rows)
    for k, n in top.most_common():
        print(f"| {k} | {n} |")

    print("\n## Residential sub-categories")
    for k, n in sorted(collections.Counter(r["category"] for r in rows
                                           if r["category"].startswith("Residential")).items()):
        print(f"| {k} | {n} |")

    pal_in = [r for r in rows if r["repo"].startswith("PalatialSim/")]
    print(f"\n## Palatial ({len(pal_in)} in catalog): top class labels")
    for k, n in collections.Counter(r["class_label"] for r in pal_in).most_common(15):
        print(f"| {k} | {n} |")

    # reset checklist
    print("\n## household-reset checklist (home-relevant hits; industrial categories excluded)")
    out = []
    for item, rule in CHECKLIST.items():
        hits = [r for r in rows
                if not r["category"].startswith(NON_HOME_PREFIXES)
                and (matches(rule, tokens(r["name"])) or (r["class_label"] and matches(rule, tokens(r["class_label"]))))]
        by_repo = collections.Counter(r["repo"].split("/")[0] for r in hits)
        examples = sorted({r["name"] for r in hits})[:8]
        print(f"| {item} | {len(hits)} | {dict(by_repo)} | {', '.join(examples)} |")
        out.append({"item": item, "hits": len(hits), "by_source": json.dumps(dict(by_repo)),
                    "examples": "; ".join(examples)})
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "hits", "by_source", "examples"])
        w.writeheader()
        w.writerows(out)
    print("\nwrote", args.csv)

    # package size picture (snapshot of 2,597 assets on 2026-10-16)
    mb = sorted(b / 1e6 for b in sizes.values())
    print(f"\n## package sizes: n={len(mb)} median={mb[len(mb)//2]:.1f} MB p90={mb[int(len(mb)*.9)]:.0f} MB "
          f"total={sum(mb)/1e3:.0f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
