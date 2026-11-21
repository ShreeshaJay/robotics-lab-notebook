"""Cost rows for COSTS.md from the runs' own container clocks (results/raw/**/*.meta.json + lerobot run JSONs).

  python analysis/costs.py      -> prints markdown rows and the total of the metered GPU runs
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
GPU = {"NVIDIA L4": 0.000222, "NVIDIA A10": 0.000306, "NVIDIA A10G": 0.000306, "NVIDIA B200": 0.001736, "NVIDIA H200": 0.001261, "NVIDIA H100 80GB HBM3": 0.001097}
CPU_CORE, MEM_GIB = 0.0000131, 0.00000222
REQ = {"groot": (4, 24), "lerobot": (8, 32), "pear": (16, 64)}  # requested physical cores, GiB


def rate(gpu: str, app: str) -> float:
    c, m = REQ[app]
    return GPU.get(gpu, 0.0) + c * CPU_CORE + m * MEM_GIB


def rows() -> list[tuple]:
    out = []
    for p in sorted((RAW / "groot").glob("*.meta.json")):
        for s in json.loads(p.read_text()):
            out.append(("groot", p.stem.replace(".meta", ""), f"shard {s['shard']}", s["gpu"], s["container_s"]))
    for p in sorted((RAW / "pear").glob("*.meta.json")):
        s = json.loads(p.read_text())
        if "container_s" in s:
            out.append(("pear", p.stem.replace(".meta", ""), f"conc {s.get('conc')}", s.get("gpu") or "NVIDIA B200", s["container_s"]))
    for p in sorted((RAW / "lerobot").glob("*.json")):
        m = json.loads(p.read_text())["meta"]
        gpu = (m.get("timing") or {}).get("gpu", "NVIDIA L4")
        out.append(("lerobot", p.stem, m["mode"], gpu, m["container_s"]))
    return out


def main() -> None:
    total = 0.0
    for app, tag, part, gpu, sec in rows():
        usd = sec * rate(gpu, app)
        total += usd
        print(f"| t3-tut01-{app} | {tag} ({part}) | {gpu.replace('NVIDIA ', '')} | {sec:.0f} | {usd:.2f} |")
    print(f"metered total: ${total:.2f}")


if __name__ == "__main__":
    main()
