#!/usr/bin/env python3
"""Draw a few `lerobot-annotate` bounding boxes onto the source frames (PNG) and dump subtasks per episode.

Usage (inside the lerobot venv): python draw_boxes.py <annotated_root> <pristine_root> <out_dir> <camera> <ep,ep,...>
Reads the rewritten parquet (language_persistent / language_events), decodes the matching frame from the
pristine copy with LeRobotDataset, and writes <out_dir>/ep<E>_t<T>.png plus <out_dir>/subtasks.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def rows(cell) -> list[dict]:
    if cell is None:
        return []
    try:
        return [dict(r) for r in list(cell)]
    except TypeError:
        return []


def main() -> None:
    ann_root, pristine_root, out_dir, camera, eps = sys.argv[1:6]
    episodes = [int(e) for e in eps.split(",")]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{ann_root}/data/chunk-*/file-*.parquet")))
    info = json.loads(Path(pristine_root, "meta", "info.json").read_text())
    ds = LeRobotDataset(repo_id="local/pristine", root=pristine_root, episodes=episodes)
    summary: dict = {}
    for ep in episodes:
        g = df[df["episode_index"] == ep].sort_values("frame_index")
        persistent = rows(g.iloc[0]["language_persistent"]) if "language_persistent" in g else []
        subtasks = sorted((r for r in persistent if r.get("style") == "subtask"), key=lambda r: r["timestamp"])
        # language_events rows are stored on the frame they belong to; the frame carries the timestamp.
        events = [{**e, "timestamp": float(t)} for t, cell in zip(g["timestamp"], g["language_events"])
                  for e in rows(cell)] if "language_events" in g else []
        vqa = [e for e in events if e.get("style") == "vqa"]
        boxes = []
        for i, e in enumerate(vqa):
            if e.get("role") != "assistant":
                continue
            try:
                ans = json.loads(e["content"])
            except Exception:
                continue
            if isinstance(ans, dict) and isinstance(ans.get("detections"), list) and ans["detections"]:
                q = next((u["content"] for u in vqa[:i][::-1] if u.get("role") == "user"
                          and u.get("timestamp") == e.get("timestamp") and u.get("camera") == e.get("camera")), "")
                boxes.append({"timestamp": e["timestamp"], "camera": e.get("camera"), "question": q, "answer": ans})
        summary[ep] = {
            "duration_s": round(float(g["timestamp"].max()), 2),
            "task": None,
            "subtasks": [{"t": round(float(r["timestamp"]), 2), "text": r.get("content")} for r in subtasks],
            "n_vqa_pairs": sum(1 for e in vqa if e.get("role") == "assistant"),
            "n_bbox_answers": len(boxes),
            "vqa_examples": [
                {"t": round(float(e["timestamp"]), 2), "role": e["role"], "content": e["content"][:300]} for e in vqa[:10]
            ],
        }
        # Draw up to 2 box answers per episode, spread out in time.
        picks = boxes[:: max(1, len(boxes) // 2)][:2]
        for b in picks:
            ts = float(b["timestamp"])
            row = g.iloc[(g["timestamp"] - ts).abs().argmin()]
            # Map the global frame index to the filtered dataset's position.
            pos = [int(x) for x in ds.hf_dataset["index"]].index(int(row["index"]))
            item = ds[pos]
            img = (item[b["camera"] or camera].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            im = Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2))
            d = ImageDraw.Draw(im)
            W, H = im.size
            for det in b["answer"]["detections"]:
                bb = det.get("bbox")
                if not (isinstance(bb, list) and len(bb) == 4):
                    continue
                x0, y0, x1, y1 = bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H
                d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3)
                d.text((x0 + 3, max(0, y0 - 12)), str(det.get("label", ""))[:40], fill=(255, 255, 0))
            active = [s for s in subtasks if float(s["timestamp"]) <= ts]
            caption = f"ep{ep} t={ts:.1f}s | subtask: {active[-1]['content'] if active else '-'}"
            d.rectangle([0, H - 18, W, H], fill=(0, 0, 0))
            d.text((4, H - 15), caption[:110], fill=(255, 255, 255))
            name = f"ep{ep}_t{ts:.1f}.png"
            im.save(out / name)
            b["png"] = name
        summary[ep]["drawn"] = [{k: b[k] for k in ("timestamp", "question", "answer", "png")} for b in picks]
    summary["fps"] = info.get("fps")
    (out / "subtasks.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: (len(v["subtasks"]), v["n_bbox_answers"]) for k, v in summary.items() if k != "fps"}))


if __name__ == "__main__":
    main()
