#!/usr/bin/env python3
"""Merge Flora's ~3 s action segments into FineART-style steps (~9 s) and compare both with EPIC narrations.

CPU only, standard library only. Usage:
  python merge_steps.py CAPTIONS.json EPIC_GT.json [--out steps.json]
  python merge_steps.py --batch results/more_clips/captions results/more_clips  # every <id>.json with a GT file

How the merge works (deliberately simple, all thresholds fixed before looking at the results):
  1. One timeline. Sort segments by start. A segment that starts inside the previous one is trimmed to start
     where the previous one ends; if almost nothing is left (< 0.3 s) it is folded into the previous segment.
  2. Parse each caption into clauses ("pick up glass with left hand, place glass in rack" -> two clauses),
     drop the hand phrases, and take a verb (first word, or a two-word phrasal verb such as "pick up") and an
     object (the last word before the first preposition). "pick up glass" -> ("pick up", "glass").
  3. Greedy left to right: the next segment joins the current step when it is adjacent (gap <= 1 s), the step
     would stay <= 15 s, and the two share
        a (verb, object) pair, or the object, or caption words (Jaccard >= 0.3 on content words).
  4. Clean-up: a step shorter than 4 s joins the neighbour it shares more words with, if that keeps it <= 15 s.
  5. Step text: the first clause of the step's longest segment, hand phrases removed, as one imperative.
  6. FineART format: start-only, contiguous, non-overlapping; a step lasts until the next one starts.

Comparison with EPIC narrations (human, atomic, ~3 s):
  - boundary error: for each internal step boundary, the distance in seconds to the nearest EPIC narration start
    or end. Reported for the raw Flora segments too.
  - matched human steps: an EPIC narration is matched when one step covers at least half of it and that step
    mentions the narration's verb or noun (a small synonym table maps e.g. tap -> faucet). "strict" looks only at
    the step text; "lenient" also looks at the Flora captions merged into the step.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

GAP_S, MAX_STEP_S, MIN_STEP_S, JACCARD, MIN_PIECE_S = 1.0, 15.0, 4.0, 0.3, 0.3

PHRASAL = {"pick", "put", "turn", "take", "wipe", "pull", "push", "throw", "set", "hang", "close", "open", "lift"}
PARTICLES = {"up", "down", "on", "off", "out", "away", "back", "over"}
PREPS = {"in", "into", "on", "onto", "from", "with", "to", "under", "inside", "at", "off", "of", "over", "across",
         "through", "near", "next", "behind", "toward", "towards", "along", "using", "for", "by", "against"}
STOP = PREPS | PARTICLES | {"the", "a", "an", "and", "both", "hand", "hands", "left", "right", "it", "its", "them",
                            "while", "then", "is", "are", "of"}
HAND = re.compile(r"\b(with|using|in|by)\s+(the\s+)?(left|right|both)\s+hands?\b|\b(left|right|both)\s+hands?\b")
# Small synonym table for matching EPIC nouns/verbs to Flora words (EPIC uses British kitchen words).
SYN = {
    "tap": {"faucet", "water", "tap"}, "cooker": {"stove", "stovetop", "cooktop", "hob", "cooker"},
    "hob": {"stove", "stovetop", "cooktop", "hob"}, "bin": {"trash", "bin", "garbage"},
    "cupboard": {"cabinet", "cupboard", "shelf"}, "cupboards": {"cabinet", "cupboard", "shelf"},
    "counter": {"counter", "countertop", "worktop"}, "top": {"counter", "countertop"},
    "sponge": {"sponge", "cloth", "scrubber"}, "cloth": {"cloth", "sponge", "towel", "rag"},
    "rubbish": {"trash", "debris", "waste", "rubbish"}, "dish": {"dish", "plate", "bowl"}, "plate": {"plate", "dish"},
    "block": {"board", "block"}, "chopping": {"board", "cutting"}, "pot": {"pot", "pan", "saucepan"},
    "lids": {"lid", "lids"}, "glass": {"glass", "cup"}, "mug": {"mug", "cup"},
    "wash": {"wash", "scrub", "clean", "rinse"}, "clean": {"clean", "wipe", "wash", "scrub"},
    "rinse": {"rinse", "wash"}, "wipe": {"wipe", "clean"}, "chop": {"chop", "cut", "slice", "dice"},
    "cut": {"cut", "chop", "slice"}, "take": {"take", "pick", "grab"}, "put": {"put", "place", "set"},
    "throw": {"throw", "discard", "drop", "place"}, "squeeze": {"squeeze", "wring"}, "open": {"open"},
    "close": {"close", "shut"}, "oven": {"oven", "microwave"}, "microwave": {"microwave", "oven"},
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def strip_hands(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", HAND.sub("", text)).strip(" ,.")
    return re.sub(r"(\s+(to|from|with|in|into|by|using|between))+(?=\s*,|\s*$)", "", text)  # "pass cloth to" -> "pass cloth"


def clauses(caption: str) -> list[tuple[str, str]]:
    """'pick up glass with left hand, place glass in rack' -> [('pick up','glass'), ('place','glass')]."""
    out = []
    for part in re.split(r",|;|\bthen\b", strip_hands(caption)):
        w = words(part)
        if not w:
            continue
        verb, rest = (w[0] + " " + w[1], w[2:]) if len(w) > 1 and w[0] in PHRASAL and w[1] in PARTICLES else (w[0], w[1:])
        obj = []
        for x in rest:
            if x in PREPS:
                break
            obj.append(x)
        obj = [x for x in obj if x not in STOP]
        out.append((verb, obj[-1] if obj else ""))
    return out


def content(caption: str) -> set[str]:
    return {x for x in words(strip_hands(caption)) if x not in STOP}


def one_timeline(segs: list[dict]) -> list[dict]:
    segs = sorted((dict(s) for s in segs if s.get("caption")), key=lambda s: (s["start_s"], -s["end_s"]))
    out: list[dict] = []
    for s in segs:
        s["captions"] = [s["caption"]]
        if out and s["start_s"] < out[-1]["end_s"]:
            if s["end_s"] - out[-1]["end_s"] < MIN_PIECE_S:
                out[-1]["captions"].append(s["caption"])  # contained duplicate: fold into the previous segment
                continue
            s["start_s"] = out[-1]["end_s"]
        out.append(s)
    return out


def related(a: dict, b: dict) -> bool:
    ca = {c for cap in a["captions"] for c in clauses(cap)}
    cb = {c for cap in b["captions"] for c in clauses(cap)}
    if ca & cb:
        return True
    if {o for _, o in ca if o} & {o for _, o in cb if o}:
        return True
    wa = set().union(*(content(c) for c in a["captions"]))
    wb = set().union(*(content(c) for c in b["captions"]))
    return bool(wa | wb) and len(wa & wb) / len(wa | wb) >= JACCARD


def overlap_words(a: dict, b: dict) -> int:
    return len(set().union(*(content(c) for c in a["captions"])) & set().union(*(content(c) for c in b["captions"])))


def merge(segs: list[dict]) -> list[dict]:
    steps: list[dict] = []
    for s in one_timeline(segs):
        piece = {"start_s": s["start_s"], "end_s": s["end_s"], "captions": list(s["captions"]),
                 "parts": [(s["end_s"] - s["start_s"], s["caption"])]}
        cur = steps[-1] if steps else None
        if (cur and s["start_s"] - cur["end_s"] <= GAP_S and s["end_s"] - cur["start_s"] <= MAX_STEP_S
                and related(cur, piece)):
            cur["end_s"] = s["end_s"]
            cur["captions"] += piece["captions"]
            cur["parts"] += piece["parts"]
        else:
            steps.append(piece)
    changed = True
    while changed:  # clean-up pass for very short steps
        changed = False
        for i, st in enumerate(steps):
            if st["end_s"] - st["start_s"] >= MIN_STEP_S:
                continue
            cands = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(steps):
                    lo, hi = min(steps[j]["start_s"], st["start_s"]), max(steps[j]["end_s"], st["end_s"])
                    if hi - lo <= MAX_STEP_S:
                        cands.append((overlap_words(st, steps[j]), -(hi - lo), j))
            if cands:
                _, _, j = max(cands)
                t = steps[j]
                t["start_s"], t["end_s"] = min(t["start_s"], st["start_s"]), max(t["end_s"], st["end_s"])
                t["captions"] = (st["captions"] + t["captions"]) if j > i else (t["captions"] + st["captions"])
                t["parts"] += st["parts"]
                del steps[i]
                changed = True
                break
    for st in steps:  # step text: the clause about the object that takes up most of the step
        weight: dict[str, float] = {}
        for dur, cap in st["parts"]:
            cl = clauses(cap)
            for _, obj in cl:
                if obj:
                    weight[obj] = weight.get(obj, 0.0) + dur / len(cl)
        main_obj = max(weight, key=weight.get) if weight else ""
        best = ""
        for dur, cap in sorted(st["parts"], reverse=True):
            for part in re.split(r",|;", strip_hands(cap)):
                if not main_obj or main_obj in words(part):
                    best = part.strip()
                    break
            if best:
                break
        st["content"] = best[:1].upper() + best[1:] + "."
        st.pop("parts")
    return steps


def fineart(steps: list[dict], duration: float) -> list[dict]:
    """Start-only, contiguous rows like FineART's language_persistent subtask entries."""
    return [{"role": "assistant", "style": "subtask", "timestamp": round(s["start_s"], 2), "content": s["content"],
             "camera": None, "tool_calls": None} for s in steps]


def lex(word: str) -> set[str]:
    out = {word, word.rstrip("s")}
    for w in list(out):
        out |= SYN.get(w, set())
    return out


def evaluate(units: list[dict], gt: list[dict], text_keys: tuple[str, ...]) -> dict:
    bounds = sorted({round(x, 2) for g in gt for x in (g["start_s"], g["end_s"])})
    internal = [u["start_s"] for u in units[1:]]
    errs = [min(abs(b - x) for x in bounds) for b in internal] if bounds else []
    matched = {}
    for mode, keys in (("strict", text_keys[:1]), ("lenient", text_keys)):
        hit = 0
        for g in gt:
            dur = max(g["end_s"] - g["start_s"], 1e-6)
            best = max(units, key=lambda u: min(u["end_s"], g["end_s"]) - max(u["start_s"], g["start_s"]))
            cover = (min(best["end_s"], g["end_s"]) - max(best["start_s"], g["start_s"])) / dur
            text = " ".join(" ".join(best[k]) if isinstance(best[k], list) else best[k] for k in keys)
            tw = set(words(text)) | {w.rstrip("s") for w in words(text)}
            want = set()
            for w in words(g["narration"]):
                if w not in STOP:
                    want |= lex(w)
            if cover >= 0.5 and want & tw:
                hit += 1
        matched[mode] = hit
    lens = [u["end_s"] - u["start_s"] for u in units]
    # Chance baseline: the same number of boundaries, evenly spaced. EPIC narrations are dense (a boundary every
    # ~1.5 s), so even arbitrary boundaries land close to one; the gap to this baseline is what matters.
    end = max(u["end_s"] for u in units)
    grid = [end * k / len(units) for k in range(1, len(units))]
    gerr = [min(abs(b - x) for x in bounds) for b in grid] if bounds and grid else []
    return {
        "units": len(units), "mean_len_s": round(statistics.mean(lens), 2), "median_len_s": round(statistics.median(lens), 2),
        "boundary_error_s": {"mean": round(statistics.mean(errs), 2) if errs else None,
                             "median": round(statistics.median(errs), 2) if errs else None,
                             "within_1s": round(sum(e <= 1.0 for e in errs) / len(errs), 2) if errs else None,
                             "even_grid_median": round(statistics.median(gerr), 2) if gerr else None},
        "human_steps": len(gt), "matched_strict": matched["strict"], "matched_lenient": matched["lenient"],
    }


def run_one(cap_path: Path, gt_path: Path, out: Path | None) -> dict:
    cap = json.loads(cap_path.read_text())
    gt = json.loads(gt_path.read_text())["narrations"]
    segs = [s for s in cap["segments"] if s.get("status") == "complete"]
    raw = [dict(s, captions=[s["caption"]]) for s in sorted(segs, key=lambda s: s["start_s"])]
    steps = merge(segs)
    res = {
        "video": cap_path.stem, "duration_s": cap["duration_s"], "params": {
            "gap_s": GAP_S, "max_step_s": MAX_STEP_S, "min_step_s": MIN_STEP_S, "jaccard": JACCARD},
        "raw_flora": evaluate(raw, gt, ("caption", "captions")),
        "merged": evaluate(steps, gt, ("content", "captions")),
        "steps": [{"start_s": round(s["start_s"], 2), "end_s": round(s["end_s"], 2), "content": s["content"],
                   "flora_captions": s["captions"]} for s in steps],
        "fineart_rows": fineart(steps, cap["duration_s"]),
    }
    if out:
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("captions", type=Path, help="Flora annotation JSON, or a folder of them with --batch")
    p.add_argument("gt", type=Path, help="EPIC ground-truth JSON, or the folder with epic_ground_truth_<id>.json")
    p.add_argument("--out", type=Path, help="steps JSON (single mode) or output folder (batch mode)")
    p.add_argument("--batch", action="store_true")
    a = p.parse_args()
    pairs = []
    if a.batch:
        for c in sorted(a.captions.glob("*.json")):
            g = a.gt / f"epic_ground_truth_{c.stem}.json"
            if g.exists():
                pairs.append((c, g, (a.out or a.gt) / f"steps_{c.stem}.json"))
    else:
        pairs.append((a.captions, a.gt, a.out))
    if not pairs:
        sys.exit("no caption/ground-truth pairs found")
    print("| clip | units | mean len s | boundary err median s (even grid) | within 1 s | human steps | matched strict | matched lenient |")
    print("|-|-|-|-|-|-|-|-|")
    for c, g, o in pairs:
        r = run_one(c, g, o)
        for kind in ("raw_flora", "merged"):
            e = r[kind]
            print(f"| {r['video']} {kind} | {e['units']} | {e['mean_len_s']} | {e['boundary_error_s']['median']} ({e['boundary_error_s']['even_grid_median']}) | "
                  f"{e['boundary_error_s']['within_1s']} | {e['human_steps']} | {e['matched_strict']} | {e['matched_lenient']} |")


if __name__ == "__main__":
    main()
