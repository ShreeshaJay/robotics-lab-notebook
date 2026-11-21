"""Executable acceptance gate: a JSON spec + checks over an audit result. Each check returns pass/fail and the
episodes it flags, so a vendor gets a list of what to fix, not just a red X."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from .audit import AuditResult, run_audit
from .profiles import ROOT, get_profile

SPEC_DIR = ROOT / "specs"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    flagged: list[int] = field(default_factory=list)


def load_spec(name: str) -> dict:
    return json.loads((SPEC_DIR / f"{name}.json").read_text())


def run_checks(res: AuditResult, spec: dict, info: dict | None = None) -> dict[str, Check]:
    df, s = res.episodes, res.summary
    out: dict[str, Check] = {}
    ids = lambda mask: [int(x) for x in df.episode_index[mask]]

    def add(name, mask, detail_ok, limit=0):
        fl = ids(mask)
        out[name] = Check(name, len(fl) <= limit, f"{len(fl)} episode(s) flagged (limit {limit}). {detail_ok}", fl)

    if info is not None:
        feats = info.get("features", {})
        bad = [k for k, shp in spec["required_features"].items() if k not in feats or list(feats[k].get("shape", [])) != shp]
        fps_ok = float(info.get("fps", -1)) == float(spec["fps"])
        out["schema"] = Check("schema", not bad and fps_ok, f"missing/mis-shaped: {bad}; fps ok: {fps_ok}")
    add("frame_index_contiguous", df.fi_noncontig > 0, "frame_index must be 0..n-1 in row order.")
    add("timestamps_regular", (df.ts_gaps > 0) | (df.ts_nonmono > 0), "No gap > 1.5/fps, strictly increasing.")
    add("meta_length_matches", df.meta_len_mismatch > 0, "meta/episodes length must equal data rows.")
    add("no_nan", df.has_nan > 0, "No NaN/inf in action or state.")
    add("dynamic_continuity", (df.jump_rate > spec["max_jump_rate"]) & (df.n_jumps >= spec.get("min_jumps", 1)),
        f"Speed spikes vs both neighbours (hidden dropped frames): flag if > {spec['max_jump_rate']} per moving frame "
        f"and at least {spec.get('min_jumps', 1)} spikes.")
    add("gripper_sign_convention", df.gripper_sign_corr < spec["min_gripper_sign_corr"],
        f"corr(gripper command, gripper state 0.3 s later) >= {spec['min_gripper_sign_corr']}.")
    lo, hi = np.array(spec["action_min"]), np.array(spec["action_max"])
    oob = df.apply(lambda r: bool(np.any(np.array(r.action_min) < lo - 1e-3) or np.any(np.array(r.action_max) > hi + 1e-3)), axis=1)
    add("action_bounds", oob, "Every action within the declared bounds.")
    hyg = s["hygiene"]
    allowed = set(spec.get("allowed_typo_words", []))
    bad_words = {w for w in hyg["typo_examples"] if w not in allowed}
    from .hygiene import normalize
    # in a near-duplicate group, the string with the most episodes is taken as canonical; the others are flagged
    n_by_norm = df.task.fillna("").map(normalize).value_counts()
    near_strings = set()
    for g in hyg["near_duplicate_examples"]:
        canon = max(g, key=lambda x: n_by_norm.get(x, 0))
        near_strings |= {x for x in g if x != canon}
    word = re.compile(r"[a-z]+")
    def bad_task(t):
        t = t or ""
        return (not t.strip()) or t != t.strip() or "  " in t or any(w in bad_words for w in word.findall(t.lower())) \
            or normalize(t) in near_strings
    add("task_labels_clean", df.task.map(bad_task),
        f"No empty/whitespace/duplicate/near-duplicate task strings, no typo words (found: {sorted(bad_words)[:12]}, "
        f"near-dup groups: {hyg['n_near_duplicate_groups']}, dup groups: {hyg['n_duplicate_groups_after_normalisation']}).")
    # the task table is judged as a whole, even for strings with no episode in the audited slice
    table_dirty = bool(bad_words) or hyg["n_near_duplicate_groups"] or hyg["n_duplicate_groups_after_normalisation"]         or hyg["n_empty"] or hyg["n_whitespace_issues"] or hyg["n_control_chars"]
    out["task_labels_clean"].passed = out["task_labels_clean"].passed and not table_dirty
    out["task_labels_clean"].detail += f" Task table clean: {not table_dirty}."
    tc = df.task_episodes_full if "task_episodes_full" in df else df.task_index.map(df.task_index.value_counts())
    add("task_min_episodes", tc < spec["min_task_episodes"],
        f"Every task has >= {spec['min_task_episodes']} episodes in the full dataset (counted from meta/episodes).")
    add("no_exact_duplicates", df.exact_dup_copy > 0, "No episode is a byte-level copy of another.")
    add("no_near_duplicates", df.near_dup_copy > 0, "No episode within RMS 0.02 std of another (and length within 10%).")
    if s["task_consistency"]["applicable"]:
        add("task_trajectory_consistency", df.task_knn_flag > 0,
            "Each episode's 5 nearest trajectories mostly share its task label.", limit=spec.get("max_task_knn_flags", 0))
    add("idle_fraction", df.idle_frac > spec["max_idle_frac"], f"Idle fraction per episode <= {spec['max_idle_frac']}.")
    add("episode_length", (df.duration_s < spec["min_episode_s"]) | (df.duration_s > spec["max_episode_s"]),
        f"Episode length within [{spec['min_episode_s']}, {spec['max_episode_s']}] s.")
    return out


def run_gate(dataset: str, root: str | Path | None = None, spec_name: str | None = None, log=print):
    spec = load_spec(spec_name or dataset.replace("_corrupt", ""))
    p = get_profile(dataset, root)
    res = run_audit(p, scales=M.Scales.from_json(spec["scales"]), log=log)
    info = json.loads((Path(p.root) / "meta" / "info.json").read_text())
    return res, run_checks(res, spec, info)
