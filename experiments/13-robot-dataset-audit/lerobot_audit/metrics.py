"""Per-episode metrics. All motion metrics run on a common 10 Hz grid with per-dimension scales, so the
same thresholds mean the same thing on a 10 Hz sim dataset, a 15 Hz Franka dataset and a 30 Hz ALOHA dataset."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .load import Episode
from .profiles import Profile

HZ = 10.0                 # common analysis rate
IDLE_TAU = 0.05           # idle if every arm dim moves < 5% of its typical fast speed (p95) ...
GRIP_TAU = 0.02           # ... and gripper closure changes < 2% of its range per 0.1 s
IDLE_MIN_S = 0.5          # ... for at least 0.5 s
MOVE_TAU = 0.10           # frames counted as "moving" for jerk statistics
REGRASP_GAP_S = 2.5       # open -> close again within 2.5 s = a re-grasp (retry)
JUMP_RATIO = 1.7          # raw-rate speed spike vs both neighbours (hidden dropped frame signature)
JUMP_MIN = 0.3
FP_POINTS = 32            # trajectory fingerprint length


@dataclass
class Scales:
    vel: np.ndarray            # p95 |delta arm state| per 0.1 s, per arm dim
    pos_std: np.ndarray        # std of arm state, per arm dim
    act: np.ndarray            # p95 of |action velocity| per 0.1 s (delta: |a|), per arm action dim
    grip_cmd: list             # [(lo, hi)] per gripper, raw command units
    grip_state: list           # [(lo, hi)] per gripper, raw state units

    def to_json(self):
        return {"vel": self.vel.tolist(), "pos_std": self.pos_std.tolist(), "act": self.act.tolist(),
                "grip_cmd": self.grip_cmd, "grip_state": self.grip_state}

    @classmethod
    def from_json(cls, d):
        return cls(np.array(d["vel"]), np.array(d["pos_std"]), np.array(d["act"]),
                   [tuple(x) for x in d["grip_cmd"]], [tuple(x) for x in d["grip_state"]])


def unwrap_axis_angle(r: np.ndarray) -> np.ndarray:
    """Axis-angle vectors flip (theta -> theta - 2pi) near pi; pick the representation closest to the previous frame."""
    out = r.copy()
    for t in range(1, len(out)):
        th = np.linalg.norm(out[t])
        if th > 1e-6:
            alt = out[t] * (1.0 - 2.0 * np.pi / th)
            if np.linalg.norm(alt - out[t - 1]) < np.linalg.norm(out[t] - out[t - 1]):
                out[t] = alt
    return out


def arm_state(p: Profile, ep: Episode) -> np.ndarray:
    s = ep.state[:, p.arm_state_dims].copy()
    if p.axis_angle_state_dims:
        idx = [p.arm_state_dims.index(d) for d in p.axis_angle_state_dims]
        s[:, idx] = unwrap_axis_angle(s[:, idx])
    return s


def grip_state_raw(g, ep: Episode) -> np.ndarray:
    if len(g.state_dims) == 2:
        return ep.state[:, g.state_dims[0]] - ep.state[:, g.state_dims[1]]
    return ep.state[:, g.state_dims[0]]


def resample(t: np.ndarray, x: np.ndarray, hz: float = HZ) -> np.ndarray:
    t = t - t[0]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / hz)
    if x.ndim == 1:
        return np.interp(grid, t, x)
    return np.stack([np.interp(grid, t, x[:, d]) for d in range(x.shape[1])], axis=1)


def smooth3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x
    k = np.ones(3) / 3
    pad = np.pad(x, ((1, 1), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, d], k, mode="valid") for d in range(x.shape[1])], axis=1)


def compute_scales(p: Profile, eps: list[Episode]) -> Scales:
    dv, pos, av, gc, gs = [], [], [], [[] for _ in p.grippers], [[] for _ in p.grippers]
    for ep in eps:
        if ep.n < 4:
            continue
        s = resample(ep.t, arm_state(p, ep))
        dv.append(np.abs(np.diff(smooth3(s), axis=0)))
        pos.append(s)
        a = resample(ep.t, ep.action[:, p.arm_action_dims])
        av.append(np.abs(a) if p.action_kind == "delta" else np.abs(np.diff(a, axis=0)))
        for i, g in enumerate(p.grippers):
            gc[i].append(ep.action[:, g.action_dim])
            gs[i].append(grip_state_raw(g, ep))
    dv, pos, av = np.concatenate(dv), np.concatenate(pos), np.concatenate(av)
    q = lambda x, lo, hi: (float(np.percentile(x, lo)), float(np.percentile(x, hi)))
    return Scales(vel=np.maximum(np.percentile(dv, 95, axis=0), 1e-6), pos_std=np.maximum(pos.std(axis=0), 1e-6),
                  act=np.maximum(np.percentile(av, 95, axis=0), 1e-6),
                  grip_cmd=[q(np.concatenate(x), 1, 99) for x in gc], grip_state=[q(np.concatenate(x), 1, 99) for x in gs])


def _runs(mask: np.ndarray):
    """(start, end) of True runs, end exclusive."""
    if not len(mask):
        return []
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _norm01(x, lo, hi, closed_high):
    y = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    y = np.clip(y, 0, 1)
    return y if closed_high else 1 - y


def _binarize(c: np.ndarray, lo=0.35, hi=0.65) -> np.ndarray:
    out = np.zeros(len(c), dtype=bool)
    state = c[0] > 0.5
    for i, v in enumerate(c):
        if state and v < lo:
            state = False
        elif not state and v > hi:
            state = True
        out[i] = state
    return out


def sparc(speed: np.ndarray, fs: float = HZ, padlevel: int = 4, fc: float = 5.0, amp_th: float = 0.05) -> float:
    """Spectral arc length (Balasubramanian et al. 2015). Closer to 0 = smoother; about -1.6 for a minimum-jerk reach."""
    n = len(speed)
    nfft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mf = np.abs(np.fft.fft(speed, nfft))
    if mf.max() <= 0:
        return np.nan
    mf = mf / mf.max()
    sel = f <= fc
    f, mf = f[sel], mf[sel]
    inx = np.flatnonzero(mf >= amp_th)
    f, mf = f[inx[0]:inx[-1] + 1], mf[inx[0]:inx[-1] + 1]
    if len(f) < 2:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mf) ** 2)))


def fingerprint(p: Profile, ep: Episode, sc: Scales) -> np.ndarray:
    s = resample(ep.t, arm_state(p, ep)) / sc.pos_std
    feats = [s]
    for i, g in enumerate(p.grippers):
        lo, hi = sc.grip_cmd[i]
        feats.append(_norm01(resample(ep.t, ep.action[:, g.action_dim]), lo, hi, g.action_closed_high)[:, None])
    x = np.concatenate(feats, axis=1)
    u = np.linspace(0, len(x) - 1, FP_POINTS)
    return np.stack([np.interp(u, np.arange(len(x)), x[:, d]) for d in range(x.shape[1])], axis=1).ravel()


def episode_hash(ep: Episode) -> str:
    h = hashlib.sha1()
    h.update(np.round(ep.action, 5).astype(np.float32).tobytes())
    h.update(np.round(ep.state, 5).astype(np.float32).tobytes())
    return h.hexdigest()


def episode_metrics(p: Profile, ep: Episode, sc: Scales, fps: float) -> dict:
    r: dict = {"episode_index": ep.index, "task_index": ep.task_index, "n_frames": ep.n}
    # --- integrity (raw rate) ---
    dt = np.diff(ep.t)
    r["ts_gaps"] = int(np.sum(dt > 1.5 / fps))
    r["ts_nonmono"] = int(np.sum(dt <= 0))
    r["fi_noncontig"] = int(np.any(ep.frame_index != np.arange(ep.n)))
    r["has_nan"] = int(not (np.isfinite(ep.action).all() and np.isfinite(ep.state).all()))
    r["duration_s"] = float(ep.n / fps)
    r["success"] = ep.extra.get("success", None)
    r["action_min"] = np.round(ep.action.min(0), 4).tolist()
    r["action_max"] = np.round(ep.action.max(0), 4).tolist()
    s_raw = arm_state(p, ep)
    u = np.linalg.norm(np.diff(s_raw, axis=0) / (sc.vel * HZ / fps), axis=1) / np.sqrt(len(sc.vel))
    if len(u) >= 3:
        mid = u[1:-1]
        jump = (mid >= JUMP_MIN) & (mid >= JUMP_RATIO * u[:-2]) & (mid >= JUMP_RATIO * u[2:])
        moving = max(int(np.sum(u >= JUMP_MIN)), 1)
        r["n_jumps"] = int(jump.sum())
        r["jump_rate"] = float(jump.sum() / moving)
    else:
        r["n_jumps"], r["jump_rate"] = 0, 0.0
    if ep.n < 4 or ep.t[-1] - ep.t[0] < 0.4:
        r.update(idle_s=0.0, idle_frac=0.0, lead_idle_s=0.0, trail_idle_s=0.0, n_pauses=0)
        return r
    # --- motion on the 10 Hz grid ---
    s = resample(ep.t, s_raw)
    ds = np.diff(smooth3(s), axis=0) / sc.vel
    m = np.abs(ds).max(axis=1)
    closures, cmds = [], []
    for i, g in enumerate(p.grippers):
        lo, hi = sc.grip_state[i]
        closures.append(_norm01(resample(ep.t, grip_state_raw(g, ep)), lo, hi, g.state_closed_high))
        lo, hi = sc.grip_cmd[i]
        cmds.append(_norm01(resample(ep.t, ep.action[:, g.action_dim]), lo, hi, g.action_closed_high))
    gm = np.max([np.abs(np.diff(c)) for c in closures], axis=0) if closures else np.zeros_like(m)
    still = (m < IDLE_TAU) & (gm < GRIP_TAU)
    runs = [(a, b) for a, b in _runs(still) if (b - a) / HZ >= IDLE_MIN_S]
    idle = np.zeros(len(m), dtype=bool)
    for a, b in runs:
        idle[a:b] = True
    T = len(m)
    r["idle_s"] = float(idle.sum() / HZ)
    r["idle_frac"] = float(idle.mean())
    r["lead_idle_s"] = float((runs[0][1] - runs[0][0]) / HZ) if runs and runs[0][0] == 0 else 0.0
    r["trail_idle_s"] = float((runs[-1][1] - runs[-1][0]) / HZ) if runs and runs[-1][1] == T else 0.0
    r["n_pauses"] = int(sum(1 for a, b in runs if a > 0 and b < T))
    if p.eef_pos and "eef_pos" in ep.extra:
        e = resample(ep.t, ep.extra["eef_pos"])
        sp = np.linalg.norm(np.diff(e, axis=0), axis=1) * HZ
        r["eef_idle_frac_1cm"] = float(np.mean(sp < 0.01))
        r["eef_path_m"] = float(np.sum(sp) / HZ)
    # --- smoothness ---
    mov = m >= MOVE_TAU
    v = np.diff(s, axis=0) / sc.vel
    jerk = np.linalg.norm(np.diff(v, n=2, axis=0), axis=1) / np.sqrt(v.shape[1])
    mj = mov[1:-1] if len(mov) > 2 else np.zeros(0, bool)
    r["state_jerk"] = float(np.sqrt(np.mean(jerk[mj[:len(jerk)]] ** 2))) if mj.any() else np.nan
    a = resample(ep.t, ep.action[:, p.arm_action_dims])
    av = (a if p.action_kind == "delta" else np.diff(a, axis=0)) / sc.act
    aj = np.linalg.norm(np.diff(av, n=2, axis=0), axis=1) / np.sqrt(av.shape[1])
    k = min(len(aj), len(mj))
    r["action_jerk"] = float(np.sqrt(np.mean(aj[:k][mj[:k]] ** 2))) if mj[:k].any() else np.nan
    # --- gripper events and retries ---
    n_close = n_open = n_regrasp = n_empty = 0
    corr = []
    seg_cuts = set()
    for i, (c, g) in enumerate(zip(cmds, closures)):
        b = _binarize(c)
        ch = np.flatnonzero(np.diff(b.astype(np.int8)))
        closes = [j + 1 for j in ch if b[j + 1]]
        opens = [j + 1 for j in ch if not b[j + 1]]
        n_close += len(closes)
        n_open += len(opens)
        seg_cuts.update(closes + opens)
        for cj in closes:
            prev_open = [o for o in opens if o < cj]
            if prev_open and any(cl < prev_open[-1] for cl in closes) and (cj - prev_open[-1]) / HZ <= REGRASP_GAP_S:
                n_regrasp += 1
            nxt = [o for o in opens if o > cj]
            end = nxt[0] if nxt else len(g)
            if end - cj >= 5 and g[min(cj + 5, end - 1):end].max() >= 0.95:
                n_empty += 1
        if c.std() > 0.05 and g.std() > 0.02 and len(c) > 6:
            corr.append(float(np.corrcoef(c[:-3], g[3:])[0, 1]))
    r["n_close"], r["n_open"], r["n_regrasp"], r["n_empty_close"] = n_close, n_open, n_regrasp, n_empty
    r["gripper_active"] = int(len(corr) > 0)
    r["gripper_sign_corr"] = float(min(corr)) if corr else np.nan
    r["closes_per_min"] = float(n_close / max(T / HZ / 60, 1e-9))
    # sub-movement SPARC: split at idle runs and gripper events, keep moving pieces >= 1 s
    speed = np.linalg.norm(v, axis=1) / np.sqrt(v.shape[1]) * HZ
    cuts = sorted(seg_cuts | {a for a, _ in runs} | {b for _, b in runs} | {0, len(speed)})
    vals = []
    for a0, b0 in zip(cuts[:-1], cuts[1:]):
        seg = speed[a0:b0]
        if len(seg) >= 10 and not idle[a0:b0].all():
            vals.append(sparc(seg))
    r["sparc"] = float(np.median(vals)) if vals else np.nan
    r["n_segments"] = len(vals)
    return r
