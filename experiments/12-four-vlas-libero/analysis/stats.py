"""Statistics for Tutorial 1 (standard library only): Wilson intervals, exact McNemar, paired bootstrap, power."""

from __future__ import annotations

import math
import random
from statistics import NormalDist

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar (binomial test on discordant pairs)."""
    assert len(a) == len(b)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)  # A succeeds, B fails
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    m = n10 + n01
    if m == 0:
        return {"a_only": n10, "b_only": n01, "p": 1.0}
    k = min(n10, n01)
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / 2**m
    return {"a_only": n10, "b_only": n01, "p": min(1.0, 2 * p)}


def paired_bootstrap(a: list[bool], b: list[bool], groups: list[int] | None = None, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """95% percentile CI of mean(a) - mean(b), resampling episodes (or whole groups, e.g. tasks) with replacement."""
    rng = random.Random(seed)
    idx = list(range(len(a)))
    if groups is None:
        units = [[i] for i in idx]
    else:
        by = {}
        for i, g in zip(idx, groups):
            by.setdefault(g, []).append(i)
        units = list(by.values())
    diffs = []
    for _ in range(n_boot):
        sample = [i for _ in units for i in rng.choice(units)]
        diffs.append(sum(a[i] - b[i] for i in sample) / len(sample))
    diffs.sort()
    obs = sum(x - y for x, y in zip(a, b)) / len(a)
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]
    p_two = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / n_boot
    return {"diff": obs, "ci95": [lo, hi], "p_boot": min(1.0, p_two), "unit": "episode" if groups is None else "task"}


def n_unpaired(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Episodes per model to detect p1 vs p2 with two independent binomial samples (two-sided z-test)."""
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def n_paired(p_a_only: float, p_b_only: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired episodes (same initial states) for McNemar, from the discordant-pair rates (Connor 1987)."""
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    pd = p_a_only + p_b_only
    delta = p_a_only - p_b_only
    return math.ceil((za * math.sqrt(pd) + zb * math.sqrt(pd - delta**2)) ** 2 / delta**2)


def power_unpaired(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    za = NormalDist().inv_cdf(1 - alpha / 2)
    se0 = math.sqrt(2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2) / n)
    se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return NormalDist().cdf((abs(p1 - p2) - za * se0) / se1)


if __name__ == "__main__":
    # Power calculator:  python analysis/stats.py 0.5 0.6 [alpha] [power] [discordant_rate]
    import sys

    p1, p2 = float(sys.argv[1]), float(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    pw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    print(f"unpaired: {n_unpaired(p1, p2, alpha, pw)} episodes per model to detect {p1:.2f} vs {p2:.2f}")
    if len(sys.argv) > 5:
        disc = float(sys.argv[5])
        d = abs(p2 - p1)
        print(f"paired (same init states, {disc:.0%} discordant): {n_paired((disc + d) / 2, (disc - d) / 2, alpha, pw)} episodes")
