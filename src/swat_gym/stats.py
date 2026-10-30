"""Statistics over overlapping weather windows (CLAUDE.md rule 7).

Consecutive window start years share calendar years, so windows are not independent: every
interval uses the effective sample size (ESS) of the window set, never sd / sqrt(n).
"""
from __future__ import annotations

import numpy as np


def overlap_fraction(a: int, b: int, length: int) -> float:
    """Share of calendar years two windows of ``length`` years have in common, in ``[0, 1]``."""
    ya, yb = set(range(a, a + length)), set(range(b, b + length))
    return len(ya & yb) / length


def overlap_matrix(starts: list[int], length: int) -> list[list[float]]:
    """Correlation proxy: fraction of shared calendar years."""
    return [[overlap_fraction(a, b, length) for b in starts] for a in starts]


def effective_n(starts: list[int], length: int) -> float:
    """ESS = n² / ΣΣ ρ_ij, with ρ_ij the share of calendar years windows i and j share."""
    n = len(starts)
    if n <= 1:
        return float(n)
    total = sum(sum(row) for row in overlap_matrix(starts, length))
    return float(n * n / total) if total > 0 else float(n)


def paired(a: list[dict], b: list[dict], *, n_boot: int = 10_000, seed: int = 0,
           window_len: int) -> dict:
    """Paired per-window difference a - b of each row's ``value``, matched by ``start_year``.
    Quote se_ess (windows overlap; rule 7).

    No ``se`` key on purpose; se_naive is kept only to show the ratio. ``ci95_boot`` is a
    moving-block bootstrap over windows in start-year order, block length round(n / ESS)
    (iid when windows don't overlap).
    """
    if [r["start_year"] for r in a] != [r["start_year"] for r in b]:
        raise ValueError("paired() needs the same windows in the same order")
    d = np.array([r["value"] for r in a]) - np.array([r["value"] for r in b])
    n = int(d.size)
    ess = effective_n([r["start_year"] for r in a], window_len) if n > 1 else float(n)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se_naive = sd / np.sqrt(n) if n > 1 else float("nan")
    se_ess = sd / np.sqrt(ess) if n > 1 else float("nan")
    block = max(1, min(n, int(round(n / ess)))) if n > 1 else 1
    if n > 1:
        ordered = d[np.argsort([r["start_year"] for r in a], kind="stable")]
        starts = np.random.default_rng(seed).integers(0, n - block + 1,
                                                      size=(n_boot, -(-n // block)))
        idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
        means = ordered[idx].mean(axis=1)
        ci_boot = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        ci_ess = [float(d.mean() - 1.96 * se_ess), float(d.mean() + 1.96 * se_ess)]
    else:
        ci_boot = ci_ess = [float("nan"), float("nan")]
    return {"mean": float(d.mean()), "n": n, "ess": float(ess), "sd": sd,
            "se_naive": float(se_naive), "se_ess": float(se_ess),
            "ci95_boot": ci_boot, "boot_block": block, "ci95_ess": ci_ess,
            "resolvable": bool(n > 1 and ci_ess[0] * ci_ess[1] > 0),
            "per_window": [round(float(v), 1) for v in d]}
