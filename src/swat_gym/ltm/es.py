"""OpenAI-ES (Salimans et al. 2017, arXiv:1703.03864) for LTM policies: the legacy crop benchmark's
ES (legacy/src/swat_gym/experiments/crops_ga.py) with the LTM episode, observation and score.

Chosen over PPO and the GA on that benchmark (runs/bench_*_xl.json, 9 crop x lever
cases, 4 seeds, 60k candidate-seasons): never resolvably worse than PPO, resolvably better on
3 of 4 N cases (best-of-runs picked on train, ci95_boot excludes 0; wheat N identical for every
method), unresolved on all 5 I cases.

Policy: OBS_DIM -> 64 tanh -> 64 tanh -> A (A = 6 crop logits for R, else 1 residual), the SB3
actor initialisation (output gain 0.01), so the zero genome replays measured practice.
Each generation plays theta +- sigma*eps (mirrored, P/2 pairs) on the same W train windows;
fitness = mean strategy score; gradient from centred ranks drives Adam with L2 decay toward
the zero genome. The centre is snapshotted at ``curve_points``; the final pick is the snapshot
with the best mean score on all train windows (budgeted). Budget unit: HRU-episode steps
simulated (LtmRunner.n_hru_runs), the unit CMA-ES is matched in.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..engine import EngineError
from ..experiments.common import MAX_FAILURE_FRAC, MIN_FAILURES
from .env import OBS_DIM, run_policy
from .econ import LtmPrices
from .practice import CROPS, decisions
from .runner import LtmRunner
from .score import Reference, episode
from .windows import TEST, TRAIN

#: version 2 (2026-10-28): every train window each generation (was 3 drawn at random: ES
#: optimised a noisy subset of the objective CMA-ES sees exactly, and its curves drifted), and
#: tied fitness gets averaged ranks (v1 ranked ties by list order: a population tied on a
#: plateau, e.g. Exp 2 applying no N at all, then took full Adam steps along a fake gradient
#: and never left it; all 4 seeds ended identical).
ES = {"version": 2, "population": 16, "windows": len(TRAIN), "sigma": 0.02, "hidden": [64, 64],
      "lr": 0.01, "adam": [0.9, 0.999], "l2": 0.005, "mirrored": True,
      "shaping": "centered ranks in [-0.5, 0.5], ties averaged", "curve_points": 8,
      "final": "best snapshot centre by mean train score",
      "init": "orthogonal, gain sqrt(2) hidden / 0.01 output (SB3 actor)"}


def _orthogonal(rng, rows: int, cols: int, gain: float) -> np.ndarray:
    """Orthogonal init (as SB3/PyTorch): a QR of a Gaussian matrix, sign-fixed, times gain."""
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    q = q if rows >= cols else q.T
    return gain * q[:rows, :cols]


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of x scaled to [-0.5, 0.5], ties sharing their mean rank, so a tied population
    gives no gradient (only the L2 decay toward the zero genome = measured practice acts).
    (The legacy crop benchmark's version broke ties by order.)"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                      # average the ranks within each tie
        tie = x == v
        r[tie] = r[tie].mean()
    return r / (len(x) - 1) - 0.5


def out_dim(lever: str) -> int:
    return len(CROPS) if lever == "R" else 1


def shapes(lever: str) -> list[tuple[int, int]]:
    sizes = [OBS_DIM, *ES["hidden"], out_dim(lever)]
    return list(zip(sizes[:-1], sizes[1:]))


def n_params(lever: str) -> int:
    return sum(i * o + o for i, o in shapes(lever))


def init_genome(lever: str, rng) -> np.ndarray:
    parts, sh = [], shapes(lever)
    for j, (i, o) in enumerate(sh):
        gain = 0.01 if j == len(sh) - 1 else np.sqrt(2.0)
        parts += [_orthogonal(rng, i, o, gain).ravel(), np.zeros(o)]
    return np.concatenate(parts)


def forward(lever: str, genomes: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Raw outputs [P, A] for P genomes on P observations."""
    h, at = obs.astype(float), 0
    sh = shapes(lever)
    for j, (i, o) in enumerate(sh):
        w = genomes[:, at:at + i * o].reshape(-1, i, o)
        at += i * o
        b = genomes[:, at:at + o]
        at += o
        h = np.einsum("pi,pio->po", h, w) + b
        if j < len(sh) - 1:
            h = np.tanh(h)
    return h


def evaluate(lever: str, genomes: np.ndarray, runner: LtmRunner, start: int,
             prices: LtmPrices) -> list[dict]:
    """Episode dicts of every genome on one window (closed loop)."""
    plans, outs, xs = run_policy(lambda obs: forward(lever, genomes, obs), lever, runner,
                                 len(genomes), start)
    return [{**episode(p, o, prices, start), "x": x.tolist()}
            for p, o, x in zip(plans, outs, xs)]


def scores(lever, genomes, runner, windows, prices, ref: Reference, w: float) -> np.ndarray:
    """scores[genome, window]."""
    cols = [[ref.score(e, w) for e in evaluate(lever, genomes, runner, s, prices)]
            for s in windows]
    return np.array(cols).T


#: Engine crashes (exp 3, 2026-10-26: one SWAT+ run exited -5 after ~9 h and took the whole
#: experiment down). A crashed batch is re-run one genome at a time; a genome that still
#: crashes scores FAILED_SCORE (ranked last) for that generation. Abort past the same limits as
#: experiments.common.minimise. Without a crash the path, and so the result, is unchanged.
FAILED_SCORE = -1e30
CRASH_DIR = Path(__file__).resolve().parents[3] / "runs" / "ltm_crash"
CRASH_KEEP = 5


def _keep_crash(err: EngineError, tally: dict) -> None:
    """Copy the crashed run's directory (inputs as rendered) to runs/ltm_crash/."""
    m = re.search(r" in (\S+)", str(err))
    if not m or tally["kept"] >= CRASH_KEEP:
        return
    dst = CRASH_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}_{tally['kept']}"
    try:
        shutil.copytree(m.group(1), dst)
        (dst / "error.txt").write_text(str(err))
        tally["kept"] += 1
    except OSError:
        pass


def safe_scores(lever, genomes, runner, windows, prices, ref, w, tally: dict) -> np.ndarray:
    """``scores`` that survives engine crashes (counted in ``tally``)."""
    try:
        return scores(lever, genomes, runner, windows, prices, ref, w)
    except EngineError as e:
        _keep_crash(e, tally)
        tally["crashed_batches"] += 1
        tally["first_error"] = tally["first_error"] or str(e)[:2000]
    out = np.empty((len(genomes), len(windows)))
    for i in range(len(genomes)):
        try:
            out[i] = scores(lever, genomes[i:i + 1], runner, windows, prices, ref, w)[0]
        except EngineError as e:
            _keep_crash(e, tally)
            out[i] = FAILED_SCORE
            tally["failed"] += 1
    if tally["failed"] >= max(MIN_FAILURES, MAX_FAILURE_FRAC * tally["planned"]):
        raise RuntimeError(f"ES: {tally['failed']} failed genome evaluations (limit "
                           f"{MAX_FAILURE_FRAC:.0%} of {tally['planned']}); first:\n"
                           f"{tally['first_error']}")
    return out


def es_seed(args) -> dict:
    """Pool job: one ES run of ``budget`` HRU-episode steps for one lever and strategy."""
    lever, w, prices, ref_key, budget, seed = args
    ref = Reference.from_key(ref_key)
    rng = np.random.default_rng(seed)
    P, W, sig = ES["population"], ES["windows"], ES["sigma"]
    T = len(decisions(lever))
    gen_cost = P * W * T
    n_snap = ES["curve_points"]
    final_cost = n_snap * len(TRAIN) * T
    n_gens = max(1, (budget - final_cost) // gen_cost)
    snap_at = set(np.linspace(n_gens / n_snap, n_gens, n_snap).round().astype(int))
    theta = init_genome(lever, rng)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2 = ES["adam"]
    log, snaps, curve = [], [], []
    tally = {"planned": n_gens * P, "failed": 0, "crashed_batches": 0, "kept": 0,
             "first_error": None}
    batched = "mixed" if lever == "R" else True   # alfalfa plans run alone (runner docstring)
    t0 = time.time()
    # R: discrete plans repeat across the population and generations -> replay cached outputs
    cache = 1024 if lever == "R" else 0
    with LtmRunner(P, batched=batched, cache=cache) as run_r, \
            LtmRunner(1, batched=True, cache=cache) as test_r:
        for gen in range(1, n_gens + 1):
            eps = rng.standard_normal((P // 2, theta.size))
            pop = np.concatenate([theta + sig * eps, theta - sig * eps])
            wins = TRAIN if W == len(TRAIN) else rng.choice(TRAIN, size=W, replace=False)
            fit = safe_scores(lever, pop, run_r, wins, prices, ref, w, tally).mean(1)
            u = centered_ranks(fit)
            g = ((u[:P // 2] - u[P // 2:]) @ eps) / (P * sig) - ES["l2"] * theta
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            theta = theta + ES["lr"] * (m / (1 - b1 ** gen)) / (np.sqrt(v / (1 - b2 ** gen)) + 1e-8)
            log.append({"generation": gen, "hru_runs": int(run_r.n_hru_runs),
                        "fit_median": float(np.median(fit)), "fit_max": float(fit.max())})
            if gen in snap_at:
                snaps.append(theta.copy())
                curve.append({"generation": gen, "hru_runs": int(run_r.n_hru_runs)})
        cand = np.stack(snaps)
        full = safe_scores(lever, cand, run_r, TRAIN, prices, ref, w, tally).mean(1)
        best = int(np.argmax(full))
        spent = int(run_r.n_hru_runs)
        genome = cand[best]
        train_eps = [evaluate(lever, genome[None], test_r, s, prices)[0] for s in TRAIN]
        test_eps = [evaluate(lever, genome[None], test_r, s, prices)[0] for s in TEST]
        cached = int(run_r.n_cached)
    return {"seed": seed, "generations": int(n_gens), "hru_runs": spent, "hru_runs_cached": cached,
            "train_seconds": time.time() - t0, "fitness": log, "curve": curve,
            "picked_snapshot": best, "picked_train_score": float(full[best]),
            "train": train_eps, "test": test_eps, "genome": genome.tolist(),
            "engine_failures": {k: tally[k] for k in ("failed", "crashed_batches",
                                                      "first_error")}}
