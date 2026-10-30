"""Experiment machinery shared by the drivers: parallel CMA-ES and reusable stage files.

A stage (one ES seed, one CMA-ES search, a reference run, ...) is saved as
``{"config": ..., "result": ...}`` and reused only while its config matches exactly, so a
changed model, price, setting or code version recomputes it instead of mixing results.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np


# -- CMA-ES ---------------------------------------------------------------------------

#: Abort a search once this share of evaluations has raised (and at least MIN_FAILURES).
MAX_FAILURE_FRAC = 0.01
MIN_FAILURES = 3
#: Objective assigned to a failed evaluation so CMA-ES ranks it last.
FAILED = 1e30


def _guarded(task):
    """Pool task: (fn, x, window) -> (value, traceback or None)."""
    fn, x, w = task
    try:
        return float(fn(x) if w is None else fn(x, w)), None
    except Exception:
        return FAILED, traceback.format_exc()


def _guarded_batch(task):
    """Pool task: (fn, xs, window) -> ([values], traceback or None); one call per window."""
    fn, xs, w = task
    try:
        return [float(v) for v in fn(xs, w)], None
    except Exception:
        return [FAILED] * len(xs), traceback.format_exc()


def default_workers() -> int:
    return os.cpu_count() or 1


def minimise(fn, x0, *, evals: int, windows=None, batched: bool = False, seed: int = 0,
             workers: int = 1, sigma0: float = 0.25, label: str = "CMA-ES"):
    """Minimise a picklable objective over [0, 1]^n, evaluating each generation in parallel.

    ``fn(x)``, or with ``windows`` ``fn(x, window)`` averaged over windows; each
    (candidate, window) pair is its own pool task so all workers stay busy. ``batched``:
    ``fn(xs, window) -> [values]`` scores the whole generation on one window per task (crop
    models run it as one multi-HRU engine run). Evaluations that raise are counted and
    ranked last; past MAX_FAILURE_FRAC the search aborts with the first traceback.
    Returns (best_x, n_evals, history of {evals, best_f, failed}). Deterministic given seed.
    """
    if batched and windows is None:
        raise ValueError("batched evaluation needs windows")
    import warnings
    warnings.filterwarnings("ignore", message="Could not import matplotlib")
    import cma

    x0 = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, {"bounds": [0.0, 1.0], "maxfevals": evals,
                                                     "seed": seed + 1, "verbose": -9})
    ws = [None] if windows is None else list(windows)
    best_x, best_f, history, failed, first_tb = x0, float("inf"), [], 0, None
    pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
    t0 = time.time()
    try:
        while not es.stop():
            xs = es.ask()
            if batched:
                tasks = [(fn, [np.asarray(x) for x in xs], w) for w in ws]
                out = pool.map(_guarded_batch, tasks) if pool else list(map(_guarded_batch, tasks))
                # Re-shape to the per-(candidate, window) layout below.
                res = [(vals[i], tb) for i in range(len(xs)) for vals, tb in out]
            else:
                tasks = [(fn, np.asarray(x), w) for x in xs for w in ws]
                if pool:
                    res = pool.map(_guarded, tasks,
                                   chunksize=max(1, len(tasks) // (4 * workers)))
                else:
                    res = list(map(_guarded, tasks))
            fs = []
            for i in range(len(xs)):
                part = res[i * len(ws):(i + 1) * len(ws)]
                bad = [tb for _, tb in part if tb is not None]
                if bad:
                    failed += 1
                    first_tb = first_tb or bad[0]
                    fs.append(FAILED)
                else:
                    fs.append(float(np.mean([v for v, _ in part])))
            if failed >= max(MIN_FAILURES, MAX_FAILURE_FRAC * evals):
                raise RuntimeError(f"{label}: {failed} of {es.countevals + len(xs)} evaluations "
                                   f"raised; first:\n{first_tb}")
            for x, f in zip(xs, fs):
                if f < best_f:
                    best_f, best_x = f, np.asarray(x).copy()
            es.tell(xs, fs)
            history.append({"evals": int(es.countevals), "best_f": best_f, "failed": failed})
            print(f"  {label}: {es.countevals}/{evals} evals  best objective {best_f:.6g}  "
                  f"failed {failed}  {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
    if failed:
        print(f"  {label}: {failed} failed evaluation(s); first:\n{first_tb}", flush=True)
    return best_x, int(es.countevals), history


# -- stage reuse ------------------------------------------------------------------------

def load_stage(path: Path, cfg: dict):
    """Stage result if ``path`` exists and was produced with the same config."""
    if path.is_file():
        d = json.loads(path.read_text())
        if d.get("config") == cfg:
            print(f"  reusing {path.name}", flush=True)
            return d["result"]
        print(f"  {path.name} has a different config; recomputing", flush=True)
    return None


def save_stage(path: Path, cfg: dict, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": cfg, "result": result}, indent=2))
