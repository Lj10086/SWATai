"""CMA-ES driver: deterministic, monotone history, parallel == serial."""
from __future__ import annotations

import numpy as np
import pytest

from swat_gym.experiments.common import minimise


def rosen(x) -> float:
    x = np.asarray(x) * 4.0 - 2.0
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def test_deterministic_and_parallel_matches_serial():
    a = minimise(rosen, np.full(5, 0.5), evals=120, seed=3)
    b = minimise(rosen, np.full(5, 0.5), evals=120, seed=3, workers=2)
    assert np.allclose(a[0], b[0]) and a[1] == b[1]


def test_history_is_monotone():
    _, _, hist = minimise(rosen, np.full(4, 0.5), evals=150, seed=2)
    fs = [h["best_f"] for h in hist]
    assert fs == sorted(fs, reverse=True)


def always_fails(x) -> float:
    raise RuntimeError("engine is broken")


_CALLS = {"n": 0}


def fails_twice(x) -> float:
    """Serial only: calls 5 and 40 raise, everything else is the Rosenbrock function."""
    _CALLS["n"] += 1
    if _CALLS["n"] in (5, 40):
        raise RuntimeError("transient")
    return rosen(x)


def per_window(x, w) -> float:
    return rosen(x) + w


def test_a_broken_objective_aborts_with_its_traceback():
    """Failures are not scores: past MAX_FAILURE_FRAC the search stops and says why."""
    with pytest.raises(RuntimeError, match="(?s)raised.*engine is broken"):
        minimise(always_fails, np.full(3, 0.5), evals=200, seed=0)


def test_rare_failures_are_counted_and_ranked_last():
    _CALLS["n"] = 0
    _, n, hist = minimise(fails_twice, np.full(3, 0.5), evals=300, seed=1)
    assert hist[-1]["failed"] == 2 and n >= 290
    assert all(np.isfinite(h["best_f"]) and h["best_f"] < 1e29 for h in hist)


def test_windows_average_per_task_and_match_serial():
    """(candidate, window) tasks: mean over windows, and the pool changes nothing."""
    a = minimise(per_window, np.full(4, 0.5), evals=80, windows=[0.0, 2.0], seed=4)
    b = minimise(per_window, np.full(4, 0.5), evals=80, windows=[0.0, 2.0], seed=4, workers=3)
    c = minimise(rosen, np.full(4, 0.5), evals=80, seed=4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
    assert a[2][-1]["best_f"] == pytest.approx(c[2][-1]["best_f"] + 1.0)


def per_window_batch(xs, w):
    return [rosen(x) + w for x in xs]


def test_batched_generations_equal_per_candidate_tasks():
    a = minimise(per_window, np.full(4, 0.5), evals=80, windows=[0.0, 2.0], seed=4)
    b = minimise(per_window_batch, np.full(4, 0.5), evals=80, windows=[0.0, 2.0], seed=4,
                 batched=True, workers=2)
    assert np.array_equal(a[0], b[0]) and a[2] == b[2]
