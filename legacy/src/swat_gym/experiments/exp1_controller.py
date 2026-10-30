"""CMA-ES-fitted feedback irrigation controller (separates "no value in adapting" from "PPO failed").

    depth = clip(a + b * (SW_TARGET - sw) + c * (PRECIP_NORM - precip), 0, MONTH_DEPTH_MAX)

using last month's soil water and precip. Rolled out through the same env as PPO.

    uv run python -m swat_gym.experiments.exp1_controller --budget 300000

The budget is engine runs (one per env step), matched to CMA-ES and each PPO seed (rule 2).
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np

from ..env import SwatEnv, rollout
from ..plan import MONTH_DEPTH_MAX
from ..rewarders import average
from ..windows import TEST_YEARS, TRAIN_YEARS
from .common import RUNS, default_workers, mean, minimise, row

SW_TARGET = 250.0
PRECIP_NORM = 20.0
X0 = np.array([0.2, 0.3, 0.3])

_ENV: dict = {}


def decode(abc) -> tuple[float, float, float]:
    """Unit box -> (baseline mm, soil-water gain, precip-deficit gain)."""
    a, b, c = (float(v) for v in np.clip(np.asarray(abc, dtype=float), 0.0, 1.0))
    return a * MONTH_DEPTH_MAX, b * 2.0, c * 2.0


def controller(abc):
    a, b, c = decode(abc)

    def act(env, obs):
        s = env.last
        depth = a + b * max(0.0, SW_TARGET - s["sw"]) + c * max(0.0, PRECIP_NORM - s["precip"])
        return [float(np.clip(depth, 0.0, MONTH_DEPTH_MAX)) / MONTH_DEPTH_MAX]
    return act


def env_for(prices, no3_price) -> SwatEnv:
    """This process's irrigation env (reused across rollouts)."""
    key = (repr(prices), no3_price)
    if key not in _ENV:
        _ENV[key] = SwatEnv("I", prices=prices, no3_price=no3_price)
    return _ENV[key]


def run_controller(abc, start_year, prices, no3_price=0.0) -> dict:
    return rollout(env_for(prices, no3_price), controller(abc), start_year)


def neg_profit(abc, start_year, prices, no3_price) -> float:
    """CMA-ES objective on one window (minimise averages over windows)."""
    return -run_controller(abc, start_year, prices, no3_price)["profit"]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=300_000, help="engine runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--no3-price", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=RUNS / "exp1_controller.json")
    args = ap.parse_args(argv)

    prices, train, test = average(), list(TRAIN_YEARS), list(TEST_YEARS)
    fn = functools.partial(neg_profit, prices=prices, no3_price=args.no3_price)
    runs_per_eval = len(train) * len(SwatEnv.steps_for("I"))
    best, n_evals, history = minimise(fn, X0, evals=max(1, args.budget // runs_per_eval),
                                      windows=train, seed=args.seed, workers=args.workers,
                                      label="controller")
    test_rows = [row(run_controller(best, sy, prices, args.no3_price), sy) for sy in test]
    summary = {"abc": list(map(float, best)), "decoded": dict(zip("abc", decode(best))),
               "no3_price": args.no3_price, "prices": prices.label, "n_evals": n_evals,
               "engine_runs": n_evals * runs_per_eval,
               "history": history, "test": mean(test_rows), "per_window_test": test_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"controller test {summary['test']:.0f} $/ha  {summary['decoded']}  -> {args.out}")
    return summary


if __name__ == "__main__":
    main()
