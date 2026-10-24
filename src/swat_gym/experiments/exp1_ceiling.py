"""Exp 1 gate: per-window oracle schedule vs one shared schedule = foresight value.

Both are fitted on the test windows they are scored on, with equal search: every oracle
and the shared schedule get the same number of CMA-ES evaluations, so the shared fit and
the oracles together use equal engine runs. The gap is then only what knowing each window's
weather buys, not a train/test or budget difference. An estimate, not an upper bound
(oracles can be under-converged). If within the ~250 $/ha noise floor, skip PPO.

    uv run python -m swat_gym.experiments.exp1_ceiling --per-window-budget 2000
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

from ..plan import default_x
from ..rewarders import average
from ..windows import TEST_YEARS
from .common import (RUNS, default_workers, load_stage, mean, minimise, neg_profit, paired,
                     save_stage, score, stage_config)

NOISE_FLOOR = 250.0


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-window-budget", type=int, default=2_000,
                    help="CMA-ES evals per oracle, and for the shared schedule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--out", type=Path, default=RUNS / "exp1_ceiling.json")
    args = ap.parse_args(argv)

    prices, test = average(), list(TEST_YEARS)
    cfg = stage_config(stage="ceiling", per_window_budget=args.per_window_budget,
                       seed=args.seed, prices=prices.key(), test_years=test)
    done = load_stage(args.out, cfg)
    if done is not None:
        return done
    obj = functools.partial(neg_profit, lever="I", prices=prices, no3_price=0.0)

    shared_x, n_shared, _ = minimise(obj, default_x("I"), evals=args.per_window_budget,
                                     windows=test, seed=args.seed, workers=args.workers,
                                     label="shared")
    shared_test = score("I", shared_x, test, prices, 0.0)

    oracle_test, n_oracle = [], 0
    for sy in test:
        x, n, _ = minimise(obj, default_x("I"), evals=args.per_window_budget, windows=[sy],
                           seed=args.seed + sy, workers=args.workers, label=f"oracle {sy}")
        oracle_test += score("I", x, [sy], prices, 0.0)
        n_oracle += n

    foresight = mean(oracle_test) - mean(shared_test)
    summary = {"test_years": test, "noise_floor": NOISE_FLOOR, "prices": prices.label,
               "shared_x": list(map(float, shared_x)), "shared_test": mean(shared_test),
               "oracle_test": mean(oracle_test), "foresight_test": foresight,
               "paired_test": paired(oracle_test, shared_test),
               "engine_runs": {"shared": n_shared * len(test), "oracles": n_oracle},
               "gate_pass": bool(foresight > NOISE_FLOOR)}
    save_stage(args.out, cfg, summary)
    print(f"foresight {foresight:+.0f} $/ha  gate_pass={summary['gate_pass']}  -> {args.out}")
    return summary


if __name__ == "__main__":
    main()
