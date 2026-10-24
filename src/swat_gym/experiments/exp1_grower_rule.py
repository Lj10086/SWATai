"""Grower irrigation as the exp1_controller rule, fitted to logged 2013-2019 monthly depths.

Same (a, b, c) controller as the profit row, fitted to behaviour, not profit. Loss = monthly
SSE + rule 5 total-water gate. Fit years overlap test years; fine because it is a reference
row, never searched on profit.

    uv run python -m swat_gym.experiments.exp1_grower_rule --evals 220
    uv run python -m swat_gym.experiments.exp1_grower_rule --from-fit runs/exp1_grower_rule.json \\
        --no3-price 16 --out runs/exp1_grower_rule_n16.json
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..plan import MEASURED_ROTATION, MONTH_DEPTH_MAX, N_GROWING, N_YEARS, year_events
from ..rewarders import average
from ..schedule import GROWING_MONTHS, _md
from ..windows import TEST_YEARS, TRAIN_YEARS
from .common import ROOT, RUNS, default_workers, mean, minimise, row
from .exp1_controller import X0, decode, run_controller

IRR_CSV = ROOT / "data" / "irrigation_gracenet.csv"
FIT_START = 2012                                   # spin-up 2012, scored 2013-2019
FIT_YEARS = list(range(FIT_START + 1, FIT_START + 1 + N_YEARS))
GATE_TOL = 0.01                                    # rule 5
TOTAL_PENALTY = 1e3                                # per mm² beyond GATE_TOL


def logged_monthly_mm() -> np.ndarray:
    """Logged irrigation as (N_YEARS, N_GROWING) monthly totals, Apr-Sep."""
    df = pd.read_csv(IRR_CSV)
    df = df[df.year.isin(FIT_YEARS)]
    mon = pd.to_datetime(df["date"]).dt.month
    out = np.zeros((N_YEARS, N_GROWING))
    for (yr, m), mm in df.groupby([df.year, mon])["mm"].sum().items():
        if m in GROWING_MONTHS:
            out[FIT_YEARS.index(yr), GROWING_MONTHS.index(m)] = mm
    return out


def logged_total_mm() -> float:
    """All logged mm in the scored years, including October."""
    df = pd.read_csv(IRR_CSV)
    return float(df[df.year.isin(FIT_YEARS)]["mm"].sum())


def rendered_monthly_mm(x) -> np.ndarray:
    """Lever-I vector -> monthly totals as actually rendered by year_events."""
    month_mm = np.asarray(x, dtype=float).reshape(N_YEARS, N_GROWING) * MONTH_DEPTH_MAX
    out = np.zeros_like(month_mm)
    for y in range(N_YEARS):
        for doy, mm in year_events(month_mm[y], MEASURED_ROTATION[y]):
            m = _md(doy)[0]
            if m in GROWING_MONTHS:
                out[y, GROWING_MONTHS.index(m)] += mm
    return out


def fit_loss(rendered: np.ndarray, target: np.ndarray, total: float) -> float:
    sse = float(((rendered - target) ** 2).sum())
    excess = max(0.0, abs(float(rendered.sum()) - total) - GATE_TOL * total)
    return sse + TOTAL_PENALTY * excess ** 2


def neg_fit(abc, prices) -> float:
    sim = run_controller(abc, FIT_START, prices)
    return fit_loss(rendered_monthly_mm(sim["x"]), logged_monthly_mm(), logged_total_mm())


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--from-fit", type=Path, default=None, help="reuse (a, b, c); only re-score")
    ap.add_argument("--no3-price", type=float, default=0.0, help="for scoring only")
    ap.add_argument("--out", type=Path, default=RUNS / "exp1_grower_rule.json")
    args = ap.parse_args(argv)

    prices, test = average(), list(TEST_YEARS)
    target, total = logged_monthly_mm(), logged_total_mm()
    if args.from_fit:
        best = np.asarray(json.loads(args.from_fit.read_text())["abc"])
    else:
        best, _, _ = minimise(functools.partial(neg_fit, prices=prices), X0, evals=args.evals,
                              seed=args.seed, workers=args.workers, label="grower fit")

    sim = run_controller(best, FIT_START, prices)
    rendered = rendered_monthly_mm(sim["x"])
    resid = rendered - target
    err = (float(sim["irrigation_mm"]) - total) / total
    fit = {"r2_monthly": float(1 - (resid ** 2).sum() / ((target - target.mean()) ** 2).sum()),
           "rmse_monthly_mm": float(np.sqrt((resid ** 2).mean())),
           "fitted_annual_mm": rendered.sum(axis=1).round(1).tolist(),
           "target_annual_mm": target.sum(axis=1).round(1).tolist(),
           "logged_total_mm": total, "simulated_total_mm": float(sim["irrigation_mm"]),
           "total_err_pct": 100 * err, "gate_pass": bool(abs(err) <= GATE_TOL)}
    summary = {"abc": list(map(float, best)), "decoded": dict(zip("abc", decode(best))),
               "fit": fit, "fit_years": FIT_YEARS, "no3_price": args.no3_price,
               "prices": prices.label, "train_years": list(TRAIN_YEARS), "test_years": test}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fit: {fit['simulated_total_mm']:.1f} mm vs logged {total:.1f} "
          f"({fit['total_err_pct']:+.2f} %), R² {fit['r2_monthly']:.2f}", flush=True)
    if not fit["gate_pass"]:
        summary["scored"] = False
        args.out.write_text(json.dumps(summary, indent=2))
        print("GATE FAILED (rule 5): not scored.")
        sys.exit(3)

    test_rows = [row(run_controller(best, sy, prices, args.no3_price), sy) for sy in test]
    summary.update(scored=True, test=mean(test_rows), per_window_test=test_rows)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"grower rule test {summary['test']:.0f} $/ha  -> {args.out}")
    return summary


if __name__ == "__main__":
    main()
