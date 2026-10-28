#!/usr/bin/env python3
"""Fit the LTM auto-irrigation trigger to the trial's own water use.

User decision 2026-10-25: auto-irrigation should be "realistic, moderate, standard practice".
The schedule (swat_gym.ltm.practice.AUTO_IRR) irrigates one 1-inch pass whenever the profile's
plant-available water is below sw_frac x its field-capacity value, inside each season's window.
sw_frac is the one free number; it is root-found so that, under the trial's rotation and
windows on the trial's weather (window 2012 = seasons 2013-2020) at practice.MEASURED_IRR_EFF, the
rule applies the trial's measured total (gross) irrigation (data/ltm/irrigation_daily.csv, 2013-2020). "Standard practice" is
therefore this field's practice. Per-year totals are reported against measured.

    uv run python scripts/ltm_fit_auto_irr.py          # report
    uv run python scripts/ltm_fit_auto_irr.py --apply  # -> model/ltm/auto_irr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swat_gym.ltm import practice, runner as rmod  # noqa: E402
from swat_gym.ltm.render import EpisodePlan  # noqa: E402
from swat_gym.ltm.runner import LtmRunner  # noqa: E402

START = 2012
BOUNDS = (0.30, 0.99)


def measured() -> pd.Series:
    irr = pd.read_csv(ROOT / "data" / "ltm" / "irrigation_daily.csv", parse_dates=["date"])
    irr = irr[irr.date.dt.year.between(2013, 2020)]
    return irr.groupby(irr.date.dt.year).irr_mm.sum()


def simulated(r: LtmRunner, sw_frac: float) -> pd.Series:
    rmod.AUTO_IRR["sw_frac"] = sw_frac            # this process only
    out = r.run([EpisodePlan(irr_auto=True, irr_eff=practice.MEASURED_IRR_EFF)], START)[0]
    wb = out["hru_wb_yr.txt"]
    return wb.set_index("yr").irr / practice.MEASURED_IRR_EFF    # gross


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    meas = measured()
    target = float(meas.sum())
    with LtmRunner(1) as r:
        f = lambda x: float(simulated(r, x).sum()) - target  # noqa: E731
        lo, hi = f(BOUNDS[0]), f(BOUNDS[1])
        if lo > 0 or hi < 0:
            raise SystemExit(f"target {target:.0f} mm not bracketed: {lo + target:.0f} .. "
                             f"{hi + target:.0f} mm on {BOUNDS}")
        x = brentq(f, *BOUNDS, xtol=1e-4)
        sim = simulated(r, x)
        mgt = r.run([EpisodePlan(irr_auto=True, irr_eff=practice.MEASURED_IRR_EFF)], START)[0]["mgt"]
    ev = mgt[mgt.operation == "IRRIGATE"].groupby("yr").size()
    table = pd.DataFrame({"measured_mm": meas, "auto_mm": sim, "auto_events": ev})
    irr = pd.read_csv(ROOT / "data" / "ltm" / "irrigation_daily.csv", parse_dates=["date"])
    irr = irr[(irr.irr_mm > 0) & irr.date.dt.year.between(2013, 2020)]
    table["measured_events"] = irr.groupby(irr.date.dt.year).size()
    print(f"sw_frac {x:.4f}: auto {sim.sum():.0f} mm vs measured {target:.0f} mm (2013-2020)")
    print(table.round(1).to_string())
    if args.apply:
        rec = {"sw_frac": round(x, 4), "pass_mm": practice.IRR_PASS_MM,
               "efficiency": practice.MEASURED_IRR_EFF, "gross": True,
               "target_mm_2013_2020": target, "auto_mm": float(sim.sum()),
               "per_year": {int(y): {"measured": float(meas[y]), "auto": float(sim[y]),
                                     "auto_events": int(table.auto_events.get(y, 0)),
                                     "measured_events": int(table.measured_events.get(y, 0))}
                            for y in meas.index},
               "fitted": date.today().isoformat()}
        practice.AUTO_IRR_FILE.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"wrote {practice.AUTO_IRR_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
