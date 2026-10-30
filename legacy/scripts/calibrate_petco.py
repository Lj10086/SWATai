"""Fit hydrology.hyd pet_co to measured AgriMet ETos (grass reference).

PET is linear in pet_co: pet_co* = pet_co_run * sum(ETos) / sum(PET). Percolation, NO3 and
yield are reported alongside, not fitted.

    uv run python scripts/calibrate_petco.py
    uv run python scripts/calibrate_petco.py --values 0.85 0.90 0.95 1.00 --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from swat_gym import FastRunner  # noqa: E402
from swat_gym.fastrunner import EDITABLE, TXTINOUT  # noqa: E402
from swat_gym.params import CALIBRATABLE, set_value  # noqa: E402

from calib_report import collect, pbias  # noqa: E402

FILE, ROW, COLUMN = "hydrology.hyd", "hyd1", "pet_co"

#: Default sweep values.
DEFAULT_VALUES = (0.81, 0.90, 0.95, 1.00)


def score(df: pd.DataFrame) -> dict[str, float]:
    """PET vs ETos (objective) plus perc, NO3 and yield (reported)."""
    meas = df[df["yld_meas"].notna()]
    per_crop = {c: pbias(g["yld"], g["yld_meas"]) for c, g in meas.groupby("crop")}
    return {
        "pet_vs_meas": pbias(df["pet"], df["pet_meas"]),
        "et_vs_ref": pbias(df["et"], df["et_ref"]),
        "perc_mm": df["perc"].mean(),
        "no3_rchg": df["no3"].mean(),
        "yld_vs_meas": pbias(meas["yld"], meas["yld_meas"]),
        "yld_worst": max(abs(v) for v in per_crop.values()),
    }


def implied_optimum(df: pd.DataFrame, pet_co: float) -> float:
    """Solve for the pet_co that zeroes PET bias, exploiting PET's linearity in it."""
    keep = df["pet"].notna() & df["pet_meas"].notna()
    return pet_co * float(df.loc[keep, "pet_meas"].sum() / df.loc[keep, "pet"].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", nargs="+", type=float, default=list(DEFAULT_VALUES),
                    help="pet_co values to sweep")
    ap.add_argument("--apply", action="store_true",
                    help="write the measurement-implied optimum into hydrology.hyd")
    args = ap.parse_args()

    src = (TXTINOUT / FILE).read_text()
    rows, optima = [], {}

    with FastRunner(editable=EDITABLE | CALIBRATABLE) as r:
        for v in args.values:
            r.run({FILE: set_value(src, FILE, ROW, COLUMN, v)})
            df = collect(r)
            rows.append({"pet_co": v, **score(df)})
            optima[v] = implied_optimum(df, v)

        best = round(sum(optima.values()) / len(optima), 3)
        r.run({FILE: set_value(src, FILE, ROW, COLUMN, best)})
        verify = collect(r)
        rows.append({"pet_co": best, **score(verify)})

    table = pd.DataFrame(rows)
    print("\npet_co sweep -- objective is |pet_vs_meas|, everything else is a reported cost\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+9.2f}"))

    spread = max(optima.values()) - min(optima.values())
    print(f"\nimplied optimum from each run: "
          + ", ".join(f"{k}->{v:.3f}" for k, v in optima.items()))
    print(f"  spread {spread:.4f} "
          f"({'linear, as expected' if spread < 0.01 else 'NON-LINEAR -- do not trust the solve'})")
    print(f"\npet_co* = {best}  "
          f"(verified: PET {table.iloc[-1]['pet_vs_meas']:+.1f} % vs measured ETos, "
          f"perc {table.iloc[-1]['perc_mm']:.1f} mm/yr, "
          f"no3_rchg {table.iloc[-1]['no3_rchg']:.2f} kg/ha/yr)")

    if args.apply:
        (TXTINOUT / FILE).write_text(set_value(src, FILE, ROW, COLUMN, best))
        print(f"\n{FILE}: {COLUMN} -> {best} (written)")
    else:
        print(f"\n(not written -- re-run with --apply to set {COLUMN} = {best})")


if __name__ == "__main__":
    main()
