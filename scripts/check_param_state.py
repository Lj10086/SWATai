#!/usr/bin/env python3
"""Check model params on disk match ownership rules (rule 11b). Exits non-zero on drift.

Workbook-owned (port_crops.COLMAP + gn_* HARV_EFF): asserted. Optimizer-owned (optimize.PARAMS):
reported. Shared params are not allowed; pin them in OVERRIDES and assert.

    uv run python scripts/check_param_state.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.abspath(str(ROOT / "scripts")))

from swat_gym.params import get_value  # noqa: E402

from port_crops import COLMAP, CROP_MAP, OVERRIDES  # noqa: E402

TIO = ROOT / "model" / "rotation" / "TxtInOut"
CROP_CSV = ROOT / "data" / "crop_params_swat.csv"
HARVEST = {"CSIL": "gn_corn", "BARL": "gn_barl", "ALFA": "gn_alfa"}
TOL = 1e-4


def workbook_drift() -> list[tuple[str, str, float, float]]:
    """Every workbook-owned value that no longer matches the workbook."""
    txt = (TIO / "plants.plt").read_text()
    header = txt.splitlines()[1].split()
    wb = pd.read_csv(CROP_CSV).set_index("CPNM")
    out = []
    for code, plt in CROP_MAP.items():
        for idx, (field, scale) in COLMAP.items():
            col = header[idx]
            cur = get_value(txt, "plants.plt", plt, col)
            book = float(wb.loc[code, field]) * scale
            if abs(cur - book) > TOL * max(1.0, abs(book)):
                out.append((plt, col, cur, book))
    return out


def harvest_drift() -> list[tuple[str, str, float, float]]:
    """``harv.ops`` harvest indices against the workbook's ``HARV_EFF``."""
    wb = pd.read_csv(CROP_CSV).set_index("CPNM")
    txt = (TIO / "harv.ops").read_text()
    out = []
    for code, row in HARVEST.items():
        cur = get_value(txt, "harv.ops", row, "harv_idx")
        book = float(wb.loc[code, "HARV_EFF"])
        if abs(cur - book) > TOL:
            out.append((row, "harv_idx", cur, book))
    return out


def shared_drift() -> list[tuple[str, str, float, float]]:
    """Parameters both writers touch. They must agree or re-porting silently reverts a fit."""
    txt = (TIO / "plants.plt").read_text()
    out = []
    for (row, col), (value, _why) in OVERRIDES.items():
        cur = get_value(txt, "plants.plt", row, col)
        if abs(cur - value) > TOL * max(1.0, abs(value)):
            out.append((row, col, cur, value))
    return out


def integer_columns_intact() -> list[str]:
    """days_mat/yrs_mat must be integer tokens (else the row mis-parses silently)."""
    txt = (TIO / "plants.plt").read_text()
    bad = []
    for plt in CROP_MAP.values():
        line = next(l for l in txt.splitlines() if l.split()[:1] == [plt])
        # Column indices from the header, not hard-coded.
        header = txt.splitlines()[1].split()
        for idx, name in ((header.index("days_mat"), "days_mat"),
                          (header.index("yrs_mat"), "yrs_mat")):
            tok = line.split()[idx]
            if "." in tok:
                bad.append(f"{plt}.{name} = {tok!r}")
    return bad


def main() -> None:
    problems = 0
    for title, rows, fix in (
        ("workbook-owned (rule 11b)", workbook_drift(),
         "uv run python scripts/port_crops.py"),
        ("harv.ops harvest indices", harvest_drift(),
         "uv run python scripts/port_crops.py  # then re-check harv.ops"),
        ("shared with port_crops.OVERRIDES", shared_drift(),
         "update OVERRIDES to the last applied fit, or re-run optimize.py --apply"),
    ):
        if rows:
            problems += len(rows)
            print(f"\nDRIFT — {title}:")
            print(f"  {'row':<10}{'column':<14}{'in model':>12}{'expected':>12}")
            for r, c, cur, exp in rows:
                print(f"  {r:<10}{c:<14}{cur:>12.5f}{exp:>12.5f}")
            print(f"  fix: {fix}")
        else:
            print(f"ok — {title}")

    bad = integer_columns_intact()
    if bad:
        problems += len(bad)
        print(f"\nDRIFT — integer columns written as floats (rev 62 column shift): {bad}")
    else:
        print("ok — days_mat/yrs_mat still integer tokens")

    # Reported, never asserted: no source of truth beyond the last fit.
    txt = (TIO / "plants.plt").read_text()
    print("\noptimizer-owned (reported, not checked):")
    for row, col in (("corn", "days_mat"), ("barl", "days_mat")):
        print(f"  {row}.{col:<10} = {get_value(txt, 'plants.plt', row, col)}")
    for f, row, col in (("hydrology.hyd", "hyd1", "epco"), ("hydrology.hyd", "hyd1", "pet_co"),
                        ("parameters.bsn", "", "orgn_min"), ("parameters.bsn", "", "n_perc"),
                        ("nutrients.sol", "soilnut1", "fr_hum_act")):
        print(f"  {f}:{col:<12} = {get_value((TIO / f).read_text(), f, row, col)}")

    print(f"\n{'PASS' if problems == 0 else f'FAIL — {problems} drifted value(s)'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
