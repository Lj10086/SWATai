"""Port initial soil nutrients, topography, hydrology and basin N-cycling params (ArcSWAT
reference HRU 119; GRACEnet measurements win where they exist).

pet_co is owned by calibrate_petco.py; only changed here if --pet-co is passed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIO = ROOT / "model" / "rotation" / "TxtInOut"
SOIL_CSV = ROOT / "data" / "soil_gracenet.csv"

HRU_SOILNUT, HRU_TOPO, HRU_HYD = "soilnut1", "topohru1", "hyd1"

# 000140001.hru
REF_SLOPE = 0.012          # HRU_SLP
REF_SLOPE_LEN = 121.951    # SLSUBBSN
REF_DIST_CHA = 35.0        # DIS_STREAM
REF_CANMX = 0.0            # CANMX (the SWAT+ template ships 1.0)

# basins.bsn -> parameters.bsn (SWAT2012 name in comments).
REF_BASIN = {
    "orgn_min": 0.0010,    # CMN     humus active organic-N mineralisation rate factor
    "n_uptake": 10.0,      # N_UPDIS nitrogen uptake distribution parameter
    "n_perc": 0.20,        # NPERCO  nitrogen percolation coefficient
    "denit_exp": 0.001,    # CDN     denitrification exponential rate coefficient
    "surq_lag": 1.0,       # SURLAG  surface runoff lag time (days)
}


def measured_soil_nutrients() -> tuple[float, float]:
    """Profile-mean initial NO3 and labile P (ppm) from GRACEnet April-2013, weighted by layer mass."""
    df = pd.read_csv(SOIL_CSV).sort_values("depth_mm")
    thick = df["depth_mm"].diff().fillna(df["depth_mm"])
    mass = thick * df["bd_g_cm3"]
    wmean = lambda col: float((df[col] * mass).sum() / mass.sum())   # noqa: E731
    return wmean("no3_n_mg_kg"), wmean("olsen_p_mg_kg")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _set_tokens(path: Path, row_name: str, updates: dict[int, float]) -> None:
    """Rewrite the row whose first token is row_name, replacing tokens by index."""
    lines = path.read_text().splitlines(keepends=True)
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks[:1] != [row_name]:
            continue
        for idx, val in updates.items():
            toks[idx] = f"{val:.5f}"
        desc = toks[-1] if not _is_float(toks[-1]) else ""
        nums = toks[1:-1] if desc else toks[1:]
        lines[i] = (f"{row_name:<22}" + "".join(f"{t:>14}" for t in nums)
                    + (f"  {desc}" if desc else "") + "\n")
        path.write_text("".join(lines))
        return
    raise SystemExit(f"{path.name}: row {row_name!r} not found")


def _set_basin(path: Path, updates: dict[str, float]) -> None:
    """parameters.bsn is one header row plus one value row, matched by name."""
    lines = path.read_text().splitlines()
    header, values = lines[1].split(), lines[2].split()
    for name, val in updates.items():
        if name not in header:
            raise SystemExit(f"{path.name}: parameter {name!r} not in header")
        values[header.index(name)] = f"{val:.5f}"
    lines[2] = "  ".join(values)
    path.write_text("\n".join(lines) + "\n")


def _set_nyskip(path: Path, nyskip: int) -> None:
    """print.prt line index 2 is the nyskip / date-range value row."""
    lines = path.read_text().splitlines()
    toks = lines[2].split()
    toks[0] = str(nyskip)
    lines[2] = "".join(f"{t:<10}" for t in toks)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet-co", type=float, default=None,
                    help="PET multiplier. Default: leave whatever calibrate_petco.py set. "
                         "Pass a value only to override that deliberately.")
    args = ap.parse_args()

    # --- initial soil nutrients (nutrients.sol: lab_p = token 2, nitrate = token 3) ---
    nitrate, lab_p = measured_soil_nutrients()
    _set_tokens(TIO / "nutrients.sol", HRU_SOILNUT, {2: lab_p, 3: nitrate})
    print(f"nutrients.sol: nitrate -> {nitrate:.2f} ppm, lab_p -> {lab_p:.2f} ppm "
          f"(GRACEnet measured April 2013, profile mass-weighted)")

    # --- topography (slp = 1, slp_len = 2, lat_len = 3, dist_cha = 4) ---
    _set_tokens(TIO / "topography.hyd", HRU_TOPO,
                {1: REF_SLOPE, 2: REF_SLOPE_LEN, 3: REF_SLOPE_LEN, 4: REF_DIST_CHA})
    print(f"topography.hyd: slp -> {REF_SLOPE}, slp_len/lat_len -> {REF_SLOPE_LEN} m, "
          f"dist_cha -> {REF_DIST_CHA} m")

    # --- hydrology (can_max = token 3, pet_co = token 13) ---
    hyd = {3: REF_CANMX} | ({13: args.pet_co} if args.pet_co is not None else {})
    _set_tokens(TIO / "hydrology.hyd", HRU_HYD, hyd)
    print(f"hydrology.hyd: can_max -> {REF_CANMX}, pet_co -> "
          + (f"{args.pet_co} (overridden)" if args.pet_co is not None
             else "left as calibrated (see calibrate_petco.py)"))

    # --- basin N-cycling parameters ---
    _set_basin(TIO / "parameters.bsn", REF_BASIN)
    print("parameters.bsn: " + ", ".join(f"{k} -> {v}" for k, v in REF_BASIN.items()))

    # --- output: skip the 2012 spin-up year only (reference NYSKIP=1) ---
    _set_nyskip(TIO / "print.prt", 1)
    print("print.prt: nyskip -> 1 (2012 spin-up only; 2013 keeps its HRU diagnostics)")


if __name__ == "__main__":
    main()
