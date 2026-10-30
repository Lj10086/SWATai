"""Calibration scorecard. Tier 1 = measured (GRACEnet yield per year, AgriMet ETos, April soil
NO3). Tier 2 = ArcSWAT reference output (model-vs-model, not validation).

    uv run python scripts/calib_report.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from swat_gym import FastRunner

ROOT = Path(__file__).resolve().parents[1]
REF_CSV = ROOT / "data" / "reference_hru119.csv"

#: Measured GRACEnet dry yield, t/ha, all seven years.
GRACENET = {
    2013: 22.100,    # corn
    2014: 5.900,     # barley
    2015: 9.000,     # alfalfa
    2016: 14.785,    # alfalfa
    2017: 16.158,    # alfalfa
    2018: 22.935,    # corn
    2019: 8.776,     # barley
}

#: Measured N uptake, kg/ha. Not a target: alfalfa is per cutting, barley is grain only.
GRACENET_NUPTAKE_RAW = {2013: 270.7, 2014: 135.1, 2015: 147.2}

#: AgriMet TWFI reference ET. Compare PET to etos (grass), not etrs (alfalfa).
ET_CSV = ROOT / "data" / "observed_et_agrimet.csv"

CROP = {"CSIL": "corn", "BARL": "barl", "ALFA": "alfa"}


def pbias(sim: pd.Series, obs: pd.Series) -> float:
    keep = obs.notna() & sim.notna()
    if not keep.any():
        return float("nan")
    return 100.0 * (sim[keep] - obs[keep]).sum() / obs[keep].sum()


def collect(runner: FastRunner) -> pd.DataFrame:
    """Join a completed run against the reference and the measurements, year by year."""
    ref = pd.read_csv(REF_CSV).set_index("year")
    obs_et = (pd.read_csv(ET_CSV).groupby("year")[["et_mm", "etos_mm", "etrs_mm"]].sum()
              if ET_CSV.is_file() else None)
    yld = runner.yields().set_index("year")
    wb = runner.read("hru_wb_yr.txt").set_index("yr")
    nb = runner.read("basin_nb_yr.txt").set_index("yr")
    aqu = runner.read("basin_aqu_yr.txt").set_index("yr")

    df = pd.DataFrame(index=ref.index)
    df["crop"] = [CROP.get(c, c) for c in ref["crop"]]
    df["yld"] = yld["yld(t)"]
    df["yld_ref"] = ref["yld_t_ha"]
    df["yld_meas"] = [GRACENET.get(y) for y in ref.index]
    for col, src, refcol in [
        ("et", wb["et"], "et_mm"), ("pet", wb["pet"], "pet_mm"),
        ("perc", wb["perc"], "perc_mm"), ("nup", nb["nuptake"], "nup"),
        ("no3", aqu["no3_rchg"], "no3l"),
    ]:
        df[col] = src
        df[f"{col}_ref"] = ref[refcol]
    if obs_et is not None:
        df["pet_meas"] = obs_et["etos_mm"]     # grass-reference ET, measured
        df["etrs_meas"] = obs_et["etrs_mm"]    # alfalfa-reference, for context only
    return df


def report(df: pd.DataFrame, runner: FastRunner | None = None) -> None:
    show = ["crop", "yld", "yld_ref", "yld_meas", "et", "et_ref", "pet", "pet_ref",
            "nup", "nup_ref", "no3", "no3_ref"]
    print(df[show].to_string(float_format=lambda v: f"{v:8.1f}"))

    print("\n-- TIER 1: MEASURED " + "-" * 52)
    meas = df[df["yld_meas"].notna()]
    err = 100.0 * (meas["yld"] - meas["yld_meas"]) / meas["yld_meas"]
    print(f"  GRACEnet dry-matter yield ({len(meas)} years, all rotation years):")
    # Per-year first; crop-level PBIAS hides cancellation.
    print("    year   crop      sim     meas      err")
    for year, row in meas.iterrows():
        print(f"    {year}   {row['crop']:<6} {row['yld']:7.1f}  {row['yld_meas']:7.1f}  "
              f"{err[year]:+7.1f} %")
    print(f"    RMS per-year error : {(err ** 2).mean() ** 0.5:6.1f} %   "
          f"(worst {err.abs().max():.1f} %)")
    print(f"    all measured years : {pbias(meas['yld'], meas['yld_meas']):+6.1f} %  "
          f"<- bias only; cancels within a crop")
    for crop, g in df.groupby("crop"):
        if g["yld_meas"].notna().any():
            print(f"    {crop:<16} : {pbias(g['yld'], g['yld_meas']):+6.1f} %  "
                  f"(n={g['yld_meas'].notna().sum()})")
    if "pet_meas" in df:
        n_et = int((df["pet"].notna() & df["pet_meas"].notna()).sum())
        print(f"  AgriMet TWFI reference ET ({n_et} years):")
        print(f"    PET vs meas ETos : {pbias(df['pet'], df['pet_meas']):+6.1f} %   "
              f"({df['pet'].mean():.0f} vs {df['pet_meas'].mean():.0f})")
        print(f"      (the ArcSWAT reference model scores "
              f"{pbias(df['pet_ref'], df['pet_meas']):+.1f} % on the same target)")

    if runner is not None:
        from n_trajectory import report as no3_report
        from n_trajectory import trajectory
        print("  GRACEnet April soil-nitrate profile (7 samplings, 0-122 cm, 4 plots):")
        no3_report(trajectory(runner))

    print("\n-- TIER 2: reference model, simulated (prior, not validation) " + "-" * 14)
    print(f"  yield, all years   : {pbias(df['yld'], df['yld_ref']):+6.1f} %")
    for crop, g in df.groupby("crop"):
        print(f"  {crop:<18} : {pbias(g['yld'], g['yld_ref']):+6.1f} %")
    for var, label in [("et", "ET"), ("pet", "PET"), ("perc", "deep perc"),
                       ("nup", "N uptake"), ("no3", "NO3 leached")]:
        print(f"  {label:<18} : {pbias(df[var], df[f'{var}_ref']):+6.1f} %   "
              f"({df[var].mean():.1f} vs {df[f'{var}_ref'].mean():.1f})")


def main() -> None:
    with FastRunner() as r:
        r.run()
        report(collect(r), runner=r)


if __name__ == "__main__":
    main()
