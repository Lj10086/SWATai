"""Reconstruct soil NO3 from fluxes (SWAT+ doesn't print the pool) and score vs measured April.

Applied ammonium fate is unreported, so the pool is bracketed: lo = all volatilises, hi = all
nitrifies. Alfalfa years (no fertiliser) are exact. Scored one-step-ahead from the measured
pool; free-run also reported.

    uv run python scripts/n_trajectory.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MEAS_CSV = ROOT / "data" / "soil_no3_gracenet.csv"

#: Sampled in April, before the ~10 April fertiliser.
SAMPLE_MONTH = "April"


def measured() -> pd.DataFrame:
    """The April profile-nitrate stock, kg/ha, indexed by year."""
    return pd.read_csv(MEAS_CSV).set_index("year")


def fertiliser_split(txtinout: Path) -> tuple[float, float]:
    """(mineral frac of N, NH3 frac of mineral N) from fertilizer.frt; asserts all manures agree."""
    rows = {}
    for line in (txtinout / "fertilizer.frt").read_text().splitlines()[2:]:
        t = line.split()
        if len(t) >= 6 and t[0].startswith("gn"):
            min_n, org_n, nh3 = float(t[1]), float(t[3]), float(t[5])
            rows[t[0]] = (min_n / (min_n + org_n), nh3)
    if not rows:
        raise SystemExit("fertilizer.frt: no gn* GRACEnet manure rows found")
    splits = set(round(v[0], 6) for v in rows.values())
    nh3s = set(round(v[1], 6) for v in rows.values())
    if len(splits) != 1 or len(nh3s) != 1:
        raise SystemExit(
            f"GRACEnet manures no longer share one mineral/NH3 split: {rows}. "
            "The bracket in n_trajectory.py assumes a single pair; make it per-year."
        )
    return splits.pop(), nh3s.pop()


def fluxes(runner) -> pd.DataFrame:
    """Per-year soil-nitrate inputs and outputs, kg/ha, from a completed run."""
    nb = runner.read("basin_nb_yr.txt").set_index("yr")
    ls = runner.read("basin_ls_yr.txt").set_index("yr")
    aqu = runner.read("basin_aqu_yr.txt").set_index("yr")
    min_frac, nh3_frac = fertiliser_split(Path(runner.workdir))

    d = pd.DataFrame(index=nb.index)
    d["mineralised"] = nb["act_nit_n"]              # active humus organic N -> nitrate
    d["fert_min"] = nb["fertn"] * min_frac          # mineral N applied (NH4 + NO3)
    d["fert_nh4"] = d["fert_min"] * nh3_frac        # ...of which ammonium: fate unreported
    d["fert_no3"] = d["fert_min"] - d["fert_nh4"]   # ...and nitrate: enters the pool directly
    d["fixed"] = nb["fixn"]                         # goes to the plant, never to the soil pool
    # Uptake is whole-plant; the share met by fixation never passed through the soil pool.
    d["soil_uptake"] = (nb["nuptake"] - nb["fixn"]).clip(lower=0.0)
    d["losses"] = (nb["denit"] + ls["surqno3"] + ls["lat3no3"] + ls["tileno3"]
                   + aqu["no3_rchg"])
    d["net_lo"] = d["mineralised"] + d["fert_no3"] - d["soil_uptake"] - d["losses"]
    d["net_hi"] = d["net_lo"] + d["fert_nh4"]
    return d


def trajectory(runner) -> pd.DataFrame:
    """Join the reconstruction to the measurement, one-step-ahead and free-running."""
    obs, flux = measured(), fluxes(runner)
    years = [y for y in flux.index if y in obs.index and (y + 1) in obs.index]

    rows = []
    free = float(obs.loc[years[0], "no3_kg_ha"]) if years else float("nan")
    for y in years:
        f, start = flux.loc[y], float(obs.loc[y, "no3_kg_ha"])
        target = float(obs.loc[y + 1, "no3_kg_ha"])
        lo, hi = start + f["net_lo"], start + f["net_hi"]
        rows.append({
            "year": y,
            "obs_start": start,
            "obs_end": target,
            "obs_change": target - start,
            "sim_lo": lo,
            "sim_hi": hi,
            "exact": f["fert_nh4"] < 1e-9,          # no ammonium applied -> bracket is a point
            # Signed distance outside the bracket; 0 when the measurement is inside it.
            "miss": 0.0 if lo <= target <= hi else (target - hi if target > hi else target - lo),
            "free_lo": free + f["net_lo"],
            "free_hi": free + f["net_hi"],
        })
        free = max(0.0, free + 0.5 * (f["net_lo"] + f["net_hi"]))
    return pd.DataFrame(rows).set_index("year")


def score(traj: pd.DataFrame, exact_only: bool = True) -> float:
    """RMS miss outside the bracket, kg N/ha. ``exact_only``: fertiliser-free years only."""
    g = traj[traj["exact"]] if exact_only else traj
    if g.empty:
        return float("nan")
    return float((g["miss"] ** 2).mean() ** 0.5)


#: Mineralisation context values (PROVENANCE §5e); not targets.
MINERALISATION_CONTEXT = (
    (209.8, "measured buried-bag net N min, 0-60 cm, LT Manure 30_Annual field (8 yr)"),
    (117.0, "ArcSWAT reference A_MN on *this* field, 2013-2019 mean"),
)


def report(traj: pd.DataFrame, flux: pd.DataFrame | None = None) -> None:
    show = traj[["obs_start", "obs_end", "obs_change", "sim_lo", "sim_hi", "miss", "exact"]]
    print(show.to_string(float_format=lambda v: f"{v:9.1f}"))
    if flux is not None:
        ours = float(flux["mineralised"].mean())
        print(f"\n  humus mineralisation, this model : {ours:6.1f} kg N/ha/yr")
        for value, label in MINERALISATION_CONTEXT:
            print(f"    vs {value:6.1f}  ({ours / value - 1:+.0%})  {label}")
    n_exact = int(traj["exact"].sum())
    print(f"\n  RMS miss, fertiliser-free years only (n={n_exact}) : "
          f"{score(traj):8.1f} kg N/ha")
    print(f"  RMS miss, all years (n={len(traj)})                  : "
          f"{score(traj, exact_only=False):8.1f} kg N/ha")
    inside = int((traj["miss"] == 0).sum())
    print(f"  years landing inside their bracket               : {inside}/{len(traj)}")


def main() -> None:
    from swat_gym import FastRunner

    with FastRunner() as r:
        r.run()
        traj, flux = trajectory(r), fluxes(r)
    print("measured April soil nitrate (kg N/ha, 0-122 cm) vs reconstructed balance\n")
    report(traj, flux)


if __name__ == "__main__":
    main()
