#!/usr/bin/env python3
"""Score the LTM scenario against the trial's measurements. Fits nothing.

Runs model/ltm/TxtInOut once (three treatment HRUs) and compares, per treatment and year:

* biomass at harvest (mgt_out plant_bioms) vs measured 52A whole-plant biomass;
* harvested yield (mgt_out op_var, dry) vs measured grain DM (barley, wheat) and sugarbeet
  root DM (P2 Table 5, 4-yr 2-field means, so compared to the simulated 2018/2020 mean);
* plant N and P uptake (hru_nb_yr) vs measured 52A uptake;
* soil NO3 and solution P at the measured sampling dates (0-kg probe ops) vs measured
  profile NO3 and Olsen P: depth and extractant differ, so trajectories are compared by
  shape (Pearson r) and the 2020 treatment contrast by ratio;
* N mineralisation (act_nit_n + rsd_nitorg_n) vs the measured buried-bag net IN, 52A;
* irrigation applied vs measured (gate: within 1 %).

Losses (NO3 leached, P leached/runoff) have no on-site measurement and are reported only.

    uv run python scripts/ltm_validate.py      # -> runs/ltm_validation.json, .csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swat_gym.fastrunner import FastRunner  # noqa: E402
from swat_gym.ltm.practice import MEASURED_IRR_EFF  # noqa: E402

TIO = ROOT / "model" / "ltm" / "TxtInOut"
DATA = ROOT / "data" / "ltm"
OUT = ROOT / "runs"
HRU = {1: "52A", 2: "fert", 3: "control"}
CROP_PLANT = {"barley": "barl", "sugarbeet": "sgbt", "wheat": "wwht", "potato": "pota"}
IRR_TOL = 0.01


def mgt_out(path: Path) -> pd.DataFrame:
    """mgt_out.txt rows: hru, date, op name, operation, then the numeric columns."""
    cols = ["phubase", "phuplant", "soil_water", "plant_bioms", "surf_rsd", "soil_no3",
            "soil_solp", "op_var", "var1", "var2", "var3", "var4"]
    recs = []
    for ln in path.read_text().splitlines()[3:]:
        t = ln.split()
        if len(t) < 7:
            continue
        nums = [float(x) for x in t[6:6 + len(cols)]]
        recs.append(dict(hru=int(t[0]), date=pd.Timestamp(int(t[1]), int(t[2]), int(t[3])),
                         name=t[4], operation=t[5], **dict(zip(cols, nums))))
    return pd.DataFrame(recs)


def pearson(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 else None


def main() -> None:
    with FastRunner(TIO, trim_outputs=False) as fr:
        fr.run()
        mg = mgt_out(fr.workdir / "mgt_out.txt")
        nb, ls, wb = (fr.read(t) for t in ("hru_nb_yr.txt", "hru_ls_yr.txt", "hru_wb_yr.txt"))
    mg["treatment"] = mg.hru.map(HRU)
    for df in (nb, ls, wb):
        df["treatment"] = df.unit.map(HRU)
    report: dict = {"build": json.loads((TIO.parent / "build.json").read_text())}

    # -- harvest: biomass and yield ------------------------------------------------------
    hv = mg[mg.operation == "HARV/KILL"].assign(year=lambda d: d.date.dt.year)
    hv = hv[hv.year >= 2013]
    up = pd.read_csv(DATA / "uptake.csv")
    yl = pd.read_csv(DATA / "yield.csv")
    meas_y = yl[yl.year > 0].groupby(["year", "crop"]).yield_dm_mg_ha.mean()
    rows = []
    for r in hv.itertuples():
        crop = next(k for k, v in CROP_PLANT.items() if v == r.name)
        rec = dict(year=r.year, treatment=r.treatment, crop=crop,
                   sim_biomass_t=r.plant_bioms / 1000, sim_yield_dm_t=r.op_var / 1000)
        if r.treatment == "52A":
            u = up[up.year == r.year]
            rec["meas_biomass_t"] = float(u.biomass_mg_ha.iloc[0]) if len(u) else np.nan
            rec["meas_yield_dm_t"] = float(meas_y.get((r.year, crop), np.nan))
        rows.append(rec)
    harvest = pd.DataFrame(rows)
    harvest["biomass_pct_err"] = 100 * (harvest.sim_biomass_t / harvest.meas_biomass_t - 1)
    harvest["yield_pct_err"] = 100 * (harvest.sim_yield_dm_t / harvest.meas_yield_dm_t - 1)
    report["harvest"] = harvest.round(3).to_dict("records")

    # sugarbeet root DM by treatment: P2 Table 5 (2014/16/18/20, two fields) vs simulated
    # 2018/2020 (2014 beets were not harvested; 2016 was potato on this field)
    t5 = yl[(yl.year == 0) & (yl.source == "P2 Table 5")].set_index("treatment").yield_dm_mg_ha
    sb = harvest[(harvest.crop == "sugarbeet") & harvest.year.isin([2018, 2020])]
    sim_sb = sb.groupby("treatment").sim_yield_dm_t.mean()
    report["sugarbeet_root_dm"] = {
        t: {"meas_t": float(t5[t]), "sim_t": float(sim_sb[t]),
            "pct_err": float(100 * (sim_sb[t] / t5[t] - 1))} for t in HRU.values()}
    report["sugarbeet_contrast"] = {
        "meas_fert_over_52A": float(t5["fert"] / t5["52A"]),
        "sim_fert_over_52A": float(sim_sb["fert"] / sim_sb["52A"]),
        "meas_control_over_52A": float(t5["control"] / t5["52A"]),
        "sim_control_over_52A": float(sim_sb["control"] / sim_sb["52A"])}

    # -- uptake --------------------------------------------------------------------------
    n52 = nb[nb.treatment == "52A"].set_index("yr")
    upt = up.set_index("year")
    report["uptake_52A"] = [
        dict(year=int(y), crop=upt.crop[y], meas_n=float(upt.n_up_kg_ha[y]),
             sim_n=float(n52.nuptake[y]), meas_p=float(upt.p_up_kg_ha[y]),
             sim_p=float(n52.puptake[y])) for y in upt.index if y in n52.index]

    # -- soil trajectories (probes) -----------------------------------------------------
    ss = pd.read_csv(DATA / "soil_series.csv")
    meas = ss[ss["plot"] == "mean"].dropna(subset=["date"]).copy()
    meas["date"] = pd.to_datetime(meas.date)
    pr = mg[(mg.name == "elem_n") & (mg.op_var == 0.0)]
    soil = {}
    for var, simcol in (("profile_no3_mg_kg", "soil_no3"), ("olsen_p_mg_kg_0_30", "soil_solp")):
        m = meas[meas["var"] == var].set_index("date")["value"]
        s = pr[pr.treatment == "52A"].set_index("date")[simcol]
        both = pd.concat([m.rename("meas"), s.rename("sim")], axis=1).dropna()
        soil[var] = {"pearson_r": pearson(both.meas, both.sim),
                     "series": [dict(date=str(d.date()), meas=float(v.meas), sim_kg_ha=float(v.sim))
                                for d, v in both.iterrows()]}
    # 2020 treatment contrast, soil NO3 (P1 Table 5, 0-60 cm, spring 2020)
    t5n = ss[(ss["var"] == "no3_mg_kg_0_60") & (ss.year == 2020)].set_index("treatment").value
    s20 = pr[pr.date == pd.Timestamp("2020-04-01")].set_index("treatment").soil_no3
    soil["no3_2020_contrast"] = {
        t: {"meas_ratio_to_52A": float(t5n[t] / t5n["52A"]),
            "sim_ratio_to_52A": float(s20[t] / s20["52A"])} for t in HRU.values()}
    pre = pr[pr.date == pd.Timestamp("2012-10-13")].set_index("treatment")
    soil["prestudy_2012_10_13"] = {"sim_no3_kg_ha": float(pre.soil_no3["52A"]),
                                   "sim_solp_kg_ha": float(pre.soil_solp["52A"]),
                                   "meas_olsen_p_mg_kg_0_30": 11.1}
    report["soil_52A"] = soil

    # -- N mineralisation ------------------------------------------------------------------
    nm = pd.read_csv(DATA / "nmin.csv")
    nm = nm[nm.treatment == "52A"].set_index("year").net_in_kg_ha
    report["nmin_52A"] = [dict(year=int(y), meas_net_in=float(nm[y]),
                               sim_act_nit=float(n52.act_nit_n[y]),
                               sim_rsd_nitorg=float(n52.rsd_nitorg_n[y]))
                          for y in nm.index if y in n52.index]

    # -- water gate + losses ---------------------------------------------------------------
    irr = pd.read_csv(DATA / "irrigation_daily.csv", parse_dates=["date"])
    meas_irr = irr[irr.date.dt.year.between(2013, 2020)].irr_mm.sum()
    # hru_wb irr is what reaches the soil (amt x eff); the gate compares gross water
    sim_irr = wb[(wb.treatment == "52A") & wb.yr.between(2013, 2020)].irr.sum() / MEASURED_IRR_EFF
    report["irrigation_gate"] = {"meas_mm": float(meas_irr), "sim_mm": float(sim_irr),
                                 "rel_err": float(sim_irr / meas_irr - 1),
                                 "pass": bool(abs(sim_irr / meas_irr - 1) <= IRR_TOL)}
    losses = (ls.groupby("treatment")[["percn", "lchlabp", "surqsolp", "sedorgp", "sedminp"]]
              .sum().round(2))
    perc = wb.groupby("treatment")[["perc", "et", "irr", "precip"]].sum().round(1)
    report["losses_2013_2020_sum"] = losses.join(perc).to_dict("index")

    OUT.mkdir(exist_ok=True)
    (OUT / "ltm_validation.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    harvest.to_csv(OUT / "ltm_validation_harvest.csv", index=False)

    # -- print ------------------------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(harvest.round(2).to_string(index=False))
    print("\nsugarbeet root DM (t/ha):", json.dumps(report["sugarbeet_root_dm"], default=float))
    print("contrast:", {k: round(v, 3) for k, v in report["sugarbeet_contrast"].items()})
    print("\nuptake 52A:")
    print(pd.DataFrame(report["uptake_52A"]).round(1).to_string(index=False))
    for k in ("profile_no3_mg_kg", "olsen_p_mg_kg_0_30"):
        print(f"\n{k}: r = {soil[k]['pearson_r']}")
        print(pd.DataFrame(soil[k]["series"]).round(1).to_string(index=False))
    print("\nNO3 2020 contrast:", soil["no3_2020_contrast"])
    print("pre-study:", soil["prestudy_2012_10_13"])
    print("\nN mineralisation 52A:")
    print(pd.DataFrame(report["nmin_52A"]).round(1).to_string(index=False))
    print("\nirrigation gate:", report["irrigation_gate"])
    print("\nlosses + water 2013-2020:")
    print(losses.join(perc).to_string())


if __name__ == "__main__":
    main()
