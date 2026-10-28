"""Port the USDA-ARS Kimberly long-term manure (LTM) trial into data/ltm/*.csv.

Sources (all on the user's machine, ~/Documents/Kimberly, Idaho/USDA Long-Term Manure/):
  [W1] LT Manure Data Summary.xlsx                     (collaborator workbook)
  [W2] LT Manure Properties & P adjustment.xlsx
  [W3] LT Manure Soil Properties and P uptake data.xlsx
  [P1] Leytem et al. 2024, SSSAJ 88(1):71-88, doi:10.1002/saj2.20599  (NminLTstudy.pdf)
       -- this field's rotation (south field: barley 2013 ... sugarbeet 2020)
  [P2] Leytem et al. 2025, Crop Science (Leytem_2025_LT_Manure_Sugarbeet_Nutrient_Uptake.pdf)
       -- sugarbeet years of BOTH adjacent fields; its 2016 beet is the north field
  [W4] ~/Downloads/Long Term Manure Project Input Data.xlsx (collaborator's model input set,
       received 2026-10-25): bulk harvest dates, 2014 kill date, manure date and target-rate
       nutrient fractions, irrigation 2013-2020 on true calendar dates

User decisions 2026-10-25: manure on W4's target-rate basis (5,200 kg/ha x W4 fractions = 52
Mg DM x concentration), not P1 Table 2's applied N (kept as n_applied_p1 for reference); W4's
dates for harvest (bulk), the 2014 kill and manure.

Treatments modelled: 52A (workbook label "30_Annual" = 30 t/ac wet ~ 23 t/ac DM = 52 Mg/ha DM),
fertilizer-only ("fert"), control. Paper-table values are transcribed below with their table.

Workbook columns that are outputs of the collaborator's earlier SWAT model (NO3, SOL_P, DNIT,
NO3L, PERC, NUP/PUP/BIOM, F_MN/A_MN) are NOT measurements and are not ported.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ltm"
SRC = Path.home() / "Documents" / "Kimberly, Idaho" / "USDA Long-Term Manure"
W1 = SRC / "LT Manure Data Summary.xlsx"
W3 = SRC / "LT Manure Soil Properties and P uptake data.xlsx"
W4 = Path.home() / "Downloads" / "Long Term Manure Project Input Data.xlsx"
#: W4 applies 5,200 kg/ha of manure with fractions scaled x10, i.e. 52 Mg DM/ha.
W4_RATE_KG, MANURE_TARGET_MG = 5200.0, 52.0

TRTS = ("52A", "fert", "control")
LB_AC = 1.12085          # lb/ac -> kg/ha
TON_AC = 2.24170         # short ton/ac -> Mg/ha
CWT_AC = 0.112085        # cwt/ac -> Mg/ha
BU_LB = {"barley": 48.0, "wheat": 60.0}  # lb per bushel

# --- [P1] Table 1: manure composition by application year (dry-weight basis) -----------------
#           DM g/kg(fresh), TN g/kg, TC g/kg, IN g/kg, orgN %, P g/kg
MANURE_T1 = {
    2012: (586, 14.8, 286, 2.3, 86.5, 4.6),
    2013: (357, 18.3, 319, 2.9, 86.3, 6.2),
    2014: (620, 12.0, 192, 3.6, 76.9, 3.6),
    2015: (515, 15.0, 220, 3.1, 82.9, 6.7),
    2016: (545, 12.4, 196, 3.6, 77.5, 5.5),
    2017: (539, 11.7, 175, 1.0, 92.1, 4.7),
    2018: (523.2, 11.0, 124.0, 0.3, 97.4, 5.0),
    2019: (568, 10.6, 137, 1.2, 89.8, 6.6),
}
# [P1] Table 2: manure N actually applied to 52A (kg N/ha), this field. Differs from the target
# rate x concentration in [W1] "Summary" (e.g. 2012: 864 vs 769.6); the applied amount wins.
MANURE_N_52A = {2012: 864, 2013: 1234, 2014: 677, 2015: 871, 2016: 659, 2017: 684,
                2018: 581, 2019: 735}
# [P1] Table 2: synthetic fertilizer N (kg N/ha) by year
FERT_N = {
    "52A":     {2013: 62, 2016: 45, 2017: 38},
    "fert":    {2013: 81, 2014: 75, 2015: 107, 2016: 365, 2017: 136, 2018: 115, 2019: 147,
                2020: 89},
    "control": {},
}
# [P1] Table S1 (saj220599-sup-0001-tabless1-s2.docx, received 2026-10-26): fertilizer P (kg P/ha)
# by treatment and year on this field. 52A and control received none. [P2] Table 3's 2014 figure
# (66) disagrees with S1's 151; S1 is this field's record (its MAP-N, 0.485 x 151 = 73 kg, fits
# inside [P1] Table 2's 75 kg N; [P2]'s 2014 N, 107, does not match this field either).
FERT_P_KNOWN = {
    "fert": {2013: 39, 2014: 151, 2015: 19, 2016: 98, 2017: 16, 2018: 38, 2019: 20, 2020: 59},
    "52A": {y: 0 for y in range(2013, 2021)},
}
FERT_P_RANGE: dict = {}
#: MAP 11-52-0: 0.11 kg N per kg, 0.52 P2O5 = 0.227 kg P per kg -> 0.485 kg N per kg P
MAP_N_PER_P = 0.11 / (0.52 / 2.291)

# --- [P1] 2.6 field operations (this field) ---------------------------------------------------
CROPS = {  # year: crop, plant [P1 2.6], fert date(s) [P1 2.6], harvest [W4: bulk], note
    2013: ("barley", "2013-04-04", ["2013-04-02"], "2013-08-05", "malt barley Moravian 69; straw baled+removed"),
    2014: ("sugarbeet", "2014-05-05", ["2014-04-16"], "2014-10-28", "NOT harvested: killed 10-28 [W4] to start decomposition; roots+tops returned"),
    2015: ("wheat", "2015-04-02", ["2015-04-02"], "2015-08-13", "hard red spring wheat Jefferson; straw baled+removed"),
    2016: ("potato", "2016-05-02", ["2016-04-20", "2016-05-25", "2016-07-12"], "2016-10-15", "Russet Burbank; N split preplant/PCU 5-25/urea 7-12, split not reported; tops returned"),
    2017: ("barley", "2017-04-06", ["2017-04-04"], "2017-08-08", "malt barley; straw baled+removed"),
    2018: ("sugarbeet", "2018-05-01", ["2018-04-18"], "2018-10-09", "bulk harvest; tops returned"),
    2019: ("wheat", "2019-04-13", ["2019-04-12"], "2019-08-16", "hard red spring wheat; straw baled+removed"),
    2020: ("sugarbeet", "2020-05-07", ["2020-05-05"], "2020-10-12", "bulk harvest; tops returned"),
}
# Manure "each fall (October or November)" [P2] 2.3, incorporated by disk/moldboard to 15 cm;
# all treatments tilled alike. Day: October 20 [W4].
MANURE_DAY = (10, 20)
PRE_CROP = {2011: "corn silage", 2012: "pinto bean"}  # [P1] 2.1, no manure history


def _wb(path):
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def _d(s: str) -> date:
    return date.fromisoformat(s)


def management() -> pd.DataFrame:
    rows = []
    for yr, (crop, plant, fdates, harv, note) in CROPS.items():
        for trt in TRTS:
            rows.append(dict(year=yr, treatment=trt, op="plant", date=plant, crop=crop,
                             note=note, source="P1 2.6"))
            rows.append(dict(year=yr, treatment=trt, op="harvest", date=harv, crop=crop,
                             note=note, source="W4 Crop plant harvest"))
            n = FERT_N[trt].get(yr, 0)
            p_known = FERT_P_KNOWN.get(trt, {}).get(yr)
            p_lo, p_hi = (0, 0) if trt == "control" else FERT_P_RANGE.get(yr, (np.nan, np.nan))
            # All fertiliser P was MAP ([P1] 2.6), whose N is inside Table 2's totals: P can be
            # at most N / MAP_N_PER_P, so no recorded N means no P.
            if p_known is None and n == 0:
                p_known = 0
            p_hi = min(p_hi, n / MAP_N_PER_P) if not np.isnan(p_hi) else p_hi
            p_lo = min(p_lo, p_hi) if not np.isnan(p_lo) else p_lo
            if n or trt != "control":
                rows.append(dict(
                    year=yr, treatment=trt, op="fert", date=fdates[0], crop=crop,
                    n_kg_ha=n, p_kg_ha=p_known if p_known is not None else np.nan,
                    p_kg_ha_lo=p_lo if p_known is None else p_known,
                    p_kg_ha_hi=p_hi if p_known is None else p_known,
                    note=("urea + MAP + KCl, incorporated same day"
                          + ("; dates " + ",".join(fdates) if len(fdates) > 1 else "")
                          + ("" if p_known is not None else "; P unknown")),
                    source="P1 Table 2 (N); P1 Table S1 (P)"))
    w4 = w4_manure()
    for yr, (dm, tn, tc, inn, orgn, p) in MANURE_T1.items():
        mon, day = MANURE_DAY
        for trt in TRTS:
            if trt == "52A":
                mn, mp, on, op = w4[yr]
                rows.append(dict(
                    year=yr, treatment=trt, op="manure", date=date(yr, mon, day).isoformat(),
                    dm_mg_ha=MANURE_TARGET_MG, n_kg_ha=round(W4_RATE_KG * (mn + on), 1),
                    min_n_kg_ha=round(W4_RATE_KG * mn, 1), org_n_kg_ha=round(W4_RATE_KG * on, 1),
                    p_kg_ha=round(W4_RATE_KG * (mp + op), 1),
                    min_p_kg_ha=round(W4_RATE_KG * mp, 1),
                    c_kg_ha=round(MANURE_TARGET_MG * tc, 0), dm_frac_fresh=dm / 1000,
                    n_applied_p1=MANURE_N_52A[yr],
                    note="target rate (user decision); dry lot, stacked; disked to 15 cm same day",
                    source="W4 Fertilizer application (fractions x 5,200 kg/ha); C, DM: P1 Table 1"))
            rows.append(dict(year=yr, treatment=trt, op="tillage",
                             date=date(yr, mon, day).isoformat(), depth_mm=150,
                             note="fall disk/moldboard plow after manure; all treatments",
                             source="P2 2.3"))
    df = pd.DataFrame(rows).sort_values(["treatment", "date", "op"]).reset_index(drop=True)
    return df


def w4_manure() -> dict[int, tuple[float, float, float, float]]:
    """[W4] per-fall (mineral N, mineral P, organic N, organic P) fractions at 5,200 kg/ha."""
    ws = _wb(W4)["Fertilizer application"]
    out = {}
    for r in ws.iter_rows(min_row=3, values_only=True):
        if isinstance(r[0], int) and 2012 <= r[0] <= 2019 and isinstance(r[1], float):
            out[r[0]] = tuple(float(v) for v in r[1:5])
    assert sorted(out) == list(range(2012, 2020)), sorted(out)
    return out


def irrigation_daily() -> pd.DataFrame:
    """2013-2020 from [W4] (true dates). 2012 (history year) from [W1], shifted one day later:
    [W1]'s 2016 and 2020 events are all exactly one day before [W4]'s (leap years; the series
    was evidently built on day of year), and 2012 is also a leap year."""
    ws = _wb(W4)["Irrigation"]
    w4 = [(r[0].date(), float(r[1])) for r in ws.iter_rows(min_row=2, values_only=True)
          if r[0] is not None and r[1]]
    ws = _wb(W1)["Irrigation"]
    w1 = [(r[0].date(), float(r[1] or 0)) for r in ws.iter_rows(min_row=2, values_only=True)
          if r[0] is not None and r[0].year == 2012 and r[1]]
    w1 = [(d + pd.Timedelta(days=1 if d >= date(2012, 2, 29) else 0).to_pytimedelta(), mm)
          for d, mm in w1]
    df = pd.DataFrame([(d, mm, "W1 Irrigation, +1 day (leap-year offset)") for d, mm in w1]
                      + [(d, mm, "W4 Irrigation") for d, mm in w4],
                      columns=["date", "irr_mm", "source"])
    # [P1] 3.1 season totals 2013-2020: 385, 538, 419, 776, 574, 788, 289, 772 mm
    tot = df.assign(y=pd.to_datetime(df.date).dt.year).groupby("y").irr_mm.sum()
    for y, ref in zip(range(2013, 2021), (385, 538, 419, 776, 574, 788, 289, 772)):
        assert abs(tot[y] - ref) <= 1.5, (y, tot[y], ref)
    return df


def _num(x):
    return float(x) if isinstance(x, (int, float)) else np.nan


def yields() -> pd.DataFrame:
    """52A per plot [W3 plant P uptake]; treatment means for sugarbeet [P2 3.4]."""
    ws = _wb(W3)["plant P uptake"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        plot, trt, rep, field, crop, yr = r[:6]
        grain_bu = _num(r[7])                   # moisture-corrected (barley 14.5 %, wheat 13.5 %)
        crop = crop.strip()
        rec = dict(year=int(yr), treatment="52A", plot=plot, field=field, crop=crop,
                   biomass_dm_mg_ha=_num(r[9]) * TON_AC, p_uptake_kg_ha=_num(r[10]) * LB_AC,
                   source="W3 plant P uptake")
        if crop in BU_LB:
            moist = 0.145 if crop == "barley" else 0.135
            rec["yield_fresh_mg_ha"] = grain_bu * BU_LB[crop] * LB_AC / 1000
            rec["yield_dm_mg_ha"] = rec["yield_fresh_mg_ha"] * (1 - moist)
            rec["yield_basis"] = f"grain at {moist:.1%} moisture"
        elif crop == "sugarbeet":
            rec["yield_fresh_mg_ha"] = _num(r[11]) * TON_AC
            rec["yield_basis"] = "fresh root"
        elif crop == "potato":
            rec["yield_fresh_mg_ha"] = _num(r[12]) * CWT_AC
            rec["yield_basis"] = "fresh total tuber"
        rows.append(rec)
    df = pd.DataFrame(rows)
    # [P2] 3.4: sugarbeet fresh root yield averaged over 2014/16/18/20 and both fields
    for trt, v in (("control", 48.7), ("fert", 70.7), ("52A", (74.9, 78.2))):
        lo, hi = v if isinstance(v, tuple) else (v, v)
        df = pd.concat([df, pd.DataFrame([dict(
            year=0, treatment=trt, crop="sugarbeet", yield_fresh_mg_ha=(lo + hi) / 2,
            yield_fresh_lo=lo, yield_fresh_hi=hi, yield_basis="fresh root, 4-yr 2-field mean",
            source="P2 3.4 (52A: range over all manure treatments)")])], ignore_index=True)
    # [P2] Table 5: sugarbeet root / foliage DM and root moisture, 4 years x 2 fields pooled
    t5 = {"52A": (17.2, 8.7, 77.8), "fert": (17.7, 5.8, 74.8), "control": (12.7, 3.8, 73.5)}
    for trt, (root, fol, moist) in t5.items():
        df = pd.concat([df, pd.DataFrame([dict(
            year=0, treatment=trt, crop="sugarbeet", yield_dm_mg_ha=root,
            foliage_dm_mg_ha=fol, root_moisture_pct=moist,
            yield_basis="root DM, 4-yr 2-field mean", source="P2 Table 5")])], ignore_index=True)
    return df


def uptake() -> pd.DataFrame:
    """52A crop-year biomass and N/P uptake means [W1 Crop Uptake, 'From April Leytem']."""
    ws = _wb(W1)["Crop Uptake"]
    rows = []
    for r in ws.iter_rows(min_row=4, max_row=10, values_only=True):
        yr, crop, bm, bm_sd, nup, nup_sd, pup, pup_sd = r[11:19]
        rows.append(dict(year=int(yr), treatment="52A", crop=crop.strip(),
                         biomass_mg_ha=bm, biomass_sd=bm_sd, n_up_kg_ha=nup, n_up_sd=nup_sd,
                         p_up_kg_ha=pup, p_up_sd=pup_sd,
                         source="W1 Crop Uptake 'From April Leytem' (measured)"))
    return pd.DataFrame(rows)


def soil_initial() -> pd.DataFrame:
    """Layered texture (4 plots, measured) and the measured physical / pre-study chemistry."""
    wb = _wb(W3)
    tex = pd.DataFrame(list(wb["soil texture"].iter_rows(min_row=2, values_only=True)))
    tex = tex[[0, 2, 4, 5, 6]].dropna(subset=[2])
    tex.columns = ["plot", "depth", "clay", "silt", "sand"]
    tex[["top_mm", "bot_mm"]] = tex.depth.str.replace("mm", "").str.split("-", expand=True).astype(float)
    g = tex.groupby(["top_mm", "bot_mm"])[["clay", "silt", "sand"]].agg(["mean", "std", "count"])
    out = []
    for (top, bot), r in g.iterrows():
        out.append(dict(top_mm=top, bot_mm=bot,
                        clay_pct=100 * r[("clay", "mean")], silt_pct=100 * r[("silt", "mean")],
                        sand_pct=100 * r[("sand", "mean")], n_plots=int(r[("clay", "count")]),
                        source="W3 soil texture (52A plots, north field)"))
    df = pd.DataFrame(out)
    # physical props, 52A plots in 2016 (manure-affected), [W3 soil phys props] 'Average' row
    phys = list(wb["soil phys props"].iter_rows(values_only=True))
    avg = next(r for r in phys if r[0] == "Average")
    hdr = phys[0]
    pick = {k: avg[hdr.index(k)] for k in hdr if k and ("Bulk Density" in k or "bar" in k
                                                      or "Saturated" in k or "Organic C" in k)}
    meas = pd.DataFrame([dict(item=k.strip(), value=v, source="W3 soil phys props, 52A 2016 mean")
                         for k, v in pick.items()])
    pre = [  # [P2] 2.2: 0-30 cm, all plots, one week before the first manure (fall 2012)
        ("pH", 8.0), ("EC dS/m", 1.02), ("total N (as printed: mg/kg; % intended)", 0.083),
        ("Olsen P mg/kg", 11.1), ("K mg/kg", 115), ("CaCO3 %", 5.9), ("OM %", 1.2),
        ("clay %", 11), ("silt %", 71), ("sand %", 18), ("BD 0-30 g/cm3", 1.34)]
    pre = pd.DataFrame([dict(item=k, value=v, source="P2 2.2 pre-study 0-30 cm") for k, v in pre])
    bd = pd.DataFrame([dict(item="BD 0-30 g/cm3", value=1.28, source="P1 2.8 field mean"),
                       dict(item="BD 30-60 g/cm3", value=1.37, source="P1 2.8 field mean")])
    return df, pd.concat([pre, bd, meas], ignore_index=True)


def soil_series() -> pd.DataFrame:
    """Spring preplant soil, 52A per plot per year [W3 CaCO3 sheet]; 52A profile means [W1];
    treatment contrasts [P1 Table 5, P2 Table 4]."""
    wb = _wb(W3)
    rows = list(wb["soil CaCO3 and Carbon"].iter_rows(values_only=True))
    hdr = [str(h or "") for h in rows[0]]
    col = lambda key: next(i for i, h in enumerate(hdr) if key in h)  # noqa: E731
    want = {"no3_mg_kg_0_30": col("Nitrate-N"), "nh4_mg_kg_0_30": col("Ammonium-N"),
            "olsen_p_mg_kg_0_30": col("NaHCO3\nP"), "tn_pct_0_30": col("Nitrogen %          0-12"),
            "c_pct_0_30": col("Carbon %     0-12"), "om_pct_0_30": col("Organic\nMatter %   0-12"),
            "ec_ds_m_0_30": col("Electrical")}
    out = []
    for r in rows[1:]:
        if r[0] is None or not isinstance(r[5], (int, float)):
            continue
        for var, i in want.items():
            out.append(dict(year=int(r[5]), treatment="52A", plot=r[0], var=var,
                            value=_num(r[i]), source="W3 soil CaCO3 and Carbon (preplant)"))
    ws = wb_w1 = _wb(W1)["N & P plots actual irr"]
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[11] is None:
            continue
        for var, (m, s) in {"profile_no3_mg_kg": (11, 12), "olsen_p_mg_kg_0_30": (13, 14),
                            "olsen_p_mg_kg_30_60": (15, 16)}.items():
            out.append(dict(year=r[0].year, date=r[0].date(), treatment="52A", plot="mean",
                            var=var, value=r[m], sd=r[s],
                            source="W1 N & P plots 'Sampled' columns (depth of profile NO3 "
                                   "not stated in workbook)"))
    del wb_w1
    t5 = {"52A": 124.0, "fert": 36.2, "control": 14.9}      # [P1] Table 5, 2020, 0-60 cm
    t4 = {"52A": (55.89, 124.98), "fert": (11.32, 10.02), "control": (9.37, 4.95)}  # [P2] T4
    for trt in TRTS:
        out.append(dict(year=2020, treatment=trt, plot="mean", var="no3_mg_kg_0_60",
                        value=t5[trt], source="P1 Table 5"))
        for var, v in zip(("no3_mg_kg_0_30", "olsen_p_mg_kg_0_30"), t4[trt]):
            out.append(dict(year=0, treatment=trt, plot="mean", var=var, value=v,
                            source="P2 Table 4 (mean of 2014/16/18/20, both fields)"))
    return pd.DataFrame(out)


def nmin() -> pd.DataFrame:
    """Net seasonal IN mineralisation 0-60 cm, 52A [W1 N Min 'Leytem'], with [P1] Table 4 dates."""
    ws = _wb(W1)["N Min"]
    dates = {2013: ("04-12", "09-12"), 2014: ("05-01", "10-09"), 2015: ("04-13", "10-12"),
             2016: ("05-05", "11-11"), 2017: ("05-03", "10-11"), 2018: ("05-08", "10-09"),
             2019: ("04-18", "10-10"), 2020: ("06-12", "10-06")}
    rows = []
    for r in ws.iter_rows(min_row=7, max_row=14, values_only=True):
        yr, v = r[6], r[8]
        rows.append(dict(year=int(yr), treatment="52A", net_in_kg_ha=v,
                         bag_in=f"{yr}-{dates[yr][0]}", bag_out=f"{yr}-{dates[yr][1]}",
                         source="W1 N Min 'Leytem' column; dates P1 Table 4"))
    # [P1] 3.5: control and fertilizer did not differ; 45-150 kg N/ha, 72 avg excluding residue years
    for trt in ("fert", "control"):
        rows.append(dict(year=0, treatment=trt, net_in_kg_ha=72, lo=45, hi=150,
                         source="P1 3.5 (summary only; per-year values are in Figure 4)"))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layers, props = soil_initial()
    tables = {"management": management(), "irrigation_daily": irrigation_daily(),
              "yield": yields(), "uptake": uptake(), "soil_layers": layers,
              "soil_props": props, "soil_series": soil_series(), "nmin": nmin()}
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"data/ltm/{name}.csv  {len(df)} rows")
    m = tables["management"]
    miss = m[(m.op == "fert") & m.p_kg_ha.isna()]
    print(f"\nmissing fertilizer P (P1 Table S1): {len(miss)} treatment-years -> "
          + ", ".join(f"{r.treatment} {r.year}" for r in miss.itertuples()))


if __name__ == "__main__":
    main()
