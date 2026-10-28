#!/usr/bin/env python3
"""Build the long-term manure (LTM) scenario, model/ltm/TxtInOut, from data/ltm/*.csv.

Three 1-ha HRUs in one engine run, one per treatment: 52A (hru1), fertilizer-only (hru2),
control (hru3). 2012 (pinto bean, the field's pre-study crop) is history -- set up as the
collaborator did (G. Richards, email 2026-10-27): auto-irrigated, 100 kg/ha N + 50 kg/ha P at
planting; 2013-2020 is the trial. Measured irrigation at practice.MEASURED_IRR_EFF, the 2012
auto-irrigation at practice.BASE_IRR_EFF. Inputs and their precedence (CLAUDE.md rules 10, 11, 11b):

* base TxtInOut: Ai-SWAT-Plus's barley monoculture build (model/crops/barl), for its codes,
  parameters.bsn, hydrology (esco 0.48 / epco 1.0 / pet_co 1.0) and file wiring;
* engine: model/ltm/engine (the patched rev-62 engine every crop fit was made on);
* crop parameters: model/ltm/crop_params/*.json, verbatim (scripts/ltm_crop_params.py);
* weather: gridMET at the LTM field (scripts/port_gridmet.py --site ltm);
* soil: SSURGO mukey 80295 (Soil Data Access at the field point) for hydraulics, the trial's
  measured texture / bulk density / pre-study chemistry for the rest (data/ltm/soil_*.csv);
* management: data/ltm/management.csv (manure, fertiliser, dates) and irrigation_daily.csv.

    uv run python scripts/build_ltm.py [--fert-p mid|lo|hi]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from swat_gym.ltm.practice import (AUTO_IRR, BASE_IRR_EFF, MEASURED_IRR_EFF,  # noqa: E402
                                   SPINUP_FERT, auto_op, spinup_window)
from swat_gym.ltm.runner import _irr_table  # noqa: E402
BASE = ROOT / "model" / "crops" / "barl" / "TxtInOut"
LTM = ROOT / "model" / "ltm"
TIO = LTM / "TxtInOut"
PARAMS = LTM / "crop_params"
DATA = ROOT / "data" / "ltm"

TREATMENTS = {"52A": "52a", "fert": "fert", "control": "ctrl"}   # treatment -> tag; hru order
COMM = "ltm_comm"
PLANTS = ("barl", "sgbt", "wwht", "pota", "ptbn", "alfa", "corn")
CROP_PLANT = {"barley": "barl", "sugarbeet": "sgbt", "wheat": "wwht", "potato": "pota"}
START, END = 2012, 2020

# 2012 history: pinto bean on Ai-SWAT-Plus's sheet dates (emerge 7/1, harvest 9/25); the
# collaborator's 100 kg/ha N + 50 kg/ha P at planting and auto-irrigation (practice.SPINUP_*).
HISTORY = [("plnt", date(2012, 7, 1), "ptbn", "null", 0.0),
           *[("fert", date(2012, 7, 1), prod, "broadcast", kg) for prod, kg in SPINUP_FERT],
           ("hvkl", date(2012, 9, 25), "ptbn", "wb_ptbn", 0.0)]
BEAN_IRR = "ltm_bean_irr"     #: lum.dtl table auto-irrigating the 2012 beans

# Fall incorporation "disking or moldboard plowing to a 15-cm depth" (P2 2.3); mixing
# efficiency and roughness from the GRACEnet disk op (port_management.TILLAGE).
TILL = ("ltm_till", 0.85, 150.0, 30.0)
# 2014 beets were not harvested and were tilled in (P1 2.6): harvest nothing, then kill.
NO_HARVEST = ("ltm_none", "biomass", 0.0, 1.0, 0.0)
# Fertiliser forms (P1 2.6: urea, MAP, KCl). MAP 11-52-0: 0.11 N, 0.52 P2O5 = 0.227 P.
MAP = ("ltm_map", 0.11, 0.227)
NH3_FRAC = 0.99   # port_management.NH3_FRAC (reference fert.dat FNH3N)
PROBE = "elem_n"
#: "One week prior to the first fall manure application in 2012" (P2 2.2); manure 10-20 [W4].
PRESTUDY_SAMPLE = date(2012, 10, 13)

# Soil layers (bottom mm). SSURGO 80295 (Soil Data Access, 2026-10-25): Ap 0-280 mm, Bk
# 280-1520 mm; values per horizon:  awc, ksat um/s, om %, caco3 %, ph, ec, kwfact
SSURGO = {"Ap": (0.20, 9.00, 1.50, 8.0, 7.9, 0.0, 0.43),
          "Bk": (0.18, 2.82, 0.75, 23.0, 8.4, 5.0, 0.64)}
LAYERS_MM = (150, 300, 600, 900, 1220, 1520)
OM_TO_OC = 1 / 1.724   # Van Bemmelen


def rows(path: Path) -> list[str]:
    return path.read_text().splitlines()


def put(name: str, lines: list[str]) -> None:
    (TIO / name).write_text("\n".join(lines) + "\n")


def load_params() -> dict:
    return {p: json.loads((PARAMS / f"{p}.json").read_text()) for p in PLANTS}


# -- crop parameters (verbatim) -----------------------------------------------------------
def write_plants(params: dict) -> None:
    lines = rows(TIO / "plants.plt")
    for p, rec in params.items():
        idx = [i for i, ln in enumerate(lines) if ln.split()[:1] == [p]]
        if len(idx) != 1:
            raise SystemExit(f"plants.plt: {len(idx)} rows for {p}")
        lines[idx[0]] = rec["plants_plt_row"]
    lines[0] = "plants.plt: swat-kimberly build_ltm.py; crop rows verbatim from Ai-SWAT-Plus"
    put("plants.plt", lines)

    cal = ["calibration.cal: heat units to maturity per crop (Ai-SWAT-Plus fits), all HRUs",
           str(len(params)),
           "NAME CHG_TYP VAL CONDS LYR1 LYR2 YEAR1 YEAR2 DAY1 DAY2 OBJ_TOT"]
    objs = " ".join(str(i + 1) for i in range(len(TREATMENTS)))
    for p, rec in params.items():
        cal.append(f"phu_mat absval {rec['phu_mat']} 1 0 0 0 0 0 0 {len(TREATMENTS)} {objs}")
        # Not "plant": rev 62's cal_conditions tests a "plant" condition against the community's
        # FIRST plant only (it exits the plant loop after one pass), so in a multi-plant
        # community only that plant would match. Any other var passes the filter, and the
        # phu_mat branch then matches cond%targc to each plant by name -- the intended per-plant
        # override. Verified by the drop test in PROVENANCE (LTM section).
        cal.append(f"phu_plant = 0 {p}")
    put("calibration.cal", cal)

    harv = rows(TIO / "harv.ops")
    keep = harv[:2] + [ln for ln in harv[2:] if ln.split() and ln.split()[0] not in
                       {r["harv_op"] for r in params.values()} | {NO_HARVEST[0]}]
    keep += [r["harv_ops_row"] for r in params.values()]
    n, typ, idx, eff, bm = NO_HARVEST
    keep.append(f"{n:<22}{typ:>14}{idx:>14.5f}{eff:>14.5f}{bm:>14.5f}  ltm_2014_beets_tilled_in")
    put("harv.ops", keep)

    lines = ["plant.ini: written by swat-kimberly build_ltm.py",
             "pcom_name          plt_cnt  rot_yr_ini          plt_name     lc_status      lai_init       bm_init      phu_init      plnt_pop      yrs_init      rsd_init  ",
             f"{COMM:<19}{len(PLANTS):>7}{1:>11}  "]
    for p in PLANTS:
        lines.append(f"{'':>40}{p:<8}{'n':>6}" + "".join(f"{0.0:>14.5f}" for _ in range(6)) + "  ")
    put("plant.ini", lines)


# -- soil ------------------------------------------------------------------------------------
def write_soil() -> None:
    tex = pd.read_csv(DATA / "soil_layers.csv")
    props = pd.read_csv(DATA / "soil_props.csv").set_index("item")["value"]
    pre_om, pre_ph = float(props["OM %"]), float(props["pH"])
    pre_caco3, pre_ec = float(props["CaCO3 %"]), float(props["EC dS/m"])
    bd_top = 1.28   # P1 2.8 field mean (data/ltm/soil_props.csv), 0-30 cm
    bd_mid = 1.37   # P1 2.8 field mean, 30-60 cm
    out, top = [], 0
    for bot in LAYERS_MM:
        mid = (top + bot) / 2
        hz = "Ap" if mid < 280 else "Bk"
        awc, ksat, om, caco3, ph, ec, kw = SSURGO[hz]
        t = tex[(tex.top_mm < bot) & (tex.bot_mm > top)]
        w = (t[["bot_mm"]].clip(upper=bot).values.ravel() - t[["top_mm"]].clip(lower=top).values.ravel())
        clay, silt, sand = ((t[c] * w).sum() / w.sum() for c in ("clay_pct", "silt_pct", "sand_pct"))
        norm = 100 / (clay + silt + sand)
        if bot <= 300:   # measured pre-study 0-30 cm chemistry (P2 2.2)
            om, caco3, ph, ec, bd = pre_om, pre_caco3, pre_ph, pre_ec, bd_top
        else:
            bd = bd_mid if bot <= 600 else 1.38   # SSURGO Bk dbthirdbar below 60 cm
        out.append((bot, bd, awc, ksat * 3.6, om * OM_TO_OC, clay * norm, silt * norm,
                    sand * norm, 0.0, 0.3, kw, ec, caco3, ph))
        top = bot
    lines = rows(TIO / "soils.sol")[:2]
    lines.append(f"{'ltm_portneuf':<32}{len(out):>3}{'C':>18}{LAYERS_MM[-1]:>14.5f}"
                 f"{0.5:>14.5f}{0.5:>14.5f}  {'SIL':<35}")
    for lyr in out:
        lines.append(f"{'':>132}" + "".join(f"{v:>14.5f}" for v in lyr))
    put("soils.sol", lines)
    nut = rows(TIO / "nutrients.sol")
    p = nut[2].split()
    olsen = float(props["Olsen P mg/kg"])
    p[2] = f"{olsen:.5f}"   # lab_p <- pre-study Olsen P 0-30 cm (P2 2.2); nitrate kept
    nut[2] = "  ".join(p)
    put("nutrients.sol", nut)


# -- management -----------------------------------------------------------------------------
def _op(typ, d: date, d1, d2="null", d3=0.0) -> tuple[date, int, str]:
    # a probe reads the soil before anything else that day; manure, then incorporate
    order = {"probe": -1, "fert": 0, "till": 1, "plnt": 2, "irrm": 3, "hvkl": 4}[typ]
    if typ == "probe":   # 0 kg elemental N: inert, prints soil NO3 / solution P to mgt_out.txt
        typ, d1, d2, d3 = "fert", PROBE, "broadcast", 0.0
    line = f"{'':>48}{typ:<10}{d.month:>6}{d.day:>10}{0.0:>14.5f}{d1:>18}{d2:>18}{d3:>14.5f}  "
    return d, order, line


def probe_dates() -> list[date]:
    """Measured soil sampling dates (data/ltm/soil_series.csv) + the pre-study sample."""
    ss = pd.read_csv(DATA / "soil_series.csv").dropna(subset=["date"])
    return sorted({date.fromisoformat(d) for d in ss.date} | {PRESTUDY_SAMPLE})


def schedules(fert_p: str, probes: bool = True) -> tuple[dict[str, list], list[str], list[str]]:
    """Ops per treatment; plus the fertilizer.frt and irr.ops rows they reference."""
    m = pd.read_csv(DATA / "management.csv")
    irr = pd.read_csv(DATA / "irrigation_daily.csv", parse_dates=["date"])
    # 2012 is auto-irrigated (HISTORY); measured events from 2013
    irr = irr[(irr.irr_mm > 0) & (irr.date.dt.year.between(START + 1, END))].reset_index(drop=True)
    irr_rows = [f"{f'ltm{i:04d}':<22}{r.irr_mm:>8.5f}{MEASURED_IRR_EFF:>14.5f}" + f"{0.0:>14.5f}" * 5
                + "  ltm_measured" for i, r in enumerate(irr.itertuples(), 1)]
    irr_rows.append(f"{auto_op(BASE_IRR_EFF):<22}{AUTO_IRR['mm']:>8.5f}{BASE_IRR_EFF:>14.5f}"
                    + f"{0.0:>14.5f}" * 5 + "  ltm_auto_irrigation_pass")
    irr_ops = [_op("irrm", r.date.date(), f"ltm{i:04d}") for i, r in enumerate(irr.itertuples(), 1)]

    frt = [f"{MAP[0]:<22}{MAP[1]:>10.5f}{MAP[2]:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}"
           f"{'null':>14}  monoammonium_phosphate_11-52-0",
           f"{SPINUP_FERT[0][0]:<22}{1.0:>10.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}"
           f"{'null':>14}  elemental_N_2012_beans",
           f"{SPINUP_FERT[1][0]:<22}{0.0:>10.5f}{1.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}"
           f"{'null':>14}  elemental_P_2012_beans"]
    out = {}
    for trt, tag in TREATMENTS.items():
        ops = [_op(t, d, a, b, c) for t, d, a, b, c in HISTORY] + list(irr_ops)
        if probes:
            ops += [_op("probe", d, None) for d in probe_dates()]
        for r in m[m.treatment == trt].itertuples():
            d = date.fromisoformat(r.date)
            if r.op == "plant":
                ops.append(_op("plnt", d, CROP_PLANT[r.crop]))
            elif r.op == "harvest":
                plt = CROP_PLANT[r.crop]
                hop = NO_HARVEST[0] if (r.year == 2014 and plt == "sgbt") else \
                    json.loads((PARAMS / f"{plt}.json").read_text())["harv_op"]
                ops.append(_op("hvkl", d, plt, hop))
            elif r.op == "tillage":
                ops.append(_op("till", d, TILL[0]))
            elif r.op == "manure":
                name = f"ltm{r.year}"
                if trt == "52A":
                    kg = r.dm_mg_ha * 1000
                    frt.append(f"{name:<22}{r.min_n_kg_ha / kg:>10.6f}"
                               f"{r.min_p_kg_ha / kg:>14.6f}"
                               f"{r.org_n_kg_ha / kg:>14.6f}{(r.p_kg_ha - r.min_p_kg_ha) / kg:>14.6f}"
                               f"{NH3_FRAC:>14.5f}{'null':>14}  LTM_dairy_drylot_{r.year}")
                    ops.append(_op("fert", d, name, "broadcast", kg))
            elif r.op == "fert":
                n_tot = float(r.n_kg_ha)
                if pd.isna(r.p_kg_ha):
                    lo, hi = float(r.p_kg_ha_lo), float(r.p_kg_ha_hi)
                    p = {"lo": lo, "hi": hi, "mid": (lo + hi) / 2}[fert_p]
                else:
                    p = float(r.p_kg_ha)
                map_kg = p / MAP[2]
                urea_n = n_tot - MAP[1] * map_kg
                if urea_n < -1e-9:
                    raise SystemExit(f"{trt} {r.year}: P {p} as MAP brings more N than the "
                                     f"{n_tot} kg N recorded")
                if map_kg > 0:
                    ops.append(_op("fert", d, MAP[0], "broadcast", map_kg))
                dates = [d] if r.year != 2016 else \
                    [date(2016, 4, 20), date(2016, 5, 25), date(2016, 7, 12)]
                for dd in dates:   # 2016 potato: split not reported -> equal parts
                    if urea_n > 1e-9:
                        ops.append(_op("fert", dd, "elem_n", "broadcast", urea_n / len(dates)))
        ops.sort(key=lambda t: (t[0], t[1]))
        out[tag] = ops
    return out, frt, irr_rows


def write_management(fert_p: str, probes: bool = True) -> None:
    scheds, frt, irr_rows = schedules(fert_p, probes)
    lines = ["management.sch: written by swat-kimberly build_ltm.py",
             "name                                       numb_ops  numb_auto            op_typ       mon       day        hu_sch          op_data1          op_data2      op_data3  "]
    for tag, ops in scheds.items():
        lines.append(f"{'ltm_' + tag:<34}{len(ops):>6}{1:>11}  ")
        lines.append(f"{'':>48}{BEAN_IRR:<18}")
        lines += [o[2] for o in ops]
    put("management.sch", lines)
    if AUTO_IRR["sw_frac"] is None:
        raise SystemExit("auto-irrigation trigger not fitted: run scripts/ltm_fit_auto_irr.py")
    dtl = (TIO / "lum.dtl").read_text()
    head, count, rest = dtl.split("\n", 2)
    table = _irr_table(BEAN_IRR, [(START, *spinup_window(START))], AUTO_IRR["sw_frac"],
                       auto_op(BASE_IRR_EFF))
    (TIO / "lum.dtl").write_text(f"{head}\n{int(count.strip()) + 1}\n{rest}".rstrip("\n")
                                 + "\n" + table)

    base = rows(TIO / "fertilizer.frt")
    names = {ln.split()[0] for ln in frt}
    put("fertilizer.frt", [ln for ln in base if not ln.split() or ln.split()[0] not in names] + frt)
    base = rows(TIO / "irr.ops")
    put("irr.ops", [ln for ln in base if not ln.split() or not ln.split()[0].startswith("ltm")]
        + irr_rows)
    til = rows(TIO / "tillage.til")
    n, eff, dp, rough = TILL
    put("tillage.til", [ln for ln in til if ln.split()[:1] != [n]]
        + [f"{n:<22}{eff:>10.5f}{dp:>14.5f}{rough:>14.5f}{0.0:>14.5f}{0.0:>14.5f}  ltm_fall_15cm"])


# -- HRUs --------------------------------------------------------------------------------------
def write_hrus() -> None:
    """k = 3 one-ha HRUs; same wiring as swat_gym.crop._write_batched_inputs."""
    k = len(TREATMENTS)
    tags = list(TREATMENTS.values())
    lum = rows(TIO / "landuse.lum")
    r = lum[2].split()
    put("landuse.lum", lum[:2] + ["  ".join([f"lum_{t}", r[1], COMM, f"ltm_{t}"] + r[4:])
                                  for t in tags])
    hd = rows(TIO / "hru-data.hru")
    r = hd[2].split()
    put("hru-data.hru", hd[:2] + ["  ".join([str(i + 1), f"hru{i + 1}", r[2], r[3], "ltm_portneuf",
                                             f"lum_{t}"] + r[6:]) for i, t in enumerate(tags)])
    hc = rows(TIO / "hru.con")
    r = hc[2].split()
    put("hru.con", hc[:2] + ["  ".join([str(i + 1), f"hru{i + 1}", str(i + 1)] + r[3:7]
                                       + [str(i + 1)] + r[8:]) for i in range(k)])
    for name in ("rout_unit.ele", "ls_unit.ele"):
        lines = rows(TIO / name)
        r = lines[2].split()
        out = []
        for i in range(k):
            rr = list(r)
            rr[0], rr[1], rr[3], rr[4] = str(i + 1), f"hru{i + 1}", str(i + 1), f"{1.0 / k:.6E}"
            out.append("  ".join(rr))
        put(name, lines[:2] + out)
    lines = rows(TIO / "rout_unit.def")
    r = lines[2].split()
    put("rout_unit.def", lines[:2] + ["  ".join([r[0], r[1], "2", "1", f"-{k}"])])
    lines = rows(TIO / "ls_unit.def")
    r = lines[3].split()
    put("ls_unit.def", lines[:3] + ["  ".join([r[0], r[1], f"{k:.5f}", "2", "1", f"-{k}"])])
    for name in ("rout_unit.con", "aquifer.con"):
        lines = rows(TIO / name)
        out = lines[:2]
        for line in lines[2:]:
            p = line.split()
            p[3] = f"{k:.5f}"
            out.append("  ".join(p))
        put(name, out)
    lines = rows(TIO / "object.cnt")
    p = lines[2].split()
    p[1], p[2], p[3], p[4] = f"{k:.5f}", f"{k:.5f}", str(int(p[3]) - 1 + k), str(k)
    put("object.cnt", lines[:2] + ["  ".join(p)])


#: Engine defaults for the active-humus fraction: fr_hum_act (initial, nutrients.sol) and
#: n_act_frac (equilibrium of the active<->stable transfer, carbon_module). The LTM fit sets
#: both to one value f, so the pool starts at its equilibrium (scripts/ltm_calibrate_n.py).
F_ACT_DEFAULT = 0.02


def set_active_fraction(nut: str, cal: str, cal_parms: str, f: float) -> tuple[str, str, str]:
    """Return (nutrients.sol, calibration.cal, cal_parms.cal) with the active fraction = f."""
    lines = nut.splitlines()
    p = lines[2].split()
    p[4] = f"{f:.5f}"                      # fr_hum_act
    lines[2] = "  ".join(p)
    nut = "\n".join(lines) + "\n"
    lines = [ln for ln in cal.splitlines() if ln.split()[:1] != ["n_act_frac"]]
    n = sum(1 for ln in lines[3:] if ln.split() and ln.split()[0] in ("phu_mat",))
    lines[1] = str(n + 1)
    lines.append(f"n_act_frac absval {f:.5f} 0 0 0 0 0 0 0 0")
    cal = "\n".join(lines) + "\n"
    lines = cal_parms.splitlines()
    if not any(ln.split()[:1] == ["n_act_frac"] for ln in lines[3:]):
        lines.append(f"{'n_act_frac':<27}{'bsn':>8}{0.001:>14.5f}{0.5:>14.5f}{'null':>18}  ")
        lines[1] = str(int(lines[1]) + 1)
    return nut, cal, "\n".join(lines) + "\n"


def apply_n_cycle() -> dict | None:
    """Soil N-cycle fit (scripts/ltm_calibrate_n.py) if applied, else the engine defaults."""
    path = LTM / "n_cycle.json"
    fit = json.loads(path.read_text()) if path.is_file() else None
    f = fit["f_act"] if fit else F_ACT_DEFAULT
    nut, cal, cp = set_active_fraction((TIO / "nutrients.sol").read_text(),
                                       (TIO / "calibration.cal").read_text(),
                                       (TIO / "cal_parms.cal").read_text(), f)
    for name, text in (("nutrients.sol", nut), ("calibration.cal", cal), ("cal_parms.cal", cp)):
        (TIO / name).write_text(text)
    return fit


def write_time() -> None:
    put("time.sim", ["time.sim: written by swat-kimberly build_ltm.py",
                     "day_start  yrc_start   day_end   yrc_end      step  ",
                     f"{0:>10}{START:>11}{0:>10}{END:>10}{0:>10}  "])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-probes", action="store_true",
                    help="omit the 0-kg soil probes (to check they are inert)")
    ap.add_argument("--fert-p", choices=("mid", "lo", "hi"), default="mid",
                    help="fertiliser P for the 10 treatment-years only Table S1 has")
    args = ap.parse_args()

    if TIO.exists():
        shutil.rmtree(TIO)
    TIO.mkdir(parents=True)
    for p in BASE.iterdir():
        if not p.name.startswith("swatplus_"):
            shutil.copy2(p, TIO / p.name)
    for p in (LTM / "engine").iterdir():
        shutil.copy2(p, TIO / p.name)

    sys.path.insert(0, str(ROOT / "scripts"))
    import port_gridmet as g
    g.use_site("ltm")
    w = g.convert(pd.read_csv(g.CACHE, index_col="date", parse_dates=True))
    elev = float(rows(TIO / "hru.con")[2].split()[6])
    g.write(TIO, w, elev)

    write_plants(load_params())
    write_soil()
    n_fit = apply_n_cycle()
    write_management(args.fert_p, not args.no_probes)
    write_hrus()
    write_time()
    (LTM / "build.json").write_text(json.dumps(
        {"fert_p": args.fert_p, "probes": not args.no_probes, "n_cycle": n_fit, "start": START, "end": END, "treatments": TREATMENTS,
         "base": str(BASE.relative_to(ROOT)), "built": date.today().isoformat()}, indent=2) + "\n")
    print(f"built {TIO.relative_to(ROOT)} (fert P policy: {args.fert_p})")


if __name__ == "__main__":
    main()
