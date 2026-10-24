"""Build the GRACEnet rotation management (dates/HI from the ArcSWAT reference; irrigation from
data/irrigation_gracenet.csv).

Writes management.sch, plant.ini, fertilizer.frt/harv.ops/irr.ops entries, landuse.lum.
Ops sorted by (year, doy); SWAT+ rolls the year on a date wrap. Simulation is 2012 spin-up +
2013-2019.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIO = ROOT / "model" / "rotation" / "TxtInOut"
# From extract_primary.py.
IRR_CSV = ROOT / "data" / "irrigation_gracenet.csv"

COMM, SCHED, LUM = "kimb_comm", "kimb_rot", "alfa_lum"

# Harvest ops: harv_idx = workbook HARV_EFF, eff 1.0, no biomass floor.
HARVEST = {          # name -> (harv_idx, harv_eff, harv_bm_min)
    "gn_corn": (0.98, 1.0, 0.0),
    "gn_barl": (0.54, 1.0, 0.0),
    "gn_alfa": (0.95, 1.0, 0.0),
}

# Manure disked in on the application date (as in the field).
TILLAGE = {"gn_disk": (0.85, 150.0, 30.0)}   # name -> (mix_eff, mix_dp mm, rough)

# Measured GRACEnet manures: name -> (rate kg/ha, min_n, min_p, org_n, org_p).
MANURES = {
    "gn2013": (43800, 0.00260, 0.004655, 0.01040, 0.000245),
    "gn2014": (48000, 0.00390, 0.006650, 0.01560, 0.000350),
    "gn2018": (88300, 0.00082, 0.001995, 0.00328, 0.000105),
    "gn2019": (54300, 0.00082, 0.007220, 0.00328, 0.000380),
}
NH3_FRAC = 0.99   # reference fert.dat FNH3N; the template default is 1.0

# (op, year, month, day, op_data1, op_data2). Use hvkl to end a crop (standalone kill is
# ignored); harv only for intermediate alfalfa cuts.
SPINUP_OPS = [
    ("plnt", 4, 1, "barl", "null"),
    ("hvkl", 8, 10, "barl", "gn_barl"),
]

CROP_OPS = [
    ("fert", 2013, 4, 10, "gn2013", "broadcast"),
    ("till", 2013, 4, 10, "gn_disk", "null"),
    ("plnt", 2013, 5, 16, "corn", "null"),
    ("hvkl", 2013, 9, 12, "corn", "gn_corn"),

    ("plnt", 2014, 4, 9, "barl", "null"),
    ("fert", 2014, 4, 10, "gn2014", "broadcast"),
    ("till", 2014, 4, 10, "gn_disk", "null"),
    ("hvkl", 2014, 8, 20, "barl", "gn_barl"),

    ("plnt", 2015, 4, 16, "alfa", "null"),
    ("harv", 2015, 7, 18, "alfa", "gn_alfa"),
    ("harv", 2015, 9, 9, "alfa", "gn_alfa"),

    ("harv", 2016, 5, 26, "alfa", "gn_alfa"),
    ("harv", 2016, 7, 13, "alfa", "gn_alfa"),
    ("harv", 2016, 9, 7, "alfa", "gn_alfa"),

    ("harv", 2017, 5, 31, "alfa", "gn_alfa"),
    ("harv", 2017, 7, 10, "alfa", "gn_alfa"),
    ("hvkl", 2017, 9, 15, "alfa", "gn_alfa"),

    ("fert", 2018, 4, 10, "gn2018", "broadcast"),
    ("till", 2018, 4, 10, "gn_disk", "null"),
    ("plnt", 2018, 5, 17, "corn", "null"),
    ("hvkl", 2018, 9, 24, "corn", "gn_corn"),

    ("fert", 2019, 4, 10, "gn2019", "broadcast"),
    ("till", 2019, 4, 10, "gn_disk", "null"),
    ("plnt", 2019, 4, 13, "barl", "null"),
    ("hvkl", 2019, 8, 8, "barl", "gn_barl"),
]

SPINUP_YEARS = (2012,)

#: Last simulated year (2020 crop unknown, no measured yield).
SIM_END_YEAR = 2019


def doy(year: int, mon: int, day: int) -> int:
    return date(year, mon, day).timetuple().tm_yday


def md(year: int, jday: int) -> tuple[int, int]:
    d = date.fromordinal(date(year, 1, 1).toordinal() + jday - 1)
    return d.month, d.day


def build_ops() -> list[tuple]:
    """Return all schedule ops as (year, doy, line-tuple), sorted by (year, doy)."""
    ops: list[tuple[int, int, tuple]] = []

    rotation = list(CROP_OPS)
    for yr in SPINUP_YEARS:
        rotation += [(op, yr, mon, day, d1, d2) for op, mon, day, d1, d2 in SPINUP_OPS]

    for op, yr, mon, day, d1, d2 in rotation:
        d3 = MANURES[d1][0] if op == "fert" else 0.0
        # op_data3 on plnt is not heat units (large values segfault).
        ops.append((yr, doy(yr, mon, day), (op, mon, day, 0.0, d1, d2, float(d3))))

    # measured daily irrigation -> one irrm op per event, referencing an irr.ops entry.
    df = pd.read_csv(IRR_CSV)
    df = df[df.year <= SIM_END_YEAR].rename(columns={"mm": "irrigation"})
    for i, r in enumerate(df.itertuples(), start=1):
        mon, day = md(int(r.year), int(r.jday))
        ops.append((int(r.year), int(r.jday),
                    ("irrm", mon, day, 0.0, f"gn{i:04d}", "null", 0.0)))

    ops.sort(key=lambda t: (t[0], t[1]))
    return ops


def write_management(ops: list[tuple]) -> None:
    lines = [
        "management.sch: written by swat-kimberly port_management.py\n",
        "name                                       numb_ops  numb_auto            op_typ       mon       day        hu_sch          op_data1          op_data2      op_data3  \n",
        f"{SCHED:<34}{len(ops):>6}{0:>11}  \n",
    ]
    for _, _, o in ops:
        typ, mon, day, hu, d1, d2, d3 = o
        lines.append(f"{'':>48}{typ:<10}{mon:>6}{day:>10}{hu:>14.5f}{d1:>18}{d2:>18}{d3:>14.5f}  \n")
    (TIO / "management.sch").write_text("".join(lines))
    print(f"management.sch: {len(ops)} ops")


#: yrs_init at sim start. Alfalfa 0 (not planted until 2015).
YRS_INIT = {"corn": 1.0, "barl": 1.0, "alfa": 0.0}


def write_plant_ini() -> None:
    plants = ["corn", "barl", "alfa"]
    lines = [
        "plant.ini: written by swat-kimberly port_management.py\n",
        "pcom_name          plt_cnt  rot_yr_ini          plt_name     lc_status      lai_init       bm_init      phu_init      plnt_pop      yrs_init      rsd_init  \n",
        f"{COMM:<19}{len(plants):>7}{1:>11}  \n",
    ]
    for p in plants:
        lines.append(f"{'':>40}{p:<8}{'n':>6}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{YRS_INIT[p]:>14.5f}{0.0:>14.5f}  \n")
    (TIO / "plant.ini").write_text("".join(lines))
    print(f"plant.ini: community {COMM} with {plants}, yrs_init {YRS_INIT}")


def append_fertilizers() -> None:
    frt = (TIO / "fertilizer.frt").read_text().rstrip("\n").splitlines()
    existing = {ln.split()[0] for ln in frt[2:]}
    for name, (_, min_n, min_p, org_n, org_p) in MANURES.items():
        if name in existing:
            continue
        frt.append(f"{name:<22}{min_n:>10.5f}{min_p:>14.5f}{org_n:>14.5f}{org_p:>14.5f}"
                   f"{NH3_FRAC:>14.5f}{'null':>14}  GRACEnet_manure")
    (TIO / "fertilizer.frt").write_text("\n".join(frt) + "\n")
    print(f"fertilizer.frt: +{len(MANURES)} GRACEnet manures (nh3_n={NH3_FRAC})")


def append_harvest_ops() -> None:
    harv = (TIO / "harv.ops").read_text().rstrip("\n").splitlines()
    existing = {ln.split()[0] for ln in harv[2:]}
    for name, (idx, eff, bm_min) in HARVEST.items():
        if name in existing:
            continue
        harv.append(f"{name:<22}{'biomass':>14}{idx:>14.5f}{eff:>14.5f}{bm_min:>14.5f}  reference_HI_override")
    (TIO / "harv.ops").write_text("\n".join(harv) + "\n")
    print(f"harv.ops: +{len(HARVEST)} reference harvest types "
          + ", ".join(f"{n} HI={v[0]}" for n, v in HARVEST.items()))


def append_tillage_ops() -> None:
    til = (TIO / "tillage.til").read_text().rstrip("\n").splitlines()
    existing = {ln.split()[0] for ln in til[2:]}
    for name, (eff, dp, rough) in TILLAGE.items():
        if name in existing:
            continue
        til.append(f"{name:<22}{eff:>10.5f}{dp:>14.5f}{rough:>14.5f}"
                   f"{0.0:>14.5f}{0.0:>14.5f}  manure_incorporation_disk_15cm")
    (TIO / "tillage.til").write_text("\n".join(til) + "\n")
    print(f"tillage.til: +{len(TILLAGE)} " + ", ".join(
        f"{n} (mix_eff {v[0]}, mix_dp {v[1]:.0f} mm)" for n, v in TILLAGE.items()))


def set_time_sim() -> None:
    """Set time.sim to SPINUP..SIM_END_YEAR."""
    path = TIO / "time.sim"
    lines = path.read_text().splitlines()
    toks = lines[2].split()
    toks[1], toks[3] = str(min(SPINUP_YEARS)), str(SIM_END_YEAR)   # yrc_start, yrc_end
    lines[2] = "".join(f"{t:>10}" for t in toks)
    path.write_text("\n".join(lines) + "\n")
    print(f"time.sim: {toks[1]}-{toks[3]} (2020 dropped)")


def append_irr_ops() -> None:
    df = pd.read_csv(IRR_CSV).rename(columns={"mm": "irrigation"})
    events = df[df.year <= SIM_END_YEAR].reset_index(drop=True)
    irr = (TIO / "irr.ops").read_text().rstrip("\n").splitlines()
    existing = {ln.split()[0] for ln in irr[2:] if ln.strip()}
    for i, r in enumerate(events.itertuples(), start=1):
        if f"gn{i:04d}" in existing:
            continue
        amt = float(r.irrigation)
        irr.append(f"{f'gn{i:04d}':<22}{amt:>8.5f}{1.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}  gracenet")
    (TIO / "irr.ops").write_text("\n".join(irr) + "\n")
    print(f"irr.ops: +{len(events)} measured irrigation events")


def point_landuse() -> None:
    lum = (TIO / "landuse.lum").read_text().splitlines(keepends=True)
    parts = lum[2].split()
    parts[2] = COMM   # plnt_com
    parts[3] = SCHED  # mgt
    lum[2] = "  ".join(parts) + "  \n"
    (TIO / "landuse.lum").write_text("".join(lum))
    print(f"landuse.lum: plnt_com={COMM}, mgt={SCHED}")


def main() -> None:
    ops = build_ops()
    write_management(ops)
    write_plant_ini()
    append_fertilizers()
    append_harvest_ops()
    append_tillage_ops()
    append_irr_ops()
    point_landuse()
    set_time_sim()


if __name__ == "__main__":
    main()
