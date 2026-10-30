#!/usr/bin/env python3
"""Water and nutrient summary of the LTM scenario, for comparison with the trial's own data.

Runs model/ltm/TxtInOut (three HRUs: 52A, fert, control) under three irrigation cases:

* measured   -- the trial's daily irrigation events, replayed at practice.MEASURED_IRR_EFF (1.0)
* auto100    -- auto-irrigation (practice.AUTO_IRR: 1-inch pass when profile plant-available
                water < sw_frac x FC, inside each season's window) at efficiency 1.0, the
                efficiency the trigger was fitted at (scripts/ltm_fit_auto_irr.py)
* auto080    -- the same rule at practice.BASE_IRR_EFF (0.80), the experiments' base case

The 2012 spin-up (beans, auto-irrigated at 0.80) is identical in every case; outputs cover
2013-2020 (print.prt skips the spin-up year). Every case adds a 0-kg elem_n probe on every day
(inert: checked against the measured case without daily probes, `probe_check` below); a
probe runs before that day's other ops, so its soil values are the start-of-day state.

Outputs (runs/ltm_nutrient_summary.html + runs/ltm_nutrient_summary/*.csv):
yearly applied N and P; monthly irrigation (gross = what reaches the soil / efficiency);
monthly NO3-N leached (hru_ls percn); biomass and yield at harvest (mgt_out HARV/KILL);
N and P uptake (hru_nb_yr); daily profile soil NO3-N and solution P (probe rows of mgt_out,
kg/ha over the whole 0-1520 mm profile, also as a profile-mean mg/kg).

    uv run python scripts/ltm_nutrient_summary.py
"""
from __future__ import annotations

import html
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_ltm as B  # noqa: E402
import ltm_figures as F  # noqa: E402
from swat_gym.fastrunner import FastRunner  # noqa: E402
from swat_gym.ltm.practice import (AUTO_IRR, BASE_IRR_EFF, MEASURED_IRR_EFF,  # noqa: E402
                                   auto_op)
from swat_gym.ltm.render import EpisodePlan, render  # noqa: E402
from swat_gym.ltm.runner import EDITABLE, _irr_table, read_mgt, read_table  # noqa: E402

TIO = B.TIO
DATA = B.DATA
OUT_HTML = ROOT / "runs" / "ltm_nutrient_summary.html"
OUT_DIR = ROOT / "runs" / "ltm_nutrient_summary"
HRU = {1: "52A", 2: "fert", 3: "control"}
TRT_LABEL = {"52A": "52A (manure)", "fert": "Fertilizer only", "control": "Control"}
#: case -> (label, efficiency of the 2013-2020 irrigation; None = measured replay)
CASES = {"measured": ("Measured irrigation (eff 1.0)", None),
         "auto100": ("Auto-irrigation, eff 1.0", 1.0),
         "auto080": ("Auto-irrigation, eff 0.80", BASE_IRR_EFF)}
CASE_COL = dict(zip(CASES, ("#2a78d6", "#eb6834", "#1baf7a")))   # ltm_figures slots 1-3
FIRST, LAST = B.START + 1, B.END
SPIN_EFF = BASE_IRR_EFF                    #: the 2012 beans' auto-irrigation (build_ltm)
TRIAL_IRR = "ltm_trial_irr"                #: lum.dtl table auto-irrigating 2013-2020
TOL = 1e-6


# -- inputs per case ------------------------------------------------------------------------
def sched_text(scheds: dict[str, list], autos: list[str]) -> str:
    """management.sch in build_ltm.write_management's layout."""
    lines = ["management.sch: written by swat-kimberly build_ltm.py",
             "name                                       numb_ops  numb_auto            op_typ       mon       day        hu_sch          op_data1          op_data2      op_data3  "]
    for tag, ops in scheds.items():
        lines.append(f"{'ltm_' + tag:<34}{len(ops):>6}{len(autos):>11}  ")
        lines += [f"{'':>48}{a:<18}" for a in autos]
        lines += [o[2] for o in ops]
    return "\n".join(lines) + "\n"


def base_schedules() -> dict[str, list]:
    """build_ltm's schedules, checked against the built model (they must not have drifted)."""
    fert_p = json.loads((TIO.parent / "build.json").read_text())["fert_p"]
    scheds, _, _ = B.schedules(fert_p, probes=True)
    if sched_text(scheds, [B.BEAN_IRR]) != (TIO / "management.sch").read_text():
        raise SystemExit("model/ltm/TxtInOut/management.sch differs from build_ltm.schedules(): "
                         "rebuild with scripts/build_ltm.py first")
    return scheds


def trial_windows() -> list[tuple[int, int, int]]:
    """Auto-irrigation windows of the trial seasons (as the env's EpisodePlan(irr_auto))."""
    _, _, auto = render(EpisodePlan(irr_auto=True), B.START, "x")
    return [w for w in auto if w[0] >= FIRST]


def daily_probes(scheds: dict[str, list]) -> dict[str, list]:
    have = set(B.probe_dates())
    days = [date(B.START, 1, 1) + timedelta(days=i)
            for i in range((date(LAST, 12, 31) - date(B.START, 1, 1)).days + 1)]
    extra = [B._op("probe", d, None) for d in days if d not in have]
    return {t: sorted(ops + extra, key=lambda o: (o[0], o[1])) for t, ops in scheds.items()}


def case_edits(scheds: dict[str, list], eff: float | None, probes: bool = True) -> dict[str, str]:
    if probes:
        scheds = daily_probes(scheds)
    if eff is None:
        return {"management.sch": sched_text(scheds, [B.BEAN_IRR])}
    if AUTO_IRR["sw_frac"] is None:
        raise SystemExit("auto-irrigation trigger not fitted: run scripts/ltm_fit_auto_irr.py")
    scheds = {t: [o for o in ops if o[2].split()[0] != "irrm"] for t, ops in scheds.items()}
    dtl = (TIO / "lum.dtl").read_text()
    head, count, rest = dtl.split("\n", 2)
    table = _irr_table(TRIAL_IRR, trial_windows(), AUTO_IRR["sw_frac"], auto_op(eff))
    dtl = f"{head}\n{int(count.strip()) + 1}\n{rest}".rstrip("\n") + "\n" + table
    irr = (TIO / "irr.ops").read_text().rstrip("\n").split("\n")
    if not any(ln.split()[:1] == [auto_op(eff)] for ln in irr):
        irr.append(f"{auto_op(eff):<22}{AUTO_IRR['mm']:>8.5f}{eff:>14.5f}"
                   + f"{0.0:>14.5f}" * 5 + "  ltm_auto_irrigation_pass")
    return {"management.sch": sched_text(scheds, [B.BEAN_IRR, TRIAL_IRR]),
            "lum.dtl": dtl, "irr.ops": "\n".join(irr) + "\n"}


def run_case(fr: FastRunner, edits: dict[str, str]) -> dict[str, pd.DataFrame]:
    fr.run(edits)
    w = fr.workdir
    out = {t: read_table(w / f"{t}.txt") for t in ("hru_wb_day", "hru_ls_day", "hru_nb_yr")}
    for t, df in out.items():
        df["treatment"] = df.unit.map(HRU)
        if "day" in t:
            df["date"] = pd.to_datetime(dict(year=df.yr, month=df.mon, day=df.day))
    mg = read_mgt(w / "mgt_out.txt")
    mg["treatment"] = mg.hru.map(HRU)
    mg["date"] = pd.to_datetime(dict(year=mg.yr, month=mg.mon, day=mg.day))
    out["mgt"] = mg
    return out


def probe_check(base: dict, noprobe: dict) -> dict:
    """Max |difference| between the measured case with and without daily probes."""
    rep = {}
    for t, cols in (("hru_wb_day", ["irr", "perc", "et", "sw_final"]),
                    ("hru_ls_day", ["percn", "lchlabp", "surqsolp"]),
                    ("hru_nb_yr", ["fertn", "fertp", "nuptake", "puptake", "act_nit_n"])):
        rep[t] = float(np.abs(base[t][cols].to_numpy() - noprobe[t][cols].to_numpy()).max())
    hv = [m[m.operation == "HARV/KILL"][["plant_bioms", "op_var"]].to_numpy()
          for m in (base["mgt"], noprobe["mgt"])]
    rep["harvest"] = float(np.abs(hv[0] - hv[1]).max())
    rep["inert"] = all(v <= TOL for v in rep.values())
    return rep


def soil_mass_kg_ha() -> float:
    """Dry soil mass of the whole profile, kg/ha (soils.sol layer depths and bulk density)."""
    lines = (TIO / "soils.sol").read_text().splitlines()
    nly = int(lines[2].split()[1])
    top, mass = 0.0, 0.0
    for ln in lines[3:3 + nly]:
        dp, bd = (float(x) for x in ln.split()[:2])
        mass += bd * 1000 * (dp - top) / 1000 * 1e4      # g/cm3 -> kg/m3, mm -> m, m2/ha
        top = dp
    return mass


# -- tables ---------------------------------------------------------------------------------
def gross_irr(wb: pd.DataFrame, eff: float | None) -> pd.Series:
    """Applied (gross) water: hru_wb irr is what reaches the soil (amount x efficiency)."""
    e = np.where(wb.yr <= B.START, SPIN_EFF, MEASURED_IRR_EFF if eff is None else eff)
    return wb.irr / e


def applied_inputs() -> pd.DataFrame:
    """N and P each treatment received per year, from data/ltm/management.csv (fertiliser P
    as build_ltm's policy for the years Table S1 gives only a range)."""
    fert_p = json.loads((TIO.parent / "build.json").read_text())["fert_p"]
    m = pd.read_csv(DATA / "management.csv")
    rows = []
    for (trt, yr), g in m[m.year.between(FIRST, LAST)].groupby(["treatment", "year"]):
        man = g[(g.op == "manure") & (trt == "52A")]
        fe = g[g.op == "fert"]
        p_syn = 0.0
        for r in fe.itertuples():
            if pd.isna(r.p_kg_ha):
                p_syn += {"lo": r.p_kg_ha_lo, "hi": r.p_kg_ha_hi,
                          "mid": (r.p_kg_ha_lo + r.p_kg_ha_hi) / 2}[fert_p]
            else:
                p_syn += r.p_kg_ha
        rows.append(dict(treatment=trt, year=yr,
                         manure_n_kg_ha=float((man.min_n_kg_ha + man.org_n_kg_ha).sum()),
                         manure_p_kg_ha=float(man.p_kg_ha.sum()),
                         synthetic_n_kg_ha=float(fe.n_kg_ha.sum()), synthetic_p_kg_ha=p_syn))
    return pd.DataFrame(rows)


def build_tables(res: dict[str, dict]) -> dict[str, pd.DataFrame]:
    yearly, monthly, daily, harvest = [], [], [], []
    up = pd.read_csv(DATA / "uptake.csv").set_index("year")
    crop_of = {v: k for k, v in B.CROP_PLANT.items()}
    for case, (_, eff) in CASES.items():
        o = res[case]
        nb = o["hru_nb_yr"]
        yearly.append(nb[["treatment", "yr", "fertn", "fertp", "nuptake", "puptake"]]
                      .rename(columns={"yr": "year", "fertn": "swat_applied_n_kg_ha",
                                       "fertp": "swat_applied_p_kg_ha",
                                       "nuptake": "n_uptake_kg_ha", "puptake": "p_uptake_kg_ha"})
                      .assign(case=case))
        wb, ls = o["hru_wb_day"], o["hru_ls_day"]
        d = pd.DataFrame({"treatment": wb.treatment, "date": wb.date,
                          "irr_gross_mm": gross_irr(wb, eff), "irr_to_soil_mm": wb.irr,
                          "precip_mm": wb.precip, "perc_mm": wb.perc})
        d = d.merge(ls[["treatment", "date", "percn"]].rename(columns={"percn": "no3n_leached_kg_ha"}),
                    on=["treatment", "date"])
        d["month"] = d.date.dt.to_period("M").dt.to_timestamp()
        monthly.append(d.groupby(["treatment", "month"], as_index=False)
                       [["irr_gross_mm", "irr_to_soil_mm", "precip_mm", "perc_mm",
                         "no3n_leached_kg_ha"]].sum().assign(case=case))
        mg = o["mgt"]
        pr = mg[(mg.name == B.PROBE) & (mg.op_var == 0.0) & mg.yr.between(FIRST, LAST)]
        pr = pr.drop_duplicates(["treatment", "date"])
        daily.append(pr[["treatment", "date", "soil_no3", "soil_solp", "soil_water"]]
                     .rename(columns={"soil_no3": "soil_no3n_kg_ha", "soil_solp": "soil_solp_kg_ha",
                                      "soil_water": "soil_water_mm"}).assign(case=case))
        hv = mg[(mg.operation == "HARV/KILL") & mg.yr.between(FIRST, LAST)]
        for r in hv.itertuples():
            meas = r.treatment == "52A" and r.yr in up.index
            harvest.append(dict(case=case, treatment=r.treatment, year=r.yr, date=r.date.date(),
                                crop=crop_of[r.name], sim_biomass_t_ha=r.plant_bioms / 1000,
                                sim_yield_dm_t_ha=r.op_var / 1000,
                                meas_biomass_t_ha=float(up.biomass_mg_ha[r.yr]) if meas else np.nan))
    yearly = pd.concat(yearly).merge(applied_inputs(), on=["treatment", "year"], how="left")
    daily = pd.concat(daily)
    mass = soil_mass_kg_ha()
    daily["soil_no3n_mg_kg_profile"] = daily.soil_no3n_kg_ha / mass * 1e6
    daily["soil_solp_mg_kg_profile"] = daily.soil_solp_kg_ha / mass * 1e6
    return {"yearly": yearly, "monthly": pd.concat(monthly), "daily": daily,
            "harvest": pd.DataFrame(harvest)}


def measured_soil() -> pd.DataFrame:
    """Dated measured soil values per treatment: 52A profile NO3 / Olsen P series (W1) and the
    spring-2020 0-60 cm NO3 of all three (P1 Table 5; dated 2020-04-01 as ltm_validate does)."""
    ss = pd.read_csv(DATA / "soil_series.csv")
    m = ss[(ss["plot"] == "mean") & ss.date.notna()].copy()
    t5 = ss[(ss["var"] == "no3_mg_kg_0_60") & (ss.year == 2020)].copy()
    t5["date"] = "2020-04-01"
    m = pd.concat([m, t5])
    m["date"] = pd.to_datetime(m.date)
    return m[["treatment", "date", "var", "value", "sd", "source"]]


# -- figures --------------------------------------------------------------------------------
X0 = pd.Timestamp(FIRST, 1, 1)


def xnum(ts) -> float:
    return (pd.Timestamp(ts) - X0).days


def year_ticks() -> tuple[list[float], dict]:
    t = [xnum(pd.Timestamp(y, 1, 1)) for y in range(FIRST, LAST + 2)]
    return t, {v: str(FIRST + i) for i, v in enumerate(t)}


SHAPES = {"profile_no3_mg_kg": ("measured profile NO₃-N (depth not stated)", "dot"),
          "no3_mg_kg_0_60": ("measured NO₃-N 0–60 cm, spring 2020", "square"),
          "olsen_p_mg_kg_0_30": ("measured Olsen P 0–30 cm", "dot"),
          "olsen_p_mg_kg_30_60": ("measured Olsen P 30–60 cm", "ring")}


def marker(x, y, shape, tip) -> str:
    t = f"<title>{html.escape(tip)}</title>"
    if shape == "dot":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{F.INK}" stroke="#fff" stroke-width="2">{t}</circle>'
    if shape == "ring":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#fff" stroke="{F.INK}" stroke-width="2">{t}</circle>'
    return f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" fill="{F.INK}" stroke="#fff" stroke-width="2">{t}</rect>'


def line_panel(title: str, series: list[tuple[str, str, pd.Series]], ylab: str, unit: str,
               points: list[tuple[pd.Timestamp, float, str, str]] = (), step: bool = False,
               w: int = 960, h: int = 230) -> str:
    """Lines over 2013-2020 (one per case), optional measured markers. Hover: a crosshair
    reads every series at the nearest x (the chart's JSON in data-*, ltm_hover JS)."""
    left, top = 70, 30
    xt, xlab = year_ticks()
    ymax = max([float(s.max()) for _, _, s in series] + [p[1] for p in points] + [1e-9])
    yt = F.ticks(0, ymax * 1.05)
    ax = F.Axes(left, top, w - left - 20, h - top - 40, (xt[0], xt[-1]), (yt[0], yt[-1]))
    body = [F.text(left, 16, title, 13, weight="bold"),
            ax.grid(xt, yt, "", ylab, xfmt=lambda v: xlab.get(v, ""))]
    xs_all = None
    hover = []
    for label, col, s in series:
        xs = [xnum(i) for i in s.index]
        if step:      # monthly totals drawn as steps across each month
            ends = [xnum(i + pd.offsets.MonthBegin(1)) for i in s.index]
            pts = []
            for a, b, v in zip(xs, ends, s.values):
                pts += [(ax.x(a), ax.y(v)), (ax.x(b), ax.y(v))]
        else:
            pts = [(ax.x(a), ax.y(v)) for a, v in zip(xs, s.values)]
        body.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" '
                    f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        xs_all = xs
        hover.append({"label": label, "color": col, "y": [round(float(v), 3) for v in s.values]})
    for ts, v, shape, tip in points:
        body.append(marker(ax.x(xnum(ts)), ax.y(v), shape, tip))
    data = {"x0": X0.strftime("%Y-%m-%d"), "x": xs_all, "monthly": step, "unit": unit,
            "px": [ax.x0, ax.w, ax.y0, ax.h, ax.xd[0], ax.xd[1]], "s": hover}
    return (f'<div class="chart" data-chart=\'{html.escape(json.dumps(data))}\'>'
            + F.svg(w, h, "".join(body), title) + "</div>")


def legend_row(items: list[tuple[str, str, str]]) -> str:
    return F.svg(960, 24, F.legend(items, 70, 16), "legend")


def bar_panel(title: str, s: pd.Series, col: str, ylab: str, ymax: float,
              w: int = 960, h: int = 200) -> str:
    """Monthly bars; hover via <title> on each bar."""
    left, top = 70, 30
    xt, xlab = year_ticks()
    yt = F.ticks(0, ymax * 1.05)
    ax = F.Axes(left, top, w - left - 20, h - top - 40, (xt[0], xt[-1]), (yt[0], yt[-1]))
    body = [F.text(left, 16, title, 13, weight="bold"),
            ax.grid(xt, yt, "", ylab, xfmt=lambda v: xlab.get(v, ""))]
    for m, v in s.items():
        if v <= 0:
            continue
        a, b = ax.x(xnum(m)), ax.x(xnum(m + pd.offsets.MonthBegin(1)))
        y = ax.y(v)
        body.append(f'<rect x="{a + 1:.1f}" y="{y:.1f}" width="{max(b - a - 2, 1):.1f}" '
                    f'height="{ax.y(0) - y:.1f}" fill="{col}">'
                    f'<title>{m:%b %Y}: {v:,.0f} mm</title></rect>')
    return F.svg(w, h, "".join(body), title)


HOVER_JS = """
<div id="tip" style="position:fixed;display:none;background:#fff;border:1px solid #c9c8c4;
padding:4px 8px;font-size:12px;pointer-events:none"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.chart').forEach(div => {
  const d = JSON.parse(div.dataset.chart), svg = div.querySelector('svg');
  const [x0, w, y0, h, a, b] = d.px, ns = 'http://www.w3.org/2000/svg';
  const cross = document.createElementNS(ns, 'line');
  cross.setAttribute('stroke', '#52514e'); cross.setAttribute('y1', y0);
  cross.setAttribute('y2', y0 + h); cross.style.display = 'none'; svg.appendChild(cross);
  const base = new Date(d.x0 + 'T00:00:00Z');
  svg.addEventListener('mousemove', ev => {
    const r = svg.getBoundingClientRect(), sx = svg.viewBox.baseVal.width / r.width;
    const px = (ev.clientX - r.left) * sx;
    if (px < x0 || px > x0 + w) { cross.style.display = 'none'; tip.style.display = 'none'; return; }
    const xv = a + (px - x0) / w * (b - a);
    let i = 0, best = Infinity;
    d.x.forEach((v, j) => { const dd = d.monthly ? (xv >= v ? xv - v : Infinity) : Math.abs(v - xv);
      if (dd < best) { best = dd; i = j; } });
    const X = x0 + (d.x[i] - a) / (b - a) * w;
    cross.setAttribute('x1', X); cross.setAttribute('x2', X); cross.style.display = '';
    const dt = new Date(base.getTime() + d.x[i] * 864e5);
    const lab = d.monthly ? dt.toLocaleDateString('en-US', {month: 'short', year: 'numeric', timeZone: 'UTC'})
                          : dt.toISOString().slice(0, 10);
    tip.innerHTML = '<b>' + lab + '</b>' + d.s.map(s => '<br><span style="color:' + s.color +
      '">&#9632;</span> ' + s.label + ': ' + s.y[i].toFixed(2) + ' ' + d.unit).join('');
    tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 10) + 'px';
  });
  svg.addEventListener('mouseleave', () => { cross.style.display = 'none'; tip.style.display = 'none'; });
});
</script>
"""


# -- page -----------------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x))


def num(x, nd=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def table(head: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def case_legend() -> str:
    return legend_row([(CASES[c][0], CASE_COL[c], "square") for c in CASES])


def page(T: dict, meas: pd.DataFrame, check: dict, mass: float, auto_check: dict) -> str:
    Y, M, D, H = T["yearly"], T["monthly"], T["daily"], T["harvest"]
    parts = []

    # 1. applied N and P
    rows = []
    y0 = Y[Y.case == "measured"]
    for trt in HRU.values():
        for r in y0[y0.treatment == trt].sort_values("year").itertuples():
            rows.append([esc(TRT_LABEL[trt]), r.year, num(r.manure_n_kg_ha), num(r.synthetic_n_kg_ha),
                         num(r.swat_applied_n_kg_ha), num(r.manure_p_kg_ha), num(r.synthetic_p_kg_ha),
                         num(r.swat_applied_p_kg_ha)])
    parts.append("<h2>1. Yearly applied N and P</h2>"
                 "<p>Same in every irrigation case. Manure is applied in the fall (Oct 20) and counted in "
                 "its calendar year. Manure N is mineral + organic N; manure P is total P. "
                 "<i>SWAT applied</i> is hru_nb <code>fertn</code>/<code>fertp</code>, i.e. what the "
                 "engine actually added.</p>"
                 + table(["Treatment", "Year", "Manure N", "Synthetic N", "SWAT applied N",
                          "Manure P", "Synthetic P", "SWAT applied P"], rows)
                 + "<p>All values kg/ha. The fall-2012 manure (52A) falls in the spin-up year and is "
                   "not in this table.</p>")

    # 2. monthly irrigation
    ymax = M.irr_gross_mm.max()
    yt_ = M.assign(year=M.month.dt.year).groupby(["case", "year", "treatment"]).irr_gross_mm.sum()
    same_auto = all(np.ptp(yt_[(c, y)].to_numpy()) < 0.5 for c in CASES if c != "measured"
                    for y in range(FIRST, LAST + 1))
    auto_note = ("In this run auto-irrigation also fired identically on all three plots."
                 if same_auto else "Their auto-irrigation differs by plot; see the table and CSV.")
    bars = [bar_panel(f"{CASES[c][0]}, 52A", M[(M.case == c) & (M.treatment == "52A")]
                      .set_index("month").irr_gross_mm, CASE_COL[c], "Gross mm/month", ymax)
            for c in CASES]
    yr = M.assign(year=M.month.dt.year).groupby(["case", "treatment", "year"]).irr_gross_mm.sum()
    rows = [[esc(TRT_LABEL[t]), esc(CASES[c][0])] + [num(yr[(c, t, y)], 0) for y in range(FIRST, LAST + 1)]
            + [num(sum(yr[(c, t, y)] for y in range(FIRST, LAST + 1)), 0)]
            for t in HRU.values() for c in CASES]
    parts.append("<h2>2. Monthly irrigation</h2>"
                 "<p>Gross applied water (mm/month). SWAT+ puts amount × efficiency into the soil; "
                 "gross = hru_wb <code>irr</code> ÷ efficiency. The bars show 52A; the fertilizer-only "
                 "and control plots get the same measured events. " + auto_note
                 + " Hover a bar for its value.</p>" + "".join(bars)
                 + "<h3>Yearly gross irrigation (mm)</h3>"
                 + table(["Treatment", "Case"] + [str(y) for y in range(FIRST, LAST + 1)] + ["Total"], rows)
                 + f"<p>Check: auto-irrigation at eff 1.0 applies {auto_check['sim_mm']:,.1f} mm to 52A "
                   f"over 2013–2020; scripts/ltm_fit_auto_irr.py recorded {auto_check['fit_mm']:,.1f} mm "
                   f"(model/ltm/auto_irr.json). Measured events total {auto_check['meas_mm']:,.1f} mm.</p>")

    # 3. monthly leached N
    panels = []
    for t in HRU.values():
        ser = [(CASES[c][0], CASE_COL[c], M[(M.case == c) & (M.treatment == t)].set_index("month")
                .no3n_leached_kg_ha) for c in CASES]
        panels.append(line_panel(TRT_LABEL[t], ser, "kg N/ha/month", "kg/ha", step=True))
    lr = M.assign(year=M.month.dt.year).groupby(["case", "treatment", "year"]).no3n_leached_kg_ha.sum()
    rows = [[esc(TRT_LABEL[t]), esc(CASES[c][0])] + [num(lr[(c, t, y)]) for y in range(FIRST, LAST + 1)]
            + [num(sum(lr[(c, t, y)] for y in range(FIRST, LAST + 1)))] for t in HRU.values() for c in CASES]
    parts.append("<h2>3. Monthly NO₃-N leached</h2>"
                 "<p>NO₃-N leaving the bottom of the 1.52 m profile (hru_ls <code>percn</code>), "
                 "summed by month. <b>Leaching is not validated on this site</b>: there is no on-site "
                 "measurement.</p>" + case_legend() + "".join(panels)
                 + "<h3>Yearly NO₃-N leached (kg/ha)</h3>"
                 + table(["Treatment", "Case"] + [str(y) for y in range(FIRST, LAST + 1)] + ["Total"], rows))

    # 4 + 5. harvest biomass, uptake
    hv = H.set_index(["case", "treatment", "year"])
    rows = []
    for t in HRU.values():
        for y in range(FIRST, LAST + 1):
            r0 = hv.loc[("measured", t, y)]
            rows.append([esc(TRT_LABEL[t]), y, esc(r0.crop), num(r0.meas_biomass_t_ha, 2)]
                        + [num(hv.loc[(c, t, y)].sim_biomass_t_ha, 2) for c in CASES]
                        + [num(hv.loc[(c, t, y)].sim_yield_dm_t_ha, 2) for c in CASES])
    short = {"measured": "meas. irr.", "auto100": "auto 1.0", "auto080": "auto 0.80"}
    parts.append("<h2>4. Biomass at harvest</h2>"
                 "<p>Above-ground biomass on the harvest/kill day (mgt_out <code>plant_bioms</code>) and "
                 "the harvested dry yield (<code>op_var</code>), t/ha. Measured biomass exists for 52A "
                 "only (2013–2019). The 2014 sugarbeet was not harvested (tilled in), so its yield is 0. "
                 "<b>Crop parameters are Ai-SWAT-Plus's, copied verbatim and not fitted here</b>: "
                 "potato under-yields, and spring wheat uses Ai-SWAT-Plus <code>wwht</code>.</p>"
                 + table(["Treatment", "Year", "Crop", "Biomass, measured"]
                         + [f"Biomass, {short[c]}" for c in CASES]
                         + [f"Yield DM, {short[c]}" for c in CASES], rows))
    up = pd.read_csv(DATA / "uptake.csv").set_index("year")
    yi = Y.set_index(["case", "treatment", "year"])
    same = [y for y in range(FIRST, LAST + 1)
            if all(np.ptp([yi.loc[(c, t, y)].n_uptake_kg_ha for t in HRU.values()]) < 0.05
                   and np.ptp([hv.loc[(c, t, y)].sim_biomass_t_ha for t in HRU.values()]) < 0.005
                   for c in CASES)]
    same_note = (f"<p>Simulated biomass and N uptake are the same on all three treatments in "
                 f"{', '.join(map(str, same))} (every irrigation case): in those years the crops "
                 "are not N-limited even on the control.</p>" if same else "")
    rows = []
    for t in HRU.values():
        for y in range(FIRST, LAST + 1):
            mn = up.n_up_kg_ha.get(y) if t == "52A" else None
            mp = up.p_up_kg_ha.get(y) if t == "52A" else None
            rows.append([esc(TRT_LABEL[t]), y, esc(hv.loc[("measured", t, y)].crop), num(mn)]
                        + [num(yi.loc[(c, t, y)].n_uptake_kg_ha) for c in CASES] + [num(mp)]
                        + [num(yi.loc[(c, t, y)].p_uptake_kg_ha) for c in CASES])
    parts.append("<h2>5. N and P uptake</h2>"
                 "<p>Plant N and P uptake over the calendar year (hru_nb <code>nuptake</code>/"
                 "<code>puptake</code>), kg/ha. Each year has one crop. Measured uptake is 52A only, "
                 "2013–2019.</p>" + same_note
                 + table(["Treatment", "Year", "Crop", "N, measured"] + [f"N, {short[c]}" for c in CASES]
                         + ["P, measured"] + [f"P, {short[c]}" for c in CASES], rows))

    # 6 + 7. daily soil
    def soil(col_kg, col_mg, vars_, name):
        out = [f"<h3>{name}: whole profile, kg/ha</h3>", case_legend()]
        for t in HRU.values():
            ser = [(CASES[c][0], CASE_COL[c], D[(D.case == c) & (D.treatment == t)].set_index("date")[col_kg])
                   for c in CASES]
            out.append(line_panel(TRT_LABEL[t], ser, "kg/ha (0–1520 mm)", "kg/ha"))
        out.append(f"<h3>{name}: profile-mean mg/kg, with measured values</h3>")
        out.append(legend_row([(CASES[c][0], CASE_COL[c], "square") for c in CASES]))
        out.append(legend_row([(SHAPES[v][0], F.INK, SHAPES[v][1]) for v in vars_]))
        for t in HRU.values():
            ser = [(CASES[c][0], CASE_COL[c], D[(D.case == c) & (D.treatment == t)].set_index("date")[col_mg])
                   for c in CASES]
            mm = meas[(meas.treatment == t) & meas["var"].isin(vars_)]
            pts = [(r.date, r.value, SHAPES[r.var][1],
                    f"{r.date:%Y-%m-%d} {SHAPES[r.var][0]}: {r.value:.1f} mg/kg") for r in mm.itertuples()]
            out.append(line_panel(TRT_LABEL[t], ser, "mg/kg", "mg/kg", points=pts))
        return "".join(out)

    parts.append("<h2>6. Soil nitrate-N, daily</h2>"
                 "<p>Read from a 0-kg probe on every day (mgt_out <code>soil_no3</code>, the start-of-day "
                 "state). SWAT+ reports only the <b>whole-profile total (0–1520 mm)</b>; no output on "
                 "disk gives soil NO₃ by layer. The mg/kg panels divide that total by the profile's dry "
                 f"soil mass ({mass / 1e6:,.2f} × 10⁶ kg/ha, soils.sol bulk density × depth). That is a "
                 "1.52 m mean, so it is <b>not depth-matched</b> to the measured samples (depth not "
                 "stated, or 0–60 cm). Compare shape and timing, not level.</p>"
                 + soil("soil_no3n_kg_ha", "soil_no3n_mg_kg_profile",
                        ["profile_no3_mg_kg", "no3_mg_kg_0_60"], "Soil NO₃-N"))
    parts.append("<h2>7. Soil soluble P, daily</h2>"
                 "<p>SWAT+ solution (labile) P from the same probes (mgt_out <code>soil_solp</code>), "
                 "whole profile. Measured values are <b>Olsen P</b> (NaHCO₃-extractable) at 0–30 and "
                 "30–60 cm. That is a different pool and depth from SWAT solution P, so compare trends "
                 "only.</p>"
                 + soil("soil_solp_kg_ha", "soil_solp_mg_kg_profile",
                        ["olsen_p_mg_kg_0_30", "olsen_p_mg_kg_30_60"], "Soil solution P"))

    files = "".join(f"<li><code>runs/ltm_nutrient_summary/{n}.csv</code></li>"
                    for n in ("yearly", "monthly", "daily", "harvest", "measured_soil"))
    status = "inert" if check["inert"] else "NOT inert"
    head = ("<h1>LTM scenario: water and nutrients under measured and auto-irrigation</h1>"
            "<p>SWAT+ rev 62, model/ltm/TxtInOut (south field, 2013–2020 after a 2012 bean spin-up), "
            "with three HRUs: 52A (52 Mg/ha DM manure every fall), fertilizer-only and control. "
            "Weather is gridMET at the field. Three irrigation cases, all on the measured rotation, "
            "fertilizer and manure:</p><ul>"
            "<li><b>Measured irrigation</b>: the trial's daily events, replayed at efficiency 1.0.</li>"
            f"<li><b>Auto-irrigation, eff 1.0</b>: one 25.4 mm pass whenever profile plant-available "
            f"water is below {AUTO_IRR['sw_frac']:.4f} × its field-capacity value, inside each crop's "
            "irrigation window. The trigger was fitted at this efficiency to apply the trial's "
            "2013–2020 total.</li>"
            "<li><b>Auto-irrigation, eff 0.80</b>: the same rule at the experiments' base efficiency. "
            "Only 80 % of each pass reaches the soil, so it pumps more gross water.</li></ul>"
            f"<p>Daily probes: {status} (max |Δ| vs no daily probes: "
            + ", ".join(f"{k} {v:.2g}" for k, v in check.items() if k != "inert")
            + f"). Data files:</p><ul>{files}</ul>")
    style = ("<style>body{font-family:Calibri,sans-serif;background:#fff;color:#000;margin:24px;"
             "max-width:1000px}table{border-collapse:collapse;font-size:13px;margin:8px 0}"
             "th,td{border:1px solid #ccc;padding:2px 6px;text-align:right}"
             "th:first-child,td:first-child{text-align:left}code{font-family:Calibri,sans-serif}"
             "</style>")
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>LTM water and nutrient summary</title>" + style + "</head><body>"
            + head + "".join(parts) + HOVER_JS + "</body></html>\n")


def main() -> None:
    scheds = base_schedules()
    res = {}
    with FastRunner(TIO, trim_outputs=False, editable=EDITABLE) as fr:
        for case, (_, eff) in CASES.items():
            res[case] = run_case(fr, case_edits(scheds, eff))
            print(f"ran {case}")
        noprobe = run_case(fr, case_edits(scheds, None, probes=False))
    check = probe_check(res["measured"], noprobe)
    print("probe check:", check)
    if not check["inert"]:
        raise SystemExit("daily probes changed the simulation; not writing the summary")

    T = build_tables(res)
    meas = measured_soil()
    mass = soil_mass_kg_ha()
    m = T["monthly"]
    irr = pd.read_csv(DATA / "irrigation_daily.csv", parse_dates=["date"])
    auto_check = {
        "sim_mm": float(m[(m.case == "auto100") & (m.treatment == "52A")].irr_gross_mm.sum()),
        "fit_mm": float(json.loads((TIO.parent / "auto_irr.json").read_text())["auto_mm"]),
        "meas_mm": float(irr[irr.date.dt.year.between(FIRST, LAST)].irr_mm.sum())}
    print("auto-irrigation check:", auto_check)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in T.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, float_format="%.4f")
    meas.to_csv(OUT_DIR / "measured_soil.csv", index=False)
    OUT_HTML.write_text(page(T, meas, check, mass, auto_check))
    print(f"wrote {OUT_HTML.relative_to(ROOT)} and {OUT_DIR.relative_to(ROOT)}/*.csv")


if __name__ == "__main__":
    main()
