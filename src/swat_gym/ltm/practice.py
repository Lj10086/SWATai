"""Management templates: the trial's 52A practice per season, per crop, and the lever geometry.

Season k (1..8) of any window replays trial year 2012 + k: its crop, dates, fertiliser, its
measured daily irrigation (by calendar day) and the manure applied the fall before it (fall
k - 1; fall 0 is the spin-up year's). This is "measured practice", the normaliser of every
score. Two departures from the field record, both because a window is a generic year:
2014's beets, killed unharvested on 10-28 after a field problem (Leytem et al. 2024 sec. 2.6;
the validation build does that), are harvested on the sugarbeet template date (2018's 10-09),
and the 2016 potato N split (not reported) is in equal thirds.

Exp 1 (rotation) needs each crop's practice outside its trial year: a trial crop takes its
template year (barley 2013, sugarbeet 2018, wheat 2015, potato 2016), its 52A fertiliser and
that year's measured irrigation summed by month; alfalfa and corn silage (not in the trial)
take the Ai-SWAT-Plus spec dates, no synthetic N (manure supplies it, as for 52A in most
years), and the crop model's own monthly irrigation (model/ltm/exp1_irr_monthly.json).

Auto-irrigation (user decision 2026-10-25: every Exp 1 season, and an Exp 3 baseline): a
soil-water-triggered table (``AUTO_IRR``) inside each season's irrigation window(s) --
trial crops: the first..last measured event of the season's (template) year; alfalfa and
silage: the Ai-SWAT-Plus spec windows. Rendered per HRU by ``render``/``runner``.

Manure rate x in [0, 1]: x = 1 is the trial's 52 Mg/ha DM target with that fall's measured
composition (target-rate basis, user decision 2026-10-25), so x = 1 replays measured 52A.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from ..fastrunner import PROJECT_ROOT
from .windows import SEASONS

DATA = PROJECT_ROOT / "data" / "ltm"
PARAMS = PROJECT_ROOT / "model" / "ltm" / "crop_params"
TXTINOUT = PROJECT_ROOT / "model" / "ltm" / "TxtInOut"
#: alfalfa's and corn silage's monthly irrigation (Exp 1; not trial crops), see the file's "source"
EXP1_IRR_MONTHLY = PROJECT_ROOT / "model" / "ltm" / "exp1_irr_monthly.json"

CROPS = ("barl", "sgbt", "wwht", "pota", "corn", "alfa")
TRIAL_ROTATION = ("barl", "sgbt", "wwht", "pota", "barl", "sgbt", "wwht", "sgbt")
TRIAL_START = 2012                        #: season k is trial year TRIAL_START + k
TEMPLATE_YEAR = {"barl": 2013, "sgbt": 2018, "wwht": 2015, "pota": 2016}
CROP_NAME = {"barley": "barl", "sugarbeet": "sgbt", "wheat": "wwht", "potato": "pota"}
ALFA_STAND = 3                            #: years per stand (user decision 2026-10-25)
NO_REPEAT = ("sgbt", "pota")              #: the rule first proposed (kept for stage keys)
#: Rotation rules (user decision 2026-10-27, replacing "no rules": without them Exp 1 grew
#: potato 8 years running, which SWAT+ cannot penalise -- it has no soilborne disease or
#: nematode module). (1) No crop two years running, except inside one 3-year alfalfa stand (a
#: stand may not follow a stand). (2) Potato at most once in any ROTATION_GAP["pota"]
#: consecutive years, i.e. two non-potato years between potato crops -- the most common Idaho
#: practice, "potatoes followed by two years of grain" (38 % of growers; Patterson & Bohl 1995,
#: UI A.E. Extension Series 95-4); the recommendation is "not more than one crop of potatoes in
#: succession" (McDole & Dallimore 1973, UI Current Information Series 200). Sugarbeet is held to
#: rule (1) only: the trial itself grows it in 2018 and 2020 (user decision 2026-10-27).
ROTATION_NO_REPEAT: tuple[str, ...] = CROPS
ROTATION_GAP = {"pota": 3}
MANURE_TARGET_MG = 52.0
MANURE_DAY = (10, 20)                     #: [W4] (data/ltm/management.csv)
#: Auto-irrigation (user decision 2026-10-25: "realistic, moderate, standard practice"): a
#: 1-inch gross pass (the trial's typical event; IRR_PASS_MM) at the plan's efficiency
#: whenever profile plant-available water < sw_frac x its field-capacity value, inside a window.
#: sw_frac is fitted so the rule applies the trial's own 2013-2020 water under its rotation at
#: MEASURED_IRR_EFF
#: (scripts/ltm_fit_auto_irr.py -> model/ltm/auto_irr.json). Not Ai-SWAT-Plus's table: that row's
#: lim_op "-" makes its trigger FC - 0.95 mm, i.e. irrigate whenever the profile is below FC.
AUTO_IRR_FILE = TXTINOUT.parent / "auto_irr.json"
IRR_CUTOFF_DAYS = 7

#: Irrigation application efficiency: SWAT+ puts amt x eff in the soil (rev 62 mgt_sched /
#: actions: applied = amt_mm * eff * (1 - surq)); the rest leaves the field. Water is costed and
#: scored on the gross amount. "Potential" application efficiencies, Irmak et al. 2011 (UNL
#: Extension EC732) Table 2, range midpoints: conventional furrow 45-65 %, center pivot 75-85 %
#: (solid set 70-85 %), surface drip 85-95 %.
#: User decision 2026-10-27: the measured replay stays at MEASURED_IRR_EFF = 1.0 (the trial's
#: depths are what reached the plots; at 0.8 the 2013-2020 profile never drains and yields drop
#: 11-46 %, PROVENANCE section 7). Auto-irrigation and every lever-generated schedule (Exp 3's
#: default / fixed / policy) run at the plan's ``irr_eff``: base 0.80 (the collaborators' "medium
#: efficiency"; Exp 1 and the Exp 3 base case), Exp 3 cases 0.55 / 0.90. The auto trigger is
#: fitted at 1.0 (the rule reproduces the trial's water), so at 0.8 it pumps more.
IRR_EFF_CASES = {"low": 0.55, "medium": 0.80, "high": 0.90}
BASE_IRR_EFF = IRR_EFF_CASES["medium"]
MEASURED_IRR_EFF = 1.0
#: The efficiency plans default to. The Exp 3 driver sets it (and LTM_IRR_EFF, which spawned
#: workers read on import) per efficiency case.
IRR_EFF = float(os.environ.get("LTM_IRR_EFF", BASE_IRR_EFF))


def set_irr_eff(eff: float) -> None:
    global IRR_EFF
    IRR_EFF = float(eff)
    os.environ["LTM_IRR_EFF"] = repr(float(eff))


def auto_op(eff: float) -> str:
    """irr.ops name of the auto-irrigation pass at efficiency ``eff``."""
    return f"ltm_auto{round(eff * 100):03d}"


#: Spin-up year (collaborator's setup, G. Richards email 2026-10-27): pinto beans (the field's
#: pre-study crop) on the Ai-SWAT-Plus sheet dates, auto-irrigated, 100 kg/ha N and 50 kg/ha P
#: (applied at planting: user decision 2026-10-27). Elemental N/P under their own names so
#: scoring never counts them as the scored seasons' fertiliser.
SPINUP_CROP = "ptbn"
SPINUP_FERT = (("ltm_bnn", 100.0), ("ltm_bnp", 50.0))


def _auto_irr() -> dict:
    base = {"mm": 25.4}
    if AUTO_IRR_FILE.is_file():
        return {**base, "sw_frac": float(json.loads(AUTO_IRR_FILE.read_text())["sw_frac"])}
    return {**base, "sw_frac": None}


AUTO_IRR = _auto_irr()
SGBT_2014_HARVEST_FROM = 2018
TREATMENT = "52A"
#: Irrigation passes: at most this many mm on one day (the trial's events are mostly 1 inch).
IRR_PASS_MM = 25.4
N_MONTH_MAX = 150.0                       #: kg N/ha per month (as the legacy crop models)
N_PRODUCT, MAP = "elem_n", "ltm_map"
MAP_N, MAP_P = 0.11, 0.227


def _d(s: str) -> date:
    return date.fromisoformat(s)


@dataclass(frozen=True)
class Season:
    """One crop-season, dates as (month, day) in its own calendar year."""
    crop: str
    plant: tuple[int, int]
    harvest: tuple[int, int]
    harv_op: str
    cuts: tuple[tuple[int, int], ...] = ()          #: alfalfa: all cut dates (last = harvest)
    #: [(mon, day, product, kg)] synthetic fertiliser of measured practice (MAP and urea)
    fert: tuple[tuple[int, int, str, float], ...] = ()
    #: measured irrigation [(mon, day, mm)] (trial seasons) -- None for template seasons
    irr_daily: tuple[tuple[int, int, float], ...] | None = None
    #: monthly irrigation of the template {mon: mm}
    irr_monthly: tuple[tuple[int, float], ...] = ()
    #: auto-irrigation windows, (first, last) day of year
    irr_windows: tuple[tuple[int, int], ...] = ()


@lru_cache(maxsize=None)
def _management() -> pd.DataFrame:
    m = pd.read_csv(DATA / "management.csv")
    return m[m.treatment == TREATMENT]


@lru_cache(maxsize=None)
def _irrigation() -> pd.DataFrame:
    irr = pd.read_csv(DATA / "irrigation_daily.csv", parse_dates=["date"])
    return irr[irr.irr_mm > 0]


@lru_cache(maxsize=None)
def _crop_params(plant: str) -> dict:
    return json.loads((PARAMS / f"{plant}.json").read_text())


def fert_p_policy() -> str:
    build = json.loads((TXTINOUT.parent / "build.json").read_text())
    return build["fert_p"]


def _fert_ops(year: int) -> tuple:
    """52A synthetic fertiliser of a trial year, as build_ltm renders it (MAP + urea)."""
    m = _management()
    r = m[(m.year == year) & (m.op == "fert")]
    if r.empty:
        return ()
    r = r.iloc[0]
    if pd.isna(r.p_kg_ha):
        lo, hi = float(r.p_kg_ha_lo), float(r.p_kg_ha_hi)
        p = {"lo": lo, "hi": hi, "mid": (lo + hi) / 2}[fert_p_policy()]
    else:
        p = float(r.p_kg_ha)
    d = _d(r.date)
    out = []
    map_kg = p / MAP_P
    if map_kg > 0:
        out.append((d.month, d.day, MAP, map_kg))
    urea = float(r.n_kg_ha) - MAP_N * map_kg
    dates = [d] if year != 2016 else [date(2016, 4, 20), date(2016, 5, 25), date(2016, 7, 12)]
    if urea > 1e-9:
        out += [(dd.month, dd.day, N_PRODUCT, urea / len(dates)) for dd in dates]
    return tuple(out)


def _irr_daily(year: int) -> tuple:
    irr = _irrigation()
    y = irr[irr.date.dt.year == year]
    return tuple((r.date.month, r.date.day, float(r.irr_mm)) for r in y.itertuples())


def _monthly(daily: tuple) -> tuple:
    by: dict[int, float] = {}
    for mon, _, mm in daily:
        by[mon] = by.get(mon, 0.0) + mm
    return tuple(sorted(by.items()))


@lru_cache(maxsize=None)
def trial_season(k: int) -> Season:
    """Season k (1..8): trial year 2012 + k under 52A."""
    year = TRIAL_START + k
    m = _management()
    pl = m[(m.year == year) & (m.op == "plant")].iloc[0]
    hv = m[(m.year == year) & (m.op == "harvest")].iloc[0]
    crop = CROP_NAME[pl.crop]
    p, h = _d(pl.date), _d(hv.date)
    if year == 2014:   # killed unharvested in the field; a generic year harvests it
        h2 = _d(m[(m.year == SGBT_2014_HARVEST_FROM) & (m.op == "harvest")].iloc[0].date)
        h = date(year, h2.month, h2.day)
    daily = _irr_daily(year)
    doys = [date(year, mon, day).timetuple().tm_yday for mon, day, _ in daily]
    # window: first measured event .. a week before harvest (standard cut-off; the trial's
    # post-harvest events, e.g. 2013-09-16 after an 08-05 barley harvest, wet bare stubble)
    last = min(max(doys), date(year, h.month, h.day).timetuple().tm_yday - IRR_CUTOFF_DAYS) \
        if doys else 0
    return Season(crop, (p.month, p.day), (h.month, h.day), _crop_params(crop)["harv_op"],
                  fert=_fert_ops(year), irr_daily=daily, irr_monthly=_monthly(daily),
                  irr_windows=((min(doys), last),) if doys else ())


@lru_cache(maxsize=None)
def crop_season(crop: str) -> Season:
    """Exp 1: the crop's practice in any season (its template year, or the Ai-SWAT-Plus spec)."""
    if crop in TEMPLATE_YEAR:
        t = trial_season(TEMPLATE_YEAR[crop] - TRIAL_START)
        return Season(t.crop, t.plant, t.harvest, t.harv_op, fert=t.fert,
                      irr_monthly=t.irr_monthly, irr_windows=t.irr_windows)
    monthly = tuple((int(m), float(mm)) for m, mm in
                    json.loads(EXP1_IRR_MONTHLY.read_text())[crop])
    rec = _crop_params(crop)["spec"]
    cuts = tuple(tuple(h) for h in rec["harvests"])
    return Season(crop, tuple(rec["emerge"]), cuts[-1], _crop_params(crop)["harv_op"],
                  cuts=cuts if rec["perennial"] else (), irr_monthly=monthly,
                  irr_windows=tuple(tuple(w) for w in rec["irrigation_windows_doy"]))


@lru_cache(maxsize=None)
def spinup_season() -> Season:
    """The spin-up year's beans (SPINUP_CROP): spec dates, harvest op, auto-irrigation window
    from emergence to a week before harvest (as trial seasons)."""
    rec = _crop_params(SPINUP_CROP)
    sp = rec["spec"]
    p, h = tuple(sp["emerge"]), tuple(sp["harvests"][-1])
    return Season(SPINUP_CROP, p, h, rec["harv_op"])


def spinup_window(year: int) -> tuple[int, int]:
    s = spinup_season()
    doy = lambda md: date(year, *md).timetuple().tm_yday  # noqa: E731
    return doy(s.plant), doy(s.harvest) - IRR_CUTOFF_DAYS


@lru_cache(maxsize=None)
def manure(fall: int) -> tuple[str, float, float]:
    """Fall ``fall`` (0..7, 0 = spin-up year): (fertilizer.frt name, kg DM/ha at x = 1, kg N)."""
    year = TRIAL_START + fall
    m = _management()
    r = m[(m.year == year) & (m.op == "manure")].iloc[0]
    return f"ltm{year}", float(r.dm_mg_ha) * 1000.0, float(r.n_kg_ha)


def manure_kg(fall: int, x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) * manure(fall)[1] if x > MANURE_DEAD_ZONE else 0.0


#: As N_DEAD_ZONE: CMA-ES "zeros" (1e-6) must not buy a manure pass.
MANURE_DEAD_ZONE = 0.05

#: Synthetic-N lever values at or below this share of the range apply nothing; above it the
#: range is rescaled so the maximum is unchanged. An application pass costs the same at any
#: rate, so without a dead zone an optimizer whose samples never land exactly on 0 (CMA-ES
#: returns 1e-6, not 0) pays a full pass for sub-gram applications. Same mapping on every path.
N_DEAD_ZONE = 0.05


def n_amount(x, max_amount: float):
    """Unit-box N lever value(s) -> applied amount, with the dead zone."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x <= N_DEAD_ZONE, 0.0, (x - N_DEAD_ZONE) / (1.0 - N_DEAD_ZONE) * max_amount)


def n_unit(amount, max_amount: float):
    """Inverse of n_amount: applied amount(s) -> unit-box value (0 for none)."""
    a = np.clip(np.asarray(amount, dtype=float) / max_amount, 0.0, 1.0)
    return np.where(a <= 0.0, 0.0, N_DEAD_ZONE + (1.0 - N_DEAD_ZONE) * a)


# -- lever geometry ---------------------------------------------------------------------------

def _days_in(year: int, mon: int) -> int:
    nxt = date(year + (mon == 12), mon % 12 + 1, 1)
    return (nxt - date(year, mon, 1)).days


@dataclass(frozen=True)
class Decision:
    """One decision point: season k (0 = spin-up), kind, month (monthly levers)."""
    k: int
    kind: str              #: "crop", "manure", "n", "irr"
    mon: int = 0
    day: int = 1           #: decision date within the season's year (the probe date)


def irr_months(s: Season) -> tuple[int, ...]:
    """Months a season can be irrigated: plant month (or first measured event) .. harvest."""
    first = min([s.plant[0]] + [m for m, _, _ in (s.irr_daily or ())]
                + [m for m, _ in s.irr_monthly])
    return tuple(range(first, s.harvest[0] + 1))


def n_months(s: Season) -> tuple[int, ...]:
    """Months whose 15th lies in [plant - 30 d, harvest) -- pre-plant N included."""
    ref = 2015
    plant = date(ref, *s.plant) - timedelta(days=30)
    harv = date(ref, *s.harvest)
    return tuple(m for m in range(1, 13) if plant <= date(ref, m, 15) < harv)


#: Cap per calendar month (mm) of the irrigation lever: 1.5 x the most the trial applied in
#: that month in any season (x = 1/1.5 replays the wettest measured month).
def irr_cap() -> dict[int, float]:
    caps: dict[int, float] = {}
    for k in range(1, SEASONS + 1):
        for mon, mm in trial_season(k).irr_monthly:
            caps[mon] = max(caps.get(mon, 0.0), 1.5 * mm)
    return caps


def decisions(lever: str) -> tuple[Decision, ...]:
    """Decision points of a lever, in calendar order (one env step each)."""
    out: list[Decision] = []
    if lever == "R":
        return tuple(Decision(k, "crop", 1, 2) for k in range(1, SEASONS + 1))
    for k in range(0, SEASONS + 1):
        if k >= 1:
            s = trial_season(k)
            months = n_months(s) if lever == "N" else irr_months(s)
            caps = irr_cap()
            for mon in months:
                if lever == "I" and caps.get(mon, 0.0) <= 0:
                    continue
                out.append(Decision(k, "n" if lever == "N" else "irr", mon, 1))
        if lever == "N" and k < SEASONS:
            out.append(Decision(k, "manure", MANURE_DAY[0], MANURE_DAY[1] - 1))
    return tuple(out)


def measured_x(lever: str) -> np.ndarray:
    """The measured practice's value at every decision (the zero-policy / CMA-ES start)."""
    xs = []
    caps = irr_cap()
    for d in decisions(lever):
        if d.kind == "manure":
            xs.append(1.0)
        elif d.kind == "n":
            kg = sum(kg for mon, _, prod, kg in trial_season(d.k).fert
                     if prod == N_PRODUCT and mon == d.mon)
            xs.append(float(n_unit(kg, N_MONTH_MAX)))
        elif d.kind == "irr":
            mm = dict(trial_season(d.k).irr_monthly).get(d.mon, 0.0)
            xs.append(mm / caps[d.mon])
        else:
            xs.append(float(CROPS.index(TRIAL_ROTATION[d.k - 1])))
    return np.asarray(xs, dtype=float)
