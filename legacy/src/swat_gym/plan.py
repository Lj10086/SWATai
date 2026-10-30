"""Management plans: the measured default, the two levers, and open-loop scoring.

A lever maps a unit-box vector ``x`` to a full 7-year plan; everything it doesn't control
stays at measured practice.
  I: 42 monthly irrigation depths (Apr-Sep x 7 years).
  N: per annual-crop year, 6 monthly mineral-N rates + 1 April manure rate (4 x 7 = 28).
     Irrigation stays at measured practice; alfalfa years get no N.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .rewarders import Prices, nass, profit
from .schedule import CALENDAR, GROWING_MONTHS, YearAction, _doy, _md, build

N_YEARS = 7                 # scored years
SPINUP = 1                  # 2012-style spin-up year, skipped by print.prt nyskip
WEATHER_YEARS = (1995, 2025)
N_GROWING = len(GROWING_MONTHS)

MEASURED_ROTATION = ("corn", "barl", "alfa", "alfa", "alfa", "corn", "barl")
#: Measured GRACEnet manure per year: (Mg/ha, source). Applied 10 April (DOY 100).
MEASURED_MANURE = ((43.8, "gn2013"), (48.0, "gn2014"), (0.0, "gn2013"), (0.0, "gn2013"),
                   (0.0, "gn2013"), (88.3, "gn2018"), (54.3, "gn2019"))
MANURE_DOY = 100
#: Measured-practice irrigation: every 7 days from DOY 130 at 32.824 mm -> 3,938.8 mm total.
DEFAULT_IRR = {"irr_start_doy": 130, "irr_interval": 7, "irr_depth": 32.824}
MEASURED_IRRIGATION_MM = 3938.8

MONTH_DEPTH_MAX = 200.0     # mm per month (lever I range)
MAX_EVENT_MM = 40.0         # mm per pass; big single events fake leaching in SWAT+
MONTH_N_MAX = 150.0         # kg N/ha per month (lever N range)
MANURE_MAX = 90.0           # Mg/ha per year (lever N range)

ANNUAL_YEARS = tuple(y for y, c in enumerate(MEASURED_ROTATION) if c != "alfa")
#: Bump when the lever -> schedule mapping changes meaning; part of every stage config.
#: 2: no irrigation carry-back between months; no N after harvest (2026-10-23).
#: 3: nitrogen dead zone (N_DEAD_ZONE) (2026-10-24).
RENDER_VERSION = 3

#: Nitrogen lever values at or below this share of the range apply nothing; above it the
#: range is rescaled so the maximum is unchanged. An application pass costs the same at any
#: rate, so without a dead zone an optimizer whose samples never land exactly on 0 (CMA-ES
#: returns 1e-6, not 0) pays a full pass for sub-gram applications; residual-action PPO, which
#: clips to exactly 0, did not. Same mapping on every code path. Irrigation has no pass cost.
N_DEAD_ZONE = 0.05


def n_amount(x, max_amount: float):
    """Unit-box N lever value(s) -> applied amount, with the dead zone."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x <= N_DEAD_ZONE, 0.0, (x - N_DEAD_ZONE) / (1.0 - N_DEAD_ZONE) * max_amount)


def n_unit(amount, max_amount: float):
    """Inverse of n_amount: applied amount(s) -> unit-box value (0 for none)."""
    a = np.clip(np.asarray(amount, dtype=float) / max_amount, 0.0, 1.0)
    return np.where(a <= 0.0, 0.0, N_DEAD_ZONE + (1.0 - N_DEAD_ZONE) * a)
DIMS = {"I": N_YEARS * N_GROWING, "N": len(ANNUAL_YEARS) * (N_GROWING + 1)}


def default_plan() -> list[YearAction]:
    return [YearAction(crop=c, manure_mg=mg, manure_doy=MANURE_DOY, manure_src=src,
                       **DEFAULT_IRR)
            for c, (mg, src) in zip(MEASURED_ROTATION, MEASURED_MANURE)]


def time_sim(start_year: int, n_years: int = N_YEARS + SPINUP) -> str:
    return ("time.sim: written by swat_gym.plan\n"
            "day_start  yrc_start   day_end   yrc_end      step  \n"
            f"{0:>10}{start_year:>11}{0:>10}{start_year + n_years - 1:>10}{0:>10}  \n")


# -- irrigation rendering -------------------------------------------------------------

def _season_end(crop: str) -> int:
    cal = CALENDAR[crop]
    return _doy(*cal["cuts"][-1]) if crop == "alfa" else _doy(*cal["harvest"])


def _days_in_month(mon: int) -> int:
    return _doy(mon + 1, 1) - _doy(mon, 1) if mon < 12 else 31


def _spread(volume: float, first: int, last: int, max_event: float):
    """Equal passes on distinct days in [first, last]; returns (passes, undeliverable volume)."""
    usable = last - first + 1
    if volume <= 0 or usable <= 0:
        return [], max(0.0, volume)
    n = min(max(1, int(np.ceil(volume / max_event))), usable)
    per = volume / n
    overflow = 0.0
    if per > max_event:
        overflow = volume - usable * max_event
        per, n = max_event, usable
    return [(first + int(k * usable / n), per) for k in range(n)], overflow


def _month_days(mon: int, crop: str) -> tuple[int, int]:
    """[first, last] irrigable day of ``mon`` for ``crop`` (last < first: none, harvested)."""
    first = _doy(mon, 1)
    return first, min(first + _days_in_month(mon) - 1, _season_end(crop) - 1)


def irrigable_months(crop: str) -> tuple[int, ...]:
    """Indices into GROWING_MONTHS with at least one day before the season ends."""
    return tuple(mi for mi, mon in enumerate(GROWING_MONTHS)
                 if _month_days(mon, crop)[1] >= _month_days(mon, crop)[0])


def fertilisable_months(crop: str) -> tuple[int, ...]:
    """Indices into GROWING_MONTHS whose mid-month N date falls before the season ends."""
    return tuple(mi for mi, mon in enumerate(GROWING_MONTHS) if _doy(mon, 15) < _season_end(crop))


def year_events(month_mm_year, crop: str, max_event: float = MAX_EVENT_MM):
    """Six monthly volumes -> ((doy, mm), ...) passes <= max_event, before harvest.

    Deterministic and state-independent. Each month's water stays in that month: volume
    above max_event x its usable days (or in a month after harvest) is dropped, never moved,
    so deciding a later month cannot change an earlier one (see undeliverable_mm).
    """
    events = []
    for mi, mon in enumerate(GROWING_MONTHS):
        events += _spread(float(month_mm_year[mi]), *_month_days(mon, crop), max_event)[0]
    return tuple(sorted(events))


def undeliverable_mm(month_mm_year, crop: str, max_event: float = MAX_EVENT_MM) -> float:
    """Requested mm that year_events drops (it never reaches the field or the bill)."""
    return float(sum(_spread(float(month_mm_year[mi]), *_month_days(mon, crop), max_event)[1]
                     for mi, mon in enumerate(GROWING_MONTHS)))


def default_monthly_irr() -> np.ndarray:
    """Measured-practice irrigation as monthly mm, shape (N_YEARS, N_GROWING)."""
    out = np.zeros((N_YEARS, N_GROWING))
    for y, a in enumerate(default_plan()):
        end = _season_end(a.crop)
        doy = a.irr_start_doy
        while doy < end:
            mon = _md(doy)[0]
            if mon in GROWING_MONTHS:
                out[y, GROWING_MONTHS.index(mon)] += a.irr_depth
            doy += a.irr_interval
    return out


# -- levers ---------------------------------------------------------------------------

def default_x(lever: str) -> np.ndarray:
    """Measured practice in the lever's unit box (the CMA-ES start point)."""
    if lever == "I":
        return np.clip(default_monthly_irr() / MONTH_DEPTH_MAX, 0.0, 1.0).ravel()
    x = np.zeros((len(ANNUAL_YEARS), N_GROWING + 1))
    for i, y in enumerate(ANNUAL_YEARS):
        x[i, -1] = n_unit(MEASURED_MANURE[y][0], MANURE_MAX)
    return x.ravel()


def plan_from(lever: str, x: Sequence[float]) -> list[YearAction]:
    """Unit-box vector -> full plan."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    base = default_plan()
    if lever == "I":
        month_mm = x.reshape(N_YEARS, N_GROWING) * MONTH_DEPTH_MAX
    else:
        month_mm = default_monthly_irr()
    plan = [replace(a, irr_day_depths=year_events(month_mm[y], a.crop))
            for y, a in enumerate(base)]
    if lever == "N":
        xn = x.reshape(len(ANNUAL_YEARS), N_GROWING + 1)
        for i, y in enumerate(ANNUAL_YEARS):
            live = fertilisable_months(plan[y].crop)
            splits = tuple((_doy(mon, 15), float(n_amount(xn[i, m], MONTH_N_MAX)))
                           for m, mon in enumerate(GROWING_MONTHS) if m in live)
            plan[y] = replace(plan[y], fert_splits=splits,
                              manure_mg=float(n_amount(xn[i, -1], MANURE_MAX)))
    return plan


def evaluate(lever: str, x: Sequence[float], runner, *, prices: Prices | None = None,
             start_year: int = 2013, no3_price: float = 0.0) -> dict:
    """Simulate the lever's plan on one weather window and return profit terms."""
    edits = build(plan_from(lever, x))
    edits["time.sim"] = time_sim(start_year)
    runner.run(edits)
    out = profit(runner, prices or nass(2024), no3_price=no3_price)
    out["start_year"] = start_year
    out["irr_dropped_mm"] = irr_dropped_mm(lever, x)
    return out


def irr_dropped_mm(lever: str, x: Sequence[float]) -> float:
    """Requested irrigation the renderer could not place (0 for lever N's measured water)."""
    if lever != "I":
        return 0.0
    month_mm = np.clip(np.asarray(x, dtype=float), 0.0, 1.0).reshape(N_YEARS, N_GROWING)
    return float(sum(undeliverable_mm(month_mm[y] * MONTH_DEPTH_MAX, c)
                     for y, c in enumerate(MEASURED_ROTATION)))
