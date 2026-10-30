"""Episode plan -> one HRU's 9-year schedule (spin-up year + 8 seasons).

Ops per calendar year in date order, each year closed by a ``skip``
(the convention of the legacy crop models). Same-day order: probe, fertiliser/manure, tillage, plant, irrigation, harvest.
Probes are 0-kg ``elem_n`` passes (inert: verified against no-probe runs, PROVENANCE LTM)
whose mgt_out rows give the soil NO3 / solution P / soil water a decision observes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from . import practice as _pr
from .practice import (ROTATION_GAP, ROTATION_NO_REPEAT, ALFA_STAND, CROPS, IRR_PASS_MM, MANURE_DAY,
                       MEASURED_IRR_EFF, N_PRODUCT, SPINUP_FERT, TRIAL_ROTATION, Season, _days_in, crop_season,
                       irr_months, manure, manure_kg, n_months, spinup_season, spinup_window,
                       trial_season)
from .windows import SEASONS

TILL = "ltm_till"
PROBE = "elem_n"
ORDER = {"probe": 0, "fert": 1, "till": 2, "plnt": 3, "irrm": 4, "harv": 5, "hvkl": 6}


@dataclass(frozen=True)
class EpisodePlan:
    """Management of one HRU for one window. Defaults = measured 52A practice."""
    crops: tuple[str, ...] = TRIAL_ROTATION
    manure_x: tuple[float, ...] = (1.0,) * SEASONS          #: falls 0..7, nominal 52A = 1
    #: per season: None = measured fertiliser; else urea kg per n_months(season) (MAP kept)
    n_kg: tuple = (None,) * SEASONS
    #: per season: None = measured daily (trial crop) / template monthly; else mm per irr_months
    irr_mm: tuple = (None,) * SEASONS
    probes: tuple = field(default=(), compare=True)         #: decision points to probe
    #: True: every season auto-irrigated in its window(s) (practice.AUTO_IRR); irr_mm ignored
    irr_auto: bool = False
    #: application efficiency of auto-irrigation and generated (monthly) schedules
    #: (practice.IRR_EFF at creation); measured daily events replay at MEASURED_IRR_EFF
    irr_eff: float = field(default_factory=lambda: _pr.IRR_EFF)


def season_irr_eff(plan: EpisodePlan, k: int) -> float:
    """Efficiency of season k's irrigation: measured daily replay or generated/auto."""
    if plan.irr_auto or plan.irr_mm[k - 1] is not None or season_of(plan, k).irr_daily is None:
        return plan.irr_eff
    return MEASURED_IRR_EFF


def season_of(plan: EpisodePlan, k: int) -> Season:
    """Season k (1..8): the trial's own season when the plan grows the trial's crop."""
    crop = plan.crops[k - 1]
    return trial_season(k) if crop == TRIAL_ROTATION[k - 1] else crop_season(crop)


def stand_year(crops: tuple[str, ...], k: int) -> int:
    """1-based year within its alfalfa stand for season k (0 if not alfalfa)."""
    if crops[k - 1] != "alfa":
        return 0
    j = 1
    while k - j >= 1 and crops[k - j - 1] == "alfa":
        j += 1
    return (j - 1) % ALFA_STAND + 1


def validate(crops: tuple[str, ...]) -> None:
    """Rotation rules (practice.ROTATION_NO_REPEAT / ROTATION_GAP / ALFA_STAND)."""
    if len(crops) != SEASONS or any(c not in CROPS for c in crops):
        raise ValueError(f"bad rotation {crops}")
    for k in range(2, SEASONS + 1):
        c = crops[k - 1]
        if c == crops[k - 2] and c in ROTATION_NO_REPEAT and not \
                (c == "alfa" and stand_year(crops, k) > 1):
            raise ValueError(f"{c} two years running at season {k}: {crops}")
        gap = ROTATION_GAP.get(c, 0)
        if gap and c in crops[max(0, k - gap):k - 1]:
            raise ValueError(f"{c} twice within {gap} years at season {k}: {crops}")
    k = 1
    while k <= SEASONS:
        if crops[k - 1] == "alfa":
            run = 0
            while k + run <= SEASONS and crops[k + run - 1] == "alfa":
                run += 1
            if run % ALFA_STAND:
                raise ValueError(f"alfalfa stand of {run} years at season {k} (must be "
                                 f"{ALFA_STAND}): {crops}")
            k += run
        else:
            k += 1


def _line(typ: str, mon: int, day: int, d1: str, d2: str = "null", d3: float = 0.0) -> str:
    return f"{'':>48}{typ:<10}{mon:>6}{day:>10}{0.0:>14.5f}{d1:>18}{d2:>18}{d3:>14.5f}  "


def _passes(year: int, mon: int, mm: float, first: date, last: date) -> list[tuple[int, float]]:
    """``mm`` in one month as equal passes of <= IRR_PASS_MM on evenly spaced days."""
    days = [d for d in range(1, _days_in(year, mon) + 1)
            if first <= date(year, mon, d) <= last]
    if mm <= 1e-9 or not days:
        return []
    n = min(len(days), int(-(-mm // IRR_PASS_MM)))
    return [(days[int(i * len(days) / n)], mm / n) for i in range(n)]


def render(plan: EpisodePlan, start: int, tag: str) -> tuple[list[str], list[str], list]:
    """(schedule op lines, irr.ops rows, auto-irrigation windows [(year, first doy, last doy)])
    for one HRU and the window starting at ``start``."""
    validate(plan.crops)
    ops: list[tuple[int, int, int, int, str]] = []   # (year, mon, day, order, line)
    irr_rows: list[str] = []
    auto: list[tuple[int, int, int]] = []

    def add(year, typ, mon, day, *args):
        kind = "probe" if (typ == "fert" and args[0] == PROBE and args[2] == 0.0) else typ
        ops.append((year, mon, day, ORDER[kind], _line(typ, mon, day, *args)))

    def irrigate(year, mon, day, mm, eff):
        name = f"{tag}{len(irr_rows):05d}"
        irr_rows.append(f"{name:<22}{mm:>8.5f}{eff:>14.5f}" + f"{0.0:>14.5f}" * 5 + "  ltm")
        add(year, "irrm", mon, day, name)

    for k in range(0, SEASONS + 1):
        year = start + k
        if k == 0:   # spin-up beans: practice.SPINUP_*, always auto-irrigated
            b = spinup_season()
            add(year, "plnt", *b.plant, b.crop)
            for prod, kg in SPINUP_FERT:
                add(year, "fert", *b.plant, prod, "broadcast", kg)
            add(year, "hvkl", *b.harvest, b.crop, b.harv_op)
            auto.append((year, *spinup_window(year)))
        if k >= 1:
            s = season_of(plan, k)
            j = stand_year(plan.crops, k)
            if s.crop == "alfa":
                if j == 1:
                    add(year, "plnt", *s.plant, "alfa")
                for i, cut in enumerate(s.cuts):
                    last = (j == ALFA_STAND and i == len(s.cuts) - 1)
                    add(year, "hvkl" if last else "harv", *cut, "alfa", s.harv_op)
            else:
                add(year, "plnt", *s.plant, s.crop)
                add(year, "hvkl", *s.harvest, s.crop, s.harv_op)
            # synthetic fertiliser
            n_plan = plan.n_kg[k - 1]
            for mon, day, prod, kg in s.fert:
                if prod != N_PRODUCT or n_plan is None:
                    add(year, "fert", mon, day, prod, "broadcast", kg)
            if n_plan is not None:
                for mon, kg in zip(n_months(s), n_plan):
                    if kg > 1e-9:
                        add(year, "fert", mon, 15, N_PRODUCT, "broadcast", float(kg))
            # irrigation
            i_plan = plan.irr_mm[k - 1]
            if plan.irr_auto:
                auto += [(year, a, b) for a, b in s.irr_windows]
            elif i_plan is None and s.irr_daily is not None:
                for mon, day, mm in s.irr_daily:
                    irrigate(year, mon, day, mm, MEASURED_IRR_EFF)
            else:
                months = irr_months(s)
                amounts = dict(s.irr_monthly) if i_plan is None else dict(zip(months, i_plan))
                first_ev = min([date(year, *s.plant)] + [date(year, m, d) for m, d, _ in
                                                         (s.irr_daily or ())])
                first = min(first_ev, date(year, months[0], 1))
                last = date(year, *s.harvest) - timedelta(days=1)
                for mon in months:
                    for day, mm in _passes(year, mon, amounts.get(mon, 0.0), first, last):
                        irrigate(year, mon, day, mm, plan.irr_eff)
        # fall manure + incorporation (none into standing alfalfa)
        if k < SEASONS:
            standing = k >= 1 and plan.crops[k - 1] == "alfa" and \
                stand_year(plan.crops, k) < ALFA_STAND
            if not standing:
                name, _, _ = manure(k)
                kg = manure_kg(k, plan.manure_x[k])
                if kg > 0:
                    add(year, "fert", *MANURE_DAY, name, "broadcast", kg)
                add(year, "till", *MANURE_DAY, TILL)
    # probes: decision points, plus the first and last day of the scored seasons
    for d in plan.probes:
        add(start + d.k, "fert", d.mon, d.day, PROBE, "broadcast", 0.0)
    add(start + 1, "fert", 1, 2, PROBE, "broadcast", 0.0)
    add(start + SEASONS, "fert", 12, 30, PROBE, "broadcast", 0.0)

    ops.sort(key=lambda o: o[:4])
    lines: list[str] = []
    for year in range(start, start + SEASONS + 1):
        lines += [o[4] for o in ops if o[0] == year]
        lines.append(_line("skip", 12, 31, "null"))
    return lines, irr_rows, auto


def manure_applied(plan: EpisodePlan) -> tuple[float, float]:
    """(kg DM, kg N) of manure the plan applies over the 8 scored falls' seasons (falls 0..7)."""
    dm = n = 0.0
    for k in range(SEASONS):
        standing = k >= 1 and plan.crops[k - 1] == "alfa" and \
            stand_year(plan.crops, k) < ALFA_STAND
        if standing:
            continue
        kg = manure_kg(k, plan.manure_x[k])
        if kg > 0:
            _, kg1, n1 = manure(k)
            dm += kg
            n += n1 * kg / kg1
    return dm, n

