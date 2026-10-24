"""Monoculture crop models (``model/crops/<name>``): levers, rendering, scoring, runner.

An episode is one weather window: a spin-up year, then the scored season (print.prt skips
the spin-up year). Levers replace one of the model's own decision tables with a scheduled
monthly plan; everything else keeps the model's own management.

  I: one value per month of the irrigation windows = depth / (usable days x pass mm),
     rendered as equal passes of <= ``spec.irr_mm`` on distinct window days (x = 1 is the
     most the windows hold, so every x is deliverable). The irrigation table is removed;
     ``auto_fert_n`` keeps responding to N stress.
  N: one value per month whose 15th lies in the season -> kg N up to N_MONTH_MAX through the
     dead zone of plan.n_amount (x <= 0.05 applies nothing), as elemental N
     (the decision table's product) on the 15th. ``auto_fert_n`` is removed; the
     irrigation table keeps running. Annual crops only.

The spin-up year of a lever plan replays ``default_x`` (the decision tables' monthly
average on train windows), since the removed table is gone from the whole schedule.

Runs go through ``CropRunner``: K plans as K 1-ha HRUs of one engine run when the spec is
batchable (exact; see tests/test_batch.py), else K single-HRU runs. Outputs are read per
HRU: yield ``hru_pw_yr.yield``, water ``hru_wb_yr.irr``, N/P ``hru_nb_yr.fertn/fertp``,
leaching ``hru_ls_yr.percn`` (== the shallow aquifer's no3_rchg for one HRU).
"""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .fastrunner import PROJECT_ROOT, FastRunner
from .plan import n_amount, n_unit
from .rewarders import Prices, ipcc_n2o
from .schedule import _op
from .spec import CropSpec, check_windows
from .windows import CROP_WINDOW_LEN

#: Bump when x -> schedule changes meaning, per lever; part of stage configs and the
#: default_x cache. N 2: the nitrogen dead zone (plan.N_DEAD_ZONE, 2026-10-24).
CROP_RENDER_VERSION = {"I": 1, "N": 2}
N_MONTH_MAX = 150.0           #: kg N/ha per month (lever N range), as the rotation's
N_PRODUCT = "elem_n"          #: the decision table's product (100 % mineral N)
YIELD_KG_PER_MG = 1000.0

#: Tables the crop path reads, per HRU.
CROP_OUTPUTS: dict[str, set[str]] = {
    "hru_wb": {"monthly", "yearly"},   # irr, sw_final, precip, pet
    "hru_pw": {"monthly", "yearly"},   # yield, strsn, strsw
    "hru_nb": {"monthly", "yearly"},   # fertn, fertp
    "hru_ls": {"yearly"},              # percn
}
TABLES = ("hru_wb_mon.txt", "hru_wb_yr.txt", "hru_pw_mon.txt", "hru_pw_yr.txt",
          "hru_nb_mon.txt", "hru_nb_yr.txt", "hru_ls_yr.txt")
EDITABLE = frozenset({"management.sch", "irr.ops", "time.sim"})
CACHE = PROJECT_ROOT / "runs" / "cache"


def _md(doy: int) -> tuple[int, int]:
    d = date(2015, 1, 1) + timedelta(days=int(doy) - 1)
    return d.month, d.day


def _doy(mon: int, day: int) -> int:
    return (date(2015, mon, day) - date(2015, 1, 1)).days + 1


# -- lever geometry -------------------------------------------------------------------

@dataclass(frozen=True)
class Month:
    mon: int
    days: tuple[int, ...]       #: lever I: usable window days; lever N: (the 15th,)


def lever_months(spec: CropSpec, lever: str) -> tuple[Month, ...]:
    """The lever's decision months, in calendar order (one x entry and one env step each)."""
    if lever == "I":
        window_days = sorted({d for a, b in spec.irrigation_windows for d in range(a, b + 1)
                              if d < spec.season_end})
        by_mon: dict[int, list[int]] = {}
        for d in window_days:
            by_mon.setdefault(_md(d)[0], []).append(d)
        return tuple(Month(m, tuple(ds)) for m, ds in sorted(by_mon.items()))
    if lever == "N" and "N" in spec.levers:
        start = _doy(*spec.emerge)
        return tuple(Month(m, (_doy(m, 15),)) for m in range(1, 13)
                     if start <= _doy(m, 15) < spec.season_end)
    raise ValueError(f"{spec.name} has no lever {lever!r} (has {spec.levers})")


def dims(spec: CropSpec, lever: str) -> int:
    return len(lever_months(spec, lever))


def to_amount(spec: CropSpec, lever: str, x) -> np.ndarray:
    """Unit-box plan -> mm (I, linear) or kg N (N, with the dead zone) per month."""
    if lever == "N":
        return n_amount(x, N_MONTH_MAX)
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0) * month_cap(spec, lever)


def to_unit(spec: CropSpec, lever: str, amounts) -> np.ndarray:
    """Inverse of to_amount (amounts beyond the cap clip to 1)."""
    if lever == "N":
        return n_unit(amounts, N_MONTH_MAX)
    return np.clip(np.asarray(amounts, dtype=float) / month_cap(spec, lever), 0.0, 1.0)


def month_cap(spec: CropSpec, lever: str) -> np.ndarray:
    """Physical amount at x = 1 per month: mm (lever I) or kg N (lever N)."""
    if lever == "I":
        return np.array([len(m.days) * spec.irr_mm for m in lever_months(spec, lever)])
    return np.full(dims(spec, lever), N_MONTH_MAX)


def lever_ops(spec: CropSpec, lever: str, x, tag: str) -> tuple[list, list[str], float]:
    """x -> ([(doy, op line)], irr.ops lines, scheduled kg N) for one year."""
    amounts = to_amount(spec, lever, x)
    ops, irr, kg_n = [], [], 0.0
    for m, amt in zip(lever_months(spec, lever), amounts):
        if amt <= 1e-9:
            continue
        if lever == "N":
            doy = m.days[0]
            ops.append((doy, _op("fert", *_md(doy), N_PRODUCT, "broadcast", float(amt))))
            kg_n += float(amt)
            continue
        n = min(len(m.days), int(np.ceil(amt / spec.irr_mm - 1e-9)))
        per = amt / n
        for k in range(n):
            doy = m.days[int(k * len(m.days) / n)]
            name = f"{tag}d{doy:03d}"
            ops.append((doy, _op("irrm", *_md(doy), name)))
            irr.append(f"{name:<22}{per:>10.5f}{spec.irr_eff:>14.5f}{0.0:>14.5f}"
                       f"{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}  \n")
    return ops, irr, kg_n


# -- plans ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """One HRU's management for one window. ``lever=None``: the model's own (default)."""
    lever: str | None = None
    x: tuple[float, ...] = ()
    spinup_x: tuple[float, ...] = ()


@dataclass(frozen=True)
class Rendered:
    autos: list[str]
    lines: list[str]            #: op lines for both years, each ending in a skip
    irr: list[str]              #: irr.ops entries the irrm ops name
    scored_n_kg: float          #: scheduled mineral N in the scored season
    scored_n_events: int        #: scheduled N passes in the scored season


def render(spec: CropSpec, plan: Plan, tag: str) -> Rendered:
    """Plan -> one HRU's two-year schedule (spin-up year, scored year)."""
    removed = {"I": spec.auto_irrigation, "N": spec.auto_fert_n}.get(plan.lever)
    autos = [a for a in spec.autos if a != removed]
    lines, irr, scored_n, scored_events = [], [], 0.0, 0
    for year, xs in ((0, plan.spinup_x), (1, plan.x)):
        ops = [] if (spec.perennial and year) else [(d, line.rstrip("\n") + "\n")
                                                     for d, line in spec.template]
        if plan.lever is not None:
            lo, li, kg = lever_ops(spec, plan.lever, xs, f"{tag}y{year}")
            ops += lo
            irr += li
            if year:
                scored_n = kg
                scored_events = sum(1 for _, line in lo if line.split()[0] == "fert")
        # Template ops first on a shared day (plant before the lever's first pass).
        lines += [line for _, line in sorted(ops, key=lambda t: t[0])]
        lines.append(_op("skip", 12, 31, "null"))
    return Rendered(autos, lines, irr, scored_n, scored_events)


# -- runner ---------------------------------------------------------------------------

def _rows(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _write_batched_inputs(spec: CropSpec, dest: Path, k: int) -> None:
    """Copy spec's TxtInOut to ``dest`` with k 1-ha HRUs (identical soils/routing).

    HRU i uses landuse ``lum_k<i>`` -> schedule ``<spec.schedule>_<i>``. Every HRU drains to
    the same routing unit and aquifer at 1/k of it; per-HRU outputs are exact for annual
    crops (tests/test_batch.py), which is why only per-HRU tables are ever read.
    """
    src = spec.txtinout
    shutil.copytree(src, dest)

    def put(name, lines):
        (dest / name).write_text("\n".join(lines) + "\n")

    lum = _rows(src / "landuse.lum")
    row = lum[2].split()
    put("landuse.lum", lum[:2] + ["  ".join([f"lum_k{i}", row[1], row[2],
                                             f"{spec.schedule}_{i}"] + row[4:])
                                  for i in range(k)])
    hd = _rows(src / "hru-data.hru")
    r = hd[2].split()
    put("hru-data.hru", hd[:2] + ["  ".join([str(i + 1), f"hru{i + 1}"] + r[2:5] + [f"lum_k{i}"]
                                            + r[6:]) for i in range(k)])
    hc = _rows(src / "hru.con")
    r = hc[2].split()
    put("hru.con", hc[:2] + ["  ".join([str(i + 1), f"hru{i + 1}", str(i + 1)] + r[3:7]
                                       + [str(i + 1)] + r[8:]) for i in range(k)])
    for name in ("rout_unit.ele", "ls_unit.ele"):
        lines = _rows(src / name)
        r = lines[2].split()
        rows = []
        for i in range(k):
            rr = list(r)
            rr[0], rr[1], rr[3], rr[4] = str(i + 1), f"hru{i + 1}", str(i + 1), f"{1.0 / k:.6E}"
            rows.append("  ".join(rr))
        put(name, lines[:2] + rows)
    lines = _rows(src / "rout_unit.def")
    r = lines[2].split()
    put("rout_unit.def", lines[:2] + ["  ".join([r[0], r[1], "2", "1", f"-{k}"])])
    lines = _rows(src / "ls_unit.def")
    r = lines[3].split()
    put("ls_unit.def", lines[:3] + ["  ".join([r[0], r[1], f"{k:.5f}", "2", "1", f"-{k}"])])
    for name in ("rout_unit.con", "aquifer.con"):
        lines = _rows(src / name)
        out = lines[:2]
        for line in lines[2:]:
            p = line.split()
            p[3] = f"{k:.5f}"
            out.append("  ".join(p))
        put(name, out)
    lines = _rows(src / "object.cnt")
    p = lines[2].split()
    p[1], p[2], p[3], p[4] = f"{k:.5f}", f"{k:.5f}", str(int(p[3]) - 1 + k), str(k)
    put("object.cnt", lines[:2] + ["  ".join(p)])
    lines = _rows(src / "calibration.cal")
    out = lines[:3]
    for line in lines[3:]:
        p = line.split()
        if len(p) >= 11 and p[10] == "1" and p[-1] == "1":  # applies to object 1 only
            p = p[:10] + [str(k)] + [str(i + 1) for i in range(k)]
            line = "  ".join(p)
        out.append(line)
    put("calibration.cal", out)


def time_sim(start_year: int) -> str:
    return ("time.sim: written by swat_gym.crop\n"
            "day_start  yrc_start   day_end   yrc_end      step  \n"
            f"{0:>10}{start_year:>11}{0:>10}{start_year + CROP_WINDOW_LEN - 1:>10}{0:>10}  \n")


class CropRunner:
    """Runs up to ``k`` plans per call on one window: one k-HRU engine run if the spec is
    batchable, else one single-HRU run per plan. Not thread/process safe."""

    def __init__(self, spec: CropSpec, k: int = 1) -> None:
        self.spec, self.k = spec, int(k)
        self.batched = self.k > 1 and spec.batchable
        self._tmp = None
        if self.batched:
            self._tmp = tempfile.TemporaryDirectory(prefix=f"swat_gym_{spec.name}_k{k}_")
            src = Path(self._tmp.name) / "TxtInOut"
            _write_batched_inputs(spec, src, self.k)
        else:
            src = spec.txtinout
        self.runner = FastRunner(src, keep=CROP_OUTPUTS, editable=EDITABLE)
        self._irr_base = (src / "irr.ops").read_text().rstrip("\n") + "\n"
        self._sch_head = "\n".join(_rows(src / "management.sch")[:2]) + "\n"
        #: Candidate-seasons simulated: the unit CMA-ES and PPO budgets are matched in.
        self.n_seasons = 0

    @property
    def n_runs(self) -> int:
        return self.runner.n_runs

    @property
    def n_skipped(self) -> int:
        return self.runner.n_skipped

    def _edits(self, plans: Sequence[Plan], start_year: int, names: Sequence[str]):
        blocks, irr, rendered = [], [], []
        for i, (plan, name) in enumerate(zip(plans, names)):
            r = render(self.spec, plan, f"h{i}")
            blocks.append(f"{name:<37}{len(r.lines):>7}{len(r.autos):>11}  \n"
                          + "".join(f"{'':48}{a}\n" for a in r.autos) + "".join(r.lines))
            irr += r.irr
            rendered.append(r)
        return ({"management.sch": self._sch_head + "".join(blocks),
                 "irr.ops": self._irr_base + "".join(irr),
                 "time.sim": time_sim(start_year)}, rendered)

    def run(self, plans: Sequence[Plan], start_year: int) -> list[dict[str, pd.DataFrame]]:
        """Simulate each plan on the window starting ``start_year``; per-plan output tables."""
        plans = list(plans)
        if not plans:
            return []
        if len(plans) > self.k:
            raise ValueError(f"{len(plans)} plans for a runner built for k={self.k}")
        check_windows(self.spec, [start_year], CROP_WINDOW_LEN)
        self.n_seasons += len(plans)
        if self.batched:
            # Unused HRUs replay the first plan: the HRU count is fixed by the inputs.
            padded = plans + [plans[0]] * (self.k - len(plans))
            names = [f"{self.spec.schedule}_{i}" for i in range(self.k)]
            edits, sched = self._edits(padded, start_year, names)
            self.runner.run(edits)
            by_unit = {t: dict(tuple(self.runner.read(t).groupby("unit"))) for t in TABLES}
            return [_per_hru(by_unit, i + 1, sched[i]) for i in range(len(plans))]
        out = []
        for plan in plans:
            edits, sched = self._edits([plan], start_year, [self.spec.schedule])
            self.runner.run(edits)
            by_unit = {t: dict(tuple(self.runner.read(t).groupby("unit"))) for t in TABLES}
            out.append(_per_hru(by_unit, 1, sched[0]))
        return out

    def close(self) -> None:
        self.runner.close()
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self) -> CropRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _per_hru(by_unit: dict[str, dict[int, pd.DataFrame]], unit: int, r: Rendered) -> dict:
    out = {}
    for name, parts in by_unit.items():
        if unit not in parts:
            from .engine import EngineError
            raise EngineError(f"{name} has no rows for HRU {unit}")
        out[name] = parts[unit].reset_index(drop=True)
    out["scheduled_n_kg"] = r.scored_n_kg
    out["scheduled_n_events"] = r.scored_n_events
    return out


# -- scoring --------------------------------------------------------------------------

def profit(spec: CropSpec, out: dict, prices: Prices, *, no3_price: float = 0.0) -> dict:
    """Scored-season profit ($/ha) of one HRU's outputs, with separable terms."""
    from .engine import EngineError

    yr = {k: out[k].iloc[0] for k in ("hru_wb_yr.txt", "hru_pw_yr.txt", "hru_nb_yr.txt",
                                      "hru_ls_yr.txt")}
    if spec.plant not in prices.crop:
        raise KeyError(f"no price for {spec.plant!r}; have {sorted(prices.crop)}")
    yld = float(yr["hru_pw_yr.txt"]["yield"]) / YIELD_KG_PER_MG
    if yld <= 0:
        raise EngineError(f"{spec.name}: zero yield; is the schedule wired to the HRU?")
    irr = float(yr["hru_wb_yr.txt"]["irr"])
    fertn = float(yr["hru_nb_yr.txt"]["fertn"])
    fertp = float(yr["hru_nb_yr.txt"]["fertp"])
    no3 = float(yr["hru_ls_yr.txt"]["percn"])
    sched_n = float(out["scheduled_n_kg"])
    # Decision-table passes from mass: every auto event applies exactly auto_*_kg.
    auto_n = (fertn - sched_n) / spec.auto_n_kg if spec.auto_n_kg else 0.0
    auto_p = fertp / spec.auto_p_kg if spec.auto_p_kg else 0.0
    for label, v in (("auto N", auto_n), ("auto P", auto_p)):
        if abs(v - round(v)) > 1e-3 or round(v) < 0:
            raise EngineError(f"{spec.name}: {label} events = {v:.4f}, not a whole number; "
                              "did a decision table's amount change?")
    n_events = int(round(auto_n)) + int(round(auto_p)) + int(out["scheduled_n_events"])
    revenue = yld * prices.crop[spec.plant]
    costs = {"water_cost": irr * prices.water, "fert_cost": fertn * prices.fert_n,
             "p_cost": fertp * prices.fert_p, "op_cost": n_events * prices.fert_op,
             "leach_cost": no3 * no3_price}
    return {"label": prices.label, "revenue": revenue, **costs,
            "profit": revenue - sum(costs.values()),
            "yield_mg": yld, "irrigation_mm": irr, "fert_n_kg": fertn, "fert_p_kg": fertp,
            "n_fert_events": n_events, "auto_n_events": int(round(auto_n)),
            "no3_leached_kg": no3,
            **ipcc_n2o(manure_n_kg=0.0, mineral_n_kg=fertn, no3_leached_kg=no3)}

