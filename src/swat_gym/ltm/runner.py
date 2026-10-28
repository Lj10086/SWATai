"""Run K episode plans on one window: K 1-ha HRUs of one engine run, or K single-HRU runs.

Batching is exact for annual crops (the 3-treatment check in PROVENANCE LTM; tests/
test_ltm.py) but not where alfalfa is grown (SWAT+ rev 62 keeps N/P demand per plant, not per
HRU; test_batch.py). ``batched="mixed"`` runs a call's annual-only plans as one k-HRU run
(padded to k with copies) and only its alfalfa plans as single-HRU runs (Exp 1).

Budget unit: HRU-episodes simulated (``n_hru_runs``), the unit ES and CMA-ES are matched in.
"""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..fastrunner import FastRunner
from ..manifest import input_files
from ..printprt import trim
from .practice import AUTO_IRR, TXTINOUT, auto_op
from .render import EpisodePlan, render
from .windows import WINDOW_LEN

EDITABLE = frozenset({"management.sch", "irr.ops", "time.sim", "lum.dtl"})
#: Only what env (observations) and score (episode) read; yearly pw/nb were never used.
KEEP = {"hru_wb": {"monthly", "yearly"}, "hru_pw": {"monthly"}, "hru_nb": {"monthly"},
        "hru_ls": {"yearly"}}
TABLES = ("hru_wb_mon.txt", "hru_wb_yr.txt", "hru_pw_mon.txt", "hru_nb_mon.txt", "hru_ls_yr.txt")
MGT_COLS = ["phubase", "phuplant", "soil_water", "plant_bioms", "surf_rsd", "soil_no3",
            "soil_solp", "op_var"]


def _rows(p: Path) -> list[str]:
    return p.read_text().splitlines()


def _put(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines) + "\n")


def make_dir(dest: Path, k: int, src: Path = TXTINOUT) -> None:
    """``src`` (the 3-treatment LTM model) rewired as k 1-ha HRUs ``hru<i>`` -> ``ltm_k<i>``."""
    dest.mkdir(parents=True)
    for name in input_files(src) | {p.name for p in src.iterdir()
                                    if p.name.startswith("swatplus_") and not p.name.endswith(".bak")}:
        shutil.copy2(src / name, dest / name)
    lum = _rows(dest / "landuse.lum")
    r = lum[2].split()
    _put(dest / "landuse.lum", lum[:2] + ["  ".join([f"lum_k{i}", r[1], r[2], f"ltm_k{i}"] + r[4:])
                                          for i in range(k)])
    hd = _rows(dest / "hru-data.hru")
    r = hd[2].split()
    _put(dest / "hru-data.hru", hd[:2] + ["  ".join([str(i + 1), f"hru{i + 1}"] + r[2:5]
                                                    + [f"lum_k{i}"] + r[6:]) for i in range(k)])
    hc = _rows(dest / "hru.con")
    r = hc[2].split()
    _put(dest / "hru.con", hc[:2] + ["  ".join([str(i + 1), f"hru{i + 1}", str(i + 1)] + r[3:7]
                                               + [str(i + 1)] + r[8:]) for i in range(k)])
    for name in ("rout_unit.ele", "ls_unit.ele"):
        lines = _rows(dest / name)
        r = lines[2].split()
        out = []
        for i in range(k):
            rr = list(r)
            rr[0], rr[1], rr[3], rr[4] = str(i + 1), f"hru{i + 1}", str(i + 1), f"{1.0 / k:.6E}"
            out.append("  ".join(rr))
        _put(dest / name, lines[:2] + out)
    lines = _rows(dest / "rout_unit.def")
    r = lines[2].split()
    _put(dest / "rout_unit.def", lines[:2] + ["  ".join([r[0], r[1], "2", "1", f"-{k}"])])
    lines = _rows(dest / "ls_unit.def")
    r = lines[3].split()
    _put(dest / "ls_unit.def", lines[:3] + ["  ".join([r[0], r[1], f"{k:.5f}", "2", "1", f"-{k}"])])
    for name in ("rout_unit.con", "aquifer.con"):
        lines = _rows(dest / name)
        out = lines[:2]
        for line in lines[2:]:
            p = line.split()
            p[3] = f"{k:.5f}"
            out.append("  ".join(p))
        _put(dest / name, out)
    lines = _rows(dest / "object.cnt")
    p = lines[2].split()
    k0 = int(float(p[4]))
    p[1], p[2], p[3], p[4] = f"{k:.5f}", f"{k:.5f}", str(int(p[3]) - k0 + k), str(k)
    _put(dest / "object.cnt", lines[:2] + ["  ".join(p)])
    cal = _rows(dest / "calibration.cal")
    out = cal[:3]
    for line in cal[3:]:
        q = line.split()
        if q[:1] == ["phu_mat"]:
            q = q[:10] + [str(k)] + [str(i + 1) for i in range(k)]
            line = " ".join(q)
        out.append(line)
    _put(dest / "calibration.cal", out)
    pp = dest / "print.prt"
    pp.write_text(trim(pp.read_text(), KEEP, mgtout=True))


def _dtl_row(var, obj, obj_num, lim_var, lim_op, const, alts) -> str:
    """One condition row in lum.dtl's fixed layout (Ai-SWAT-Plus monoculture._dtl_row)."""
    return (f"{var:<27}{obj:>3}{obj_num:>10}{lim_var:>18}{lim_op:>18}{const:>14.5f}"
            + "".join(f"{a:>10}" for a in alts) + "  \n")


def irr_table(name: str, windows: list[tuple[int, int, int]], op: str) -> str:
    """Soil-water-triggered irrigation, one alternative per (year, first doy, last doy).

    Conditioned on calendar year + day of year, never on the plant: rev 62 resolves dtl
    plant conditions to the community's first plant (memory: dtl-plant-conditions)."""
    if AUTO_IRR["sw_frac"] is None:
        raise RuntimeError("auto-irrigation trigger not fitted: run scripts/ltm_fit_auto_irr.py")
    return _irr_table(name, windows, AUTO_IRR["sw_frac"], op)


def _irr_table(name: str, windows: list[tuple[int, int, int]], sw_frac: float, op: str) -> str:
    n = len(windows)
    # "*": target = sw_frac x the profile's field-capacity water ("-" would be FC - sw_frac mm)
    body = _dtl_row("soil_water", "hru", 0, "fc", "*", sw_frac, ["<"] * n)
    for i, (year, a, b) in enumerate(windows):
        for var, const, cmp in (("year_cal", year, "="), ("jday", a - 1, ">"), ("jday", b + 1, "<")):
            alts = ["-"] * n
            alts[i] = cmp
            body += _dtl_row(var, "hru", 0, "null", "-", float(const), alts)
    return (f"\nname                     conds      alts      acts       !{name}: LTM auto-irrigation\n"
            f"{name:<22}{1 + 3 * n:>8}{n:>10}{1:>10}  \n"
            f"{'var':<22}{'obj':>8}{'obj_num':>10}{'lim_var':>18}{'lim_op':>18}{'lim_const':>14}"
            + "".join(f"{f'alt{i + 1}':>10}" for i in range(n)) + "  \n" + body
            + f"{'act_typ':<22}{'obj':>8}{'obj_num':>10}{'name':>18}{'option':>18}{'const':>14}"
              f"{'const2':>14}{'fp':>18}  outcome\n"
            + f"{'irr_demand':<27}{'hru':>3}{0:>10}{'sprinkler':>18}{op:>18}"
              f"{AUTO_IRR['mm']:>14.5f}{20.0:>14.5f}{'unlim':>18}  " + "".join(f"{'y':<4}" for _ in range(n))
            + "\n")


def time_sim(start: int, last: int | None = None) -> str:
    """Simulate Jan 1 ``start`` .. Dec 31 ``last`` (default: the whole window)."""
    last = start + WINDOW_LEN - 1 if last is None else last
    return ("time.sim: written by swat_gym.ltm\n"
            "day_start  yrc_start   day_end   yrc_end      step  \n"
            f"{0:>10}{start:>11}{0:>10}{last:>10}{0:>10}  \n")


def read_table(path: Path) -> pd.DataFrame:
    """A SWAT+ output table, as FastRunner.read but parsed in one pass: the units row (if
    any) is skipped before parsing, so numeric columns come out numeric directly."""
    with open(path) as f:
        f.readline()
        f.readline()
        third = f.readline().split()
    units = bool(third) and not _is_number(third[0])
    return pd.read_csv(path, sep=r"\s+", skiprows=[0, 2] if units else [0], header=0,
                       engine="c", float_precision="round_trip")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_mgt(path: Path) -> pd.DataFrame:
    recs = []
    for ln in path.read_text().splitlines()[3:]:
        t = ln.split()
        if len(t) < 6 + len(MGT_COLS):
            continue
        recs.append([int(t[0]), int(t[1]), int(t[2]), int(t[3]), t[4], t[5]]
                    + [float(x) for x in t[6:6 + len(MGT_COLS)]])
    return pd.DataFrame(recs, columns=["hru", "yr", "mon", "day", "name", "operation", *MGT_COLS])


class LtmRunner:
    """Not thread/process safe; one per worker."""

    def __init__(self, k: int, batched: bool | str = True, cache: int = 0) -> None:
        """``cache``: remember the outputs of up to this many (plan, window, last year) runs
        and replay them instead of re-running (SWAT+ is deterministic; Exp 1's discrete plans
        repeat constantly). Budget accounting (``n_hru_runs``) counts replays too."""
        self.k = int(k)
        self.cache_size = int(cache)
        self._cache: dict = {}
        self.n_cached = 0
        self.mixed = batched == "mixed"
        self.batched = bool(batched) and not self.mixed
        self._tmp = tempfile.TemporaryDirectory(prefix=f"swat_gym_ltm_k{k}_")
        root = Path(self._tmp.name)
        self.batch_runner = None
        if self.batched or self.mixed:
            make_dir(root / "T", self.k)
            self.batch_runner = FastRunner(root / "T", trim_outputs=False, editable=EDITABLE)
        if not self.batched:
            make_dir(root / "S", 1)
            self.singles = [FastRunner(root / "S", trim_outputs=False, editable=EDITABLE)
                            for _ in range(self.k)]
        else:
            self.singles = []
        self.runners = ([self.batch_runner] if self.batch_runner else []) + self.singles
        tpl = root / ("T" if self.batch_runner else "S")
        self.head = "\n".join(_rows(tpl / "management.sch")[:2]) + "\n"
        self.irr_base = "\n".join(ln for ln in _rows(tpl / "irr.ops")
                                  if not ln.split()[:1] or not ln.split()[0].startswith(("e", "ltm_auto")))
        self.dtl_base = (tpl / "lum.dtl").read_text()
        self._pool = ThreadPoolExecutor(self.k) if self.singles and self.k > 1 else None
        self.n_hru_runs = 0

    def _edits(self, plans: Sequence[EpisodePlan], start: int, offset: int = 0,
               last: int | None = None) -> dict:
        sch, irr, tables, effs = [self.head], [self.irr_base], [], set()
        for i, plan in enumerate(plans):
            lines, irr_rows, auto = render(plan, start, f"e{i + offset:02d}")
            autos = []
            if auto:
                name = f"ltm_irr_k{i}"
                tables.append(irr_table(name, auto, auto_op(plan.irr_eff)))
                effs.add(plan.irr_eff)
                autos = [f"{'':>48}{name:<18}"]
            sch.append(f"{f'ltm_k{i}':<34}{len(lines):>6}{len(autos):>11}  \n"
                       + "".join(a + "\n" for a in autos) + "\n".join(lines) + "\n")
            irr += irr_rows
        dtl = self.dtl_base
        if tables:
            for eff in sorted(effs):
                irr.append(f"{auto_op(eff):<22}{AUTO_IRR['mm']:>8.5f}{eff:>14.5f}"
                           + f"{0.0:>14.5f}" * 5 + "  ltm_auto_irrigation_pass")
            head, count, rest = dtl.split("\n", 2)
            dtl = f"{head}\n{int(count.strip()) + len(tables)}\n{rest}".rstrip("\n") + "\n" + "".join(tables)
        return {"management.sch": "".join(sch), "irr.ops": "\n".join(irr) + "\n",
                "time.sim": time_sim(start, last), "lum.dtl": dtl}

    @staticmethod
    def _read(fr: FastRunner) -> dict:
        out = {t: read_table(fr.workdir / t) for t in TABLES}
        out["mgt"] = read_mgt(fr.workdir / "mgt_out.txt")
        return out

    def run(self, plans: Sequence[EpisodePlan], start: int,
            last: int | None = None) -> list[dict]:
        """Per-plan outputs (each table filtered to its HRU, unit/hru renumbered to 1), for the
        window from ``start`` simulated through Dec 31 of ``last`` (default: all of it)."""
        plans = list(plans)
        if len(plans) > self.k:
            raise ValueError(f"{len(plans)} plans for a runner of k={self.k}")
        self.n_hru_runs += len(plans)
        if not self.cache_size:
            return self._run(plans, start, last)
        keys = [(p, start, last) for p in plans]
        # answer from a local map: evicting below may drop a key this call still needs
        got = {k: self._cache[k] for k in keys if k in self._cache}
        todo = list(dict.fromkeys(k for k in keys if k not in got))
        if todo:
            for k, o in zip(todo, self._run([k[0] for k in todo], start, last)):
                got[k] = o
                if len(self._cache) >= self.cache_size:
                    self._cache.pop(next(iter(self._cache)))      # oldest first
                self._cache[k] = o
        self.n_cached += len(plans) - len(todo)
        return [got[k] for k in keys]

    def _run(self, plans: list, start: int, last: int | None) -> list[dict]:
        if self.batched:
            return self._batch(plans, start, last)
        single = range(len(plans)) if not self.mixed else \
            [i for i, p in enumerate(plans) if "alfa" in p.crops]
        annual = [i for i in range(len(plans)) if i not in set(single)]
        out: list = [None] * len(plans)
        if annual:
            for i, o in zip(annual, self._batch([plans[i] for i in annual], start, last)):
                out[i] = o

        def one(j, i):
            fr = self.singles[j]
            fr.run(self._edits([plans[i]], start, last=last))
            return i, self._read(fr)

        jobs = list(enumerate(single))
        res = map(lambda a: one(*a), jobs) if self._pool is None else \
            self._pool.map(lambda a: one(*a), jobs)
        for i, o in res:
            out[i] = o
        return out

    def _batch(self, plans: list, start: int, last: int | None = None) -> list[dict]:
        """One k-HRU run; fewer than k plans are padded with copies of the first."""
        n = len(plans)
        padded = plans + [plans[0]] * (self.k - n)
        fr = self.batch_runner
        fr.run(self._edits(padded, start, last=last))
        return self._split(self._read(fr), n)

    @staticmethod
    def _split(whole: dict, n: int) -> list[dict]:
        """Per-HRU tables 1..n (one groupby pass per table)."""
        out: list[dict] = [{} for _ in range(n)]
        for t, df in whole.items():
            col = "hru" if t == "mgt" else "unit"
            groups = {int(k): g for k, g in df.groupby(col, sort=False)}
            for i in range(n):
                g = groups.get(i + 1)
                out[i][t] = (g if g is not None else df.iloc[:0]).reset_index(drop=True)
        return out

    @property
    def n_runs(self) -> int:
        return sum(r.n_runs for r in self.runners)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
        for r in self.runners:
            r.close()
        self._tmp.cleanup()

    def __enter__(self) -> LtmRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
