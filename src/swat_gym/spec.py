"""Model specs: what the gym needs to know about a SWAT+ model to optimise on it.

Two kinds of model plug in:

- the Kimberly **rotation** (paper Exp 1/2): ``plan.py`` + ``env.py``, unchanged;
- **monoculture crop** models (``model/crops/<name>/``): a 1-HRU SWAT+ TxtInOut plus a
  ``spec.json`` (written by ``scripts/copy_crop_models.py``), driven by ``swat_gym.crop``.

``load_crop`` checks everything that would otherwise fail silently (the engine exits 0 with
a mis-wired schedule, a missing decision table, or a window outside the weather record) and
raises with the fix. Adding a model = drop its TxtInOut and spec.json in model/crops/<name>/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .fastrunner import PROJECT_ROOT

CROPS_DIR = PROJECT_ROOT / "model" / "crops"


class SpecError(ValueError):
    """A model spec disagrees with its SWAT+ inputs."""


@dataclass(frozen=True)
class CropSpec:
    name: str
    txtinout: Path
    plant: str                      #: plants.plt name; yield is priced under this key
    perennial: bool
    emerge: tuple[int, int]         #: (month, day) planting/green-up
    harvests: tuple[tuple[int, int], ...]   #: annual: harvest-kill; perennial: cuttings
    irrigation_windows: tuple[tuple[int, int], ...]  #: (first doy, last doy), inclusive
    schedule: str                   #: management.sch name landuse.lum wires to the HRU
    autos: tuple[str, ...]          #: decision tables the shipped schedule runs
    auto_irrigation: str            #: the irrigation decision table (removed by lever I)
    auto_fert_n: str | None         #: the N decision table (removed by lever N), if run
    auto_n_kg: float                #: kg N per auto_fert_n event
    auto_p_kg: float                #: kg P per auto_fert_p event (0 if no P table)
    irr_mm: float                   #: mm per auto-irrigation event (also the lever's pass)
    irr_eff: float                  #: eff_frac of the auto-irrigation irr.ops entry
    template: tuple[tuple[int, str], ...]   #: (doy, op line) of one year's own ops
    weather_years: tuple[int, int]  #: first and last calendar year of the weather record
    batchable: bool                 #: several candidates may share one engine run as HRUs

    @property
    def levers(self) -> tuple[str, ...]:
        """I always; N for annual crops (alfalfa fixes its own N; its N table never fires)."""
        return ("I",) if self.perennial or not self.auto_fert_n else ("I", "N")

    @property
    def season_end(self) -> int:
        """Day of year the season ends (harvest-kill, or the last cutting)."""
        return _doy(*self.harvests[-1])


def _doy(mon: int, day: int) -> int:
    return (date(2015, mon, day) - date(2015, 1, 1)).days + 1


def crop_names() -> list[str]:
    return sorted(p.name for p in CROPS_DIR.iterdir() if (p / "spec.json").is_file())


def _rows(path: Path, header_line: int = 1) -> list[list[str]]:
    lines = path.read_text().splitlines()
    return [line.split() for line in lines[header_line + 1:] if line.split()]


def _dtl_blocks(text: str) -> dict[str, list[str]]:
    """lum.dtl decision tables by name -> their lines."""
    blocks, name, cur = {}, None, []
    for line in text.splitlines()[2:]:
        t = line.split()
        if not t:
            if name:
                blocks[name] = cur
            name, cur = None, []
            continue
        if name is None and len(t) >= 4 and t[1].isdigit():
            name, cur = t[0], []
            continue
        if name:
            cur.append(line)
    if name:
        blocks[name] = cur
    return blocks


def _dtl_const(block: list[str], action: str) -> float:
    """``const`` of the first ``action`` row of a decision table (e.g. kg per fertilize)."""
    for line in block:
        t = line.split()
        if t and t[0] == action:
            return float(t[5])
    raise SpecError(f"decision table has no {action!r} action")


def _schedule(text: str, name: str) -> tuple[list[str], list[tuple[int, str]]]:
    """(autos, first year's ops as (doy, line)) of schedule ``name`` in management.sch."""
    lines = text.splitlines()
    for i, line in enumerate(lines[2:], start=2):
        t = line.split()
        if t and t[0] == name and len(t) == 3:
            n_ops, n_auto = int(t[1]), int(t[2])
            autos = [lines[i + 1 + k].split()[0] for k in range(n_auto)]
            ops = []
            for op in lines[i + 1 + n_auto:i + 1 + n_auto + n_ops]:
                p = op.split()
                if p[0] == "skip":
                    break
                ops.append((_doy(int(p[1]), int(p[2])), op))
            return autos, ops
    raise SpecError(f"management.sch has no schedule named {name!r}")


def _weather_years(tio: Path) -> tuple[int, int]:
    """Calendar years every weather series covers (the engine silently generates the rest)."""
    spans = []
    for cli in ("pcp.cli", "tmp.cli", "slr.cli", "hmd.cli", "wnd.cli"):
        for t in _rows(tio / cli, 1):
            series = (tio / t[0]).read_text().splitlines()
            nbyr = int(series[2].split()[0])
            first = int(series[3].split()[0])
            last = int(series[-1].split()[0])
            if last != first + nbyr - 1:
                raise SpecError(f"{t[0]}: header says {nbyr} years from {first}, data end {last}")
            spans.append((first, last))
    return max(s[0] for s in spans), min(s[1] for s in spans)


def load_crop(name: str, root: Path = CROPS_DIR) -> CropSpec:
    """Load and validate ``root/<name>/spec.json`` against its TxtInOut."""
    d = root / name
    raw = json.loads((d / "spec.json").read_text())
    tio = d / "TxtInOut"
    where = f"model spec {name!r} ({d})"

    cnt = _rows(tio / "object.cnt", 1)[0]
    if int(cnt[4]) != 1:
        raise SpecError(f"{where}: crop models must have exactly 1 HRU, object.cnt has {cnt[4]}")
    lum = _rows(tio / "landuse.lum", 1)
    hru = _rows(tio / "hru-data.hru", 1)
    if len(lum) != 1 or len(hru) != 1 or hru[0][5] != lum[0][0]:
        raise SpecError(f"{where}: expected one landuse.lum row used by the one HRU")
    schedule = lum[0][3]

    autos, template = _schedule((tio / "management.sch").read_text(), schedule)
    dtl = _dtl_blocks((tio / "lum.dtl").read_text())
    missing = [a for a in autos if a not in dtl]
    if missing:
        raise SpecError(f"{where}: schedule {schedule} runs {missing}, absent from lum.dtl")
    for key in ("auto_irrigation", "auto_fert_n"):
        if raw.get(key) and raw[key] not in autos:
            raise SpecError(f"{where}: {key}={raw[key]!r} is not one of the schedule's {autos}")

    plants = {r[0] for r in _rows(tio / "plants.plt", 1)}
    if raw["plant"] not in plants:
        raise SpecError(f"{where}: plant {raw['plant']!r} not in plants.plt")
    emerge = tuple(raw["emerge"])
    harvests = tuple(tuple(h) for h in raw["harvests"])
    ops = {(p.split()[0], _doy(int(p.split()[1]), int(p.split()[2]))) for _, p in template}
    if ("plnt", _doy(*emerge)) not in ops:
        raise SpecError(f"{where}: spec emerge {emerge} has no plnt op in {schedule}")
    if not raw["perennial"] and ("hvkl", _doy(*harvests[0])) not in ops:
        raise SpecError(f"{where}: spec harvest {harvests[0]} has no hvkl op in {schedule}")

    n_auto = raw["auto_fert_n"] if (raw["auto_fert_n"] in autos) else None
    auto_n = _dtl_const(dtl[n_auto], "fertilize") if n_auto else 0.0
    auto_p = _dtl_const(dtl["auto_fert_p"], "fertilize") if "auto_fert_p" in autos else 0.0
    irr_tab = dtl[raw["auto_irrigation"]]
    irr_line = next(line.split() for line in irr_tab if line.split()[0] == "irr_demand")
    irr_op = irr_line[4]
    irr_rows = {r[0]: r for r in _rows(tio / "irr.ops", 1)}
    if irr_op not in irr_rows:
        raise SpecError(f"{where}: irrigation op {irr_op!r} not in irr.ops")
    irr_mm = float(irr_line[5])
    if abs(irr_mm - float(raw["auto_irr_mm"])) > 1e-9:
        raise SpecError(f"{where}: spec auto_irr_mm {raw['auto_irr_mm']} != lum.dtl {irr_mm}")

    nyskip = int(_rows(tio / "print.prt", 1)[0][0])
    if nyskip != 1:
        raise SpecError(f"{where}: print.prt nyskip={nyskip}; the spin-up year must be skipped")

    windows = tuple(tuple(int(v) for v in w) for w in raw["irrigation_windows_doy"])
    if any(not (1 <= a <= b <= 365) for a, b in windows):
        raise SpecError(f"{where}: bad irrigation windows {windows}")

    return CropSpec(
        name=name, txtinout=tio, plant=raw["plant"], perennial=bool(raw["perennial"]),
        emerge=emerge, harvests=harvests, irrigation_windows=windows, schedule=schedule,
        autos=tuple(autos), auto_irrigation=raw["auto_irrigation"], auto_fert_n=n_auto,
        auto_n_kg=auto_n, auto_p_kg=auto_p, irr_mm=irr_mm, irr_eff=float(irr_rows[irr_op][2]),
        template=tuple(template), weather_years=_weather_years(tio),
        batchable=bool(raw.get("batchable", False)),
    )


def check_windows(spec: CropSpec, starts, length: int) -> None:
    """Raise unless every window lies inside the weather record (outside it SWAT+ silently
    runs on generated weather and exits 0)."""
    lo, hi = spec.weather_years
    bad = [s for s in starts if s < lo or s + length - 1 > hi]
    if bad:
        raise SpecError(f"{spec.name}: windows {bad} fall outside the weather record {lo}-{hi}")

