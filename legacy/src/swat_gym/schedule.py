"""YearAction list -> management.sch + irr.ops edits.

SWAT+ advances a year when an op's date precedes the previous op's, so years are emitted in
date order and concatenated. Irrigation is parameterised (decision tables crash the engine).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

CROPS = ("corn", "barl", "alfa")

#: Per-crop calendar from the measured GRACEnet schedule. Alfalfa is multi-cut.
CALENDAR = {
    "corn": {"plant": (5, 16), "harvest": (9, 20)},
    "barl": {"plant": (4, 9), "harvest": (8, 15)},
    "alfa": {"plant": (4, 16), "cuts": [(5, 28), (7, 12), (9, 8)]},
}

#: harv.ops type per crop — the reference harvest-index overrides appended by port_management.
HARV_OPS = {"corn": "gn_corn", "barl": "gn_barl", "alfa": "gn_alfa"}

#: The four measured GRACEnet manure compositions, as the action's discrete `source` choice.
MANURE_SOURCES = ("gn2013", "gn2014", "gn2018", "gn2019")

#: Mineral product is fixed to urea (the only sourced price).
MINERAL_SOURCE = "urea"

#: min_n fraction per product, to convert kg N/ha -> kg product.
MINERAL_N_FRAC = {"urea": 0.46, "elem_n": 1.00, "46_00_00": 0.46, "anh_nh3": 0.82}

#: Tillage used to incorporate manure, per Bierer et al. 2022 §2.3 (disked to 15 cm).
INCORPORATION_TILL = "gn_disk"

#: irr.ops eff_frac for generated entries. Must be 1.0 to match the measured gn#### events.
IRR_EFF = 1.0

#: Growing-season months for monthly irrigation (April–September).
GROWING_MONTHS = (4, 5, 6, 7, 8, 9)


@dataclass
class YearAction:
    """One year's management decision."""

    crop: str = "corn"
    manure_mg: float = 0.0          #: Mg/ha as applied
    manure_doy: int = 100           #: day of year
    manure_src: str = "gn2013"
    fert_n_kg: float = 0.0          #: mineral N, kg N/ha (**not** kg of product)
    fert_doy: int = 120             #: day of year
    fert_src: str = MINERAL_SOURCE
    #: ((doy, kg N/ha), ...). If non-empty, replaces fert_n_kg/fert_doy.
    fert_splits: tuple[tuple[int, float], ...] = ()
    irr_start_doy: int = 140
    irr_interval: int = 7           #: days between events
    irr_depth: float = 25.0         #: mm per event
    #: Six Apr-Sep monthly mm; overrides the fixed-interval triple.
    irr_month_depths: tuple[float, ...] | None = None
    #: Explicit ((doy, mm), ...) events; overrides both above. Post-harvest events dropped.
    irr_day_depths: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.crop not in CROPS:
            raise ValueError(f"unknown crop {self.crop!r}; expected one of {CROPS}")
        if self.manure_src not in MANURE_SOURCES:
            raise ValueError(f"unknown manure source {self.manure_src!r}")
        if self.fert_src not in MINERAL_N_FRAC:
            raise ValueError(f"unknown mineral source {self.fert_src!r}")
        if self.irr_month_depths is not None and len(self.irr_month_depths) != 6:
            raise ValueError("irr_month_depths must have length 6 (Apr–Sep)")


def _md(doy: int, year: int = 2015) -> tuple[int, int]:
    """Day-of-year -> (month, day), on a non-leap reference year."""
    d = date(year, 1, 1) + timedelta(days=int(doy) - 1)
    return d.month, d.day


def _doy(mon: int, day: int, year: int = 2015) -> int:
    return (date(year, mon, day) - date(year, 1, 1)).days + 1


def _op(typ: str, mon: int, day: int, d1: str, d2: str = "null", d3: float = 0.0) -> str:
    return (f"{'':>48}{typ:<10}{mon:>6}{day:>10}{0.0:>14.5f}"
            f"{d1:>18}{d2:>18}{d3:>14.5f}  \n")


def _year_ops(action: YearAction, *, plant: bool, terminate: bool, irr_name: str) -> list[tuple[int, str]]:
    """(day-of-year, rendered op) for one year, unsorted."""
    crop, cal = action.crop, CALENDAR[action.crop]
    ops: list[tuple[int, str]] = []

    if action.manure_mg > 0:
        doy = int(action.manure_doy)
        ops.append((doy, _op("fert", *_md(doy), action.manure_src, "broadcast",
                             action.manure_mg * 1000.0)))
        # Incorporation the day after, matching the measured practice.
        ops.append((doy + 1, _op("till", *_md(doy + 1), INCORPORATION_TILL)))

    # op_data3 is kg product. Surface-applied (broadcast urea, volatilises).
    frac = MINERAL_N_FRAC[action.fert_src]
    if action.fert_splits:
        # Split program: one application per (day-of-year, rate) entry.
        for doy, kg_n in action.fert_splits:
            if kg_n <= 0:
                continue
            doy = int(doy)
            ops.append((doy, _op("fert", *_md(doy), action.fert_src, "broadcast",
                                 kg_n / frac)))
    elif action.fert_n_kg > 0:
        doy = int(action.fert_doy)
        ops.append((doy, _op("fert", *_md(doy), action.fert_src, "broadcast",
                             action.fert_n_kg / frac)))

    if plant:
        ops.append((_doy(*cal["plant"]), _op("plnt", *cal["plant"], crop)))

    if crop == "alfa":
        cuts = cal["cuts"]
        for i, (mon, day) in enumerate(cuts):
            last = terminate and i == len(cuts) - 1
            ops.append((_doy(mon, day),
                        _op("hvkl" if last else "harv", mon, day, crop, HARV_OPS[crop])))
        end_doy = _doy(*cuts[-1])
    else:
        mon, day = cal["harvest"]
        ops.append((_doy(mon, day), _op("hvkl", mon, day, crop, HARV_OPS[crop])))
        end_doy = _doy(mon, day)

    # Irrigation precedence: day events > monthly > fixed interval.
    if action.irr_day_depths is not None:
        for doy, depth in action.irr_day_depths:
            doy = int(doy)
            if depth <= 0 or doy >= end_doy:
                continue
            ops.append((doy, _op("irrm", *_md(doy), f"{irr_name}_d{doy:03d}")))
    elif action.irr_month_depths is not None:
        for mon, depth in zip(GROWING_MONTHS, action.irr_month_depths):
            if depth <= 0:
                continue
            # Mid-month, or the day before harvest if later.
            doy = _doy(mon, 15)
            if doy >= end_doy:
                doy = end_doy - 1
                if doy < _doy(mon, 1):
                    continue  # harvest before this month starts
                mon, day = _md(doy)
            else:
                day = 15
            month_name = f"{irr_name}_m{mon}"
            ops.append((doy, _op("irrm", mon, day, month_name)))
    elif action.irr_depth > 0 and action.irr_interval > 0:
        doy = int(action.irr_start_doy)
        while doy < end_doy:
            ops.append((doy, _op("irrm", *_md(doy), irr_name)))
            doy += int(action.irr_interval)

    return ops


def build(actions: list[YearAction], *, spinup: list[tuple[int, str]] | None = None,
          name: str = "kimb_rot") -> dict[str, str]:
    """Render actions to {filename: content}. 2012 barley spin-up prepended unless overridden."""
    irr_lines: list[str] = []
    blocks: list[str] = []

    if spinup is None:
        spinup_action = YearAction(crop="barl", manure_mg=0.0, irr_depth=0.0)
        spin_ops = _year_ops(spinup_action, plant=True, terminate=True, irr_name="none")
        blocks.append("".join(op for _, op in sorted(spin_ops, key=lambda t: t[0])))

    for i, action in enumerate(actions):
        prev = actions[i - 1].crop if i else None
        nxt = actions[i + 1].crop if i + 1 < len(actions) else None
        # A perennial stand is planted once and terminated when the rotation moves on.
        plant = not (action.crop == "alfa" and prev == "alfa")
        terminate = not (action.crop == "alfa" and nxt == "alfa")

        irr_name = f"gy{i:02d}"
        if action.irr_day_depths is not None:
            for doy, depth in action.irr_day_depths:
                if depth <= 0:
                    continue
                irr_lines.append(
                    f"{f'{irr_name}_d{int(doy):03d}':<22}{float(depth):>10.5f}"
                    f"{IRR_EFF:>14.5f}{0.0:>14.5f}"
                    f"{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}  \n"
                )
        elif action.irr_month_depths is not None:
            for mon, depth in zip(GROWING_MONTHS, action.irr_month_depths):
                if depth <= 0:
                    continue
                month_name = f"{irr_name}_m{mon}"
                irr_lines.append(
                    f"{month_name:<22}{float(depth):>10.5f}{IRR_EFF:>14.5f}{0.0:>14.5f}"
                    f"{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}  \n"
                )
        else:
            irr_lines.append(
                f"{irr_name:<22}{action.irr_depth:>10.5f}{IRR_EFF:>14.5f}{0.0:>14.5f}"
                f"{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}{0.0:>14.5f}  \n"
            )
        ops = _year_ops(action, plant=plant, terminate=terminate, irr_name=irr_name)
        # Empty year block would shift the rotation.
        if not ops:
            raise ValueError(f"year {i} produced no management ops — refusing empty block")
        blocks.append("".join(op for _, op in sorted(ops, key=lambda t: t[0])))

    body = "".join(blocks)
    n_ops = body.count("\n")
    sch = (
        "management.sch: written by swat_gym.schedule\n"
        "name                                       numb_ops  numb_auto            op_typ"
        "       mon       day        hu_sch          op_data1          op_data2      op_data3  \n"
        f"{name:<37}{n_ops:>7}{0:>11}  \n"
        + body
    )
    return {"management.sch": sch, "irr.ops": _irr_ops(irr_lines)}


def _irr_ops(new_lines: list[str]) -> str:
    """Append to shipped irr.ops (keep the measured gn#### events)."""
    from .fastrunner import TXTINOUT

    base = (TXTINOUT / "irr.ops").read_text().rstrip("\n").splitlines(keepends=True)
    return "".join(base) + "\n" + "".join(new_lines)
