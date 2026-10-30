"""Read/write SWAT+ table parameters addressed as (file, row, column), e.g. ("plants.plt", "barl", "bm_e")."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Files calibration may rewrite, on top of :data:`swat_gym.fastrunner.EDITABLE`.
CALIBRATABLE = frozenset({
    "plants.plt",      # per-crop growth: bm_e, lai_pot, harv_idx, tmp_opt, days_mat
    "hydrology.hyd",   # esco, epco, perco, cn3_swf, pet_co, can_max
    "parameters.bsn",  # basin N/P cycling: orgn_min, n_uptake, n_perc, rsd_decomp
    "soils.sol",       # awc, soil_k, carbon by layer
    "nutrients.sol",   # initial pools + fr_hum_act, the active/stable humus N split
    # Operative harvest index (plants.plt harv_idx is inert for biomass harvest).
    "harv.ops",
})

#: Which line the column-name header sits on, per file. Rows follow it.
HEADER_LINE = {
    "plants.plt": 1,
    "hydrology.hyd": 1,
    "parameters.bsn": 1,
    "soils.sol": 1,
    "nutrients.sol": 1,
    "harv.ops": 1,
}

#: Fortran integer columns; must be written without a decimal point or the row mis-parses.
INTEGER_COLUMNS = frozenset({
    ("plants.plt", "days_mat"),
    ("plants.plt", "yrs_mat"),
})


@dataclass(frozen=True)
class Param:
    """One addressable scalar, with the bounds a screen or optimizer may explore."""

    file: str
    row: str
    column: str
    low: float
    high: float

    @property
    def name(self) -> str:
        return f"{self.row}.{self.column}" if self.row else self.column

    def __str__(self) -> str:
        return f"{self.name} [{self.low}, {self.high}]"


def _split(text: str) -> tuple[list[str], int, list[str]]:
    lines = text.splitlines()
    return lines, 0, lines


def _locate(lines: list[str], file: str, row: str, column: str) -> tuple[int, int]:
    """Return (line index, field index) for a parameter, or raise with a useful message."""
    hdr_i = HEADER_LINE[file]
    names = lines[hdr_i].split()
    if column not in names:
        raise KeyError(f"{file}: no column {column!r}; have {names[:12]}...")
    col_i = names.index(column)

    if not row:  # single-row files (parameters.bsn) have no row key
        return hdr_i + 1, col_i
    for i in range(hdr_i + 1, len(lines)):
        parts = lines[i].split()
        if parts and parts[0] == row:
            return i, col_i
    raise KeyError(f"{file}: no row named {row!r}")


def get_value(text: str, file: str, row: str, column: str) -> float:
    lines = text.splitlines()
    i, col_i = _locate(lines, file, row, column)
    return float(lines[i].split()[col_i])


def set_value(text: str, file: str, row: str, column: str, value: float) -> str:
    """Return ``text`` with one cell replaced, the row re-rendered at uniform width."""
    lines = text.splitlines()
    i, col_i = _locate(lines, file, row, column)
    parts = lines[i].split()
    if col_i >= len(parts):
        raise KeyError(f"{file}: row {row!r} has {len(parts)} fields, need index {col_i}")
    if (file, column) in INTEGER_COLUMNS:
        parts[col_i] = str(int(round(value)))
    else:
        parts[col_i] = f"{value:.5f}"
    lines[i] = "  ".join(parts) + "  "
    return "\n".join(lines) + "\n"


def apply(sources: Mapping[str, str], values: Mapping[Param, float]) -> dict[str, str]:
    """Apply values; returns {filename: new content} for changed files only."""
    out: dict[str, str] = {}
    for param, value in values.items():
        text = out.get(param.file, sources[param.file])
        out[param.file] = set_value(text, param.file, param.row, param.column, value)
    return out


def read_defaults(txtinout: Path, params: Iterable[Param]) -> dict[Param, float]:
    """Current value of each parameter, for centring a screen or seeding an optimizer."""
    cache: dict[str, str] = {}
    out = {}
    for p in params:
        if p.file not in cache:
            cache[p.file] = (txtinout / p.file).read_text()
        out[p] = get_value(cache[p.file], p.file, p.row, p.column)
    return out
