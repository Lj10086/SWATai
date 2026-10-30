"""Port remaining ArcSWAT reference (HRU 000140001) params: CN2, USLE_P, OV_N, snow, .gw
aquifer, basin P block, SDNCO, SPCON/SPEXP, CN_FROZ.

Not ported (GRACEnet measurements win): initial NO3/P, orgn_min, bulk density, 2014/2019 irrigation.

    uv run python scripts/port_reference_params.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIO = ROOT / "model" / "rotation" / "TxtInOut"

#: Name for site-specific rows added to shared lookup tables.
SITE_ROW = "kimb_ref"

# ---------------------------------------------------------------------------------------
# 000140001.mgt / 000140001.hru
# ---------------------------------------------------------------------------------------

#: CN2 for hydrologic group C (only cn_c is read; others keep template spacing).
REF_CN2_C = 75.0
_CN_OFFSETS = {"cn_a": -23.0, "cn_b": -9.0, "cn_c": 0.0, "cn_d": 4.0}

#: ``OV_N : Manning's "n" value for overland flow`` from ``000140001.hru``.
REF_OV_N = 0.14

#: USLE_P 1.0 = up_down_slope (pointer change in landuse.lum).
REF_CONS_PRACTICE = "up_down_slope"

# ---------------------------------------------------------------------------------------
# basins.bsn snow block -> snow.sno
# ---------------------------------------------------------------------------------------

#: SWAT+ column -> (value, SWAT2012 name).
REF_SNOW = {
    "fall_tmp": (1.0, "SFTMP"),      # already matched
    "melt_tmp": (2.0, "SMTMP"),      # was 0.5
    "melt_max": (4.5, "SMFMX"),      # already matched
    "melt_min": (0.1, "SMFMN"),      # was 4.5 -- a factor of 45
    "tmp_lag": (1.0, "TIMP"),        # already matched
    "snow_h2o": (1.0, "SNOCOVMX"),   # already matched
    "cov50": (0.5, "SNO50COV"),      # already matched
}

# ---------------------------------------------------------------------------------------
# 000140001.gw -> aquifer.aqu (shallow aquifer row)
# ---------------------------------------------------------------------------------------

#: SWAT+ column -> (value, SWAT2012 name). Depth thresholds mm -> m. hl_no3n from basin value.
REF_AQUIFER = {
    "dep_wt": (1.0, "GWHT"),            # was 3.0 m
    "alpha_bf": (0.048, "ALPHA_BF"),    # was 0.05
    "revap": (0.02, "GW_REVAP"),        # already matched
    "rchg_dp": (0.05, "RCHRG_DP"),      # already matched
    "spec_yld": (0.003, "GW_SPYLD"),    # was 0.05 -- a factor of ~17
    "hl_no3n": (5.0, "HLIFE_NGW_BSN"),  # was 30 days
    "flo_min": (1.0, "GWQMN/1000"),     # was 3.0 m
    "revap_min": (0.75, "REVAPMN/1000"),  # was 5.0 m
}
AQUIFER_SHALLOW = "aqu10"

# ---------------------------------------------------------------------------------------
# basins.bsn -> parameters.bsn / codes.bsn
# ---------------------------------------------------------------------------------------

#: P block + stragglers (N params are owned by port_nutrients.py).
REF_BASIN = {
    "p_uptake": (10.0, "P_UPDIS"),      # was 20.0
    "p_perc": (11.0, "PPERCO"),         # was 10.0
    "p_soil": (150.0, "PHOSKD"),        # was 175.0
    "p_avail": (0.30, "PSP"),           # was 0.40
    "denit_frac": (1.0, "SDNCO"),       # was 1.30
    "lin_sed": (0.0001, "SPCON"),       # was 0.0
    "exp_sed": (1.0, "SPEXP"),          # was 0.0
    "cn_froz": (0.000862, "CN_FROZ"),   # was 0.0
    "evap_adj": (1.0, "EVRCH"),         # was 0.6
}

#: ``SOL_P_MODEL = 1`` — the reference runs the newer soil-phosphorus routine.
REF_CODES = {"soil_p": (1, "SOL_P_MODEL")}


# ---------------------------------------------------------------------------------------
# Header-addressed table editing
# ---------------------------------------------------------------------------------------

def _fmt(val: float | int) -> str:
    """Format to 5 decimals, more if needed to keep the value."""
    if isinstance(val, int):
        return str(val)
    for places in range(5, 12):
        text = f"{val:.{places}f}"
        if float(text) == val:
            return text
    return repr(val)


def _read(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _set_columns(path: Path, key: str | None, updates: dict[str, float | int],
                 header_line: int = 1, key_col: int = 0) -> dict[str, tuple[str, str]]:
    """Set columns (by header name) on the row where ``key_col`` == ``key``. Returns {col: (before, after)}."""
    lines = _read(path)
    header = lines[header_line].split()
    for name in updates:
        if name not in header:
            raise SystemExit(f"{path.name}: no column {name!r}; have {header}")

    for i in range(header_line + 1, len(lines)):
        toks = lines[i].split()
        if not toks:
            continue
        if key is not None and toks[key_col] != key:
            continue

        changed: dict[str, tuple[str, str]] = {}
        for name, val in updates.items():
            col = header.index(name)
            if col >= len(toks):
                raise SystemExit(f"{path.name}: row {key!r} has {len(toks)} tokens, "
                                 f"need {col + 1} for {name!r}")
            before, after = toks[col], _fmt(val)
            if float(before) != float(after):
                changed[name] = (before, after)
            toks[col] = after

        # Keep any trailing description column glued to the numbers it annotates.
        lines[i] = "  ".join(toks)
        path.write_text("\n".join(lines) + "\n")
        return changed

    raise SystemExit(f"{path.name}: row {key!r} not found")


def _upsert_row(path: Path, row: str, values: dict[str, float], description: str) -> str:
    """Add ``row`` to a lookup table, or update it if a previous run already added it."""
    lines = _read(path)
    header = lines[1].split()
    ordered = [_fmt(values[c]) for c in header[1:] if c in values]
    new = "  ".join([row, *ordered, description])

    for i in range(2, len(lines)):
        toks = lines[i].split()
        if toks and toks[0] == row:
            verb = "updated" if lines[i].split() != new.split() else "unchanged"
            lines[i] = new
            path.write_text("\n".join(lines) + "\n")
            return verb

    lines.append(new)
    path.write_text("\n".join(lines) + "\n")
    return "added"


def _report(label: str, changed: dict[str, tuple[str, str]],
            names: dict[str, tuple[float | int, str]]) -> None:
    if not changed:
        print(f"{label}: already reconciled")
        return
    for col, (before, after) in changed.items():
        print(f"{label}: {col} {before} -> {after}   ({names[col][1]})")


def main() -> None:
    # --- CN2, OV_N, USLE_P: two new lookup rows plus three pointer changes -------------
    cn = {c: REF_CN2_C + off for c, off in _CN_OFFSETS.items()}
    verb = _upsert_row(TIO / "cntable.lum", SITE_ROW, cn,
                       "Kimberly_reference_CN2  ArcSWAT_000140001.mgt  ----")
    print(f"cntable.lum: row {SITE_ROW} {verb} "
          f"(cn_c {REF_CN2_C} from CN2; was legr_strow_g cn_c 81)")

    verb = _upsert_row(TIO / "ovn_table.lum", SITE_ROW,
                       {"ovn_mean": REF_OV_N, "ovn_min": REF_OV_N, "ovn_max": REF_OV_N},
                       "Kimberly_reference_OV_N")
    print(f"ovn_table.lum: row {SITE_ROW} {verb} (ovn {REF_OV_N} from OV_N; was 0.19)")

    lines = _read(TIO / "landuse.lum")
    header, toks = lines[1].split(), lines[2].split()
    for col, val in (("cn2", SITE_ROW), ("ov_mann", SITE_ROW),
                     ("cons_prac", REF_CONS_PRACTICE)):
        i = header.index(col)
        if toks[i] != val:
            print(f"landuse.lum: {col} {toks[i]} -> {val}")
            toks[i] = val
    lines[2] = "  ".join(toks)
    (TIO / "landuse.lum").write_text("\n".join(lines) + "\n")

    # --- snow, groundwater, basin parameters, codes -----------------------------------
    for path, key, spec, label in (
        (TIO / "snow.sno", "snow001", REF_SNOW, "snow.sno"),
        (TIO / "aquifer.aqu", AQUIFER_SHALLOW, REF_AQUIFER, "aquifer.aqu"),
        (TIO / "parameters.bsn", None, REF_BASIN, "parameters.bsn"),
        (TIO / "codes.bsn", None, REF_CODES, "codes.bsn"),
    ):
        key_col = 1 if path.name == "aquifer.aqu" else 0
        changed = _set_columns(path, key, {k: v for k, (v, _) in spec.items()},
                               key_col=key_col)
        _report(label, changed, spec)


if __name__ == "__main__":
    main()
