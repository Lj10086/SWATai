"""Input file set of a TxtInOut, from file.cio plus weather files (named in .cli) and the engine."""
from __future__ import annotations

from pathlib import Path

FILE_CIO = "file.cio"


#: weather-sta.cli columns that name a series file (the rest are station/generator names).
WEATHER_SERIES = frozenset({"pcp", "tmp", "slr", "hmd", "wnd", "wnd_dir", "pet"})


def _tokens(line: str) -> list[str]:
    return [t for t in line.split() if t and t != "null"]


#: Inputs whose absence silently corrupts output (engine still exits 0). Fail early.
REQUIRED = frozenset({
    "file.cio", "time.sim", "print.prt",
    "plants.plt",        # crop parameter database — silent corruption when absent
    "management.sch", "irr.ops", "fertilizer.frt", "tillage.til", "harv.ops",
    "hru.con", "hru-data.hru", "soils.sol", "plant.ini",
})


def input_files(txtinout: Path) -> set[str]:
    """Names of files SWAT+ reads. A declared file that is missing is an error: the engine
    would run without it and exit 0."""
    cio = txtinout / FILE_CIO
    if not cio.is_file():
        raise FileNotFoundError(f"{cio} not found — is {txtinout} a SWAT+ TxtInOut?")

    names = {FILE_CIO}
    declared: list[tuple[str, str]] = []
    for line in cio.read_text().splitlines()[1:]:  # line 0 is the editor's title banner
        # Column 0 is the section label ("climate", "soils", …), not a filename.
        tokens = _tokens(line)
        # *_path rows name a directory ("null" when the files sit in TxtInOut).
        if tokens and not tokens[0].endswith("_path"):
            declared += [(FILE_CIO, t) for t in tokens[1:]]
    names |= {t for _, t in declared}

    # Weather series are named by the per-variable .cli indexes (header "filename", e.g.
    # pcp.cli -> tfccpcp.pcp) and by weather-sta.cli's series columns. weather-wgn.cli is data.
    for cli in sorted(n for n in names if n.endswith(".cli")):
        if not (txtinout / cli).is_file():
            continue  # reported below with the rest
        lines = (txtinout / cli).read_text().splitlines()
        header = lines[1].split() if len(lines) > 1 else []
        if header[:1] == ["filename"]:
            refs = [t for line in lines[2:] for t in _tokens(line)]
        elif cli == "weather-sta.cli":
            cols = [i for i, h in enumerate(header) if h in WEATHER_SERIES]
            refs = [line.split()[i] for line in lines[2:] for i in cols
                    if len(line.split()) > i and line.split()[i] != "null"]
        else:
            continue
        declared += [(cli, t) for t in refs]
        names |= set(refs)

    absent = sorted({f"{t} (named in {src})" for src, t in declared
                     if not (txtinout / t).is_file()})
    if absent:
        raise FileNotFoundError(f"{txtinout}: declared input(s) missing: {absent}")

    missing = sorted(n for n in REQUIRED if not (txtinout / n).is_file())
    if missing:
        raise FileNotFoundError(
            f"{txtinout} is missing required SWAT+ input(s): {missing}. "
            "These do not fail the engine — it exits 0 and writes structurally valid output "
            "with garbage in it. Restore them (e.g. `git checkout -- model/TxtInOut/`) "
            "before running anything.")
    return names
