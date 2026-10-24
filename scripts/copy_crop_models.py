#!/usr/bin/env python3
"""Copy the Ai-SWAT-Plus per-crop monoculture models into model/crops/<code>/.

Builds each crop with Ai-SWAT-Plus's own builder (``et_calibrate.CropFit._build`` +
``monoculture.apply_et_fit``, the setup ``et_raw.py`` runs), then copies only the SWAT+ inputs
and engines, plus a ``spec.json`` the gym reads (season dates, irrigation windows, cuttings,
auto decision-table names). Re-run after changing the Ai-SWAT-Plus model; nothing here is fitted.

Must run in the Ai-SWAT-Plus environment, since it imports that project's builders:

    uv run --project ~/Desktop/projects/Ai-SWAT-Plus/calibration \\
        python scripts/copy_crop_models.py [--crops CSIL ALFA]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import date

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AISWAT = Path.home() / "Desktop" / "projects" / "Ai-SWAT-Plus" / "calibration"
sys.path.insert(0, str(AISWAT / "src"))
sys.path.insert(0, str(AISWAT / "scripts"))

import et_calibrate as etc  # noqa: E402
import monoculture as mono  # noqa: E402
from swat_gym.manifest import input_files  # noqa: E402  (Ai-SWAT-Plus's copy)

sys.path.insert(0, str(ROOT / "scripts"))
import port_gridmet  # noqa: E402  (this project's weather: gridMET at the field point)

OUT = ROOT / "model" / "crops"
#: Weather covers 1995-2025; build the schedule for all of it so any window runs.
BUILD_START, BUILD_YEARS = 1995, 31
#: Growth failures with little real-world room to optimise (2026-10-23): pinto bean yields
#: 0.09 t/ha (temperature stress), potato 10 t/ha off a failed canopy. Pass --crops to override.
DROPPED = {"PTBN", "POTA"}


def _md(md: tuple[int, int]) -> list[int]:
    return [int(md[0]), int(md[1])]


def copy_crop(code: str) -> dict:
    plt = mono.CROPS[code][0]
    shared, _ = mono.load_et_fit()
    sw_frac = shared.get("irr_sw_frac", mono.IRR_SW_FRAC)
    with tempfile.TemporaryDirectory(prefix=f"aiswat_{plt}_") as tmp:
        tio = etc.CropFit._build(code, BUILD_START, BUILD_YEARS, Path(tmp) / "TxtInOut",
                                 sw_frac)
        mono.apply_et_fit(tio, plt)
        names = input_files(tio)
        names |= {p.name for p in tio.iterdir() if p.name.startswith("swatplus_")
                  and not p.name.endswith(".bak")}
        dest = OUT / code.lower()
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "TxtInOut").mkdir(parents=True)
        for n in sorted(names):
            shutil.copy2(tio / n, dest / "TxtInOut" / n)
    # Ai-SWAT-Plus builds on the AgriMet station record; this project runs every model on
    # gridMET (cached by port_gridmet.py: data/gridmet_kimberly_1995_2025.csv).
    weather = port_gridmet.convert(pd.read_csv(port_gridmet.CACHE, index_col="date",
                                               parse_dates=True))
    elev = float((dest / "TxtInOut" / "hru.con").read_text().splitlines()[2].split()[6])
    port_gridmet.write(dest / "TxtInOut", weather, elev)

    _, emerge, harvests = mono.CROPS[code]
    perennial = plt == "alfa"
    irr_table = f"irr_win_{plt}"
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode() + (dest / "TxtInOut" / n).read_bytes())
    spec = {
        "name": code.lower(),
        "plant": plt,
        "perennial": perennial,
        "emerge": _md(emerge),
        # Annual: the single harvest-kill date. Perennial: the cutting dates (cut by lum.dtl).
        "harvests": [_md(h_) for h_ in harvests],
        "irrigation_windows_doy": [list(w) for w in mono.irrigation_windows(code)],
        "auto_irrigation": irr_table,
        "auto_fert_n": "auto_fert_n",
        "auto_irr_mm": mono.IRR_MM,
        # Several candidates may run as HRUs of one engine run only where that is exact.
        # SWAT+ rev 62 keeps N/P demand per plant, not per HRU, so a perennial breaking
        # dormancy reads the previous HRU's demand (phantom auto-P events in alfalfa).
        # Verified by swat_gym/tests/test_batch.py; never set true without that gate.
        "batchable": not perennial,
        "source": {
            "built_by": "Ai-SWAT-Plus et_calibrate.CropFit._build + monoculture.apply_et_fit",
            "aiswat_dir": str(AISWAT),
            "irr_sw_frac": sw_frac,
            "weather": "gridMET at the field point (scripts/port_gridmet.py); the Ai-SWAT-Plus "
                       "ET fit was made on AgriMet TWFI and is not refitted",
            "copied": date.today().isoformat(),
            "inputs_sha256": h.hexdigest()[:16],
        },
    }
    (dest / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", nargs="*", default=[c for c in mono.CROPS if c not in DROPPED])
    args = ap.parse_args()
    for code in args.crops:
        s = copy_crop(code)
        print(f"{code}: -> model/crops/{s['name']}  inputs {s['source']['inputs_sha256']}")


if __name__ == "__main__":
    main()
