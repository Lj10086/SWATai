#!/usr/bin/env python3
"""Extract each crop's Ai-SWAT-Plus parameters into model/ltm/crop_params/<plant>.json.

Builds every crop with Ai-SWAT-Plus's own builder (``copy_crop_models.copy_crop``, i.e.
``et_calibrate.CropFit._build`` + ``monoculture.apply_et_fit``) in a temp dir and keeps only
what that build fitted for the crop: its ``plants.plt`` row, its ``calibration.cal`` heat units
to maturity, its ``harv.ops`` entry, and the spec (dates, irrigation windows). Nothing is fitted
or edited here (CLAUDE.md rule 11b). Also records the engine and the shared hydrology values
the fits were made under.

Run in the Ai-SWAT-Plus environment:

    uv run --project ~/Desktop/projects/Ai-SWAT-Plus/calibration python scripts/ltm_crop_params.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import copy_crop_models as ccm  # noqa: E402

OUT = ROOT / "model" / "ltm" / "crop_params"
ENGINE_DIR = ROOT / "model" / "ltm" / "engine"
#: Ai-SWAT-Plus crop code -> role in the LTM scenario.
CODES = {"BARL": "trial", "SGBT": "trial", "WWHT": "trial (spring wheat)", "POTA": "trial",
         "ALFA": "exp1", "CSIL": "exp1", "PTBN": "2012 history only"}


def _row(text: str, name: str) -> str:
    rows = [ln for ln in text.splitlines()[2:] if ln.split()[:1] == [name]]
    if len(rows) != 1:
        raise SystemExit(f"expected one '{name}' row, found {len(rows)}")
    return rows[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    shared = None
    with tempfile.TemporaryDirectory(prefix="ltm_params_") as tmp:
        ccm.OUT = Path(tmp)
        for code, role in CODES.items():
            spec = ccm.copy_crop(code)
            tio = Path(tmp) / code.lower() / "TxtInOut"
            plt = spec["plant"]
            cal = (tio / "calibration.cal").read_text().splitlines()
            phu_line, cond_line = cal[3], cal[4]
            if cond_line.split() != ["plant", "=", "0", plt] or phu_line.split()[0] != "phu_mat":
                raise SystemExit(f"{code}: unexpected calibration.cal\n" + "\n".join(cal))
            # the builder tags the crop's own op workbook_<CODE>_HARV_EFF (alfalfa's cuts are in
            # lum.dtl, so the schedule does not name it)
            harv = (tio / "harv.ops").read_text().splitlines()[2:]
            tagged = [ln for ln in harv if ln.split()[-1:] == [f"workbook_{code}_HARV_EFF"]]
            if len(tagged) != 1:
                raise SystemExit(f"{code}: {len(tagged)} harv.ops rows tagged for this crop")
            harv_name = tagged[0].split()[0]
            hyd = (tio / "hydrology.hyd").read_text().splitlines()
            hyd_cols, hyd_vals = hyd[1].split(), hyd[2].split()
            this_shared = {k: float(v) for k, v in zip(hyd_cols[1:], hyd_vals[1:])
                           if k in ("esco", "epco", "pet_co")}
            if shared is None:
                shared = this_shared
                for p in tio.iterdir():
                    if p.name.startswith("swatplus_") and not p.name.endswith(".bak"):
                        shutil.copy2(p, ENGINE_DIR / p.name)
            elif this_shared != shared:
                raise SystemExit(f"{code}: hydrology differs from the first crop: "
                                 f"{this_shared} vs {shared}")
            rec = {
                "code": code, "plant": plt, "role": role,
                "plants_plt_row": _row((tio / "plants.plt").read_text(), plt),
                "plants_plt_header": (tio / "plants.plt").read_text().splitlines()[1],
                "phu_mat": float(phu_line.split()[2]),
                "harv_op": harv_name,
                "harv_ops_row": _row((tio / "harv.ops").read_text(), harv_name),
                "spec": {k: spec[k] for k in ("perennial", "emerge", "harvests",
                                              "irrigation_windows_doy", "auto_irr_mm")},
                "source": spec["source"],
            }
            (OUT / f"{plt}.json").write_text(json.dumps(rec, indent=2) + "\n")
            print(f"{code:5} -> crop_params/{plt}.json  phu_mat {rec['phu_mat']}  "
                  f"harv {harv_name}  ({role})")
    h = hashlib.sha256()
    for p in sorted(ENGINE_DIR.iterdir()):
        h.update(p.name.encode() + p.read_bytes())
    (OUT / "_shared.json").write_text(json.dumps(
        {"hydrology": shared, "engine_sha256": h.hexdigest()[:16],
         "engine_files": sorted(p.name for p in ENGINE_DIR.iterdir()),
         "aiswat_dir": str(ccm.AISWAT)}, indent=2) + "\n")
    print(f"shared hydrology {shared}; engine -> {ENGINE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
