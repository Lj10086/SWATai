"""Regenerate tests/fixtures/baseline.json. Only deliberately, with a --why note.

    uv run python scripts/rebaseline.py --why "ported measured humidity and wind (AgriMet)"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

from swat_gym import FastRunner
from swat_gym.engine import find_engine
from swat_gym.fastrunner import TXTINOUT

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "src" / "swat_gym" / "tests" / "fixtures" / "baseline.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--why", required=True, help="what changed in the model, for provenance")
    ap.add_argument("--out", type=Path, default=FIXTURE)
    args = ap.parse_args()

    with FastRunner() as r:
        r.run()
        yld = r.yields()
        wb = r.read("hru_wb_yr.txt")
        pw = r.read("hru_pw_yr.txt")

    payload = {
        "_provenance": {
            "engine": find_engine(TXTINOUT).name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "regenerated": dt.date.today().isoformat(),
            "model": args.why,
            "note": "Regression gate. A different SWAT+ build (e.g. a Linux/WSL ELF engine) "
                    "must reproduce these numbers; a deviation is build disagreement, not a "
                    "code bug. Regenerate with scripts/rebaseline.py.",
        },
        "yields": [
            {"year": int(t.year), "crop": t.plant_name,
             "cuts": float(getattr(t, "_4")), "yld_t": float(getattr(t, "_5"))}
            for t in yld.itertuples()
        ],
        "water_mean": {c: float(wb[c].mean()) for c in ("precip", "et", "pet", "perc", "irr")},
        "nstress_days": {str(int(y)): float(s) for y, s in zip(pw["yr"], pw["strsn"])},
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)}")
    print(f"  {len(payload['yields'])} harvest years, "
          f"water means {({k: round(v, 1) for k, v in payload['water_mean'].items()})}")


if __name__ == "__main__":
    main()
