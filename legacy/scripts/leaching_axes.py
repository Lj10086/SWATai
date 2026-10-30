#!/usr/bin/env python3
"""Is leaching driven by nitrogen or by water? (PAPER §3.5)

Scores, on every test window: measured practice; the N lever at its maximum (every month's
mineral N and April manure at the box limit) with irrigation at measured practice; and
measured-practice water x1.2 with measured N. Writes runs/leaching_axes.json.

    uv run python scripts/leaching_axes.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swat_gym.fastrunner import FastRunner, model_hash
from swat_gym.plan import DIMS, default_x, evaluate
from swat_gym.rewarders import average
from swat_gym.windows import TEST_YEARS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "leaching_axes.json"


def main() -> None:
    prices = average()
    arms = {
        "measured practice": ("N", default_x("N")),
        "N at its maximum, measured water": ("N", np.ones(DIMS["N"])),
        "water x1.2, measured N": ("I", np.clip(default_x("I") * 1.2, 0.0, 1.0)),
    }
    out = {"model": model_hash(), "test_years": TEST_YEARS, "arms": {}}
    with FastRunner() as r:
        for name, (lever, x) in arms.items():
            rows = [evaluate(lever, x, r, prices=prices, start_year=sy) for sy in TEST_YEARS]
            out["arms"][name] = {
                k: float(np.mean([d[k] for d in rows]))
                for k in ("n_applied_kg", "irrigation_mm", "no3_leached_kg")}
            a = out["arms"][name]
            print(f"{name:34s} N {a['n_applied_kg']:6.0f} kg  water {a['irrigation_mm']:5.0f} mm"
                  f"  leached {a['no3_leached_kg']:5.2f} kg N/ha (7-yr window mean)")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
