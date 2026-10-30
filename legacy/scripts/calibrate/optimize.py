"""Fit the model to GRACEnet measurements.

Objective: per-year yield error (RMS %) + down-weighted soil NO3 bracket miss (n_trajectory).
PET is a bound only; pet_co is not free.

    uv run python scripts/calibrate/optimize.py --maxiter 60
    uv run python scripts/calibrate/optimize.py --apply      # write the fit into TxtInOut
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from swat_gym import FastRunner  # noqa: E402
from swat_gym.fastrunner import EDITABLE, TXTINOUT  # noqa: E402
from swat_gym.params import CALIBRATABLE, Param, apply, read_defaults, set_value  # noqa: E402

from calib_report import collect, pbias  # noqa: E402
from n_trajectory import score as no3_score  # noqa: E402
from n_trajectory import trajectory  # noqa: E402

#: Free parameters: N cycle only. Never add crop coefficients (rule 11b), pet_co (rule 10),
#: epco (sourced 0.50), days_mat, or inert params (perco, rsd_decomp, latq_co, harv_idx,
#: alfa.lai_min).
PARAMS = [
    # Live mineralisation knobs: orgn_min, fr_hum_act.
    Param("parameters.bsn", "", "orgn_min", 0.0005, 0.0060),
    Param("nutrients.sol", "soilnut1", "fr_hum_act", 0.010, 0.250),
    Param("parameters.bsn", "", "n_perc", 0.01, 1.00),
]

#: Both terms in %. NO3 down-weighted: alfalfa-year misses are unreachable (rule 13).
WEIGHTS = {"yield": 1.0, "no3": 0.25}
FAILED = 1e6
#: Abort once this share of evaluations has raised (and at least MIN_FAILURES).
MAX_FAILURE_FRAC = 0.01
MIN_FAILURES = 3

#: Fixed kg N/ha scale for the NO3 miss (mean measured stock).
NO3_SCALE = 160.0

#: PET may not drift more than this (percent) from measured AgriMet ETos.
PET_TOL = 10.0


def pet_ok(df) -> bool:
    if "pet_meas" not in df:
        return True
    return abs(pbias(df["pet"], df["pet_meas"])) <= PET_TOL


def terms(runner: FastRunner, sources: dict[str, str], values: dict[Param, float]) -> dict:
    runner.run(apply(sources, values))
    df = collect(runner)
    meas = df[df["yld_meas"].notna()]

    # Per-year relative error, RMS.
    err = 100.0 * (meas["yld"] - meas["yld_meas"]) / meas["yld_meas"]
    out = {"yield": float((err ** 2).mean() ** 0.5)}

    traj = trajectory(runner)
    # Fixed normaliser so small-pool years aren't overweighted.
    miss_pct = 100.0 * traj["miss"].abs() / NO3_SCALE
    out["no3"] = float((miss_pct ** 2).mean() ** 0.5)

    # Reported, never minimised — these are the numbers the paper quotes.
    out["_yield_pbias"] = pbias(meas["yld"], meas["yld_meas"])
    for crop, g in meas.groupby("crop"):
        out[f"_{crop}"] = pbias(g["yld"], g["yld_meas"])
    out["_no3_rms_kg"] = no3_score(traj, exact_only=False)
    out["_pet"] = pbias(df["pet"], df["pet_meas"]) if "pet_meas" in df else float("nan")
    out["_pet_ok"] = bool(pet_ok(df))
    return out


def score(t: dict) -> float:
    """Weighted mean squared term, in percent-squared. Rejects any fit that breaks the PET bound."""
    if not t.get("_pet_ok", True):
        return FAILED
    return sum(w * t[k] ** 2 for k, w in WEIGHTS.items()) / sum(WEIGHTS.values())


def _show(label: str, t: dict) -> None:
    print(f"  {label:<10} yield_rms {t['yield']:6.1f} %   no3_rms {t['no3']:7.1f} %   "
          f"| PBIAS all {t['_yield_pbias']:+6.1f}  corn {t['_corn']:+6.1f}  "
          f"alfa {t['_alfa']:+6.1f}  barl {t['_barl']:+6.1f}  "
          f"| PET {t['_pet']:+5.1f}  NO3 {t['_no3_rms_kg']:5.1f} kg/ha  "
          f"| SCORE {score(t):8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true",
                    help="write the fitted values into model/TxtInOut (default: report only)")
    ap.add_argument("--hold", nargs="*", default=[], metavar="NAME",
                    help="parameters to fix at their sourced value, by Param.name. They are "
                         "excluded from the search, not merely from the write — see below.")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "calibration.json")
    ap.add_argument("--from-saved", action="store_true",
                    help="re-score and apply the fit already in --out instead of refitting")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow a fitting run to replace an existing --out")
    args = ap.parse_args()

    # Refuse to overwrite the calibration record before searching.
    if not args.from_saved and args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already exists and holds the record of a previous fit. "
                         f"Archive it (runs/archive/) and pass --overwrite, or pick another --out.")

    # Reject unknown --hold names up front (e.g. use soilnut1.fr_hum_act).
    unknown = set(args.hold) - {p.name for p in PARAMS}
    if unknown:
        raise SystemExit(f"--hold names no such parameter: {sorted(unknown)}  "
                         f"(known: {sorted(p.name for p in PARAMS)})")

    sources = {f: (TXTINOUT / f).read_text() for f in CALIBRATABLE}
    defaults = read_defaults(TXTINOUT, PARAMS)
    started = time.time()
    calls = {"n": 0, "best": np.inf, "failed": 0, "first_tb": None}

    # Held params are excluded from the search, not just from the write.
    free = [p for p in PARAMS if p.name not in args.hold]
    print(f"{len(free)} free parameters, {len(PARAMS) - len(free)} held at sourced value "
          "(all unsourced; crop coefficients are the collaborator's, not fitted)\n")

    with FastRunner(editable=EDITABLE | CALIBRATABLE) as runner:
        base = terms(runner, sources, defaults)
        _show("start", base)
        print()

        def expand(x: np.ndarray) -> dict:
            """Search vector over the free parameters -> a full parameter set."""
            v = dict(zip(free, x))
            return {p: v.get(p, defaults[p]) for p in PARAMS}

        def objective(x: np.ndarray) -> float:
            calls["n"] += 1
            try:
                s = score(terms(runner, sources, expand(x)))
                if not np.isfinite(s):
                    raise ValueError(f"non-finite score {s} at {expand(x)}")
            except Exception:
                # A broken run is not a bad fit: count it, and stop before DE routes around
                # a model that no longer runs.
                calls["failed"] += 1
                calls["first_tb"] = calls["first_tb"] or traceback.format_exc()
                if calls["failed"] >= max(MIN_FAILURES, MAX_FAILURE_FRAC * calls["n"]):
                    raise RuntimeError(f"{calls['failed']} of {calls['n']} evaluations failed;"
                                       f" first:\n{calls['first_tb']}") from None
                return FAILED
            if s < calls["best"]:
                calls["best"] = s
                print(f"  [{calls['n']:>6}] score {s:8.1f}  ({time.time() - started:5.0f}s)")
            return s

        if args.from_saved:
            saved = json.loads(args.out.read_text())["parameters"]
            by_name = {p.name: p for p in PARAMS}
            missing = set(by_name) - set(saved)
            if missing:
                raise SystemExit(
                    f"{args.out.name} predates the current PARAMS list (missing {sorted(missing)}). "
                    "Refit rather than applying a stale set."
                )
            best = {by_name[n]: d["fitted"] for n, d in saved.items() if n in by_name}
            best = {p: (defaults[p] if p.name in args.hold else v) for p, v in best.items()}
        else:
            result = differential_evolution(
                objective, bounds=[(p.low, p.high) for p in free],
                maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
                polish=False, init="sobol", tol=0.01,
            )
            best = expand(result.x)
        # Scored exactly as --apply writes.
        final = terms(runner, sources, best)

    print(f"\n{calls['n']} evaluations in {time.time() - started:.0f} s\n")
    _show("start", base)
    _show("fitted", final)

    print(f"\n{'parameter':<22} {'default':>10} {'fitted':>10} {'bounds':>18}  at bound?")
    for p in PARAMS:
        span = p.high - p.low
        if p.name in args.hold:
            pinned = "  <-- HELD (not searched)"
        else:
            pinned = "  <-- PINNED" if min(best[p] - p.low, p.high - best[p]) < 0.01 * span else ""
        print(f"{p.name:<22} {defaults[p]:10.4f} {best[p]:10.4f} "
              f"{f'[{p.low}, {p.high}]':>18}{pinned}")

    if args.from_saved:
        print(f"\n(--from-saved: {args.out.name} left as written by the fitting run)")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "score": {"before": score(base), "after": score(final)},
            "terms": {"before": base, "after": final},
            "parameters": {p.name: {"file": p.file, "row": p.row, "column": p.column,
                                    "default": defaults[p], "fitted": best[p],
                                    "bounds": [p.low, p.high]} for p in PARAMS},
            "weights": WEIGHTS, "evaluations": calls["n"],
            "held_at_sourced_value": sorted(args.hold),
        }, indent=2))
        print(f"\n-> {args.out}")

    if args.apply:
        for name in {p.file for p in PARAMS}:
            text = (TXTINOUT / name).read_text()
            for p in PARAMS:
                if p.file == name:
                    text = set_value(text, name, p.row, p.column, best[p])
            (TXTINOUT / name).write_text(text)
        print(f"applied to {TXTINOUT}")
        for held in args.hold:
            print(f"  held at sourced value: {held}")
        print("re-run: uv run pytest && uv run python scripts/rebaseline.py --why '...'")


if __name__ == "__main__":
    main()
