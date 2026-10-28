#!/usr/bin/env python3
"""Benchmark RL (PPO) against CMA-ES and the model's own management on each crop model.

For every crop model in model/crops and each of its levers (I; N for annual crops):
policy quality on held-out seasons (paired $/ha vs default and vs CMA-ES, water, N,
yield, leaching; ESS-aware intervals; per-seed sign checks) and compute (seconds and
candidate-seasons per stage, engine throughput single vs batched).

    uv run python scripts/bench_rl.py                       # all crops, 3 seeds, 20k
    uv run python scripts/bench_rl.py --crops csil --levers I --seeds 1 --budget 2000
    uv run python scripts/bench_rl.py --smoke               # tiny, end to end

Finished stages (runs/bench_rl/) are reused when their config matches, so reruns resume.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swat_gym.crop import CropRunner, Plan  # noqa: E402
from swat_gym.crop_env import default_x  # noqa: E402
from swat_gym.experiments import crops as X  # noqa: E402
from swat_gym.rewarders import SGBT_DM, average  # noqa: E402
from swat_gym.spec import crop_names, load_crop  # noqa: E402
from swat_gym.windows import (CROP_TEST_YEARS, CROP_TRAIN_YEARS, CROP_WINDOW_LEN,  # noqa: E402
                              effective_n)

CAVEATS = [
    "Crop models are copied Ai-SWAT-Plus builds with that project's ET fit (pet_co, esco, "
    "irr_sw_frac); the rotation's rules 10/11b do not apply to them. Yield levels are not "
    "validated against Kimberly measurements: quote paired differences, not $/ha levels.",
    "Prices: Idaho NASS marketing-year averages 2022-2024 (final 2024); corn priced as silage "
    "from grain; sugarbeet at 25.2 % root dry matter (measured at Kimberly, synthetic-fertilizer plots, "
    "Leytem et al. 2025; manure plots 22-23 %); P from MAP net of its N at the urea price (AMS Inter-Mountain West, "
    "w/e 2025-04-04).",
    "Weather: gridMET at the field point for every model. The crop models' Ai-SWAT-Plus ET fit "
    "was made on AgriMet TWFI and is not refitted (on gridMET their PET moves +1 pp).",
    "Leaching (hru_ls percn) is unvalidated on site and is reported, not priced (no3_price 0).",
    "N2O is IPCC Tier 1 accounting on applied N and leached N, never priced.",
    "Windows: spin-up year + scored season; train seasons 1996-2014, test 2016-2025 (2015 "
    "buffer). Consecutive windows share one year, so ESS < n (test ESS "
    f"{effective_n(CROP_TEST_YEARS, CROP_WINDOW_LEN):.2f} of {len(CROP_TEST_YEARS)}).",
    "PPO envs share one weather window per round (batched engine runs); CMA-ES and PPO get "
    "equal candidate-seasons. Alfalfa runs unbatched (SWAT+ per-plant N/P demand leaks "
    "between HRUs at dormancy break).",
    "Lever plans replay default_x in the spin-up year; the default row uses the decision "
    "tables there. Decision-table irrigation windows are days of year, lever passes calendar "
    "dates (leap years: one day apart).",
    "The budget (config) is a benchmark, not a converged result: sign-inconsistent seeds are "
    "reported as unresolvable.",
]


def throughput(spec, k: int = 16, reps: int = 6) -> dict:
    """Candidate-seasons per second on one core: single-HRU runs vs one k-HRU run."""
    rng = np.random.default_rng(0)
    lever = spec.levers[0]
    x0 = tuple(default_x(spec, lever))
    plans = [Plan(lever, tuple(rng.uniform(0, 1, len(x0))), x0) for _ in range(k)]
    out = {}
    with CropRunner(spec) as one, CropRunner(spec, k=k) as many:
        one.run(plans[:1], 2000)
        t = time.perf_counter()
        for i in range(reps):
            one.run([plans[i]], 2001 + i)
        out["single_seasons_per_s"] = reps / (time.perf_counter() - t)
        many.run(plans, 2000)
        t = time.perf_counter()
        for i in range(reps):
            many.run(plans, 2001 + i)
        out[f"k{k}_seasons_per_s"] = reps * k / (time.perf_counter() - t)
        out["batched"] = many.batched
    out["speedup"] = out[f"k{k}_seasons_per_s"] / out["single_seasons_per_s"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", nargs="*", default=None)
    ap.add_argument("--levers", nargs="*", default=["I", "N"])
    ap.add_argument("--budget", type=int, default=20_000,
                    help="candidate-seasons per method (CMA-ES and each PPO seed)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=X.default_workers())
    ap.add_argument("--no3-price", type=float, default=0.0)
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, 1 seed")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench_rl.json")
    args = ap.parse_args()
    if args.smoke:
        args.budget, args.seeds = 400, 1
        args.out = args.out.with_name("bench_rl_smoke.json")
    stage_dir = args.out.with_suffix("")
    stage_dir.mkdir(parents=True, exist_ok=True)

    prices, no3 = average(), args.no3_price
    names = args.crops or crop_names()
    cases = [(n, lv) for n in names for lv in load_crop(n).levers if lv in args.levers]
    t_all = time.time()
    print(f"cases: {cases}  budget {args.budget}  seeds {args.seeds}  workers {args.workers}",
          flush=True)

    compute = {"throughput": {}, "stages": {}}
    for n in names:
        compute["throughput"][n] = throughput(load_crop(n))
        print(f"  {n}: {compute['throughput'][n]}", flush=True)

    base, fixed, ppo_jobs = {}, {}, []
    for n, lv in cases:
        spec = load_crop(n)
        t0 = time.time()
        default_test = X.score(spec, [Plan()], CROP_TEST_YEARS, prices, no3)[0]
        dx_test = X.score(spec, [X.lever_plan(spec, lv, default_x(spec, lv))],
                          CROP_TEST_YEARS, prices, no3)[0]
        base[(n, lv)] = (default_test, dx_test)
        t1 = time.time()
        fixed[(n, lv)] = X.fixed_stage(spec, lv, prices, no3, args.budget, args.seed,
                                       args.workers, stage_dir)
        compute["stages"][f"{n}_{lv}"] = {"baselines_s": t1 - t0, "cma_s": fixed[(n, lv)]["seconds"],
                                         "cma_seasons": fixed[(n, lv)]["seasons"]}
        print(f"{n} {lv}: default {X.mean(default_test):.0f}  default_x {X.mean(dx_test):.0f}"
              f"  fixed (train fit) done in {fixed[(n, lv)]['seconds']:.0f}s", flush=True)
        if not args.skip_ppo:
            cfg = X.case_config(spec, lv, prices, no3, args.budget, args.seed)
            for i in range(args.seeds):
                s = args.seed + 100 * i
                scfg = {**cfg, "ppo_seed": s, "ppo": X.CROP_PPO}
                ckpt = stage_dir / f"{n}_{lv}_ppo_s{s}_{X.config_key(scfg)}"
                ppo_jobs.append(((n, lv, prices, no3, args.budget, s, str(ckpt)), scfg))

    t_ppo = time.time()
    per_case_seed = X.ppo_stages(ppo_jobs, stage_dir, args.workers) if ppo_jobs else {}
    compute["ppo_wall_s"] = time.time() - t_ppo

    results = []
    for n, lv in cases:
        spec = load_crop(n)
        default_test, dx_test = base[(n, lv)]
        fixed_test = X.score(spec, [X.lever_plan(spec, lv, fixed[(n, lv)]["x"])],
                             CROP_TEST_YEARS, prices, no3)[0]
        seeds = {s: r for (cn, cl, s), r in per_case_seed.items() if (cn, cl) == (n, lv)}
        results.append(X.summarise(spec, lv, default_test, dx_test, fixed[(n, lv)],
                                   fixed_test, seeds))

    compute["total_wall_s"] = time.time() - t_all
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    summary = {
        "config": {"budget": args.budget, "seeds": args.seeds, "workers": args.workers,
                   "no3_price": no3, "prices": prices.key(), "sgbt_dm": SGBT_DM,
                   "train_years": CROP_TRAIN_YEARS, "test_years": CROP_TEST_YEARS,
                   "ppo": X.CROP_PPO, "git": git, "machine": platform.platform()},
        "caveats": CAVEATS, "compute": compute, "cases": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"-> {args.out}  ({compute['total_wall_s'] / 60:.1f} min)")
    for r in results:
        line = (f"{r['crop']:5s} {r['lever']}: default {r['mean_profit']['default_test']:7.0f}"
                f"  fixed {r['paired_test']['fixed_vs_default']['mean']:+6.0f}")
        if "ppo" in r:
            pol = r["ppo"]["policy_minus_default"]
            line += "  ppo-default " + "/".join(f"{v:+.0f}" for v in pol.values())
            line += f"  ppo-fixed {'/'.join(f'{v:+.0f}' for v in r['ppo']['policy_minus_fixed'].values())}"
        print(line)


if __name__ == "__main__":
    main()
