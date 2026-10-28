#!/usr/bin/env python3
"""Evolutionary arm (GA or ES) vs PPO on each crop model, at PPO's budget (candidate-seasons
per seed).

Reads the PPO benchmark (default, CMA-ES and PPO rows on test) and runs the ``--method`` arm
(``swat_gym.experiments.crops_ga``) with the same cases, seeds, prices and budget.

    uv run python scripts/bench_ga.py                         # GA vs runs/bench_rl_long.json
    uv run python scripts/bench_ga.py --method es             # -> runs/bench_es.json
    uv run python scripts/bench_ga.py --crops wwht --levers I --budget 2000 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swat_gym.experiments import crops as X  # noqa: E402
from swat_gym.experiments import crops_ga as G  # noqa: E402
from swat_gym.rewarders import average  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppo", type=Path, default=ROOT / "runs" / "bench_rl_long.json")
    ap.add_argument("--crops", nargs="*", default=None)
    ap.add_argument("--levers", nargs="*", default=["I", "N"])
    ap.add_argument("--budget", type=int, default=None, help="default: the PPO run's")
    ap.add_argument("--seeds", type=int, default=None, help="default: the PPO run's")
    ap.add_argument("--workers", type=int, default=X.default_workers())
    ap.add_argument("--method", choices=sorted(G.METHODS), default="ga")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/bench_<method>.json")
    ap.add_argument("--wall-min", type=float, default=None,
                    help="wall time to record when re-assembling already-trained runs")
    args = ap.parse_args()
    args.ppo = args.ppo.resolve()
    args.out = (args.out or ROOT / "runs" / f"bench_{args.method}.json").resolve()

    ppo = json.loads(args.ppo.read_text())
    cfg = ppo["config"]
    budget = args.budget or cfg["budget"]
    n_seeds = args.seeds or cfg["seeds"]
    prices, no3 = average(), cfg["no3_price"]
    if prices.key() != cfg["prices"]:
        raise SystemExit("prices differ from the PPO run's; the comparison would not be paired")
    cases = [c for c in ppo["cases"] if (args.crops is None or c["crop"] in args.crops)
             and c["lever"] in args.levers]
    seeds = [0 + 100 * i for i in range(n_seeds)]
    stage_dir = args.out.with_suffix("")
    ppo_dir = args.ppo.with_suffix("")
    t0 = time.time()
    print(f"{args.method} {G.METHODS[args.method][1]}  budget {budget}  seeds {seeds}  cases {len(cases)}", flush=True)
    jobs = [(c["crop"], c["lever"], prices, no3, budget, s) for c in cases for s in seeds]
    prior = json.loads(args.out.read_text()) if args.out.is_file() else {}
    res, n_ran = G.ga_stages(jobs, stage_dir, args.workers, args.method)
    # GA wall time: this run's if it trained anything, else the run that trained them.
    ga_wall = (args.wall_min * 60 if args.wall_min is not None
               else time.time() - t0 if n_ran or "wall_s" not in prior else prior["wall_s"])

    # PPO seeds' train-window profit, so both methods can pick their best run on train.
    import multiprocessing as mp
    pjobs = []
    for c in cases:
        for s in c["ppo"]["seeds"]:
            stage = ppo_dir / f"{c['crop']}_{c['lever']}_ppo_s{s}"
            ck = stage.with_name(f"{stage.name}_{X.config_key(json.loads(stage.with_suffix('.json').read_text())['config'])}")
            if not ck.is_dir():
                raise SystemExit(f"no checkpoint for {c['crop']} {c['lever']} s{s}: {ck}")
            pjobs.append((c["crop"], c["lever"], s, ck))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        tr = pool.map(X.ppo_train_profit, [(n, lv, prices, no3, str(ck)) for n, lv, _, ck in pjobs])
    ppo_train = {(n, lv, s): v for (n, lv, s, _), v in zip(pjobs, tr)}

    out_cases = []
    for c in cases:
        n, lv = c["crop"], c["lever"]
        default_test = c["per_window"]["default_test"]
        fixed_test = c["per_window"]["fixed_test"]
        ga = {s: res[(n, lv, s)] for s in seeds}
        ppo_rows = {}
        for s in c["ppo"]["seeds"]:
            st = json.loads((ppo_dir / f"{n}_{lv}_ppo_s{s}.json").read_text())["result"]
            ppo_rows[s] = st["policy_test"]
        ppo_mean = [{"start_year": default_test[i]["start_year"],
                     "profit": float(np.mean([ppo_rows[s][i]["profit"] for s in ppo_rows]))}
                    for i in range(len(default_test))]
        # Best of the runs, each method picking on train (never on test).
        pick_ppo = max(ppo_rows, key=lambda s: ppo_train[(n, lv, s)])
        pick_ga = max(ga, key=lambda s: ga[s]["picked_train_profit"])
        best = {"ppo_seed": pick_ppo, "ga_seed": pick_ga,
                "ppo_test": X.mean(ppo_rows[pick_ppo]), "ga_test": X.mean(ga[pick_ga]["policy_test"]),
                "paired_ga_vs_ppo": X.pair(ga[pick_ga]["policy_test"], ppo_rows[pick_ppo]),
                "ppo_test_best_seed": max(ppo_rows, key=lambda s: X.mean(ppo_rows[s])),
                "ga_test_best_seed": max(ga, key=lambda s: X.mean(ga[s]["policy_test"]))}
        out_cases.append({
            "best_of_runs_picked_on_train": best,
            "crop": n, "lever": lv,
            "default_test": X.mean(default_test), "fixed_test": X.mean(fixed_test),
            "ppo_test": {str(s): X.mean(r) for s, r in ppo_rows.items()},
            "ppo_train": {str(s): ppo_train[(n, lv, s)] for s in ppo_rows},
            "ga_train": {str(s): r["picked_train_profit"] for s, r in ga.items()},
            "ppo_curves": {s: t["curve"] for s, t in c["ppo"]["train"].items()},
            "ppo_train_seconds": {s: t["train_seconds"] for s, t in c["ppo"]["train"].items()},
            "ppo_seasons": {s: t["seasons"] for s, t in c["ppo"]["train"].items()},
            "ga_test": {str(s): X.mean(r["policy_test"]) for s, r in ga.items()},
            "ga_minus_default": {str(s): X.mean(r["policy_test"]) - X.mean(default_test)
                                 for s, r in ga.items()},
            "ga_minus_fixed": {str(s): X.mean(r["policy_test"]) - X.mean(fixed_test)
                               for s, r in ga.items()},
            "ga_minus_ppo_mean": {str(s): X.mean(r["policy_test"]) - X.mean(ppo_mean)
                                  for s, r in ga.items()},
            "paired_ga_vs_ppo_mean": {str(s): X.pair(r["policy_test"], ppo_mean)
                                      for s, r in ga.items()},
            "ga": {str(s): {k: r[k] for k in ("generations", "seasons", "train_seconds",
                                               "curve", "fitness", "picked_train_profit")}
                   for s, r in ga.items()},
        })
    summary = {"config": {"method": args.method, "ga": G.METHODS[args.method][1],
                          "budget": budget, "seeds": seeds, "workers": args.workers,
                          "no3_price": no3, "prices": prices.key(),
                          "ppo_source": str(args.ppo.relative_to(ROOT)),
                          "ppo_config": cfg, "genome_params": G.n_params()},
               "wall_s": ga_wall, "ppo_wall_s": ppo["compute"]["ppo_wall_s"],
               "cases": out_cases}
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"-> {args.out}  ({summary['wall_s'] / 60:.1f} min; PPO stage took "
          f"{summary['ppo_wall_s'] / 60:.1f} min)")
    for c in out_cases:
        print(f"{c['crop']:5s} {c['lever']}: ga-ppo "
              + "/".join(f"{v:+.0f}" for v in c["ga_minus_ppo_mean"].values())
              + "  ga-cma " + "/".join(f"{v:+.0f}" for v in c["ga_minus_fixed"].values()))


if __name__ == "__main__":
    main()
