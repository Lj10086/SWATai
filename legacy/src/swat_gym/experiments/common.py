"""Shared experiment machinery: stats, parallel CMA-ES, PPO, and the lever experiment.

Rows per lever: measured (shipped schedule replayed), default (measured practice via the
plan generator; quote gains vs this), fixed (CMA-ES open-loop), policy (PPO), frozen (the
policy's train-selected plan replayed open-loop). Budgets are matched in engine runs.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from ..env import ENV_VERSION, SwatEnv, rollout
from ..fastrunner import FastRunner, model_hash
from ..plan import N_YEARS, RENDER_VERSION, SPINUP, default_x, evaluate, time_sim
from ..rewarders import Prices, average, profit
from ..windows import TEST_YEARS, TRAIN_YEARS, WINDOW_LEN, effective_n

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"


# -- stats ----------------------------------------------------------------------------

def row(d: dict, start_year: int) -> dict:
    """One window's outcome with separable cost terms."""
    keys = ("profit", "revenue", "water_cost", "manure_cost", "fert_cost", "op_cost",
            "n_fert_events", "irrigation_mm", "manure_mg", "fert_n_kg", "n2o_kg")
    return {"start_year": start_year, "no3": d["no3_leached_kg"], **{k: d[k] for k in keys}}


def mean(rows) -> float:
    return float(np.mean([r["profit"] for r in rows]))


def paired(a: list[dict], b: list[dict], *, n_boot: int = 10_000, seed: int = 0,
           window_len: int = WINDOW_LEN) -> dict:
    """Paired per-window difference a - b. Quote se_ess (windows overlap; rule 7).

    No ``se`` key on purpose; se_naive is kept only to show the ratio. ``ci95_boot`` is a
    moving-block bootstrap over windows in start-year order, block length round(n / ESS)
    (iid when windows don't overlap).
    """
    if [r["start_year"] for r in a] != [r["start_year"] for r in b]:
        raise ValueError("paired() needs the same windows in the same order")
    d = np.array([r["profit"] for r in a]) - np.array([r["profit"] for r in b])
    n = int(d.size)
    ess = effective_n([r["start_year"] for r in a], window_len) if n > 1 else float(n)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se_naive = sd / np.sqrt(n) if n > 1 else float("nan")
    se_ess = sd / np.sqrt(ess) if n > 1 else float("nan")
    block = max(1, min(n, int(round(n / ess)))) if n > 1 else 1
    if n > 1:
        ordered = d[np.argsort([r["start_year"] for r in a], kind="stable")]
        starts = np.random.default_rng(seed).integers(0, n - block + 1,
                                                      size=(n_boot, -(-n // block)))
        idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
        means = ordered[idx].mean(axis=1)
        ci_boot = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        ci_ess = [float(d.mean() - 1.96 * se_ess), float(d.mean() + 1.96 * se_ess)]
    else:
        ci_boot = ci_ess = [float("nan"), float("nan")]
    return {"mean": float(d.mean()), "n": n, "ess": float(ess), "sd": sd,
            "se_naive": float(se_naive), "se_ess": float(se_ess),
            "ci95_boot": ci_boot, "boot_block": block, "ci95_ess": ci_ess,
            "resolvable": bool(n > 1 and ci_ess[0] * ci_ess[1] > 0),
            "per_window": [round(float(v), 1) for v in d]}


# -- engine access, per process ---------------------------------------------------------

_RUNNER: FastRunner | None = None


def runner() -> FastRunner:
    """This process's FastRunner (one per worker)."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = FastRunner()
    return _RUNNER


def score(lever: str, x, windows, prices: Prices, no3_price: float) -> list[dict]:
    return [row(evaluate(lever, x, runner(), prices=prices, start_year=sy,
                         no3_price=no3_price), sy) for sy in windows]


def score_measured(windows, prices: Prices, no3_price: float) -> list[dict]:
    """Shipped management.sch replayed on each window (only time.sim changes)."""
    out = []
    for sy in windows:
        runner().run({"time.sim": time_sim(sy, N_YEARS + SPINUP)})
        out.append(row(profit(runner(), prices, no3_price=no3_price), sy))
    return out


def neg_profit(x, start_year, lever, prices, no3_price) -> float:
    """CMA-ES objective on one window: -profit of an open-loop plan (minimise averages)."""
    return -evaluate(lever, x, runner(), prices=prices, start_year=start_year,
                     no3_price=no3_price)["profit"]


# -- CMA-ES ---------------------------------------------------------------------------

#: Abort a search once this share of evaluations has raised (and at least MIN_FAILURES).
MAX_FAILURE_FRAC = 0.01
MIN_FAILURES = 3
#: Objective assigned to a failed evaluation so CMA-ES ranks it last.
FAILED = 1e30


def _guarded(task):
    """Pool task: (fn, x, window) -> (value, traceback or None)."""
    fn, x, w = task
    try:
        return float(fn(x) if w is None else fn(x, w)), None
    except Exception:
        return FAILED, traceback.format_exc()


def _guarded_batch(task):
    """Pool task: (fn, xs, window) -> ([values], traceback or None); one call per window."""
    fn, xs, w = task
    try:
        return [float(v) for v in fn(xs, w)], None
    except Exception:
        return [FAILED] * len(xs), traceback.format_exc()


def default_workers() -> int:
    return os.cpu_count() or 1


def minimise(fn, x0, *, evals: int, windows=None, batched: bool = False, seed: int = 0,
             workers: int = 1, sigma0: float = 0.25, label: str = "CMA-ES"):
    """Minimise a picklable objective over [0, 1]^n, evaluating each generation in parallel.

    ``fn(x)``, or with ``windows`` ``fn(x, window)`` averaged over windows; each
    (candidate, window) pair is its own pool task so all workers stay busy. ``batched``:
    ``fn(xs, window) -> [values]`` scores the whole generation on one window per task (crop
    models run it as one multi-HRU engine run). Evaluations that raise are counted and
    ranked last; past MAX_FAILURE_FRAC the search aborts with the first traceback.
    Returns (best_x, n_evals, history of {evals, best_f, failed}). Deterministic given seed.
    """
    if batched and windows is None:
        raise ValueError("batched evaluation needs windows")
    import warnings
    warnings.filterwarnings("ignore", message="Could not import matplotlib")
    import cma

    x0 = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, {"bounds": [0.0, 1.0], "maxfevals": evals,
                                                     "seed": seed + 1, "verbose": -9})
    ws = [None] if windows is None else list(windows)
    best_x, best_f, history, failed, first_tb = x0, float("inf"), [], 0, None
    pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
    t0 = time.time()
    try:
        while not es.stop():
            xs = es.ask()
            if batched:
                tasks = [(fn, [np.asarray(x) for x in xs], w) for w in ws]
                out = pool.map(_guarded_batch, tasks) if pool else list(map(_guarded_batch, tasks))
                # Re-shape to the per-(candidate, window) layout below.
                res = [(vals[i], tb) for i in range(len(xs)) for vals, tb in out]
            else:
                tasks = [(fn, np.asarray(x), w) for x in xs for w in ws]
                if pool:
                    res = pool.map(_guarded, tasks,
                                   chunksize=max(1, len(tasks) // (4 * workers)))
                else:
                    res = list(map(_guarded, tasks))
            fs = []
            for i in range(len(xs)):
                part = res[i * len(ws):(i + 1) * len(ws)]
                bad = [tb for _, tb in part if tb is not None]
                if bad:
                    failed += 1
                    first_tb = first_tb or bad[0]
                    fs.append(FAILED)
                else:
                    fs.append(float(np.mean([v for v, _ in part])))
            if failed >= max(MIN_FAILURES, MAX_FAILURE_FRAC * evals):
                raise RuntimeError(f"{label}: {failed} of {es.countevals + len(xs)} evaluations "
                                   f"raised; first:\n{first_tb}")
            for x, f in zip(xs, fs):
                if f < best_f:
                    best_f, best_x = f, np.asarray(x).copy()
            es.tell(xs, fs)
            history.append({"evals": int(es.countevals), "best_f": best_f, "failed": failed})
            print(f"  {label}: {es.countevals}/{evals} evals  best objective {best_f:.6g}  "
                  f"failed {failed}  {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
    if failed:
        print(f"  {label}: {failed} failed evaluation(s); first:\n{first_tb}", flush=True)
    return best_x, int(es.countevals), history


# -- stage reuse ------------------------------------------------------------------------

def stage_config(**kw) -> dict:
    """Config a finished stage must match to be reused: inputs, engine, rendering, env."""
    return {**kw, "model": model_hash(), "render": RENDER_VERSION, "env": ENV_VERSION}


def load_stage(path: Path, cfg: dict):
    """Stage result if ``path`` exists and was produced with the same config."""
    if path.is_file():
        d = json.loads(path.read_text())
        if d.get("config") == cfg:
            print(f"  reusing {path.name}", flush=True)
            return d["result"]
        print(f"  {path.name} has a different config; recomputing", flush=True)
    return None


def save_stage(path: Path, cfg: dict, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": cfg, "result": result}, indent=2))


# -- PPO ------------------------------------------------------------------------------

#: Everything that shapes training; part of the stage and checkpoint key (resuming a
#: checkpoint trained under other settings would silently mix them). Rollout and minibatch
#: sizes are in episodes per env.
PPO_HPARAMS = {"rollout_episodes": 4, "batch_episodes": 2, "gamma": 1.0, "log_std_init": -2.0,
               "norm_obs": True, "norm_reward": False, "clip_obs": 10.0,
               "action": "residual around default_x"}


def _make_env(lever, prices, no3_price, seed):
    return SwatEnv(lever, prices=prices, no3_price=no3_price, seed=seed).to_gym()


def train_ppo(lever: str, prices: Prices, no3_price: float, budget: int, seed: int,
              n_envs: int, ckpt_dir: Path):
    """Train (or resume) one PPO policy; returns (model, VecNormalize stats, engine stats)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(1)  # the MLP is tiny; threads only contend with the engines
    final, stats = ckpt_dir / "final.zip", ckpt_dir / "vecnormalize.pkl"
    runs = ckpt_dir / "engine_runs.json"
    if final.is_file() and stats.is_file():
        print(f"  seed {seed}: already trained", flush=True)
        eng = json.loads(runs.read_text()) if runs.is_file() else {}
        return PPO.load(str(final)), pickle.loads(stats.read_bytes()), eng

    hp = PPO_HPARAMS
    venv = SubprocVecEnv([functools.partial(_make_env, lever, prices, no3_price, seed + i)
                          for i in range(n_envs)])
    ckpts = sorted(ckpt_dir.glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    try:
        if ckpts:
            steps = ckpts[-1].stem.split("_")[1]
            venv = VecNormalize.load(str(ckpt_dir / f"ppo_vecnormalize_{steps}_steps.pkl"),
                                     venv)
            model = PPO.load(str(ckpts[-1]), env=venv)
            print(f"  seed {seed}: resuming at {model.num_timesteps} steps", flush=True)
        else:
            venv = VecNormalize(venv, norm_obs=hp["norm_obs"], norm_reward=hp["norm_reward"],
                                clip_obs=hp["clip_obs"])
            ep = len(SwatEnv.steps_for(lever))
            model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                        n_steps=hp["rollout_episodes"] * ep, batch_size=hp["batch_episodes"] * ep,
                        gamma=hp["gamma"], policy_kwargs={"log_std_init": hp["log_std_init"]})
        remaining = budget - int(model.num_timesteps)
        if remaining > 0:
            cb = CheckpointCallback(save_freq=max(1, budget // (20 * n_envs)),
                                    save_path=str(ckpt_dir), name_prefix="ppo",
                                    save_vecnormalize=True)
            model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=cb)
        per_env = venv.env_method("runner_stats")
        eng = {"timesteps": int(model.num_timesteps),
               "engine_runs_this_session": sum(e["n_runs"] for e in per_env),
               "skipped_identical_this_session": sum(e["n_skipped"] for e in per_env)}
        model.save(str(final))
        venv.save(str(stats))
        runs.write_text(json.dumps(eng))
        return model, pickle.loads(stats.read_bytes()), eng
    finally:
        venv.close()


def _train_seed(args) -> None:
    """Process entry for one seed (spawned, not a daemonic Pool worker: it has children)."""
    train_ppo(*args)


def train_seeds(jobs: list[tuple]) -> None:
    """Train several seeds at once, one process each; train_ppo then loads the results."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_train_seed, args=(job,)) for job in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [job[4] for job, p in zip(jobs, procs) if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"PPO training failed for seed(s) {bad}; see the log above")


def policy_fn(model, vecnorm):
    """Deterministic policy for SwatEnv.step, mapped from the gym's residual actions."""
    def act(env, obs):
        action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        return env.absolute(action)
    return act


# -- the lever experiment -------------------------------------------------------------

def run(lever: str, out: Path, argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300_000,
                    help="engine runs per method (CMA-ES and each PPO seed)")
    ap.add_argument("--ppo-seeds", type=int, default=3)
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel processes for CMA-ES")
    ap.add_argument("--serial-seeds", action="store_true",
                    help="train PPO seeds one after another (default: all at once)")
    ap.add_argument("--no3-price", type=float, default=0.0,
                    help="$/kg N leached; one frontier point (report the whole sweep)")
    ap.add_argument("--water-price", type=float, default=None)
    ap.add_argument("--out", type=Path, default=out)
    args = ap.parse_args(argv)

    prices = average()
    if args.water_price is not None:
        from dataclasses import replace
        prices = replace(prices, water=args.water_price,
                         label=f"{prices.label}@w={args.water_price:g}")
    train, test, no3 = list(TRAIN_YEARS), list(TEST_YEARS), args.no3_price
    cfg = stage_config(lever=lever, budget=args.budget, seed=args.seed, no3_price=no3,
                       prices=prices.key())
    t0 = time.time()

    measured_test = score_measured(test, prices, no3)
    default_test = score(lever, default_x(lever), test, prices, no3)
    print(f"measured {mean(measured_test):.0f}  default {mean(default_test):.0f} $/ha",
          flush=True)

    fixed_path = args.out.with_name(f"{args.out.stem}_fixed.json")
    fixed = load_stage(fixed_path, cfg)
    if fixed is None:
        fn = functools.partial(neg_profit, lever=lever, prices=prices, no3_price=no3)
        x, n_evals, history = minimise(fn, default_x(lever), evals=args.budget // len(train),
                                       windows=train, seed=args.seed, workers=args.workers,
                                       label="fixed")
        fixed = {"x": list(map(float, x)), "n_evals": n_evals, "history": history,
                 "engine_runs": n_evals * len(train)}
        save_stage(fixed_path, cfg, fixed)
    fixed_test = score(lever, fixed["x"], test, prices, no3)
    fixed_train = score(lever, fixed["x"], train, prices, no3)
    print(f"fixed {mean(fixed_test):.0f} $/ha", flush=True)

    summary = {
        "lever": lever, "config": cfg, "train_years": train, "test_years": test,
        "fixed_x": fixed["x"], "cma_evals": fixed["n_evals"], "cma_history": fixed["history"],
        "engine_runs": {"fixed": fixed["n_evals"] * len(train)},
        "mean_profit": {"measured_test": mean(measured_test),
                        "default_test": mean(default_test),
                        "fixed_train": mean(fixed_train), "fixed_test": mean(fixed_test)},
        "paired_test": {"default_vs_measured": paired(default_test, measured_test),
                        "fixed_vs_default": paired(fixed_test, default_test)},
        "per_window": {"measured_test": measured_test, "default_test": default_test,
                       "fixed_test": fixed_test},
    }

    if not args.skip_ppo:
        seeds = [args.seed + 100 * i for i in range(args.ppo_seeds)]

        def seed_cfg(s):
            return {**cfg, "ppo_seed": s, "ppo": {**PPO_HPARAMS, "n_envs": args.n_envs}}

        def ckpt_dir(s):
            key = hashlib.sha256(json.dumps(seed_cfg(s), sort_keys=True).encode()).hexdigest()
            return args.out.parent / f"{args.out.stem}_ppo_s{s}_{key[:10]}"

        def seed_path(s):
            return args.out.with_name(f"{args.out.stem}_ppo_s{s}.json")

        todo = [s for s in seeds if load_stage(seed_path(s), seed_cfg(s)) is None]
        if len(todo) > 1 and not args.serial_seeds:
            train_seeds([(lever, prices, no3, args.budget, s, args.n_envs, ckpt_dir(s))
                         for s in todo])
        per_seed = {}
        for s in seeds:
            res = load_stage(seed_path(s), seed_cfg(s))
            if res is None:
                model, vn, eng = train_ppo(lever, prices, no3, args.budget, s, args.n_envs,
                                           ckpt_dir(s))
                act = policy_fn(model, vn)
                with SwatEnv(lever, prices=prices, no3_price=no3) as env:
                    pol_test = [rollout(env, act, sy) for sy in test]
                    pol_train = [rollout(env, act, sy) for sy in train]
                # Frozen: pick the policy's best train-window plan on train, replay on test.
                cand = {sy: mean(score(lever, r["x"], train, prices, no3))
                        for sy, r in zip(train, pol_train)}
                best = max(cand, key=cand.get)
                frozen_x = pol_train[train.index(best)]["x"]
                res = {"policy_test": [row(r, sy) for r, sy in zip(pol_test, test)],
                       "frozen_test": score(lever, frozen_x, test, prices, no3),
                       "frozen_from": best, "frozen_x": list(map(float, frozen_x)),
                       "engine_runs": eng}
                save_stage(seed_path(s), seed_cfg(s), res)
            per_seed[s] = res
            print(f"seed {s}: policy {mean(res['policy_test']):.0f}  "
                  f"frozen {mean(res['frozen_test']):.0f} $/ha", flush=True)

        def by_seed(f):
            return {str(s): f(per_seed[s]) for s in seeds}

        adv = by_seed(lambda r: mean(r["policy_test"]) - mean(fixed_test))
        adapt = by_seed(lambda r: mean(r["policy_test"]) - mean(r["frozen_test"]))
        rep = sorted(seeds, key=lambda s: mean(per_seed[s]["policy_test"]))[len(seeds) // 2]
        r = per_seed[rep]
        summary["mean_profit"].update(policy_test=mean(r["policy_test"]),
                                      frozen_test=mean(r["frozen_test"]))
        summary["paired_test"].update(policy_vs_fixed=paired(r["policy_test"], fixed_test),
                                      frozen_vs_policy=paired(r["frozen_test"],
                                                              r["policy_test"]))
        summary["engine_runs"]["ppo"] = {str(s): per_seed[s].get("engine_runs")
                                         for s in seeds}
        summary.update(ppo_seeds=seeds, representative_seed=rep,
                       advantage_over_fixed=adv, adaptivity_value=adapt,
                       sign_consistent={"advantage_over_fixed": len({v > 0 for v in adv.values()}) == 1,
                                        "adaptivity_value": len({v > 0 for v in adapt.values()}) == 1})

    summary["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"-> {args.out}")
    return summary
