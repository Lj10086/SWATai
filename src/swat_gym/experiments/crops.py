"""One (crop model, lever) case: default, default_x, fixed (CMA-ES), PPO seeds, frozen.

The crop analogue of ``common.run`` for monoculture models (``swat_gym.crop``).

Rows, all on test windows (scored seasons 2016-2025):
  default    the model's own decision tables (the shipped management; quote gains vs this)
  default_x  the tables' train-mean monthly amounts, open loop (CMA-ES and PPO start here)
  fixed      CMA-ES open-loop plan fitted on train
  policy     PPO, closed loop, per seed: its best curve-point snapshot on train
  frozen     each seed's best train-window plan (selected on train), replayed open loop

Budgets are candidate-seasons (one simulated window of one plan): CMA-ES gets
budget // n_train evaluations, each PPO seed ``budget`` env steps (the final pick
included). With batching several candidate-seasons share an engine run, so the unit is no
longer engine runs (rule 2, restated for crop models).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from ..crop import CROP_RENDER_VERSION, CropRunner, Plan, dims, profit
from ..crop_env import CropEnv, default_x, make_vec_env, rollout
from ..fastrunner import model_hash
from ..rewarders import Prices
from ..spec import CropSpec, load_crop
from ..windows import CROP_TEST_YEARS, CROP_TRAIN_YEARS, CROP_WINDOW_LEN
from .common import load_stage, minimise, paired, save_stage

#: PPO settings for crop models; part of the stage key. Envs share a window per round, so
#: a rollout of 4 episodes per env spans 4 weather years; minibatch = a quarter of it.
CROP_PPO = {"n_envs": 8, "rollout_episodes": 4, "minibatches": 4, "gamma": 1.0,
            "log_std_init": -2.0, "norm_obs": True, "clip_obs": 10.0,
            "action": "residual around default_x", "window": "shared per round",
            "final": "best of the 16 curve-point snapshots on all train windows (budgeted)"}
KEYS = ("profit", "revenue", "water_cost", "fert_cost", "p_cost", "op_cost", "yield_mg",
        "irrigation_mm", "fert_n_kg", "fert_p_kg", "n_fert_events", "no3_leached_kg",
        "n2o_kg")


def row(d: dict, start_year: int) -> dict:
    return {"start_year": start_year, **{k: float(d[k]) for k in KEYS}}


def mean(rows) -> float:
    return float(np.mean([r["profit"] for r in rows]))


def pair(a, b) -> dict:
    return paired(a, b, window_len=CROP_WINDOW_LEN)


# -- scoring, batched across plans per window --------------------------------------------

_RUNNERS: dict = {}


def runner_for(spec: CropSpec, k: int) -> CropRunner:
    """This process's runner for ``spec`` holding at least k plans per call."""
    key = (spec.name, k)
    if key not in _RUNNERS:
        _RUNNERS[key] = CropRunner(spec, k=k)
    return _RUNNERS[key]


def score(spec: CropSpec, plans: list[Plan], windows, prices: Prices,
          no3_price: float = 0.0) -> list[list[dict]]:
    """rows[plan][window]: every plan on every window, one batched run per window."""
    r = runner_for(spec, len(plans))
    rows = [[] for _ in plans]
    for sy in windows:
        for i, out in enumerate(r.run(plans, sy)):
            rows[i].append(row(profit(spec, out, prices, no3_price=no3_price), sy))
    return rows


def lever_plan(spec: CropSpec, lever: str, x) -> Plan:
    return Plan(lever, tuple(float(v) for v in np.clip(x, 0.0, 1.0)),
                tuple(default_x(spec, lever)))


def neg_profits(xs, start_year, name, lever, prices, no3_price) -> list[float]:
    """CMA-ES objective for a whole generation on one window (minimise averages windows)."""
    spec = load_crop(name)
    plans = [lever_plan(spec, lever, x) for x in xs]
    return [-r[0]["profit"] for r in score(spec, plans, [start_year], prices, no3_price)]


# -- PPO --------------------------------------------------------------------------------

#: Learning-curve points per seed (plus one before training); also the number of
#: snapshots the final pick chooses among (ES's ``curve_points``).
CURVE_POINTS = 16


def pick_cost(spec: CropSpec, lever: str) -> int:
    """Candidate-seasons of the final pick: every snapshot on every train window."""
    return CURVE_POINTS * len(CROP_TRAIN_YEARS) * dims(spec, lever)


def _curve_callback(spec, lever, prices, no3_price, budget, vn, snap_dir: Path):
    """Learning curve: deterministic policy on every test window at evenly spaced timesteps,
    plus the mean profit of the training episodes finished since the last point. The test
    scores are diagnostic (nothing selects on them; not budgeted). The policy at the first
    15 points after the start and at the end of training is saved to ``snap_dir`` for the
    final pick, which is on train."""
    from stable_baselines3.common.callbacks import BaseCallback

    every = max(1, budget // CURVE_POINTS)

    class Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.points, self._train, self._next, self.snaps = [], [], 0, []

        def _snap(self):
            i = len(self.snaps)
            self.model.save(str(snap_dir / f"{i}.zip"))
            vn.save(str(snap_dir / f"{i}.pkl"))
            self.snaps.append(int(self.num_timesteps))

        def _point(self):
            act = policy_fn(self.model, vn)
            with CropEnv(spec, lever, prices=prices, no3_price=no3_price) as env:
                test = [rollout(env, act, sy)["profit"] for sy in CROP_TEST_YEARS]
            self.points.append({"timesteps": int(self.num_timesteps),
                                "test_profit": float(np.mean(test)),
                                "train_episode_profit": (float(np.mean(self._train))
                                                         if self._train else None),
                                "train_episodes": len(self._train)})
            self._train = []
            self._next += every
            if self.num_timesteps > 0 and len(self.snaps) < CURVE_POINTS - 1:
                self._snap()

        def _on_training_start(self):
            snap_dir.mkdir(parents=True, exist_ok=True)
            self._point()

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self._train.append(info["profit"])
            if self.num_timesteps >= self._next:
                self._point()
            return True

        def _on_training_end(self):
            if self.points[-1]["timesteps"] != self.num_timesteps:
                self._point()
            if not self.snaps or self.snaps[-1] != self.num_timesteps:
                self._snap()

    return Curve()


def load_policy(ckpt: Path):
    """(model, VecNormalize) of a seed's deployed policy: its picked snapshot, or the final
    policy for checkpoints trained before the pick existed."""
    from stable_baselines3 import PPO

    stats = json.loads((ckpt / "stats.json").read_text())
    if "picked" in stats:
        model, norm = ckpt / "snaps" / f"{stats['picked']}.zip", ckpt / "snaps" / f"{stats['picked']}.pkl"
    else:
        model, norm = ckpt / "final.zip", ckpt / "vecnormalize.pkl"
    return PPO.load(str(model)), pickle.loads(norm.read_bytes())


def pick_snapshot(spec: CropSpec, lever: str, prices: Prices, no3_price: float,
                  snap_dir: Path, n_snaps: int) -> dict:
    """Every snapshot's deterministic policy on every train window; the best mean is kept."""
    from stable_baselines3 import PPO

    train, runner = [], CropRunner(spec)
    try:
        with CropEnv(spec, lever, prices=prices, no3_price=no3_price, runner=runner) as env:
            for i in range(n_snaps):
                act = policy_fn(PPO.load(str(snap_dir / f"{i}.zip")),
                                pickle.loads((snap_dir / f"{i}.pkl").read_bytes()))
                train.append(mean([rollout(env, act, sy) for sy in CROP_TRAIN_YEARS]))
        seasons = int(runner.n_seasons)
    finally:
        runner.close()
    best = int(np.argmax(train))
    return {"picked": best, "picked_train_profit": train[best], "snap_train_profit": train,
            "pick_seasons": seasons}


def train_ppo(spec: CropSpec, lever: str, prices: Prices, no3_price: float, budget: int,
              seed: int, ckpt: Path):
    """Train one seed on the vec env for ``budget`` minus the final pick's cost, then pick
    the best curve-point snapshot on train (as ES does); returns (model, VecNormalize,
    stats incl. curve) of the picked policy."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)
    final, norm = ckpt / "final.zip", ckpt / "vecnormalize.pkl"
    stats_path = ckpt / "stats.json"
    if final.is_file() and norm.is_file() and stats_path.is_file():
        return (*load_policy(ckpt), json.loads(stats_path.read_text()))
    hp = CROP_PPO
    ep = dims(spec, lever)
    venv = make_vec_env(spec, lever, prices=prices, n_envs=hp["n_envs"], seed=seed,
                        no3_price=no3_price)
    vn = VecNormalize(venv, norm_obs=hp["norm_obs"], norm_reward=False, clip_obs=hp["clip_obs"])
    n_steps = hp["rollout_episodes"] * ep
    model = PPO("MlpPolicy", vn, seed=seed, verbose=0, n_steps=n_steps,
                batch_size=n_steps * hp["n_envs"] // hp["minibatches"], gamma=hp["gamma"],
                policy_kwargs={"log_std_init": hp["log_std_init"]})
    t0 = time.time()
    try:
        # SB3 rounds up to whole rollouts; tiny (smoke) budgets get one rollout.
        train_budget = max(n_steps * hp["n_envs"], budget - pick_cost(spec, lever))
        curve = _curve_callback(spec, lever, prices, no3_price, train_budget, vn, ckpt / "snaps")
        model.learn(total_timesteps=train_budget, callback=curve)
        stats = {"timesteps": int(model.num_timesteps), "train_seconds": time.time() - t0,
                 **venv.runner_stats(), "curve": curve.points, "snap_timesteps": curve.snaps}
        model.save(str(final))
        vn.save(str(norm))
    finally:
        vn.close()
    t1 = time.time()
    stats.update(pick_snapshot(spec, lever, prices, no3_price, ckpt / "snaps", len(curve.snaps)),
                 pick_seconds=time.time() - t1)
    stats["train_seasons"], stats["seasons"] = stats["seasons"], stats["seasons"] + stats["pick_seasons"]
    stats_path.write_text(json.dumps(stats))
    return (*load_policy(ckpt), stats)


def policy_fn(model, vecnorm):
    def act(env: CropEnv, obs):
        a, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        return env.absolute(a)
    return act


def ppo_train_profit(args) -> float:
    """Pool job: a trained seed's deployed policy's mean profit on every train window (for
    picking among seeds on train, never on test). Loads the checkpoint; no training."""
    name, lever, prices, no3, ckpt = args
    spec, ckpt = load_crop(name), Path(ckpt)
    stats = json.loads((ckpt / "stats.json").read_text())
    if "picked_train_profit" in stats:  # the final pick already scored it on every train window
        return stats["picked_train_profit"]
    model, vn = load_policy(ckpt)
    with CropEnv(spec, lever, prices=prices, no3_price=no3) as env:
        return mean([rollout(env, policy_fn(model, vn), sy) for sy in CROP_TRAIN_YEARS])


def ppo_seed(args) -> dict:
    """Pool job: train one seed, roll it out, pick and replay its frozen plan."""
    name, lever, prices, no3, budget, seed, ckpt = args
    spec = load_crop(name)
    t0 = time.time()
    model, vn, stats = train_ppo(spec, lever, prices, no3, budget, seed, Path(ckpt))
    act = policy_fn(model, vn)
    with CropEnv(spec, lever, prices=prices, no3_price=no3) as env:
        pol_test = [rollout(env, act, sy) for sy in CROP_TEST_YEARS]
        pol_train = [rollout(env, act, sy) for sy in CROP_TRAIN_YEARS]
    t_eval = time.time()
    # Frozen: each train rollout's plan scored on all train windows; best replayed on test.
    cands = [lever_plan(spec, lever, r["x"]) for r in pol_train]
    on_train = score(spec, cands, CROP_TRAIN_YEARS, prices, no3)
    best = int(np.argmax([mean(r) for r in on_train]))
    frozen_test = score(spec, [cands[best]], CROP_TEST_YEARS, prices, no3)[0]
    return {"seed": seed, "policy_test": [row(r, sy) for r, sy in zip(pol_test, CROP_TEST_YEARS)],
            "frozen_test": frozen_test, "frozen_from": CROP_TRAIN_YEARS[best],
            "frozen_x": list(cands[best].x), "train": stats,
            "eval_seasons": len(CROP_TEST_YEARS + CROP_TRAIN_YEARS) * dims(spec, lever)
            + len(cands) * len(CROP_TRAIN_YEARS) + len(CROP_TEST_YEARS),
            "seconds": {"train": stats["train_seconds"], "pick": stats.get("pick_seconds", 0.0),
                        "rollouts": t_eval - t0 - stats["train_seconds"]
                        - stats.get("pick_seconds", 0.0), "frozen": time.time() - t_eval}}


# -- the case ---------------------------------------------------------------------------

def case_config(spec: CropSpec, lever: str, prices: Prices, no3: float, budget: int,
                seed: int) -> dict:
    return {"crop": spec.name, "lever": lever, "budget": budget, "seed": seed,
            "no3_price": no3, "prices": prices.key(), "model": model_hash(spec.txtinout),
            "render": CROP_RENDER_VERSION[lever], "train": CROP_TRAIN_YEARS,
            "test": CROP_TEST_YEARS}


def fixed_stage(spec, lever, prices, no3, budget, seed, workers, out_dir: Path) -> dict:
    cfg = case_config(spec, lever, prices, no3, budget, seed)
    path = out_dir / f"{spec.name}_{lever}_fixed.json"
    done = load_stage(path, cfg)
    if done is not None:
        return done
    import functools
    t0 = time.time()
    fn = functools.partial(neg_profits, name=spec.name, lever=lever, prices=prices,
                           no3_price=no3)
    x, n_evals, history = minimise(fn, default_x(spec, lever),
                                   evals=max(1, budget // len(CROP_TRAIN_YEARS)),
                                   windows=CROP_TRAIN_YEARS, batched=True, seed=seed,
                                   workers=workers, label=f"{spec.name} {lever} fixed")
    res = {"x": list(map(float, x)), "n_evals": n_evals, "history": history,
           "seasons": n_evals * len(CROP_TRAIN_YEARS), "seconds": time.time() - t0}
    save_stage(path, cfg, res)
    return res


def ppo_stages(jobs: list[tuple], out_dir: Path, workers: int) -> dict:
    """Run every (case, seed) PPO job not already on disk, ``workers`` at a time."""
    import multiprocessing as mp

    results, todo = {}, []
    for job, cfg in jobs:
        name, lever, *_, seed, _ = job
        path = out_dir / f"{name}_{lever}_ppo_s{seed}.json"
        done = load_stage(path, cfg)
        if done is None:
            todo.append((job, cfg, path))
        else:
            results[(name, lever, seed)] = done
    if todo:
        with mp.get_context("spawn").Pool(min(workers, len(todo)), maxtasksperchild=1) as pool:
            for (job, cfg, path), res in zip(todo, pool.imap(ppo_seed, [t[0] for t in todo])):
                save_stage(path, cfg, res)
                results[(job[0], job[1], job[5])] = res
                print(f"  ppo {job[0]} {job[1]} seed {job[5]}: policy {mean(res['policy_test']):.0f}"
                      f" frozen {mean(res['frozen_test']):.0f} $/ha", flush=True)
    return results


def summarise(spec, lever, default_test, dx_test, fixed, fixed_test, per_seed) -> dict:
    out = {
        "crop": spec.name, "plant": spec.plant, "lever": lever, "batched": spec.batchable,
        "dims": dims(spec, lever),
        "mean_profit": {"default_test": mean(default_test), "default_x_test": mean(dx_test),
                        "fixed_test": mean(fixed_test)},
        "paired_test": {"default_x_vs_default": pair(dx_test, default_test),
                        "fixed_vs_default": pair(fixed_test, default_test)},
        "means_test": {k: {r: float(np.mean([w[k] for w in rows])) for r, rows in
                           (("default", default_test), ("default_x", dx_test),
                            ("fixed", fixed_test))}
                       for k in ("irrigation_mm", "fert_n_kg", "yield_mg", "no3_leached_kg")},
        "fixed_x": fixed["x"], "cma": {k: fixed[k] for k in ("n_evals", "seasons", "seconds")},
        "per_window": {"default_test": default_test, "default_x_test": dx_test,
                       "fixed_test": fixed_test},
    }
    if per_seed:
        seeds = sorted(per_seed)
        adv = {str(s): mean(per_seed[s]["policy_test"]) - mean(fixed_test) for s in seeds}
        adapt = {str(s): mean(per_seed[s]["policy_test"]) - mean(per_seed[s]["frozen_test"])
                 for s in seeds}
        vs_default = {str(s): mean(per_seed[s]["policy_test"]) - mean(default_test)
                      for s in seeds}

        def consistent(d):
            signs = {np.sign(v) for v in d.values()}
            return len(signs) == 1

        out["ppo"] = {
            "seeds": seeds,
            "policy_test": {str(s): mean(per_seed[s]["policy_test"]) for s in seeds},
            "frozen_test": {str(s): mean(per_seed[s]["frozen_test"]) for s in seeds},
            "policy_minus_default": vs_default, "policy_minus_fixed": adv,
            "adaptivity_value": adapt,
            "sign_consistent": {"policy_minus_default": consistent(vs_default),
                                "policy_minus_fixed": consistent(adv),
                                "adaptivity_value": consistent(adapt)},
            "paired_test_by_seed": {str(s): {
                "policy_vs_fixed": pair(per_seed[s]["policy_test"], fixed_test),
                "policy_vs_default": pair(per_seed[s]["policy_test"], default_test),
                "frozen_vs_policy": pair(per_seed[s]["frozen_test"], per_seed[s]["policy_test"])}
                for s in seeds},
            "train": {str(s): per_seed[s]["train"] for s in seeds},
            "seconds": {str(s): per_seed[s]["seconds"] for s in seeds},
        }
    return out


def default_workers() -> int:
    return os.cpu_count() or 1


def config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
