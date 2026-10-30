"""Evolutionary arms for the crop-model policies, compared with PPO: a GA and an ES.

GA

A population of policy networks with PPO's actor architecture (9 -> 64 tanh -> 64 tanh -> 1,
the output a residual around ``default_x``) is evolved by mutation and selection, as in Such
et al. (2017), "Deep Neuroevolution", arXiv:1712.06567: no gradients, no crossover.

Selection is rank-based with a smooth kill gradient: after each generation the top quarter
(ranks < P/4) is never killed, and the kill probability then rises linearly to 1 at the worst
rank. Each killed slot is refilled by a mutated copy (Gaussian, sigma on every weight) of a
survivor drawn with probability proportional to P - rank. Survivors are kept unmutated.

Fitness is a running average. Every generation all individuals play the same
``GA["windows"]`` train windows (common random numbers); an individual's fitness is the mean,
over every season it has played since it was born, of its profit minus the default_x plan's
profit on that season. Subtracting the fixed per-season baseline removes the weather level, so
a survivor's long record and a newborn's 3 seasons are on one scale; the record makes a proven
policy hard to kill on one unlucky draw. (v1 used the current generation's 3 seasons only and
kill p = rank/(P-1): the truly best policy was protected ~1 time in 5 on irrigation.)

The budget counts every candidate-season simulated for fitness: the baseline, every
generation, and the final pick (the population re-scored on all train windows, best one kept),
the unit PPO's budget is in. Learning-curve points (the top ``GA["curve_top"]`` by fitness on
the test windows) are diagnostic only and not budgeted, as for PPO.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..crop import CROP_RENDER_VERSION, CropRunner, dims
from ..crop_env import OBS_DIM, _Episode, default_x
from ..fastrunner import model_hash
from ..rewarders import Prices
from ..spec import CropSpec, load_crop
from ..windows import CROP_TEST_YEARS, CROP_TRAIN_YEARS
from .common import load_stage, save_stage
from .crops import row

#: GA settings; part of the stage key. v1: kill p = rank/(P-1), fitness on the current 3
#: seasons. v2: top P/4 safe + running record (lower selection pressure; newborns still led).
#: v3: v1's kill gradient + running record + re-evaluation (racing) of the top ``race_top``.
GA = {"version": 3, "population": 16, "windows": 3, "sigma": 0.02, "hidden": [64, 64],
      "selection": "kill p = rank/(P-1); parents ~ P-rank",
      "fitness": "running mean over life of (profit - default_x profit) per season",
      "race_top": 4, "race_seasons": 12,
      "init": "orthogonal, gain sqrt(2) hidden / 0.01 output (SB3 actor)",
      "curve_points": 16, "curve_top": 4}

#: ES settings (Salimans et al. 2017, arXiv:1703.03864); part of the stage key.
ES = {"version": 1, "population": 16, "windows": 3, "sigma": 0.02, "hidden": GA["hidden"],
      "lr": 0.01, "adam": [0.9, 0.999], "l2": 0.005, "mirrored": True,
      "shaping": "centered ranks in [-0.5, 0.5]",
      "final": "best of the curve-point centres on all train windows",
      "init": GA["init"], "curve_points": 16}


def kill_probability(rank: np.ndarray, P: int, safe: int = 0) -> np.ndarray:
    """0 for ranks <= ``safe``, then linear up to 1 at rank P - 1 (rank 0 is never killed)."""
    return np.clip((rank - safe) / (P - 1 - safe), 0.0, 1.0)


# -- the policy network, batched over the population -------------------------------------

def _orthogonal(rng, rows: int, cols: int, gain: float) -> np.ndarray:
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    q = q if rows >= cols else q.T
    return gain * q[:rows, :cols]


def layer_shapes() -> list[tuple[int, int]]:
    sizes = [OBS_DIM, *GA["hidden"], 1]
    return list(zip(sizes[:-1], sizes[1:]))


def n_params() -> int:
    return sum(i * o + o for i, o in layer_shapes())


def init_genome(rng) -> np.ndarray:
    parts = []
    shapes = layer_shapes()
    for k, (i, o) in enumerate(shapes):
        gain = 0.01 if k == len(shapes) - 1 else np.sqrt(2.0)
        parts += [_orthogonal(rng, i, o, gain).ravel(), np.zeros(o)]
    return np.concatenate(parts)


def act(genomes: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Residual actions in [-1, 1] for P genomes on P observations (one each)."""
    h, at = obs.astype(float), 0
    shapes = layer_shapes()
    for k, (i, o) in enumerate(shapes):
        w = genomes[:, at:at + i * o].reshape(-1, i, o)
        at += i * o
        b = genomes[:, at:at + o]
        at += o
        h = np.einsum("pi,pio->po", h, w) + b
        if k < len(shapes) - 1:
            h = np.tanh(h)
    return np.clip(h[:, 0], -1.0, 1.0)


# -- evaluation ------------------------------------------------------------------------

def evaluate(spec: CropSpec, lever: str, runner: CropRunner, genomes: np.ndarray,
             start_year: int, prices: Prices, no3_price: float) -> list[dict]:
    """Closed-loop episodes of every genome on one window, all in the same engine runs."""
    x0 = default_x(spec, lever)
    eps = [_Episode(spec, lever, prices, no3_price, x0) for _ in genomes]
    obs = np.stack([e.reset(start_year) for e in eps])
    done, info = False, None
    while not done:
        res = act(genomes, obs)
        plans = [e.write([e.absolute(a)]) for e, a in zip(eps, res)]
        outs = runner.run(plans, start_year)
        steps = [e.observe(o) for e, o in zip(eps, outs)]
        obs = np.stack([s[0] for s in steps])
        done = steps[0][2]
        info = [s[3] for s in steps]
    return [row(d, start_year) for d in info]


def _profits(spec, lever, runner, genomes, windows, prices, no3) -> np.ndarray:
    """profits[genome, window]."""
    cols = [[r["profit"] for r in evaluate(spec, lever, runner, genomes, sy, prices, no3)]
            for sy in windows]
    return np.array(cols).T


def _baseline(spec, lever, runner, n, prices, no3) -> dict:
    """The default_x plan's profit on every train window (the zero genome replays it exactly,
    test_ga)."""
    prof = _profits(spec, lever, runner, np.zeros((1, n)), CROP_TRAIN_YEARS, prices, no3)[0]
    return dict(zip(CROP_TRAIN_YEARS, prof))


def ga_seed(args) -> dict:
    """Pool job: one GA (v3) run of ``budget`` candidate-seasons."""
    name, lever, prices, no3, budget, seed = args
    spec = load_crop(name)
    rng = np.random.default_rng(seed)
    P, W = GA["population"], GA["windows"]
    T = dims(spec, lever)
    gen_cost = P * W * T
    race_max = GA["race_top"] * (GA["race_seasons"] - W) * T   # worst case per generation
    final_cost = P * len(CROP_TRAIN_YEARS) * T
    every = budget / GA["curve_points"]
    pop = np.stack([init_genome(rng) for _ in range(P)])
    rel_sum, n_played = np.zeros(P), np.zeros(P)
    curve, fitness_log = [], []
    top = GA["curve_top"]
    t0 = time.time()
    with CropRunner(spec, k=P, parallel=True) as train_r, \
            CropRunner(spec, k=P, parallel=True) as test_r:
        base = _baseline(spec, lever, train_r, pop.shape[1], prices, no3)

        def play(idx, windows):
            prof = _profits(spec, lever, train_r, pop[idx], windows, prices, no3)
            rel_sum[idx] += (prof - np.array([base[w] for w in windows])).sum(1)
            n_played[idx] += len(windows)

        def ranked():
            order = np.argsort(-(rel_sum / np.maximum(n_played, 1)))
            rank = np.empty(P, int)
            rank[order] = np.arange(P)
            return order, rank

        def point(gen: int, order):
            test = _profits(spec, lever, test_r, pop[order[:top]], CROP_TEST_YEARS, prices,
                            no3).mean(1)
            curve.append({"generation": gen, "seasons": int(train_r.n_seasons),
                          "test_top": test.tolist(), "test_best": float(test[0])})

        point(0, np.arange(P))
        gen, n_raced = 0, 0
        while train_r.n_seasons + gen_cost + race_max + final_cost <= budget:
            gen += 1
            play(np.arange(P), rng.choice(CROP_TRAIN_YEARS, size=W, replace=False))
            # Racing: a policy in the top race_top is re-evaluated (W more seasons at a time)
            # until it has race_seasons or drops out, so a lucky newborn cannot lead on 3.
            while True:
                order, rank = ranked()
                short = [g for g in order[:GA["race_top"]] if n_played[g] < GA["race_seasons"]]
                if not short:
                    break
                n_raced += len(short)
                play(np.array(short), rng.choice(CROP_TRAIN_YEARS, size=W, replace=False))
            fit = rel_sum / n_played
            fitness_log.append({"generation": gen, "seasons": int(train_r.n_seasons),
                                "fit_best": float(fit.max()), "fit_median": float(np.median(fit)),
                                "best_seasons_played": int(n_played[order[0]])})
            last = train_r.n_seasons + gen_cost + race_max + final_cost > budget
            if last or train_r.n_seasons >= every * len(curve):
                point(gen, order)
            if last:
                break
            kill = rng.random(P) < kill_probability(rank, P)
            survivors = np.flatnonzero(~kill)
            w = (P - rank[survivors]).astype(float)
            parents = rng.choice(survivors, size=int(kill.sum()), p=w / w.sum())
            pop[kill] = pop[parents] + GA["sigma"] * rng.standard_normal((len(parents),
                                                                             pop.shape[1]))
            rel_sum[kill], n_played[kill] = 0.0, 0.0
        # Final pick on train: the whole population on every train window.
        full = _profits(spec, lever, train_r, pop, CROP_TRAIN_YEARS, prices, no3).mean(1)
        best = int(np.argmax(full))
        seasons = int(train_r.n_seasons)
        policy_test = [evaluate(spec, lever, test_r, pop[best:best + 1], sy, prices, no3)[0]
                       for sy in CROP_TEST_YEARS]
    return {"seed": seed, "generations": gen, "seasons": seasons, "raced": n_raced,
            "train_seconds": time.time() - t0, "curve": curve, "fitness": fitness_log,
            "picked_train_profit": float(full[best]), "policy_test": policy_test,
            "genome": pop[best].tolist()}


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of x scaled to [-0.5, 0.5] (Salimans et al. 2017's fitness shaping)."""
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def es_seed(args) -> dict:
    """Pool job: one OpenAI-ES run of ``budget`` candidate-seasons.

    Each generation draws P/2 directions eps and plays theta +- sigma*eps (mirrored) on the
    same W train windows, so a pair shares its weather and the update uses their difference.
    The gradient estimate sum_j (u+_j - u-_j) eps_j / (P sigma) (u = centred ranks of mean
    profit) drives Adam with L2 decay toward the zero genome (= default_x). The centre is
    never evaluated during training; at the 16 curve points it is snapshotted and scored on
    test (diagnostic). Final pick: the snapshots re-scored on all train windows, best kept
    (budgeted, the same cost as the GA's final pick)."""
    name, lever, prices, no3, budget, seed = args
    spec = load_crop(name)
    rng = np.random.default_rng(seed)
    P, W, sig = ES["population"], ES["windows"], ES["sigma"]
    T = dims(spec, lever)
    gen_cost = P * W * T
    n_snap = ES["curve_points"]
    final_cost = n_snap * len(CROP_TRAIN_YEARS) * T
    n_gens = max(1, (budget - final_cost) // gen_cost)
    snap_at = set(np.linspace(n_gens / n_snap, n_gens, n_snap).round().astype(int))
    theta = init_genome(rng)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2 = ES["adam"]
    curve, fitness_log, snaps = [], [], []
    t0 = time.time()
    with CropRunner(spec, k=P, parallel=True) as train_r, \
            CropRunner(spec, k=P, parallel=True) as test_r:

        def point(gen: int):
            test = float(_profits(spec, lever, test_r, theta[None], CROP_TEST_YEARS, prices,
                                  no3).mean())
            curve.append({"generation": gen, "seasons": int(train_r.n_seasons),
                          "test_best": test, "test_top": [test]})

        point(0)
        for gen in range(1, n_gens + 1):
            eps = rng.standard_normal((P // 2, theta.size))
            pop = np.concatenate([theta + sig * eps, theta - sig * eps])
            windows = rng.choice(CROP_TRAIN_YEARS, size=W, replace=False)
            prof = _profits(spec, lever, train_r, pop, windows, prices, no3).mean(1)
            u = centered_ranks(prof)
            g = ((u[:P // 2] - u[P // 2:]) @ eps) / (P * sig) - ES["l2"] * theta
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            theta = theta + ES["lr"] * (m / (1 - b1 ** gen)) / (np.sqrt(v / (1 - b2 ** gen)) + 1e-8)
            fitness_log.append({"generation": gen, "seasons": int(train_r.n_seasons),
                                "gen_profit_median": float(np.median(prof)),
                                "pair_spread": float(np.mean(np.abs(prof[:P // 2] - prof[P // 2:])))})
            if gen in snap_at:
                snaps.append(theta.copy())
                point(gen)
        cand = np.stack(snaps)
        full = _profits(spec, lever, train_r, cand, CROP_TRAIN_YEARS, prices, no3).mean(1)
        best = int(np.argmax(full))
        seasons = int(train_r.n_seasons)
        policy_test = [evaluate(spec, lever, test_r, cand[best:best + 1], sy, prices, no3)[0]
                       for sy in CROP_TEST_YEARS]
    return {"seed": seed, "generations": int(n_gens), "seasons": seasons,
            "train_seconds": time.time() - t0, "curve": curve, "fitness": fitness_log,
            "picked_train_profit": float(full[best]), "picked_snapshot": best,
            "policy_test": policy_test, "genome": cand[best].tolist()}


METHODS = {"ga": (ga_seed, GA), "es": (es_seed, ES)}


def ga_config(spec: CropSpec, lever: str, prices: Prices, no3: float, budget: int,
              seed: int, method: str = "ga") -> dict:
    return {"crop": spec.name, "lever": lever, "budget": budget, "seed": seed,
            "no3_price": no3, "prices": prices.key(), "model": model_hash(spec.txtinout),
            "render": CROP_RENDER_VERSION[lever], "train": CROP_TRAIN_YEARS,
            "test": CROP_TEST_YEARS, "method": method, method: METHODS[method][1]}


def ga_stages(jobs: list[tuple], out_dir: Path, workers: int,
              method: str = "ga") -> tuple[dict, int]:
    """Run every (case, seed) job of ``method`` not already on disk, ``workers`` at a time;
    returns (results, number of jobs run now)."""
    import multiprocessing as mp

    results, todo = {}, []
    for job in jobs:
        name, lever, prices, no3, budget, seed = job
        cfg = ga_config(load_crop(name), lever, prices, no3, budget, seed, method)
        path = out_dir / f"{name}_{lever}_{method}_s{seed}.json"
        done = load_stage(path, cfg)
        if done is None:
            todo.append((job, cfg, path))
        else:
            results[(name, lever, seed)] = done
    if todo:
        with mp.get_context("spawn").Pool(min(workers, len(todo)), maxtasksperchild=1) as pool:
            for (job, cfg, path), res in zip(todo, pool.imap(METHODS[method][0], [t[0] for t in todo])):
                save_stage(path, cfg, res)
                results[(job[0], job[1], job[5])] = res
                m = np.mean([r["profit"] for r in res["policy_test"]])
                print(f"  {method} {job[0]} {job[1]} seed {job[5]}: policy {m:.0f} $/ha "
                      f"({res['generations']} gens, {res['train_seconds'] / 60:.0f} min)",
                      flush=True)
    return results, len(todo)
