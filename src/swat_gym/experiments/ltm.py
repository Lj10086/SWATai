"""The LTM experiments: Exp 1 rotation (R), Exp 2 nitrogen (N), Exp 3 irrigation (I).

Per lever, for each strategy w in {1, 0.5, 0} (ltm.score.STRATEGIES):

  measured   the trial's 52A practice replayed (the normaliser: train means -> Reference)
  default    measured practice through the lever's plan generator (measured_x), open loop
             (R: the trial rotation auto-irrigated, as every Exp 1 plan is)
  auto       (I only) the trial's management with auto-irrigation in its measured windows
  fixed      the best open-loop plan on train: CMA-ES (N, I) at the ES budget in HRU-runs;
             for R, every repeating crop cycle of length 1-4 (exhaustive; count reported)
  es         closed-loop ES policy, ``--seeds`` seeds at ``--budget`` HRU-runs each
  frozen     each ES policy's decisions on train, averaged (R: per-season majority), open loop

all scored on the test windows. For R, compare against ``default`` (the trial rotation under
the same auto-irrigation rule as every Exp 1 plan): ``measured`` also changes the irrigation
method, which alone cuts leaching ~20-fold here. Paired per-window differences with the moving-block
bootstrap and ESS intervals (swat_gym.stats.paired; rule 7) for the strategy score,
relative yield (ry), profit (reported, not scored) and every sustainability term.

Relative yield needs one reference yield per crop (``crop_reference``, user decision
2026-10-30): the crop's train-window mean harvested yield when it replaces a season of the
default rotation (its own management, auto-irrigation at BASE_IRR_EFF, 52A manure), averaged
over every position the rotation rules allow (alfalfa: a 3-year stand). Shared by every lever
and efficiency case (runs/ltm_crop_ref.json), so ry is on one scale everywhere. Sign-inconsistent seeds -> "unresolvable" (rule 3).

Irrigation efficiency (practice.IRR_EFF_CASES): auto-irrigation and every lever-generated
schedule run at ``--irr-eff`` (default the base, 0.80); the measured replay is always at 1.0, so
the score's Reference (measured practice) is the same scale for every case. Exp 3 is run once per case (``ltm_exp3_eff55`` etc.; the base case keeps
``ltm_exp3``).

    uv run python -m swat_gym.experiments.ltm --lever N --budget 100000 --seeds 4
    uv run python -m swat_gym.experiments.ltm --lever I --irr-eff 0.55
    SMOKE=1 uv run python -m swat_gym.experiments.ltm --lever R
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np

from ..fastrunner import PROJECT_ROOT, model_hash
from ..ltm import econ
from ..ltm.env import fill_rotation, plan_from_x
from ..ltm.es import ES, es_seed
from ..ltm import practice
from ..ltm.practice import (ALFA_STAND, BASE_IRR_EFF, CROPS, TRIAL_ROTATION, TXTINOUT,
                             decisions, measured_x)
from ..ltm.render import EpisodePlan, validate
from ..ltm.runner import LtmRunner
from ..ltm.score import STRATEGIES, TERMS, Reference, episode, relative_yield
from ..ltm.windows import SEASONS, TEST, TEST_ESS, TRAIN, TRAIN_ESS, WINDOW_LEN
from ..stats import paired
from .common import default_workers, load_stage, minimise, save_stage

EXP = {"R": 1, "N": 2, "I": 3}
OUT = PROJECT_ROOT / "runs"
CYCLE_MAX = 4
#: Bump when the LTM env's meaning changes (render, scoring, observation).
LTM_VERSION = 4   # 2: W4 dates/manure basis, auto-irrigation (Exp 1, Exp 3 baseline)
#: (mixed batching, 2026-10-26, changes speed only: test_mixed_batching_equals_single)
#: 3 (2026-10-27): irrigation efficiency (gross water scored), spin-up beans, rotation rules
#: 4 (2026-10-30): relative yield replaces profit in the score; yield floor
REPORTED = ("score", "ry", "profit", *TERMS)


def practice_key(lever: str) -> dict:
    """Every practice setting a stage depends on (none of it is in the model hash)."""
    from ..ltm import practice as pr
    rot = {"rotation_no_repeat": list(pr.ROTATION_NO_REPEAT),
           "rotation_gap": pr.ROTATION_GAP} if lever == "R" else {}
    return {**rot, "auto_irr": pr.AUTO_IRR, "irr_eff": pr.IRR_EFF, "base_irr_eff": pr.BASE_IRR_EFF,
            "measured_irr_eff": pr.MEASURED_IRR_EFF, "spinup": [pr.SPINUP_CROP, [list(f) for f in pr.SPINUP_FERT]], "template_year": pr.TEMPLATE_YEAR,
            "alfa_stand": pr.ALFA_STAND, "no_repeat": list(pr.NO_REPEAT),
            "manure_day": list(pr.MANURE_DAY), "irr_pass_mm": pr.IRR_PASS_MM,
            "n_month_max": pr.N_MONTH_MAX, "fert_p": pr.fert_p_policy(),
            "sgbt_2014_harvest_from": pr.SGBT_2014_HARVEST_FROM}


def _rows(eps: list[dict], key: str) -> list[dict]:
    """paired() rows: one metric per window."""
    return [{"start_year": e["start"], "value": e[key]} for e in eps]


def run_plans(plans: list[EpisodePlan], windows, prices, k: int = 16) -> list[list[dict]]:
    """eps[plan][window], open loop: annual plans batched, alfalfa plans alone ("mixed")."""
    out = [[None] * len(windows) for _ in plans]
    for i0 in range(0, len(plans), k):
        chunk = plans[i0:i0 + k]
        with LtmRunner(len(chunk), batched="mixed") as r:
            for j, s in enumerate(windows):
                for i, o in enumerate(r.run(chunk, s)):
                    out[i0 + i][j] = episode(chunk[i], o, prices, s)
    return out


# -- open-loop objectives (picklable for the CMA-ES pool) -----------------------------------

_RUNNERS: dict = {}


def _neg_scores(xs, window, lever, prices, ref_key, w):
    ref = Reference.from_key(ref_key)
    k = len(xs)
    if k not in _RUNNERS:
        _RUNNERS[k] = LtmRunner(k, batched=True)
    plans = [plan_from_x(lever, np.asarray(x)) for x in xs]
    return [-ref.score(episode(p, o, prices, window), w)
            for p, o in zip(plans, _RUNNERS[k].run(plans, window))]


class _Objective:
    def __init__(self, lever, prices, ref_key, w):
        self.args = (lever, prices, ref_key, w)

    def __call__(self, xs, window):
        return _neg_scores(xs, window, *self.args)


def substitutions() -> list[tuple[str, tuple[int, ...], tuple[str, ...]]]:
    """(crop, its 1-based seasons, rotation) for every valid substitution of each crop into the
    trial rotation: alfalfa as a 3-year stand starting at k, every other crop at season k."""
    out = []
    for c in CROPS:
        span = ALFA_STAND if c == "alfa" else 1
        for k in range(1, len(TRIAL_ROTATION) - span + 2):
            ks = tuple(range(k, k + span))
            crops = tuple(c if j + 1 in ks else x for j, x in enumerate(TRIAL_ROTATION))
            try:
                validate(crops)
            except ValueError:
                continue
            out.append((c, ks, crops))
    missing = set(CROPS) - {c for c, _, _ in out}
    if missing:
        raise RuntimeError(f"no valid substitution position for {sorted(missing)}")
    return out


def crop_reference(prices) -> dict:
    """Per-crop reference yield, t/ha (module docstring); cached in runs/ltm_crop_ref.json."""
    subs = substitutions()
    pk = practice_key("R")
    pk.pop("irr_eff")
    cfg = {"stage": "crop_reference", "ltm_version": LTM_VERSION, "model": model_hash(TXTINOUT),
           "train": TRAIN, "practice": pk, "irr_eff": BASE_IRR_EFF,
           "substitutions": [[c, list(ks), list(cr)] for c, ks, cr in subs]}
    path = OUT / "ltm_crop_ref.json"
    done = load_stage(path, cfg)
    if done is not None:
        return done["crop_ref"]
    plans = [EpisodePlan(crops=cr, irr_auto=True, irr_eff=BASE_IRR_EFF) for _, _, cr in subs]
    eps = run_plans(plans, TRAIN, prices)
    per: dict = {c: [] for c in CROPS}
    for (c, ks, _), row in zip(subs, eps):
        per[c] += [e["season_yield_t"][k - 1] for e in row for k in ks]
    crop_ref = {c: float(np.mean(v)) for c, v in per.items()}
    save_stage(path, cfg, {"crop_ref": crop_ref, "n_seasons": {c: len(v) for c, v in per.items()},
                           "positions": {c: [list(ks) for cc, ks, _ in subs if cc == c] for c in CROPS}})
    return crop_ref


def cycles(max_len: int = CYCLE_MAX) -> list[tuple[str, ...]]:
    """Every distinct valid 8-season rotation that repeats a cycle of length <= max_len."""
    seen, out = set(), []
    for L in range(1, max_len + 1):
        for cyc in itertools.product(CROPS, repeat=L):
            crops = tuple((cyc * SEASONS)[:SEASONS])
            if crops in seen:
                continue
            try:
                validate(crops)
            except ValueError:
                continue
            seen.add(crops)
            out.append(crops)
    return out


def frozen_plan(lever: str, train_eps: list[dict]) -> EpisodePlan:
    """A policy's train decisions as one open-loop plan: per-season majority crop (R, kept
    valid), else the mean decision vector."""
    if lever != "R":
        return plan_from_x(lever, np.mean([np.asarray(e["x"]) for e in train_eps], axis=0))
    chosen: list[str] = []
    for k in range(len(chosen), SEASONS):
        if len(chosen) > k:
            continue
        for crop, _ in Counter(e["crops"][k] for e in train_eps).most_common():
            try:
                validate(fill_rotation(chosen + [crop]))
            except ValueError:
                continue
            chosen.append(crop)
            break
    return EpisodePlan(crops=fill_rotation(chosen), irr_auto=True)


def compare(a: list[dict], b: list[dict]) -> dict:
    return {k: paired(_rows(a, k), _rows(b, k), window_len=WINDOW_LEN) for k in REPORTED}


def _with_score(eps: list[dict], ref: Reference, w: float) -> list[dict]:
    return [{**e, "score": ref.score(e, w), "ry": ref.ry_of(e)} for e in eps]


def main(argv=None) -> dict:
    smoke = os.environ.get("SMOKE") == "1"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lever", choices=list(EXP), required=True)
    ap.add_argument("--budget", type=int, default=3000 if smoke else 100_000)
    ap.add_argument("--seeds", type=int, default=1 if smoke else 4)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ap.add_argument("--irr-eff", type=float, default=BASE_IRR_EFF,
                    help="irrigation application efficiency of every arm (Exp 3 cases)")
    args = ap.parse_args(argv)
    lever = args.lever
    practice.set_irr_eff(args.irr_eff)      # this process, and (env) the spawned workers
    eff_tag = "" if abs(args.irr_eff - BASE_IRR_EFF) < 1e-9 else f"_eff{round(args.irr_eff * 100):02d}"
    tag = f"ltm_exp{EXP[lever]}{eff_tag}" + ("_smoke" if smoke else "")
    stage_dir = OUT / tag
    stage_dir.mkdir(parents=True, exist_ok=True)
    prices = econ.prices()
    base_cfg = {"lever": lever, "ltm_version": LTM_VERSION, "model": model_hash(TXTINOUT),
                "prices": prices.key(), "train": TRAIN, "test": TEST,
                "decisions": len(decisions(lever)), "es": ES, "practice": practice_key(lever)}
    t0 = time.time()

    # 1. references ----------------------------------------------------------------------------
    cfg = {**base_cfg, "stage": "reference"}
    ref_stage = load_stage(stage_dir / "reference.json", cfg)
    if ref_stage is None:
        meas = run_plans([EpisodePlan()], TRAIN + TEST, prices)[0]
        dflt = run_plans([plan_from_x(lever, measured_x(lever))], TRAIN + TEST, prices)[0]
        ref_stage = {"measured": meas, "default": dflt}
        if lever == "I":
            ref_stage["auto"] = run_plans([EpisodePlan(irr_auto=True)], TRAIN + TEST, prices)[0]
        save_stage(stage_dir / "reference.json", cfg, ref_stage)
    n_tr = len(TRAIN)
    crop_ref = crop_reference(prices)
    # sustainability's yield floor: default practice's train-mean ry (score.FLOOR_PENALTY)
    floor = float(np.mean([relative_yield(e, crop_ref) for e in ref_stage["default"][:n_tr]]))
    ref = Reference.from_episodes(ref_stage["measured"][:n_tr], crop_ref, floor=floor)
    report: dict = {"config": {**base_cfg, "budget": args.budget, "seeds": args.seeds,
                               "train_ess": TRAIN_ESS, "test_ess": TEST_ESS,
                               "reference": ref.key()},
                    "strategies": {}}

    seeds = [100 * i for i in range(args.seeds)]
    jobs = [(lever, STRATEGIES[s], prices, ref.key(), args.budget, sd)
            for s in args.strategies for sd in seeds]
    # 2. ES (all strategies x seeds in one pool) ---------------------------------------------------
    es_results, todo = {}, []
    for job in jobs:
        s_name = next(k for k, v in STRATEGIES.items() if v == job[1])
        cfg = {**base_cfg, "stage": "es", "strategy": s_name, "budget": args.budget,
               "seed": job[5], "reference": ref.key()}
        path = stage_dir / f"es_{s_name}_s{job[5]}.json"
        done = load_stage(path, cfg)
        if done is None:
            todo.append((job, cfg, path, s_name))
        else:
            es_results[(s_name, job[5])] = done
    if todo:
        # every job at once: a pool smaller than the job list leaves cores idle while the
        # last jobs run alone (2026-10-26: Exp 1's last 2 of 12 seeds ran ~3 h on 2 cores)
        procs = len(todo)
        with mp.get_context("spawn").Pool(procs, maxtasksperchild=1) as pool:
            for (job, cfg, path, s_name), res in zip(todo, pool.imap(es_seed, [t[0] for t in todo])):
                save_stage(path, cfg, res)
                es_results[(s_name, job[5])] = res
                print(f"  ES {tag} {s_name} seed {job[5]}: train score "
                      f"{res['picked_train_score']:.3f} ({res['generations']} gens, "
                      f"{res['train_seconds'] / 60:.0f} min, "
                      f"{res.get('engine_failures', {}).get('failed', 0)} failed genomes)", flush=True)

    # 3. fixed (open loop), per strategy; R's cycle candidates are run once for all of them
    if lever == "R":
        cfg = {**base_cfg, "stage": "cycles", "cycle_max": CYCLE_MAX}
        cyc_stage = load_stage(stage_dir / "cycles.json", cfg)
        if cyc_stage is None:
            cands = cycles()
            cyc_stage = {"crops": [list(c) for c in cands],
                         "train": run_plans([EpisodePlan(crops=c, irr_auto=True) for c in cands],
                                            TRAIN, prices)}
            save_stage(stage_dir / "cycles.json", cfg, cyc_stage)
    for s_name in args.strategies:
        w = STRATEGIES[s_name]
        if lever == "R":
            means = [float(np.mean([ref.score(e, w) for e in row])) for row in cyc_stage["train"]]
            order = np.argsort(means)[::-1]
            fixed = {"crops": cyc_stage["crops"][int(order[0])],
                     "n_candidates": len(means), "train_score": means[int(order[0])],
                     "top10": [[cyc_stage["crops"][int(i)], means[int(i)]] for i in order[:10]]}
        else:
            cfg = {**base_cfg, "stage": "fixed", "strategy": s_name, "budget": args.budget,
                   "reference": ref.key()}
            fixed = load_stage(stage_dir / f"fixed_{s_name}.json", cfg)
            if fixed is None:
                evals = max(10, args.budget // len(TRAIN))
                x, n, hist = minimise(_Objective(lever, prices, ref.key(), w), measured_x(lever),
                                      evals=evals, windows=TRAIN, batched=True,
                                      workers=args.workers, label=f"{tag} {s_name}")
                fixed = {"x": list(map(float, x)), "evals": int(n), "history": hist[-20:]}
                save_stage(stage_dir / f"fixed_{s_name}.json", cfg, fixed)
        fixed_plan = EpisodePlan(crops=tuple(fixed["crops"]), irr_auto=True) if lever == "R" else \
            plan_from_x(lever, np.asarray(fixed["x"]))
        fixed_test = run_plans([fixed_plan], TEST, prices)[0]

        meas_test = _with_score(ref_stage["measured"][n_tr:], ref, w)
        dflt_test = _with_score(ref_stage["default"][n_tr:], ref, w)
        auto_test = _with_score(ref_stage["auto"][n_tr:], ref, w) if lever == "I" else None
        fixed_test = _with_score(fixed_test, ref, w)
        per_seed = {}
        for sd in seeds:
            res = es_results[(s_name, sd)]
            pol = _with_score(res["test"], ref, w)
            fz_plan = frozen_plan(lever, res["train"])
            fz = _with_score(run_plans([fz_plan], TEST, prices)[0], ref, w)
            per_seed[sd] = {
                "policy_vs_measured": compare(pol, meas_test),
                "policy_vs_default": compare(pol, dflt_test),
                "policy_vs_fixed": compare(pol, fixed_test),
                **({"policy_vs_auto": compare(pol, auto_test)} if auto_test else {}),
                "adaptivity_value": compare(pol, fz),     # closed loop - its own frozen plan
                "frozen_crops": list(fz_plan.crops) if lever == "R" else None,
                "test_means": {k: float(np.mean([e[k] for e in pol])) for k in REPORTED},
            }
        signs = {key: sorted({np.sign(v[key]["score"]["mean"]) for v in per_seed.values()})
                 for key in ("policy_vs_measured", "policy_vs_default", "policy_vs_fixed",
                             "adaptivity_value")}
        report["strategies"][s_name] = {
            "w": w,
            "test_means": {name: {k: float(np.mean([e[k] for e in eps])) for k in REPORTED}
                           for name, eps in (("measured", meas_test), ("default", dflt_test),
                                             ("fixed", fixed_test))
                           + ((("auto", auto_test),) if auto_test else ())},
            "fixed": {k: v for k, v in fixed.items() if k != "history"},
            "fixed_vs_measured": compare(fixed_test, meas_test),
            "fixed_vs_default": compare(fixed_test, dflt_test),
            "default_vs_measured": compare(dflt_test, meas_test),
            "es": per_seed,
            "resolvable": {k: ("unresolvable at this seed count" if len(v) > 1 else "consistent")
                           for k, v in signs.items()},
        }
    report["wall_s"] = time.time() - t0
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote runs/{tag}.json ({report['wall_s'] / 60:.1f} min)")
    return report


if __name__ == "__main__":
    main()
