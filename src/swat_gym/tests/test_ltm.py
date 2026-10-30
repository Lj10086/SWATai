"""LTM environment: parity (rule 2), zero-policy starts, batching exactness, rotation rules,
gross water."""
from __future__ import annotations

import numpy as np
import pytest

from swat_gym.ltm import econ
from swat_gym.ltm.env import Episode, fill_rotation, plan_from_x, run_policy
from swat_gym.ltm.es import forward, init_genome, n_params
from swat_gym.ltm.practice import CROPS, TRIAL_ROTATION, decisions, measured_x
from swat_gym.ltm.render import EpisodePlan, validate
from swat_gym.ltm.runner import LtmRunner
from swat_gym.ltm.score import TERMS, episode
from swat_gym.ltm.windows import TRAIN

START = TRAIN[0]
SCALARS = ("profit", "revenue", "water_cost", "manure_cost", "fert_n_cost", "fert_p_cost",
           "pass_cost", "operating_cost", "no3", "water", "p", "n2o")


@pytest.fixture(scope="module")
def prices():
    return econ.prices()


@pytest.mark.parametrize("lever", ["R", "N", "I"])
def test_episode_equals_open_loop(lever, prices):
    """A random policy's closed-loop episode scores exactly as its final plan run open loop."""
    rng = np.random.default_rng(1)
    g = init_genome(lever, rng)[None] + 0.3 * rng.standard_normal((1, n_params(lever)))
    with LtmRunner(1) as r:
        plans, outs, _ = run_policy(lambda o: forward(lever, g, o), lever, r, 1, START)
        closed = episode(plans[0], outs[0], prices, START)
        open_ = episode(plans[0], r.run([plans[0]], START)[0], prices, START)
    for k in SCALARS:
        assert closed[k] == pytest.approx(open_[k], rel=0, abs=1e-9), k
    assert closed["crops"] == open_["crops"]
    assert closed["season_yield_t"] == pytest.approx(open_["season_yield_t"], rel=0, abs=1e-9)


def test_zero_genome_rotation_is_the_trial():
    lever = "R"
    with LtmRunner(1) as r:
        plans, _, _ = run_policy(lambda o: np.zeros((len(o), len(CROPS))), lever, r, 1, START)
    assert plans[0].crops == TRIAL_ROTATION


@pytest.mark.parametrize("lever", ["N", "I"])
def test_zero_residual_is_measured_x(lever):
    ep = Episode(lever)
    xs = []
    while not ep.done:
        v = ep.absolute([0.0])
        ep.write(v)
        ep.t += 1
        xs.append(v)
    assert np.allclose(xs, measured_x(lever))


def test_batched_equals_single_for_annual_rotations(prices):
    plans = [EpisodePlan(), EpisodePlan(crops=fill_rotation(["pota", "barl", "corn"])),
             plan_from_x("I", 0.5 * np.ones(len(decisions("I"))))]
    with LtmRunner(len(plans), batched=True) as b, LtmRunner(len(plans), batched=False) as s:
        ob, os_ = b.run(plans, START), s.run(plans, START)
    for p, x, y in zip(plans, ob, os_):
        ex, ey = episode(p, x, prices, START), episode(p, y, prices, START)
        for k in SCALARS:
            assert ex[k] == pytest.approx(ey[k], rel=0, abs=1e-6), k


def test_rotation_rules():
    validate(TRIAL_ROTATION)
    validate(("alfa",) * 3 + TRIAL_ROTATION[3:])
    validate(("pota", "barl", "wwht", "pota") + TRIAL_ROTATION[4:])
    for bad in [("sgbt", "sgbt") + TRIAL_ROTATION[2:],            # no crop two years running
                ("alfa", "alfa") + TRIAL_ROTATION[2:],            # a stand is 3 years
                ("alfa",) * 6 + ("barl", "sgbt"),                 # a stand may not follow a stand
                ("pota", "barl", "pota") + TRIAL_ROTATION[3:]]:   # potato once in any 3 years
        with pytest.raises(ValueError):
            validate(bad)
    assert len(fill_rotation(["alfa"])) == 8 and fill_rotation(["alfa"])[:3] == ("alfa",) * 3


def test_rotation_mask_matches_rules():
    """Masked rollouts are always valid, and every valid cycle the fixed search scores is
    reachable through the masks (the closed loop and the open loop share one rule set)."""
    from swat_gym.experiments.ltm import cycles
    from swat_gym.ltm.env import feasible
    rng = np.random.default_rng(0)
    for _ in range(500):
        crops: list[str] = []
        for k in range(1, 9):
            ok = np.flatnonzero(feasible(crops, k))
            crops.append(CROPS[int(rng.choice(ok))])
        validate(tuple(crops))
        assert fill_rotation(crops[:int(rng.integers(0, 8))]) is not None
    for c in cycles():
        for k in range(1, 9):
            assert feasible(list(c[:k - 1]), k)[CROPS.index(c[k - 1])], c


def test_water_is_gross(prices):
    """The water term is gross: measured replay (efficiency 1.0) = the trial's depths whatever
    the plan efficiency; auto-irrigation at 0.8 = passes x 25.4 mm, i.e. soil water / 0.8."""
    import pandas as pd
    from swat_gym.ltm.practice import AUTO_IRR, DATA
    irr = pd.read_csv(DATA / "irrigation_daily.csv", parse_dates=["date"])
    meas = irr[irr.date.dt.year.between(2013, 2020)].irr_mm.sum()
    auto = EpisodePlan(irr_auto=True, irr_eff=0.8)
    with LtmRunner(2) as r:
        om, oa = r.run([EpisodePlan(irr_eff=0.55), auto], 2012)
    assert episode(EpisodePlan(irr_eff=0.55), om, prices, 2012)["water"] * 8 == \
        pytest.approx(meas, abs=1e-3)
    ea = episode(auto, oa, prices, 2012)
    n = int(((oa["mgt"].operation == "IRRIGATE") & oa["mgt"].yr.between(2013, 2020)).sum())
    assert ea["water"] * 8 == pytest.approx(n * AUTO_IRR["mm"], rel=1e-4)
    soil = float(oa["hru_wb_yr.txt"].query("2013 <= yr <= 2020").irr.sum())
    assert soil == pytest.approx(0.8 * n * AUTO_IRR["mm"], rel=1e-4)


def test_auto_irrigation_stays_in_its_windows():
    """Auto-irrigation fires only inside each season's own year and window(s)."""
    import pandas as pd
    from swat_gym.ltm.render import season_of
    plan = EpisodePlan(crops=("alfa",) * 3 + TRIAL_ROTATION[3:], irr_auto=True)
    with LtmRunner(1, batched=False) as r:
        mgt = r.run([plan], START)[0]["mgt"]
    ir = mgt[mgt.operation == "IRRIGATE"]
    assert len(ir) > 0
    doy = pd.to_datetime(dict(year=ir.yr, month=ir.mon, day=ir.day)).dt.dayofyear
    from swat_gym.ltm.practice import spinup_window
    for (y, d) in zip(ir.yr, doy):
        k = y - START
        assert 0 <= k <= 8
        wins = [spinup_window(y)] if k == 0 else season_of(plan, k).irr_windows
        assert any(a <= d <= b for a, b in wins), (y, d)


def test_mixed_batching_equals_single(prices):
    """Mixed mode: annual plans batched, alfalfa plans alone -- identical to all-single."""
    plans = [EpisodePlan(irr_auto=True),
             EpisodePlan(crops=("alfa",) * 3 + TRIAL_ROTATION[3:], irr_auto=True),
             EpisodePlan(crops=fill_rotation(["pota", "sgbt", "barl"]), irr_auto=True)]
    with LtmRunner(4, batched="mixed") as m, LtmRunner(4, batched=False) as s:
        om, os_ = m.run(plans, START), s.run(plans, START)
        assert m.batch_runner.n_runs == 1 and sum(r.n_runs for r in m.singles) == 1
    for p, x, y in zip(plans, om, os_):
        ex, ey = episode(p, x, prices, START), episode(p, y, prices, START)
        for k in SCALARS:
            assert ex[k] == pytest.approx(ey[k], rel=0, abs=1e-6), k


def test_es_survives_engine_crash(monkeypatch, tmp_path):
    """A crashed batch is re-run per genome; only the genome that still crashes scores last."""
    from swat_gym.engine import EngineError
    from swat_gym.ltm import es

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "file.cio").write_text("x")
    monkeypatch.setattr(es, "CRASH_DIR", tmp_path / "crash")

    def fake(lever, genomes, runner, windows, prices, ref, w):
        if (genomes[:, 0] < 0).any():
            raise EngineError(f"SWAT+ exited -5 in {run_dir}\n--- output tail ---\n")
        return np.repeat(genomes[:, :1], len(windows), axis=1)

    monkeypatch.setattr(es, "scores", fake)
    tally = {"planned": 1000, "failed": 0, "crashed_batches": 0, "kept": 0, "first_error": None}
    g = np.array([[1.0], [-1.0], [2.0]])
    out = es.safe_scores("N", g, None, [1, 2], None, None, 0.5, tally)
    assert out[0].tolist() == [1.0, 1.0] and out[2].tolist() == [2.0, 2.0]
    assert (out[1] == es.FAILED_SCORE).all()
    assert tally["failed"] == 1 and tally["crashed_batches"] == 1
    assert (next((tmp_path / "crash").iterdir()) / "file.cio").exists()
    assert (es.safe_scores("N", g[[0]], None, [1], None, None, 0.5, tally) == 1.0).all()
    tally["planned"] = 10
    tally["failed"] = 2
    with pytest.raises(RuntimeError):
        es.safe_scores("N", g, None, [1], None, None, 0.5, tally)


@pytest.mark.parametrize("lever", ["R", "N", "I"])
def test_truncated_steps_observe_the_same(lever, prices):
    """Intermediate steps simulated only through the next decision's year give exactly the
    observations (and final outputs) of full-window steps (run_policy truncate)."""
    rng = np.random.default_rng(2)
    g = init_genome(lever, rng)[None] + 0.3 * rng.standard_normal((1, n_params(lever)))
    seen: dict = {True: [], False: []}
    res = {}
    for trunc in (True, False):
        def policy(o, trunc=trunc):
            seen[trunc].append(o.copy())
            return forward(lever, g, o)
        with LtmRunner(1, batched="mixed" if lever == "R" else True) as r:
            plans, outs, _ = run_policy(policy, lever, r, 1, START, truncate=trunc)
            res[trunc] = episode(plans[0], outs[0], prices, START)
    assert len(seen[True]) == len(seen[False])
    for a, b in zip(seen[True], seen[False]):
        assert np.array_equal(a, b)
    for k in SCALARS:
        assert res[True][k] == res[False][k], k


def test_cache_replays_identical_outputs(prices):
    plans = [EpisodePlan(irr_auto=True), EpisodePlan(crops=fill_rotation(["corn"]), irr_auto=True)]
    with LtmRunner(2, cache=8) as c, LtmRunner(2) as u:
        first, again = c.run(plans, START), c.run(plans[::-1], START)
        ref = u.run(plans, START)
        assert c.n_cached == 2 and c.n_hru_runs == 4 and c.batch_runner.n_runs == 1
    for p, x, y, z in zip(plans, first, again[::-1], ref):
        ex, ey, ez = (episode(p, o, prices, START) for o in (x, y, z))
        for k in SCALARS:
            assert ex[k] == ey[k] == ez[k], k


def test_cache_smaller_than_a_call():
    """A full cache must not evict an entry the same call still returns (v3 Exp 1 crash)."""
    a, b = EpisodePlan(irr_auto=True), EpisodePlan(crops=fill_rotation(["corn"]), irr_auto=True)
    with LtmRunner(2, cache=1) as c:
        c.run([a], START)
        out = c.run([a, b], START)          # a is a hit; inserting b evicts a
        assert len(out) == 2 and c.n_cached == 1


def test_tied_fitness_gives_no_gradient():
    """ES v2: a tied population (a plateau, e.g. Exp 2 applying no N) must not take a step."""
    from swat_gym.ltm.es import centered_ranks
    x = np.random.default_rng(0).standard_normal(16)
    by_order = np.empty(16)
    by_order[np.argsort(x)] = np.arange(16) / 15 - 0.5
    assert np.allclose(centered_ranks(x), by_order)         # plain ranks without ties
    u = centered_ranks(np.ones(16))
    assert np.all(u == 0.0) and np.all(u[:8] - u[8:] == 0.0)


def _fake_ep(ys, **terms):
    return {"crops": list(TRIAL_ROTATION), "season_yield_t": list(ys), **terms}


def test_yield_floor_only_binds_sustainability():
    from swat_gym.ltm.score import FLOOR_PENALTY, Reference
    terms = {k: 1.0 for k in TERMS}
    crop_ref = {c: 10.0 for c in CROPS}
    ref = Reference(1.0, terms, crop_ref, floor=0.8)
    assert Reference.from_key(ref.key()) == ref
    ep = _fake_ep([7.0] * 8, **terms)                  # ry 0.7: 0.1 below the floor
    plain = Reference(1.0, terms, crop_ref)
    assert ref.score(ep, 0.0) == pytest.approx(plain.score(ep, 0.0) - FLOOR_PENALTY * 0.1)
    assert ref.score(ep, 0.5) == plain.score(ep, 0.5) and ref.score(ep, 1.0) == plain.score(ep, 1.0)
    above = _fake_ep([9.0] * 8, **terms)
    assert ref.score(above, 0.0) == plain.score(above, 0.0)


def test_reference_practice_scores_its_normaliser():
    """Measured practice's own train episodes average to ry / ry_ref = 1 and terms = 1, so
    their mean score is w - (1 - w) for every strategy; profit plays no part."""
    from swat_gym.ltm.score import STRATEGIES, Reference
    rng = np.random.default_rng(0)
    crop_ref = {c: float(rng.uniform(3, 30)) for c in CROPS}
    eps = [_fake_ep(rng.uniform(2, 40, 8), profit=float(rng.normal()),
                    **{k: float(rng.uniform(1, 9)) for k in TERMS}) for _ in range(7)]
    ref = Reference.from_episodes(eps, crop_ref)
    for w in STRATEGIES.values():
        assert np.mean([ref.score(e, w) for e in eps]) == pytest.approx(w - (1 - w))
    with pytest.raises(ValueError):
        Reference(1.0, {k: 1.0 for k in TERMS}, {**crop_ref, "pota": 0.0})


def test_crop_reference_substitutions_are_valid():
    from swat_gym.experiments.ltm import substitutions
    subs = substitutions()
    assert {c for c, _, _ in subs} == set(CROPS)
    for c, ks, crops in subs:
        validate(crops)
        assert all(crops[k - 1] == c for k in ks)
        assert all(x == t for j, (x, t) in enumerate(zip(crops, TRIAL_ROTATION)) if j + 1 not in ks)
