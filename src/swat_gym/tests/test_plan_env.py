"""Plans and the env: measured-practice nesting, objective parity, and state reading."""
from __future__ import annotations

import numpy as np
import pytest

from swat_gym.env import OBS_DIM, SwatEnv
from swat_gym.fastrunner import FastRunner
from swat_gym.engine import EngineError
from swat_gym.plan import (ANNUAL_YEARS, DIMS, GROWING_MONTHS, MAX_EVENT_MM,
                           MEASURED_IRRIGATION_MM, MEASURED_ROTATION, MONTH_DEPTH_MAX, N_GROWING,
                           N_YEARS, SPINUP, default_plan, default_x, evaluate,
                           fertilisable_months, irrigable_months, plan_from, time_sim,
                           year_events)
from swat_gym.rewarders import average, profit

MEASURED_N_KG = 2090.06


@pytest.fixture(scope="module")
def runner():
    with FastRunner() as r:
        yield r


def test_default_plan_is_measured_rotation_and_manure():
    plan = default_plan()
    assert tuple(a.crop for a in plan) == MEASURED_ROTATION
    assert sum(a.manure_mg for a in plan) == pytest.approx(234.4)
    assert all(a.manure_mg == 0 for a in plan if a.crop == "alfa")


@pytest.mark.parametrize("lever", ["I", "N"])
def test_default_x_reproduces_measured_practice(runner, lever):
    """Rule 5 (irrigation within 1 %) and measured N, for both levers' default."""
    d = evaluate(lever, default_x(lever), runner, prices=average())
    assert abs(d["irrigation_mm"] - MEASURED_IRRIGATION_MM) / MEASURED_IRRIGATION_MM < 0.01
    assert d["n_applied_kg"] == pytest.approx(MEASURED_N_KG, rel=1e-3)


def test_generated_default_matches_shipped_schedule_nitrogen(runner):
    runner.run({"time.sim": time_sim(2013, N_YEARS + SPINUP)})
    shipped = profit(runner, average())
    gen = evaluate("I", default_x("I"), runner, prices=average())
    assert gen["n_applied_kg"] == pytest.approx(shipped["n_applied_kg"], rel=1e-3)


def test_no_irrigation_pass_exceeds_cap():
    for x in (default_x("I"), np.ones(DIMS["I"])):
        for a in plan_from("I", x):
            assert all(mm <= MAX_EVENT_MM + 1e-9 for _, mm in a.irr_day_depths)


def test_n_lever_leaves_alfalfa_unfertilised():
    plan = plan_from("N", np.ones(DIMS["N"]))
    for y, a in enumerate(plan):
        if y not in ANNUAL_YEARS:
            assert a.manure_mg == 0 and not a.fert_splits


@pytest.mark.parametrize("lever", ["I", "N"])
def test_episode_profit_equals_open_loop(runner, lever):
    """Rule 2: the policy path and the open-loop path are the same objective."""
    rng = np.random.default_rng(0)
    prices = average()
    with SwatEnv(lever, prices=prices, runner=runner) as env:
        obs, _ = env.reset(start_year=2013)
        assert obs.shape == (OBS_DIM,)
        total, done = 0.0, False
        while not done:
            obs, r, done, _, info = env.step(rng.random(env.action_dim))
            total += r
        x = env.x.copy()
    openloop = evaluate(lever, x, runner, prices=prices, start_year=2013)
    assert info["profit"] == pytest.approx(openloop["profit"], rel=1e-9)
    assert total * 1e3 == pytest.approx(openloop["profit"], rel=1e-6)
    # Same plan, not just the same scalar: profit can agree while its terms trade off.
    for k in ("revenue", "irrigation_mm", "manure_mg", "fert_n_kg", "n_applied_kg",
              "no3_leached_kg", "n_fert_events"):
        assert info[k] == pytest.approx(openloop[k], rel=1e-12, abs=1e-12), k


def test_observation_tracks_the_decided_month(runner):
    """State comes from the month just decided, and varies within an episode."""
    with SwatEnv("I", prices=average(), runner=runner) as env:
        env.reset(start_year=2013)
        sw = []
        for _ in range(12):
            env.step([0.3])
            sw.append(env.last["sw"])
    assert len(set(np.round(sw, 3))) > 3


# --- causality (paper A2): a decision can only change the future ----------------------

def test_a_later_month_never_moves_water_into_an_earlier_one():
    """Barley is harvested 8/15: September water must be dropped, not placed in August."""
    base = np.array([0, 0, 0, 0, 120.0, 0])
    with_sep = base.copy()
    with_sep[5] = 200.0
    assert year_events(with_sep, "barl") == year_events(base, "barl")
    for crop in ("corn", "barl", "alfa"):
        for m in range(N_GROWING):
            rng = np.random.default_rng(m)
            v = rng.uniform(0, MONTH_DEPTH_MAX, N_GROWING)
            w = v.copy()
            w[m:] = MONTH_DEPTH_MAX  # max out this month and everything after
            cut = _month_start(GROWING_MONTHS[m])
            before = [e for e in year_events(v, crop) if e[0] < cut]
            assert before == [e for e in year_events(w, crop) if e[0] < cut], (crop, m)


def _month_start(mon: int) -> int:
    from swat_gym.schedule import _doy
    return _doy(mon, 1)


def test_undecided_tail_cannot_change_earlier_observations(runner):
    """Prefix replay is an MDP only if later actions leave the observed past unchanged."""
    runs = []
    for tail in (0.0, 1.0):
        with SwatEnv("I", prices=average(), runner=runner) as env:
            env.reset(start_year=2013)
            obs = []
            for t in range(len(env.steps)):
                obs.append(env.step([0.3 if t < 9 else tail])[0])
            wb = runner.read("hru_wb_mon.txt")
            runs.append((obs, wb))
    (obs_a, wb_a), (obs_b, wb_b) = runs
    for k in range(9):
        np.testing.assert_array_equal(obs_a[k], obs_b[k], err_msg=f"step {k} changed")
    # Every simulated month up to the 9th decision (barley August 2014) is identical.
    y, m = SwatEnv.steps_for("I")[8]
    yr, mon = 2013 + SPINUP + y, GROWING_MONTHS[m]
    upto = lambda df: df[(df["yr"] < yr) | ((df["yr"] == yr) & (df["mon"] <= mon))]
    assert upto(wb_a).equals(upto(wb_b))


def test_observation_reads_the_decided_month_not_the_last_row(runner):
    """Each channel comes from the month just decided, selected by (yr, mon)."""
    with SwatEnv("I", prices=average(), runner=runner) as env:
        env.reset(start_year=2013)
        for k in (1, 2, 3, 7):  # step 7 = April of year 2 (a year rollover)
            while env.t < k:
                env.step([0.5])
            y, m = env.steps[k - 1]
            yr, mon = 2013 + SPINUP + y, GROWING_MONTHS[m]
            wb, pw = runner.read("hru_wb_mon.txt"), runner.read("hru_pw_mon.txt")
            w = wb[(wb["yr"] == yr) & (wb["mon"] == mon)].iloc[0]
            p = pw[(pw["yr"] == yr) & (pw["mon"] == mon)].iloc[0]
            assert (int(wb.iloc[-1]["yr"]), int(wb.iloc[-1]["mon"])) != (yr, mon)
            assert env.last["sw"] == pytest.approx(float(w["sw_final"])), k
            assert env.last["precip"] == pytest.approx(float(w["precip"]))
            assert env.last["pet"] == pytest.approx(float(w["pet"]))
            assert env.last["strsw"] == pytest.approx(float(p["strsw"]))
            assert env.last["strsn"] == pytest.approx(float(p["strsn"]))


def test_hydrologic_channels_vary_within_an_episode(runner):
    """If sw/precip/PET are constant the policy can condition on nothing but the clock."""
    with SwatEnv("I", prices=average(), runner=runner) as env:
        env.reset(start_year=2013)
        seen = np.array([env.step([0.5])[0] for _ in range(6)])
    for ch, name in ((5, "soil water"), (8, "precip"), (9, "PET")):
        assert seen[:, ch].std() > 0, f"channel {ch} ({name}) is constant"


# --- steps and actions ---------------------------------------------------------------

def test_dead_months_are_not_steps():
    """Barley (harvest 8/15) has no September irrigation and no Aug/Sep mid-month N."""
    assert irrigable_months("barl") == (0, 1, 2, 3, 4)
    assert fertilisable_months("barl") == (0, 1, 2, 3)
    assert irrigable_months("corn") == irrigable_months("alfa") == tuple(range(N_GROWING))
    assert len(SwatEnv.steps_for("I")) == 40 and len(SwatEnv.steps_for("N")) == 20
    for a, c in zip(plan_from("N", np.ones(DIMS["N"])), MEASURED_ROTATION):
        if c == "barl":
            assert all(doy < 227 for doy, _ in a.fert_splits)  # 8/15


@pytest.mark.parametrize("lever", ["I", "N"])
def test_zero_residual_policy_is_measured_practice(runner, lever):
    """An untrained PPO (mean action 0) starts where CMA-ES does: default_x."""
    with SwatEnv(lever, prices=average(), runner=runner) as env:
        gym = env.to_gym()
        env.reset(start_year=2013)
        done = False
        while not done:
            _, _, done, _, info = gym.step(np.zeros(env.action_dim, dtype=np.float32))
        np.testing.assert_allclose(env.x, default_x(lever))
    ref = evaluate(lever, default_x(lever), runner, prices=average(), start_year=2013)
    assert info["profit"] == pytest.approx(ref["profit"], rel=1e-12)


def test_residual_actions_are_clipped_to_the_unit_box(runner):
    with SwatEnv("N", prices=average(), runner=runner) as env:
        env.reset(start_year=2013)
        assert env.absolute([5.0, -5.0]) == [1.0, 0.0]
        assert env.absolute([0.0, 0.0])[1] == pytest.approx(default_x("N")[N_GROWING])


def test_missing_state_row_raises(runner):
    """A month the run did not simulate must fail, not read as INITIAL_STATE."""
    with SwatEnv("I", prices=average(), runner=runner) as env:
        env.reset(start_year=2013)
        env.step([0.3])
        with pytest.raises(EngineError, match="expected 1"):
            env._read_state(N_YEARS + 3, 0)


def test_unwired_schedule_raises(runner):
    """Every yield zero = the plan never reached the HRU; the engine exits 0 regardless."""
    from swat_gym.rewarders import profit as price_run
    from swat_gym.schedule import build

    edits = build(plan_from("I", default_x("I")))
    edits["management.sch"] = edits["management.sch"].replace("kimb_rot", "nowhere", 1)
    edits["time.sim"] = time_sim(2013)
    runner.run(edits)
    with pytest.raises(EngineError, match="wired"):
        price_run(runner, average())


def test_nitrogen_dead_zone_round_trips_and_skips_tiny_rates():
    from swat_gym.plan import MANURE_MAX, MONTH_N_MAX, N_DEAD_ZONE, n_amount, n_unit
    for amt in (0.0, 1.0, 43.8, 88.3, MANURE_MAX):
        assert float(n_amount(n_unit(amt, MANURE_MAX), MANURE_MAX)) == pytest.approx(amt, abs=1e-9)
    assert float(n_amount(N_DEAD_ZONE, MONTH_N_MAX)) == 0.0
    assert float(n_amount(1.0, MONTH_N_MAX)) == MONTH_N_MAX
    plan = plan_from("N", np.full(DIMS["N"], 1e-6))
    assert all(a.manure_mg == 0 and all(kg == 0 for _, kg in a.fert_splits) for a in plan)
