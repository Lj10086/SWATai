"""Crop model specs: loading, fail-loud validation, rendering, pricing and the env."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from swat_gym.crop import (CROP_OUTPUTS, N_MONTH_MAX, TABLES, CropRunner, Plan, dims,
                           lever_months, month_cap, profit, time_sim)
from swat_gym.plan import n_unit
from swat_gym.crop_env import CropEnv, default_runs, default_x, make_vec_env, monthly_amounts
from swat_gym.engine import EngineError
from swat_gym.fastrunner import FastRunner
from swat_gym.rewarders import average
from swat_gym.spec import SpecError, check_windows, crop_names, load_crop
from swat_gym.windows import CROP_TEST_YEARS, CROP_TRAIN_YEARS, CROP_WINDOW_LEN

CROPS = crop_names()
PRICES = average()


def test_the_five_crops_load_and_alfalfa_has_no_n_lever():
    assert CROPS == ["alfa", "barl", "csil", "sgbt", "wwht"]
    for name in CROPS:
        s = load_crop(name)
        assert s.levers == (("I",) if name == "alfa" else ("I", "N"))
        assert s.plant in PRICES.crop
        check_windows(s, CROP_TRAIN_YEARS + CROP_TEST_YEARS, CROP_WINDOW_LEN)


@pytest.mark.parametrize("name", CROPS)
def test_rendered_own_management_is_the_original_model(name):
    """Plan() (template ops + the model's decision tables) == the shipped schedule."""
    s = load_crop(name)
    with FastRunner(s.txtinout, keep=CROP_OUTPUTS) as fr, CropRunner(s) as cr:
        fr.run({"time.sim": time_sim(2008)})
        mine = cr.run([Plan()], 2008)[0]
        for t in TABLES:
            assert fr.read(t).equals(mine[t]), t


def _copy(name, tmp_path):
    src = tmp_path / name
    shutil.copytree(load_crop(name).txtinout.parent, src)
    return src


def test_a_schedule_the_hru_does_not_use_is_refused(tmp_path):
    root = tmp_path / "crops"
    d = _copy("csil", root)
    lum = d / "TxtInOut" / "landuse.lum"
    lum.write_text(lum.read_text().replace("csil_rot", "other_rot"))
    with pytest.raises(SpecError, match="no schedule named 'other_rot'"):
        load_crop("csil", root)


def test_a_missing_decision_table_is_refused(tmp_path):
    root = tmp_path / "crops"
    d = _copy("barl", root)
    dtl = d / "TxtInOut" / "lum.dtl"
    dtl.write_text(dtl.read_text().replace("irr_win_barl", "irr_win_gone"))
    with pytest.raises(SpecError, match="irr_win_barl"):
        load_crop("barl", root)


def test_windows_outside_the_weather_record_are_refused():
    """SWAT+ would run 1700 on generated weather and exit 0."""
    s = load_crop("csil")
    with pytest.raises(SpecError, match="outside the weather record"):
        check_windows(s, [2025], CROP_WINDOW_LEN)
    with CropRunner(s) as cr, pytest.raises(SpecError):
        cr.run([Plan()], 1700)


def test_lever_months_follow_the_irrigation_windows_and_season():
    s = load_crop("csil")  # window DOY 145-261, harvest 9/25
    assert [m.mon for m in lever_months(s, "I")] == [5, 6, 7, 8, 9]
    assert len(lever_months(s, "I")[0].days) == 7          # May 25-31
    assert month_cap(s, "I")[0] == pytest.approx(7 * 9.0)  # one 9 mm pass a day
    assert [m.mon for m in lever_months(s, "N")] == [6, 7, 8, 9]
    a = load_crop("alfa")
    gaps = {d for m in lever_months(a, "I") for d in m.days}
    assert not gaps & set(range(160, 167)), "no irrigation in the 6/15 cutting gap"
    with pytest.raises(ValueError):
        lever_months(a, "N")


@pytest.mark.parametrize("name", ["csil", "wwht"])
@pytest.mark.parametrize("lever", ["I", "N"])
def test_default_x_reproduces_the_decision_tables_amounts(name, lever):
    """The crop analogue of rule 5 (within 1 %): the open-loop default applies the tables'
    mean amount. Not exact for irrigation: the tables' windows are days of year, so in leap
    years they open one calendar day earlier than the lever's calendar-date passes can
    (csil: 8 May passes vs 7, 0.26 % of the season's water)."""
    s = load_crop(name)
    x0 = default_x(s, lever)
    tables = np.mean([monthly_amounts(s, lever, o) for o in default_runs(s)], axis=0)
    with CropRunner(s) as cr:
        out = cr.run([Plan(lever, tuple(x0), tuple(x0))], CROP_TRAIN_YEARS[3])[0]
    d = profit(s, out, PRICES)
    applied = d["irrigation_mm"] if lever == "I" else out["scheduled_n_kg"]
    assert applied == pytest.approx(float(tables.sum()), rel=1e-2)
    assert applied <= float(tables.sum()) + 1e-9  # only the leap-day excess is lost


def test_decision_table_passes_are_counted_from_mass():
    s = load_crop("csil")
    with CropRunner(s) as cr:
        d = profit(s, cr.run([Plan()], 2015)[0], PRICES)
        assert d["n_fert_events"] == d["auto_n_events"] == d["fert_n_kg"] / s.auto_n_kg
        x = np.zeros(dims(s, "N"))
        x[1] = n_unit(75.0, N_MONTH_MAX)  # 75 kg N in July, the N table removed
        d2 = profit(s, cr.run([Plan("N", tuple(x), tuple(default_x(s, "N")))], 2015)[0], PRICES)
    assert d2["auto_n_events"] == 0 and d2["n_fert_events"] == 1
    assert d2["fert_n_kg"] == pytest.approx(75.0)
    assert d2["fert_cost"] == pytest.approx(75.0 * PRICES.fert_n)


def test_alfalfa_is_priced_from_the_hru_yield_not_the_empty_basin_table():
    s = load_crop("alfa")
    with CropRunner(s) as cr:
        d = profit(s, cr.run([Plan()], 2015)[0], PRICES)
    assert d["yield_mg"] > 10 and d["revenue"] == pytest.approx(d["yield_mg"] * PRICES.crop["alfa"])


@pytest.mark.parametrize("name,lever", [("csil", "I"), ("barl", "N"), ("alfa", "I")])
def test_episode_profit_equals_open_loop(name, lever):
    """Rule 2 for crop models: the env path and the open-loop path are the same objective."""
    s = load_crop(name)
    rng = np.random.default_rng(0)
    with CropEnv(s, lever, prices=PRICES) as env:
        env.reset(start_year=2005)
        total, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step([rng.random()])
            total += r
        x = env.x.copy()
    with CropRunner(s) as cr:
        ol = profit(s, cr.run([Plan(lever, tuple(x), tuple(default_x(s, lever)))], 2005)[0], PRICES)
    for k in ("profit", "revenue", "irrigation_mm", "fert_n_kg", "fert_p_kg", "no3_leached_kg",
              "n_fert_events"):
        assert info[k] == pytest.approx(ol[k], rel=1e-12, abs=1e-12), k
    assert total * 1e3 == pytest.approx(ol["profit"], rel=1e-6)


def test_later_crop_decisions_never_change_earlier_observations():
    s = load_crop("sgbt")
    seen = []
    for tail in (0.0, 1.0):
        with CropEnv(s, "I", prices=PRICES) as env:
            env.reset(start_year=2001)
            seen.append([env.step([0.4 if t < 3 else tail])[0] for t in range(env.n_steps)])
    for k in range(3):
        np.testing.assert_array_equal(seen[0][k], seen[1][k], err_msg=f"step {k}")


def test_zero_residual_vec_env_is_default_x_and_matches_single_runs():
    """Batched PPO env: zero residual = default_x, same window per round, exact per env."""
    s = load_crop("barl")
    venv = make_vec_env(s, "N", prices=PRICES, n_envs=4, seed=3)
    try:
        venv.reset()
        sy = venv.eps[0].start_year
        assert all(e.start_year == sy for e in venv.eps)
        rng = np.random.default_rng(1)
        acts = [np.zeros((4, 1))] + [rng.uniform(-1, 1, (4, 1)) for _ in range(3)]
        for a in acts:
            _, _, dones, infos = venv.step(a)
        assert dones.all()
        xs = [np.asarray(info["x"]) for info in infos]
        assert venv.runner.batched
    finally:
        venv.close()
    x0 = default_x(s, "N")
    assert np.allclose([x[0] for x in xs], x0[0])
    with CropRunner(s) as cr:
        for x, info in zip(xs, infos):
            ol = profit(s, cr.run([Plan("N", tuple(x), tuple(x0))], sy)[0], PRICES)
            assert info["profit"] == pytest.approx(ol["profit"], rel=1e-12)


def test_missing_month_row_raises():
    from swat_gym.crop_env import _Episode

    s = load_crop("csil")
    with CropRunner(s) as cr:
        out = cr.run([Plan()], 2015)[0]
    with pytest.raises(EngineError, match="expected 1"):
        _Episode._state(out, 13)


def test_near_zero_nitrogen_is_no_application_and_no_pass():
    """CMA-ES returns 1e-6, not 0; that must not buy a $37 pass for 0.0002 kg N."""
    s = load_crop("wwht")
    with CropRunner(s) as cr:
        d = profit(s, cr.run([Plan("N", tuple([1e-6] * dims(s, "N")), tuple(default_x(s, "N")))],
                             2015)[0], PRICES)
    assert d["n_fert_events"] == 0 and d["fert_n_kg"] == 0.0
