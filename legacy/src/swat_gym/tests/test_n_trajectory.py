"""Tests for measured soil NO3 and the reconstructed balance."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from n_trajectory import fertiliser_split, measured, score, trajectory  # noqa: E402

from swat_gym import FastRunner  # noqa: E402
from swat_gym.fastrunner import TXTINOUT  # noqa: E402

#: kg N/ha per ppm over 0-122 cm.
KG_HA_PER_PPM = 17.04


def test_measurement_matches_the_model_initial_condition():
    """2013 sample must match nutrients.sol nitrate."""
    obs = measured()
    ported = float(next(
        line.split()[3] for line in (TXTINOUT / "nutrients.sol").read_text().splitlines()
        if line.split()[:1] == ["soilnut1"]
    ))
    assert obs.loc[2013, "no3_ppm"] == pytest.approx(ported, abs=1e-3)


def test_the_two_units_are_mutually_consistent():
    obs = measured()
    ratio = obs["no3_kg_ha"] / obs["no3_ppm"]
    assert ratio.std() < 1e-6, "the ppm -> kg/ha factor must be constant across years"
    assert ratio.mean() == pytest.approx(KG_HA_PER_PPM, abs=0.05)


def test_all_seven_samplings_are_present():
    obs = measured()
    assert list(obs.index) == list(range(2013, 2020))
    assert (obs["n_plots"] == 4).all()
    assert (obs["no3_kg_ha"] > 0).all()


def test_manures_share_one_mineral_and_ammonium_split():
    """The bracket assumes a single pair of fractions; ``fertiliser_split`` asserts it."""
    min_frac, nh3_frac = fertiliser_split(TXTINOUT)
    assert min_frac == pytest.approx(0.20, abs=1e-6)
    assert nh3_frac == pytest.approx(0.99, abs=1e-6)


@pytest.mark.slow
def test_bracket_is_ordered_and_collapses_without_fertiliser():
    with FastRunner() as r:
        r.run()
        traj = trajectory(r)

    assert (traj["sim_lo"] <= traj["sim_hi"] + 1e-9).all(), "bracket must be ordered"
    # Alfalfa years are exact (no manure).
    exact = traj[traj["exact"]]
    assert list(exact.index) == [2015, 2016, 2017]
    assert (exact["sim_hi"] - exact["sim_lo"]).abs().max() < 1e-9
    # ...and the manure years must genuinely be uncertain, or the bracket is doing nothing.
    assert (traj[~traj["exact"]]["sim_hi"] - traj[~traj["exact"]]["sim_lo"] > 10.0).all()


@pytest.mark.slow
def test_miss_is_zero_exactly_when_the_measurement_is_inside_the_bracket():
    with FastRunner() as r:
        r.run()
        traj = trajectory(r)

    for year, row in traj.iterrows():
        inside = row["sim_lo"] <= row["obs_end"] <= row["sim_hi"]
        assert (row["miss"] == 0.0) == inside, f"{year}: miss/bracket disagree"
    assert score(traj) >= 0.0
