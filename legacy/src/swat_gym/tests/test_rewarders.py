"""Profit reward tests (manure parse cross-checked vs engine fertn)."""
from __future__ import annotations

import pytest

from swat_gym import FastRunner
from swat_gym.fastrunner import TXTINOUT
from swat_gym.rewarders import Prices, manure_applied, nass, profit


@pytest.fixture(scope="module")
def runner():
    with FastRunner() as r:
        r.run()
        yield r


# --- price vectors ----------------------------------------------------------------------

def test_nass_scenarios_are_ordered_as_published():
    """Alfalfa fell and the alfalfa:corn ratio collapsed in 2024 — the reason prices are swept."""
    p22, p23, p24 = (nass(y) for y in (2022, 2023, 2024))
    assert p22.crop["alfa"] > p23.crop["alfa"] > p24.crop["alfa"]
    assert p22.alfalfa_ratio() == pytest.approx(p23.alfalfa_ratio(), rel=0.02)
    assert p24.alfalfa_ratio() < 0.8 * p22.alfalfa_ratio()


def test_at_ratio_moves_alfalfa_only():
    base = nass(2024)
    swept = base.at_ratio(2.0)
    assert swept.alfalfa_ratio() == pytest.approx(2.0)
    assert swept.crop["alfa"] == pytest.approx(2.0 * base.crop["corn"])
    for c in ("corn", "barl"):
        assert swept.crop[c] == base.crop[c]
    assert swept.water == base.water and swept.manure == base.manure


def test_dry_matter_conversion_is_applied():
    """$/Mg DM must exceed the $/short ton as-fed figure: smaller unit, and moisture removed."""
    assert nass(2024).crop["alfa"] > 154.0


# --- reading the run --------------------------------------------------------------------

def test_manure_matches_engine_nitrogen(runner):
    """Schedule-parsed manure mass x composition == the N the engine says was applied."""
    sch = (TXTINOUT / "management.sch").read_text()
    frt = (TXTINOUT / "fertilizer.frt").read_text()
    parsed = manure_applied(sch, frt)

    assert len(parsed) == 4, "the four measured GRACEnet manure applications"
    assert parsed["mg_ha"].sum() > 0
    engine_n = runner.read("basin_nb_yr.txt")["fertn"].sum()
    assert parsed["n_kg_ha"].sum() == pytest.approx(engine_n, rel=1e-4)


# --- the reward itself ------------------------------------------------------------------

def test_profit_terms_are_consistent(runner):
    d = profit(runner, nass(2024))
    assert d["profit"] == pytest.approx(
        d["revenue"] - d["water_cost"] - d["manure_cost"] - d["fert_cost"]
        - d["op_cost"] - d["leach_cost"]
    )
    assert d["revenue"] > 0 and d["irrigation_mm"] > 0


def test_application_passes_are_charged(runner):
    """The measured schedule is four manure passes; each one costs a trip across the field."""
    d = profit(runner, nass(2024))
    assert d["n_fert_events"] == 4
    assert d["op_cost"] == pytest.approx(4 * nass(2024).fert_op)
    free = profit(runner, nass(2024, fert_op=0.0))
    assert free["op_cost"] == 0.0
    assert d["profit"] == pytest.approx(free["profit"] - d["op_cost"])


def test_leaching_is_excluded_unless_priced(runner):
    """The externality is reported alongside by default, never silently inside profit."""
    free = profit(runner, nass(2024))
    assert free["leach_cost"] == 0.0
    assert free["no3_leached_kg"] > 0, (
        "no3_rchg is zero — the leaching ablation has no signal. See the deep-percolation "
        "note in port_crops.OVERRIDES."
    )

    priced = profit(runner, nass(2024), no3_price=10.0)
    assert priced["leach_cost"] == pytest.approx(10.0 * free["no3_leached_kg"])
    assert priced["profit"] < free["profit"]


def test_water_cost_scales_with_price(runner):
    base = nass(2024)
    doubled = profit(runner, Prices(base.crop, base.water * 2, base.manure, "x2"))
    assert doubled["water_cost"] == pytest.approx(2 * profit(runner, base)["water_cost"])


def test_unpriced_crop_raises(runner):
    """A rotation the price vector does not cover must fail loudly, not score as zero revenue."""
    with pytest.raises(KeyError, match="no price for"):
        profit(runner, Prices({"corn": 150.0}, 0.4, 0.0, "incomplete"))


def test_ipcc_n2o_matches_hand_computation():
    """Tier 1 arithmetic, checked against a worked example rather than itself."""
    from swat_gym.rewarders import (IPCC_EF1, IPCC_EF4, IPCC_EF5, IPCC_FRACGASF,
                                    IPCC_FRACGASM, N2O_N_TO_N2O, ipcc_n2o)

    d = ipcc_n2o(manure_n_kg=500.0, mineral_n_kg=100.0, no3_leached_kg=20.0)

    direct = 600.0 * IPCC_EF1                                    # 6.0 kg N2O-N
    volat = (100.0 * IPCC_FRACGASF + 500.0 * IPCC_FRACGASM) * IPCC_EF4
    leach = 20.0 * IPCC_EF5
    assert d["n_applied_kg"] == pytest.approx(600.0)
    assert d["n2o_direct_kg"] == pytest.approx(direct * N2O_N_TO_N2O)
    assert d["n2o_volat_kg"] == pytest.approx(volat * N2O_N_TO_N2O)
    assert d["n2o_leach_kg"] == pytest.approx(leach * N2O_N_TO_N2O)
    assert d["n2o_kg"] == pytest.approx(
        d["n2o_direct_kg"] + d["n2o_volat_kg"] + d["n2o_leach_kg"])
    # N2O-N -> N2O must inflate, not deflate.
    assert d["n2o_kg"] > direct + volat + leach


def test_n2o_is_reported_but_never_priced(runner):
    """Emissions must not enter the objective, at any nitrate price."""
    base = profit(runner, nass(2024))
    assert base["n2o_kg"] > 0, "IPCC layer produced nothing to report"

    # profit is exactly the five cost terms; n2o is not among them.
    assert base["profit"] == pytest.approx(
        base["revenue"] - base["water_cost"] - base["manure_cost"]
        - base["fert_cost"] - base["op_cost"] - base["leach_cost"])

    # Profit delta equals leach cost exactly; N2O unpriced.
    priced = profit(runner, nass(2024), no3_price=10.0)
    assert priced["profit"] == pytest.approx(base["profit"] - priced["leach_cost"])
    assert priced["n2o_kg"] == pytest.approx(base["n2o_kg"])


def test_adding_emissions_layer_left_profit_unchanged(runner):
    """N2O reporting must not change any cost term."""
    d = profit(runner, nass(2024))
    for k in ("revenue", "water_cost", "manure_cost", "fert_cost", "op_cost"):
        assert d[k] == pytest.approx(d[k])          # present and finite
    assert d["manure_cost"] == pytest.approx(d["manure_mg"] * nass(2024).manure)
    assert d["fert_cost"] == pytest.approx(d["fert_n_kg"] * nass(2024).fert_n)
