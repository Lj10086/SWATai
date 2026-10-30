"""Model params on disk match declared ownership (rule 11b)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.path.abspath(str(ROOT / "scripts")))

check = pytest.importorskip("check_param_state")


def test_workbook_owned_parameters_match_the_workbook():
    """Rule 11b: the crop coefficients are the collaborator's and are not ours to fit."""
    drift = check.workbook_drift()
    assert not drift, (
        "workbook-owned crop parameters have been moved — run "
        "`uv run python scripts/port_crops.py`:\n"
        + "\n".join(f"  {r}.{c}: model {cur:.5f} vs workbook {exp:.5f}" for r, c, cur, exp in drift)
    )


def test_harvest_indices_match_the_workbook_harv_eff():
    """The ``gn_*`` rows are where SWAT+ actually reads the harvest index from."""
    drift = check.harvest_drift()
    assert not drift, "\n".join(
        f"{r}.{c}: model {cur:.5f} vs workbook HARV_EFF {exp:.5f}" for r, c, cur, exp in drift)


def test_shared_parameters_agree_between_their_two_writers():
    """No parameter may be written by both port_crops and optimize."""
    drift = check.shared_drift()
    assert not drift, "\n".join(
        f"{r}.{c}: model {cur:.5f} vs port_crops.OVERRIDES {exp:.5f}"
        for r, c, cur, exp in drift)


def test_maturity_columns_are_still_integer_tokens():
    """Integer columns stay integer tokens."""
    bad = check.integer_columns_intact()
    assert not bad, f"integer columns written as floats: {bad}"
