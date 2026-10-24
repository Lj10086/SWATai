"""Windows split and paired statistics (rule 7)."""
from __future__ import annotations

import numpy as np
import pytest

from swat_gym.experiments.common import paired
from swat_gym.windows import TEST_YEARS, TRAIN_YEARS, effective_n, window_years


def test_train_and_test_share_no_calendar_years():
    train = {y for s in TRAIN_YEARS for y in window_years(s)}
    test = {y for s in TEST_YEARS for y in window_years(s)}
    assert not train & test


def test_test_ess_is_1_25():
    assert effective_n(TEST_YEARS) == pytest.approx(1.25, abs=0.01)


def test_paired_has_no_se_key_and_uses_ess():
    a = [{"profit": p, "start_year": s} for p, s in zip([10, 20, 30, 40, 55], TEST_YEARS)]
    b = [{"profit": 0.0, "start_year": s} for s in TEST_YEARS]
    p = paired(a, b)
    assert "se" not in p
    assert p["se_ess"] > p["se_naive"]
    assert p["per_window"] == [10, 20, 30, 40, 55]


def test_bootstrap_resamples_blocks_of_overlapping_windows():
    """Moving-block bootstrap (PAPER §4): block = round(n / ESS) = 4 for the test windows."""
    a = [{"profit": p, "start_year": s} for p, s in zip([10, 20, 30, 40, 55], TEST_YEARS)]
    b = [{"profit": 0.0, "start_year": s} for s in TEST_YEARS]
    p = paired(a, b)
    assert p["boot_block"] == 4
    # Only two blocks exist ([0..3], [1..4]), so every replicate mean lies between them.
    lo, hi = sorted([np.mean([10, 20, 30, 40, 10]), np.mean([20, 30, 40, 55, 20])])
    assert lo - 1e-9 <= p["ci95_boot"][0] <= p["ci95_boot"][1] <= hi + 1e-9


def test_bootstrap_is_iid_when_windows_do_not_overlap():
    starts = [1995, 2003, 2011]  # 8-year windows, disjoint
    a = [{"profit": p, "start_year": s} for p, s in zip([1.0, 5.0, 9.0], starts)]
    b = [{"profit": 0.0, "start_year": s} for s in starts]
    assert paired(a, b)["boot_block"] == 1


def test_paired_refuses_mismatched_windows():
    a = [{"profit": 1.0, "start_year": s} for s in TEST_YEARS]
    with pytest.raises(ValueError):
        paired(a, list(reversed(a)))
