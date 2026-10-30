"""The batching gate: K plans as K HRUs of one engine run must equal K single runs exactly.

A spec may set ``batchable`` only if this passes. SWAT+ rev 62 keeps N/P demand per plant,
not per HRU (pl_nut_demand.f90), so a plant breaking dormancy reads the previous HRU's
demand: alfalfa gets phantom auto-P events in ~23 % of seasons and is not batchable.
"""
from __future__ import annotations

import numpy as np
import pytest

from swat_gym.crop import TABLES, CropRunner, Plan
from swat_gym.crop_env import default_x
from swat_gym.spec import crop_names, load_crop

IDS = {"unit", "gis_id", "name", "mgt_ops"}  # HRU identity and schedule name, not results
K = 8


def _plans(spec, seed):
    rng = np.random.default_rng(seed)
    plans = [Plan()]
    for lever in spec.levers:
        x0 = tuple(default_x(spec, lever))
        plans += [Plan(lever, tuple(rng.uniform(0, 1, len(x0))), x0) for _ in range(3)]
    return plans[:K]


@pytest.mark.parametrize("name", [n for n in crop_names() if load_crop(n).batchable])
@pytest.mark.parametrize("start_year", [1999, 2019])
def test_batched_hrus_equal_single_runs_exactly(name, start_year):
    spec = load_crop(name)
    plans = _plans(spec, start_year)
    with CropRunner(spec, k=K) as batch, CropRunner(spec) as one:
        assert batch.batched
        got = batch.run(plans, start_year)
        want = [one.run([p], start_year)[0] for p in plans]
    for i, (g, w) in enumerate(zip(got, want)):
        for t in TABLES:
            cols = [c for c in w[t].columns if c not in IDS]
            assert g[t][cols].reset_index(drop=True).equals(w[t][cols].reset_index(drop=True)), \
                f"{name} plan {i} {t}: batched result differs from its single run"


def test_alfalfa_never_batches():
    spec = load_crop("alfa")
    assert not spec.batchable
    with CropRunner(spec, k=K) as r:
        assert not r.batched
