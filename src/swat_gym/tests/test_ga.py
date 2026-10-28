"""The GA arm scores the same objective as PPO and CMA-ES (rule 2)."""
import numpy as np
import pytest

from swat_gym.crop import CropRunner
from swat_gym.crop_env import default_x
from swat_gym.experiments import crops as X
from swat_gym.experiments import crops_ga as G
from swat_gym.rewarders import average
from swat_gym.spec import load_crop


def test_genome_shape_and_initial_output():
    rng = np.random.default_rng(0)
    pop = np.stack([G.init_genome(rng) for _ in range(4)])
    assert pop.shape == (4, G.n_params())
    a = G.act(pop, rng.uniform(0, 1, (4, 9)))
    assert a.shape == (4,) and np.all(np.abs(a) < 0.1)  # starts near default_x


@pytest.mark.parametrize("name,lever", [("wwht", "I"), ("csil", "N")])
def test_zero_genome_equals_default_x_plan(name, lever):
    spec, prices = load_crop(name), average()
    zero = np.zeros((2, G.n_params()))
    with CropRunner(spec, k=2) as r:
        got = G.evaluate(spec, lever, r, zero, 2018, prices, 0.0)
    want = X.score(spec, [X.lever_plan(spec, lever, default_x(spec, lever))], [2018], prices)[0][0]
    for g in got:
        for k in X.KEYS:
            assert g[k] == pytest.approx(want[k], rel=1e-9, abs=1e-9), k


def test_kill_probability():
    p = G.kill_probability(np.arange(16), 16)
    assert p[0] == 0.0 and p[-1] == 1.0 and np.all(np.diff(p) > 0)
    q = G.kill_probability(np.arange(16), 16, safe=4)   # v2
    assert np.all(q[:5] == 0.0) and q[-1] == 1.0 and np.all(np.diff(q[4:]) > 0)


def test_centered_ranks():
    u = G.centered_ranks(np.array([3.0, -1.0, 10.0, 0.0]))
    assert u.tolist() == pytest.approx([1 / 6, -0.5, 0.5, -1 / 6])
