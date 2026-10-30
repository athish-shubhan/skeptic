import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skeptic import boot as bs
from skeptic.multiple import adjust, benjamini_hochberg, bonferroni, holm
from skeptic.pbo import pbo


def test_stationary_indices_shape_and_range():
    idx = bs.stationary_indices(100, 50, 5.0, np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_mean_block_length():
    idx = bs.stationary_indices(5000, 20, 10.0, np.random.default_rng(0))
    starts = (np.diff(idx, axis=1) != 1) & (np.diff(idx, axis=1) != -4999)
    assert 5000 / (starts.sum(axis=1).mean() + 1) == pytest.approx(10, rel=0.1)


def test_bootstrap_mean_ci_covers_truth():
    rng = np.random.default_rng(2)
    hits = 0
    for s in range(100):
        x = rng.standard_normal(300) + 0.1
        r = bs.bootstrap(x, bs.MEAN, n_boot=400, block=1, seed=s)
        hits += r.lower <= 0.1 <= r.upper
    assert hits >= 88


def test_block_bootstrap_by_group():
    v = np.r_[np.zeros(10), np.ones(10)]
    g = np.repeat([0, 1], 10)
    r = bs.block_bootstrap_by_group(v, g, np.mean, n_boot=500)
    assert r.estimate == 0.5 and r.lower == 0.0 and r.upper == 1.0


def test_known_adjustments():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(bonferroni(p), [0.04, 0.16, 0.12, 0.02])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=50))
def test_adjustment_ordering(p):
    a = np.array(p)
    raw, bh, ho, bo = a, adjust(a, "bh"), adjust(a, "holm"), adjust(a, "bonferroni")
    assert np.all(raw <= bh + 1e-12)
    assert np.all(bh <= ho + 1e-12)
    assert np.all(ho <= bo + 1e-12)
    assert np.all(bo <= 1)


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1600, 30))
    assert pbo(noise, n_blocks=10).pbo > 0.3
    edge = noise.copy()
    edge[:, 0] += 0.3
    assert pbo(edge, n_blocks=10).pbo < 0.05
