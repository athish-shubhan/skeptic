import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as sps

from skeptic import stats as S


def test_sharpe_se_normal_matches_lo_2002():
    # Lo (2002): Var(SR) = (1 + SR^2/2)/T for IID normal returns
    sr, n = 0.1, 500
    assert S.sharpe_se(sr, n) == pytest.approx(np.sqrt((1 + sr**2 / 2) / n))


def test_sharpe_se_by_simulation():
    rng = np.random.default_rng(1)
    n, mu = 250, 0.1
    srs = [S.sharpe(rng.standard_normal(n) + mu) for _ in range(4000)]
    assert np.std(srs) == pytest.approx(S.sharpe_se(mu, n), rel=0.06)


def test_psr_is_half_at_benchmark_and_monotone():
    assert S.probabilistic_sharpe_ratio(sr=0.1, n=100, sr_benchmark=0.1) == pytest.approx(0.5)
    a = S.probabilistic_sharpe_ratio(sr=0.1, n=100)
    b = S.probabilistic_sharpe_ratio(sr=0.1, n=1000)
    assert 0.5 < a < b < 1


def test_psr_closed_form_normal_case():
    # With skew 0 and kurtosis 3 the PSR reduces to Phi(SR sqrt(n-1) / sqrt(1 + SR^2/2)).
    sr, n = 0.08, 300
    expected = sps.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 + sr**2 / 2))
    assert S.probabilistic_sharpe_ratio(sr=sr, n=n) == pytest.approx(expected)


def test_negative_skew_lowers_psr():
    base = S.probabilistic_sharpe_ratio(sr=0.1, n=100)
    neg = S.probabilistic_sharpe_ratio(sr=0.1, n=100, skew=-2.0, kurt=10.0)
    assert neg < base


def test_psr_calibrated_under_null():
    # Under H0 (SR = 0) the p-value 1 - PSR should be ~uniform: ~5% below 0.05.
    rng = np.random.default_rng(7)
    p = [1 - S.probabilistic_sharpe_ratio(rng.standard_t(5, 500)) for _ in range(3000)]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.012)


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(0)
    n_obs, n_trials = 250, 100
    m = rng.standard_normal((300, n_trials, n_obs))
    best = (m.mean(-1) / m.std(-1, ddof=1)).max(-1).mean()
    assert S.expected_max_sharpe(n_trials, 1 / n_obs) == pytest.approx(best, rel=0.05)


def test_dsr_refuses_without_trials():
    with pytest.raises(ValueError):
        S.deflated_sharpe_ratio(np.random.default_rng(0).standard_normal(100), 0)


@settings(max_examples=40, deadline=None)
@given(st.integers(min_value=1, max_value=200), st.integers(min_value=1, max_value=200))
def test_dsr_decreases_with_trials(a, b):
    r = np.random.default_rng(3).standard_normal(500) * 0.01 + 0.002
    lo, hi = sorted((a, b))
    assert S.deflated_sharpe_ratio(r, hi) <= S.deflated_sharpe_ratio(r, lo) + 1e-12


def test_min_trl_gives_psr_at_threshold():
    r = np.random.default_rng(4).standard_normal(2000) * 0.01 + 0.001
    m = S.moments(r)
    n_star = S.min_track_record_length(r)
    psr = S.probabilistic_sharpe_ratio(sr=m.sr, n=round(n_star), skew=m.skew, kurt=m.kurt)
    assert psr == pytest.approx(0.95, abs=0.01)


def test_max_drawdown():
    assert S.max_drawdown([0.1, -0.05, -0.1, 0.2]) == pytest.approx(0.15)
    assert S.max_drawdown([0.1, 0.1]) == 0.0
