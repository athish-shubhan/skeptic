"""Sharpe ratio inference that accounts for non-normality and multiple trials.

All Sharpe ratios here are *per period* (not annualised) unless a function says
otherwise. Annualise only for display, with :func:`annualise`.

References
----------
- Lo, A. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002).
- Bailey, D. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
- Bailey, D. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def _arr(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        raise ValueError("need at least 3 finite observations")
    return a


def sharpe(returns: ArrayLike) -> float:
    """Per-period Sharpe ratio (mean / sample std, ddof=1)."""
    r = _arr(returns)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualise(sr: float, periods_per_year: float) -> float:
    return float(sr * np.sqrt(periods_per_year))


@dataclass(frozen=True)
class Moments:
    sr: float
    skew: float
    kurt: float  # non-excess (normal = 3)
    n: int


def moments(returns: ArrayLike) -> Moments:
    r = _arr(returns)
    return Moments(
        sr=sharpe(r),
        skew=float(sps.skew(r, bias=False)),
        kurt=float(sps.kurtosis(r, fisher=False, bias=False)),
        n=int(r.size),
    )


def sharpe_se(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of the per-period Sharpe ratio under IID, non-normal returns.

    Var(SR) = (1 + SR^2/2 - skew*SR + (kurt-3)/4 * SR^2) / n
    (Lo, 2002 for the normal case; Mertens, 2002 for skew and kurtosis.)
    """
    v = 1.0 + 0.5 * sr**2 - skew * sr + (kurt - 3.0) / 4.0 * sr**2
    return float(np.sqrt(max(v, 1e-12) / n))


def sharpe_ci(returns: ArrayLike, alpha: float = 0.05) -> tuple[float, float, float]:
    """(SR, lower, upper) per-period confidence interval, skew/kurtosis adjusted."""
    m = moments(returns)
    se = sharpe_se(m.sr, m.n, m.skew, m.kurt)
    z = sps.norm.ppf(1 - alpha / 2)
    return m.sr, m.sr - z * se, m.sr + z * se


def probabilistic_sharpe_ratio(
    returns: ArrayLike | None = None,
    sr_benchmark: float = 0.0,
    *,
    sr: float | None = None,
    n: int | None = None,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """PSR: probability that the true Sharpe exceeds ``sr_benchmark``.

    Either pass ``returns`` or the summary statistics (sr, n, skew, kurt).
    """
    if returns is not None:
        m = moments(returns)
        sr, n, skew, kurt = m.sr, m.n, m.skew, m.kurt
    if sr is None or n is None:
        raise ValueError("pass returns, or sr and n")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(max(denom, 1e-12))
    return float(sps.norm.cdf(z))


def sharpe_pvalue(returns: ArrayLike, sr_benchmark: float = 0.0) -> float:
    """One-sided p-value for H0: SR <= sr_benchmark (= 1 - PSR)."""
    return 1.0 - probabilistic_sharpe_ratio(returns, sr_benchmark)


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` Sharpe estimates under the null of zero skill.

    Bailey and Lopez de Prado (2014), eq. for SR_0 (False Strategy Theorem).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return float(sr_mean)
    g = EULER_GAMMA
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_mean + np.sqrt(sr_variance) * ((1 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    returns: ArrayLike,
    n_trials: int,
    sr_variance: float | None = None,
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected best-of-N null Sharpe.

    ``sr_variance`` is the cross-sectional variance of the per-period Sharpe
    ratios of all trials. If unknown, it defaults to the sampling variance of a
    zero-skill Sharpe estimate, 1/n, which is the right null for independent trials.
    """
    if n_trials is None or n_trials < 1:
        raise ValueError("deflated Sharpe needs the number of trials; refusing to guess")
    m = moments(returns)
    var = sr_variance if sr_variance is not None else 1.0 / m.n
    sr0 = expected_max_sharpe(n_trials, var)
    return probabilistic_sharpe_ratio(sr=m.sr, n=m.n, skew=m.skew, kurt=m.kurt, sr_benchmark=sr0)


def min_track_record_length(
    returns: ArrayLike, sr_benchmark: float = 0.0, alpha: float = 0.05
) -> float:
    """Minimum number of observations for PSR(sr_benchmark) to reach 1 - alpha."""
    m = moments(returns)
    if m.sr <= sr_benchmark:
        return float("inf")
    z = sps.norm.ppf(1 - alpha)
    denom = 1.0 - m.skew * m.sr + (m.kurt - 1.0) / 4.0 * m.sr**2
    return float(1.0 + denom * (z / (m.sr - sr_benchmark)) ** 2)


def max_drawdown(returns: ArrayLike) -> float:
    """Maximum drawdown of cumulative *additive* returns (positive number)."""
    r = np.asarray(returns, dtype=float)
    eq = np.cumsum(np.nan_to_num(r))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.max(peak - eq)) if eq.size else 0.0
