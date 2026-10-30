"""Self-tests by simulation: Skeptic measures its own error rates.

Each canary returns plain numbers so they can be asserted in tests and plotted
in the report. Nothing here touches market data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .multiple import adjust
from .splits import PurgedKFold
from .stats import deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio


def _sharpe_rows(m: np.ndarray) -> np.ndarray:
    return m.mean(axis=1) / m.std(axis=1, ddof=1)


def _pvalues(m: np.ndarray) -> np.ndarray:
    """One-sided p-values for H0: SR <= 0, one per row, skew/kurtosis adjusted."""
    sr = _sharpe_rows(m)
    n = m.shape[1]
    sk = sps.skew(m, axis=1, bias=False)
    ku = sps.kurtosis(m, axis=1, fisher=False, bias=False)
    denom = np.maximum(1 - sk * sr + (ku - 1) / 4 * sr**2, 1e-12)
    return 1 - sps.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(denom))


@dataclass(frozen=True)
class NoiseResult:
    n_strategies: int
    n_obs: int
    pass_rate: dict[str, float]
    best_sr: float
    best_psr: float
    best_dsr: float


def noise_canary(n_strategies: int = 1000, n_obs: int = 1260, alpha: float = 0.05,
                 heavy_tails: bool = True, seed: int = 0) -> NoiseResult:
    """``n_strategies`` zero-skill strategies (Student-t returns if heavy_tails).

    Before correction about ``alpha`` should pass; after Holm/Bonferroni almost none;
    and the best one should look great on PSR but fail the deflated Sharpe.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_t(5, size=(n_strategies, n_obs)) if heavy_tails else \
        rng.standard_normal((n_strategies, n_obs))
    m = m * 0.01
    p = _pvalues(m)
    rates = {k: float(np.mean(adjust(p, k) < alpha)) for k in ("none", "bonferroni", "holm", "bh")}
    sr = _sharpe_rows(m)
    best = int(np.argmax(sr))
    return NoiseResult(n_strategies, n_obs, rates, float(sr[best]),
                       probabilistic_sharpe_ratio(m[best]),
                       deflated_sharpe_ratio(m[best], n_strategies, float(sr.var(ddof=1))))


def planted_power(sr_annual: float, years: float, periods_per_year: int = 252,
                  n_trials: int = 1, alpha: float = 0.05, n_sims: int = 400,
                  seed: int = 0) -> float:
    """Share of simulations in which a true annual Sharpe of ``sr_annual`` is detected.

    With n_trials = 1 the test is PSR > 1 - alpha. With n_trials > 1 the planted
    strategy must beat the deflated threshold implied by that many null trials.
    """
    rng = np.random.default_rng(seed)
    n = int(years * periods_per_year)
    mu = sr_annual / np.sqrt(periods_per_year)
    m = rng.standard_normal((n_sims, n)) + mu
    sr = _sharpe_rows(m)
    sr0 = expected_max_sharpe(n_trials, 1.0 / n) if n_trials > 1 else 0.0
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 + 0.5 * sr**2)
    return float(np.mean(sps.norm.cdf(z) > 1 - alpha))


def best_of_n(ns: list[int], n_obs: int = 252, n_sims: int = 200,
              seed: int = 0) -> dict[str, np.ndarray]:
    """Best in-sample annualised Sharpe of N zero-skill strategies, against N.

    Also returns the theoretical expectation and the average DSR of the winner.
    """
    rng = np.random.default_rng(seed)
    nmax = max(ns)
    best_sr, dsr = np.zeros((len(ns), n_sims)), np.zeros((len(ns), n_sims))
    for s in range(n_sims):
        m = rng.standard_normal((nmax, n_obs))
        sr = _sharpe_rows(m)
        for i, n in enumerate(ns):
            k = int(np.argmax(sr[:n]))
            best_sr[i, s] = sr[k]
            sr0 = expected_max_sharpe(n, 1.0 / n_obs)
            z = (sr[k] - sr0) * np.sqrt(n_obs - 1) / np.sqrt(1 + 0.5 * sr[k] ** 2)
            dsr[i, s] = sps.norm.cdf(z)
    ann = np.sqrt(252)
    theory = np.array([expected_max_sharpe(n, 1.0 / n_obs) for n in ns]) * ann
    return {"n": np.array(ns), "best_sr_mean": best_sr.mean(axis=1) * ann,
            "best_sr_p95": np.quantile(best_sr, 0.95, axis=1) * ann, "theory": theory,
            "psr_of_best": sps.norm.cdf(best_sr.mean(axis=1) * np.sqrt(n_obs - 1)),
            "dsr_of_best": dsr.mean(axis=1)}


@dataclass(frozen=True)
class LeakageResult:
    naive_cv_lookahead: float
    pit_aligned_lookahead: float
    shuffled_kfold_overlap: float
    purged_kfold_overlap: float


def leakage_canary(n: int = 6000, horizon: int = 20, seed: int = 0) -> LeakageResult:
    """Two leaks on a pure random walk, where true accuracy is 50%.

    1. Lookahead feature: a feature that includes the next bar's return. Naive CV
       looks great; building the same feature through the point-in-time rule
       (available only once the bar has closed) removes the edge.
    2. Overlapping labels: forward ``horizon``-bar returns with a smooth feature.
       Shuffled K-fold leaks through overlap; purged K-fold with embargo does not.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score

    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n + horizon + 1)
    # --- leak 1: one future bar inside the feature
    y1 = (r[1:n + 1] > 0).astype(int)            # label: sign of next bar return
    leaked = r[1:n + 1] + 2.0 * rng.standard_normal(n)  # uses r[t+1]: lookahead
    pit = np.r_[np.nan, leaked[:-1]]              # same feature, available one bar later
    noise = rng.standard_normal((n, 4))
    x_leak = np.c_[leaked, noise]
    x_pit = np.c_[np.nan_to_num(pit), noise]
    lr = LogisticRegression(max_iter=500)
    naive = cross_val_score(lr, x_leak, y1, cv=KFold(5, shuffle=True, random_state=0)).mean()
    aligned = cross_val_score(lr, x_pit, y1, cv=KFold(5, shuffle=True, random_state=0)).mean()
    # --- leak 2: overlapping forward labels
    price = np.cumsum(r)
    fwd = price[horizon:horizon + n] - price[:n]
    y2 = (fwd > 0).astype(int)
    t = np.arange(n)
    # smooth features: samples close in time look alike, and their labels share most of
    # their forward window, so a flexible model can memorise neighbours
    x2 = np.c_[np.sin(t / 50.0), np.cos(t / 37.0), np.sin(t / 23.0)]
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=2)
    shuffled = cross_val_score(rf, x2, y2, cv=KFold(5, shuffle=True, random_state=0)).mean()
    accs = []
    for sp in PurgedKFold(5, embargo=horizon).split(t, t + horizon):
        rf.fit(x2[sp.train], y2[sp.train])
        accs.append(float((rf.predict(x2[sp.test]) == y2[sp.test]).mean()))
    return LeakageResult(float(naive), float(aligned), float(shuffled), float(np.mean(accs)))
