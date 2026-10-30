"""Stationary block bootstrap (Politis and Romano, 1994), fully vectorised.

Blocks start at uniform random positions and have geometric lengths with mean
``block``. The series is wrapped circularly, so every resample has length T.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def stationary_indices(
    n: int, n_boot: int, block: float, rng: np.random.Generator
) -> np.ndarray:
    """Return an (n_boot, n) array of resampling indices.

    Vectorised: draw a block-start flag per position (prob 1/block) and a random
    start index per position; each position takes the start index of its most
    recent block start, plus its offset into that block, modulo n.
    """
    if block < 1:
        raise ValueError("mean block length must be >= 1")
    p = 1.0 / block
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.broadcast_to(np.arange(n), (n_boot, n))
    last_start_pos = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    offset = pos - last_start_pos
    start_val = np.take_along_axis(starts, last_start_pos, axis=1)
    return (start_val + offset) % n


def default_block(n: int) -> float:
    """Rule-of-thumb mean block length, n^(1/3)."""
    return float(max(1.0, round(n ** (1.0 / 3.0))))


@dataclass(frozen=True)
class BootResult:
    estimate: float
    lower: float
    upper: float
    samples: np.ndarray

    def __repr__(self) -> str:
        return f"BootResult(estimate={self.estimate:.4g}, ci=[{self.lower:.4g}, {self.upper:.4g}])"


def bootstrap(
    x: ArrayLike,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    n_boot: int = 2000,
    block: float | None = None,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 500,
) -> BootResult:
    """Percentile CI for ``statistic`` under the stationary bootstrap.

    ``statistic`` must accept a 2-D array (n_boot, n) and reduce along axis 1,
    so the whole computation stays vectorised. Use :data:`SHARPE`, :data:`MEAN`
    or :data:`MAX_DD`, or write your own.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    b = block or default_block(n)
    rng = np.random.default_rng(seed)
    out = []
    remaining = n_boot
    while remaining > 0:
        k = min(chunk, remaining)
        idx = stationary_indices(n, k, b, rng)
        out.append(statistic(a[idx]))
        remaining -= k
    samples = np.concatenate(out)
    est = float(statistic(a[None, :])[0])
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return BootResult(est, float(lo), float(hi), samples)


def _sharpe2d(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, m.mean(axis=1) / sd, 0.0)


def _mean2d(m: np.ndarray) -> np.ndarray:
    return m.mean(axis=1)


def _median2d(m: np.ndarray) -> np.ndarray:
    return np.median(m, axis=1)


def _maxdd2d(m: np.ndarray) -> np.ndarray:
    eq = np.cumsum(m, axis=1)
    peak = np.maximum.accumulate(np.maximum(eq, 0.0), axis=1)
    return (peak - eq).max(axis=1)


SHARPE = _sharpe2d
MEAN = _mean2d
MEDIAN = _median2d
MAX_DD = _maxdd2d


def block_bootstrap_by_group(
    values: ArrayLike,
    groups: ArrayLike,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootResult:
    """Bootstrap that resamples whole groups (e.g. days) with replacement.

    Use when observations within a group are dependent and groups are the unit
    of independence. ``statistic`` takes a 1-D array of pooled values.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    buckets = [v[g == u] for u in uniq]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(buckets), size=len(buckets))
        samples[i] = statistic(np.concatenate([buckets[j] for j in pick]))
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return BootResult(float(statistic(v)), float(lo), float(hi), samples)
