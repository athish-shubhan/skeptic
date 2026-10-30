"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado and Zhu, 2017).

Given a (T x N) matrix of per-period returns for N strategy configurations,
split time into S equal blocks. For every way of choosing S/2 blocks as in-sample
(IS), pick the configuration with the best IS Sharpe and record where it ranks
out-of-sample (OOS). PBO is the share of splits in which the IS winner ranks
below the OOS median.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike


def _sharpe_cols(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, m.mean(axis=0) / sd, 0.0)


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    is_best_sr: np.ndarray
    oos_sr_of_best: np.ndarray
    n_splits: int

    @property
    def prob_oos_loss(self) -> float:
        """Share of splits where the IS winner has a negative OOS Sharpe."""
        return float(np.mean(self.oos_sr_of_best < 0))


def pbo(returns: ArrayLike, n_blocks: int = 16, max_splits: int | None = 5000,
        seed: int = 0) -> PBOResult:
    m = np.asarray(returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("returns must be (T, N) with N >= 2 configurations")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    t = m.shape[0] - m.shape[0] % n_blocks
    blocks = np.array_split(m[:t], n_blocks)
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    if max_splits and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), size=max_splits, replace=False)
        combos = [combos[i] for i in pick]
    n = m.shape[1]
    logits, is_best, oos_best = [], [], []
    for is_idx in combos:
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        is_sr = _sharpe_cols(np.vstack([blocks[i] for i in is_idx]))
        oos_sr = _sharpe_cols(np.vstack([blocks[i] for i in oos_idx]))
        best = int(np.argmax(is_sr))
        rank = (oos_sr < oos_sr[best]).sum() + 0.5 * ((oos_sr == oos_sr[best]).sum() - 1) + 1
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(is_sr[best])
        oos_best.append(oos_sr[best])
    lg = np.asarray(logits)
    return PBOResult(float(np.mean(lg <= 0)), lg, np.asarray(is_best), np.asarray(oos_best),
                     len(combos))
