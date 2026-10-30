"""Pluggable transaction cost models.

A cost model maps a position path to a per-period cost (in return units),
charged when the position changes. Reports always show gross and net side by side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class CostModel(Protocol):
    name: str

    def cost(self, positions: np.ndarray) -> np.ndarray: ...


def _turnover(positions: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(positions, dtype=float))
    return np.abs(np.diff(p, prepend=0.0))


@dataclass(frozen=True)
class FixedBps:
    """A flat cost of ``bps`` per unit of turnover (e.g. fee + half-spread)."""

    bps: float
    name: str = "fixed"

    def cost(self, positions: np.ndarray) -> np.ndarray:
        return _turnover(positions) * self.bps * 1e-4


@dataclass(frozen=True)
class SpreadCost:
    """Half the quoted spread (in bps, per period) plus a fee, per unit turnover."""

    spread_bps: np.ndarray
    fee_bps: float = 0.0
    name: str = "spread"

    def cost(self, positions: np.ndarray) -> np.ndarray:
        sp = np.asarray(self.spread_bps, dtype=float)
        return _turnover(positions) * (0.5 * sp + self.fee_bps) * 1e-4


@dataclass(frozen=True)
class ImpactCost:
    """Cost from a size-dependent impact curve, e.g. the book-walk curve from ExecLab.

    ``curve(notional) -> bps`` gives the all-in cost of trading ``notional``.
    Position units are multiplied by ``notional_per_unit`` before calling it.
    """

    curve: Callable[[np.ndarray], np.ndarray]
    notional_per_unit: float
    fee_bps: float = 0.0
    name: str = "impact"

    def cost(self, positions: np.ndarray) -> np.ndarray:
        to = _turnover(positions)
        bps = np.where(to > 0, self.curve(to * self.notional_per_unit), 0.0)
        return to * (bps + self.fee_bps) * 1e-4


def square_root_curve(y: float, sigma_bps: float, adv: float) -> Callable[[np.ndarray], np.ndarray]:
    """Square-root impact reference: cost_bps = y * sigma_bps * sqrt(Q / ADV)."""

    def f(q: np.ndarray) -> np.ndarray:
        return y * sigma_bps * np.sqrt(np.asarray(q, dtype=float) / adv)

    return f


def backtest(positions: ArrayLike, returns: ArrayLike, cost_model: CostModel) -> dict[str, np.ndarray]:
    """Gross and net per-period PnL for positions decided at t and held over t+1.

    ``positions[t]`` must only use information up to t; it earns ``returns[t+1]``.
    The cost of moving to ``positions[t]`` is charged at t, when the trade happens,
    so a position always pays to get in before it can earn anything.
    """
    pos = np.nan_to_num(np.asarray(positions, dtype=float))
    ret = np.nan_to_num(np.asarray(returns, dtype=float))
    gross = np.zeros_like(ret)
    gross[1:] = pos[:-1] * ret[1:]
    c = cost_model.cost(pos)
    return {"gross": gross, "cost": c, "net": gross - c, "turnover": _turnover(pos)}
