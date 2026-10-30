"""The usual signals, written so position[t] only uses bars up to and including t.

Each rule takes a bar DataFrame with at least ``close`` (and ``high``, ``low``,
``volume``, ``taker_buy_volume`` where needed) and returns a position series in
[-1, 1] aligned to the bars. :func:`skeptic.costs.backtest` then earns the
next bar's return on it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

Rule = Callable[..., pd.Series]


def _logret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def tsmom(bars: pd.DataFrame, lookback: int) -> pd.Series:
    """Time-series momentum: sign of the trailing ``lookback``-bar return."""
    return np.sign(np.log(bars["close"]).diff(lookback)).fillna(0.0)


def zscore_mr(bars: pd.DataFrame, lookback: int, entry: float) -> pd.Series:
    """Mean reversion: short when price is ``entry`` std above its rolling mean, and vice versa."""
    c = bars["close"]
    z = (c - c.rolling(lookback).mean()) / c.rolling(lookback).std()
    return pd.Series(np.where(z > entry, -1.0, np.where(z < -entry, 1.0, 0.0)), index=bars.index)


def bollinger(bars: pd.DataFrame, lookback: int, k: float) -> pd.Series:
    """Breakout version of Bollinger bands: long above the upper band, short below the lower."""
    c = bars["close"]
    m, s = c.rolling(lookback).mean(), c.rolling(lookback).std()
    return pd.Series(np.where(c > m + k * s, 1.0, np.where(c < m - k * s, -1.0, 0.0)), index=bars.index)


def rsi(bars: pd.DataFrame, lookback: int, band: float) -> pd.Series:
    """RSI contrarian: long below 50-band, short above 50+band (Wilder smoothing)."""
    d = bars["close"].diff()
    up = d.clip(lower=0).ewm(alpha=1 / lookback, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / lookback, adjust=False).mean()
    r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return pd.Series(np.where(r < 50 - band, 1.0, np.where(r > 50 + band, -1.0, 0.0)), index=bars.index)


def vol_breakout(bars: pd.DataFrame, lookback: int, k: float) -> pd.Series:
    """Trade in the direction of a bar return larger than ``k`` trailing std, hold one bar."""
    r = _logret(bars["close"])
    s = r.rolling(lookback).std().shift(1)
    return pd.Series(np.where(r > k * s, 1.0, np.where(r < -k * s, -1.0, 0.0)), index=bars.index)


def flow_imbalance(bars: pd.DataFrame, lookback: int, threshold: float) -> pd.Series:
    """Trade-based order flow imbalance from taker buy volume, contrarian or momentum by sign of threshold.

    imbalance = (2 * taker_buy - volume) / volume, summed over ``lookback`` bars.
    threshold > 0: follow the flow; threshold < 0: fade it.
    """
    v = bars["volume"].rolling(lookback).sum()
    b = bars["taker_buy_volume"].rolling(lookback).sum()
    imb = (2 * b - v) / v.replace(0, np.nan)
    t = abs(threshold)
    sgn = np.sign(threshold)
    return pd.Series(sgn * np.where(imb > t, 1.0, np.where(imb < -t, -1.0, 0.0)), index=bars.index)


RULES: dict[str, Rule] = {
    "tsmom": tsmom,
    "zscore_mr": zscore_mr,
    "bollinger": bollinger,
    "rsi": rsi,
    "vol_breakout": vol_breakout,
    "flow_imbalance": flow_imbalance,
}


def features(bars: pd.DataFrame) -> pd.DataFrame:
    """Feature set shared by the fitted models; every column uses data up to t only."""
    c = bars["close"]
    r = _logret(c)
    f = pd.DataFrame(index=bars.index)
    for lb in (1, 5, 15, 60, 240):
        f[f"ret_{lb}"] = np.log(c).diff(lb)
    for lb in (15, 60):
        f[f"vol_{lb}"] = r.rolling(lb).std()
        f[f"z_{lb}"] = (c - c.rolling(lb).mean()) / c.rolling(lb).std()
    if "taker_buy_volume" in bars:
        v = bars["volume"].replace(0, np.nan)
        f["imb_1"] = (2 * bars["taker_buy_volume"] - bars["volume"]) / v
        f["imb_15"] = f["imb_1"].rolling(15).mean()
        f["vol_rel"] = bars["volume"] / bars["volume"].rolling(240).mean()
    idx = pd.DatetimeIndex(bars.index)
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    return f


class TimeOfDay:
    """Fitted seasonality: go with the sign of the mean next-bar return in each UTC hour,
    estimated on training data only, and only where |t| > t_min."""

    def __init__(self, t_min: float) -> None:
        self.t_min = t_min
        self.table: pd.Series | None = None

    def fit(self, bars: pd.DataFrame) -> TimeOfDay:
        fwd = _logret(bars["close"]).shift(-1)
        hour = pd.DatetimeIndex(bars.index).hour
        g = fwd.groupby(hour)
        t = g.mean() / (g.std() / np.sqrt(g.count()))
        self.table = np.sign(g.mean()).where(t.abs() > self.t_min, 0.0)
        return self

    def predict(self, bars: pd.DataFrame) -> pd.Series:
        assert self.table is not None, "fit first"
        hour = pd.DatetimeIndex(bars.index).hour
        return pd.Series(self.table.reindex(hour).fillna(0.0).to_numpy(), index=bars.index)


class XGBDirection:
    """Gradient-boosted classifier for next-bar direction; position = sign(p - 0.5) beyond a margin."""

    def __init__(self, margin: float, max_depth: int = 3, n_estimators: int = 200, seed: int = 0) -> None:
        self.margin = margin
        self.kw: dict[str, Any] = dict(max_depth=max_depth, n_estimators=n_estimators,
                                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       random_state=seed, n_jobs=2, verbosity=0)
        self.model: Any = None

    def fit(self, bars: pd.DataFrame) -> XGBDirection:
        from xgboost import XGBClassifier

        x = features(bars)
        y = (_logret(bars["close"]).shift(-1) > 0).astype(int)
        ok = x.notna().all(axis=1) & _logret(bars["close"]).shift(-1).notna()
        self.model = XGBClassifier(**self.kw).fit(x[ok], y[ok])
        return self

    def predict(self, bars: pd.DataFrame) -> pd.Series:
        x = features(bars)
        p = pd.Series(0.5, index=bars.index)
        ok = x.notna().all(axis=1)
        p[ok] = self.model.predict_proba(x[ok])[:, 1]
        return pd.Series(np.where(p > 0.5 + self.margin, 1.0, np.where(p < 0.5 - self.margin, -1.0, 0.0)),
                         index=bars.index)
