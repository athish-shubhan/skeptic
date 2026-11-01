"""Loaders for the bar data used in the signal zoo."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _ts(x: pd.Series) -> pd.DatetimeIndex:
    """Binance switched spot timestamps from ms to us in 2025; detect by magnitude."""
    if x.iloc[0] > 1e14:
        return pd.DatetimeIndex(pd.to_datetime(x, unit="us", utc=True))
    return pd.DatetimeIndex(pd.to_datetime(x, unit="ms", utc=True))


def read_binance_klines(paths: list[str | Path]) -> pd.DataFrame:
    """Read one or more data.binance.vision kline zips into a bar frame indexed by open time (UTC)."""
    frames = []
    for p in sorted(map(Path, paths)):
        with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLS)
        if not str(df.iloc[0, 0]).isdigit():  # some files ship a header row
            df = df.iloc[1:].astype({c: float for c in KLINE_COLS[1:6]})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = df["open_time"].astype("int64")
    idx = _ts(df["open_time"])
    out = df[["open", "high", "low", "close", "volume", "taker_buy_volume", "trades"]].astype(float)
    out.index = idx
    out = out[~out.index.duplicated()].sort_index()
    return out


def resample_bars(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "taker_buy_volume": "sum", "trades": "sum"}
    return bars.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
