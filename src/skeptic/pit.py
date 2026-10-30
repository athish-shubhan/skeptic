"""Point-in-time data model.

Every record carries an ``event_time`` (what the data describes) and an
``available_at`` time (when you could first have known it). Feature code never
touches raw frames: it asks a :class:`PITStore` for data *as of* a decision
time, and the store only returns rows whose ``available_at`` is at or before
that time. Lookahead is prevented by the API rather than by discipline.

Example
-------
>>> store = PITStore()
>>> store.add("cpi", df, event_time="ref_month", available_at="release_ts")
>>> view = store.view(pd.Timestamp("2025-03-01", tz="UTC"))
>>> view.get("cpi")            # only releases published by 1 March 2025
>>> store.align("cpi", decision_times, ["value"])   # vectorised as-of join
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class LookaheadError(RuntimeError):
    """Raised when code asks for data that was not available at the decision time."""


@dataclass
class _Table:
    frame: pd.DataFrame
    event_time: str
    available_at: str


@dataclass
class PITStore:
    tables: dict[str, _Table] = field(default_factory=dict)

    def add(self, name: str, frame: pd.DataFrame, *, event_time: str,
            available_at: str | None = None, latency: Any = None) -> None:
        """Register a table.

        If ``available_at`` is not a column, it is derived as event_time + latency
        (latency is required in that case, so you must decide it explicitly).
        """
        df = frame.copy()
        if available_at is None:
            if latency is None:
                raise ValueError(
                    f"{name}: give an available_at column or an explicit latency; "
                    "Skeptic will not assume data was known the instant it happened")
            df["_available_at"] = df[event_time] + latency
            available_at = "_available_at"
        if (df[available_at] < df[event_time]).any():
            raise ValueError(f"{name}: some rows are available before they happen")
        df = df.sort_values(available_at, kind="stable").reset_index(drop=True)
        self.tables[name] = _Table(df, event_time, available_at)

    def view(self, decision_time: Any) -> PITView:
        return PITView(self, decision_time)

    def align(self, name: str, decision_times: Sequence[Any] | pd.Series | np.ndarray,
              columns: Sequence[str], by: str | None = None) -> pd.DataFrame:
        """For each decision time, the latest row available at or before it.

        Vectorised with ``merge_asof`` on ``available_at``; returns one row per
        decision time (NaN where nothing was available yet).
        """
        t = self.tables[name]
        left = pd.DataFrame({"decision_time": np.asarray(decision_times)})
        left["_order"] = np.arange(len(left))
        left = left.sort_values("decision_time", kind="stable")
        cols = [t.available_at, t.event_time, *columns] + ([by] if by else [])
        right = t.frame[list(dict.fromkeys(cols))]
        out = pd.merge_asof(left, right, left_on="decision_time", right_on=t.available_at,
                            direction="backward", allow_exact_matches=True)
        return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


@dataclass
class PITView:
    store: PITStore
    decision_time: Any

    def get(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        t = self.store.tables[name]
        idx = int(np.searchsorted(t.frame[t.available_at].to_numpy(),
                                  np.asarray(self.decision_time), side="right"))
        df = t.frame.iloc[:idx]
        return df if columns is None else df[list(columns)]

    def latest(self, name: str) -> pd.Series:
        df = self.get(name)
        if df.empty:
            raise LookaheadError(f"{name}: nothing available at {self.decision_time}")
        return df.iloc[-1]

    def assert_no_lookahead(self, name: str, used_event_times: Sequence[Any]) -> None:
        """Check that every event time a feature claims to have used was available."""
        avail = self.get(name)[self.store.tables[name].event_time]
        missing = set(pd.Index(used_event_times)) - set(pd.Index(avail))
        if missing:
            raise LookaheadError(f"{name}: used {len(missing)} rows not yet available")


def build_features(store: PITStore, decision_times: Sequence[Any],
                   fn: Callable[[PITView], dict[str, float]]) -> pd.DataFrame:
    """Loop-based feature builder: ``fn`` only ever sees a PITView.

    Slower than :meth:`PITStore.align` but safe for arbitrary feature logic.
    """
    rows = [fn(store.view(t)) | {"decision_time": t} for t in decision_times]
    return pd.DataFrame(rows)
