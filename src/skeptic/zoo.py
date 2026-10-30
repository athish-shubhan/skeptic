"""Run a grid of signals under one protocol and report the survival funnel.

Protocol: positions are computed on the full bar history (rules only look back),
but performance is measured only on the out-of-sample window. Fitted models are
refit on an expanding window at each refit date and predict only the next block.
Every configuration is logged to the ledger, and the funnel counts how many
survive each successive filter.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import signals as sg
from .costs import FixedBps, backtest
from .ledger import Ledger
from .multiple import adjust
from .stats import deflated_sharpe_ratio, moments, probabilistic_sharpe_ratio


@dataclass
class Trial:
    name: str
    params: dict[str, Any]
    gross: np.ndarray
    net: np.ndarray
    turnover: float
    meta: dict[str, Any] = field(default_factory=dict)


def grid(spec: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(spec)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*(spec[k] for k in keys))]


def walk_forward_fit(model_factory: Any, bars: pd.DataFrame, oos_start: pd.Timestamp,
                     refit: str = "MS") -> pd.Series:
    """Expanding-window refit at each period start in the OOS window; predict the next block."""
    pos = pd.Series(0.0, index=bars.index)
    starts = pd.date_range(oos_start, bars.index[-1], freq=refit, tz=bars.index.tz)
    if len(starts) == 0 or starts[0] != oos_start:
        starts = starts.insert(0, oos_start)
    edges = [*starts, bars.index[-1] + pd.Timedelta(1, "ns")]
    for a, b in itertools.pairwise(edges):
        train = bars[bars.index < a].iloc[:-1]  # drop the last bar: its label is in the test block
        test = bars[(bars.index >= a) & (bars.index < b)]
        if len(test) == 0:
            continue
        hist = bars[bars.index < b]
        m = model_factory().fit(train)
        pos.loc[test.index] = m.predict(hist).loc[test.index]
    return pos


def run_grid(bars: pd.DataFrame, rule_grid: dict[str, dict[str, Iterable[Any]]],
             fitted_grid: dict[str, tuple[Any, dict[str, Iterable[Any]]]],
             oos_start: pd.Timestamp, cost_bps: float, tag: str = "") -> list[Trial]:
    ret = np.log(bars["close"]).diff().fillna(0.0).to_numpy()
    oos = (bars.index >= oos_start)
    cm = FixedBps(cost_bps)
    out: list[Trial] = []

    def record(name: str, params: dict[str, Any], pos: pd.Series) -> None:
        p = pos.to_numpy(dtype=float, copy=True)
        p[~oos] = 0.0  # no exposure before the OOS window
        bt = backtest(p, ret, cm)
        out.append(Trial(name, params | ({"data": tag} if tag else {}), bt["gross"][oos],
                         bt["net"][oos], float(bt["turnover"][oos].mean())))

    for name, spec in rule_grid.items():
        for params in grid(spec):
            record(name, params, sg.RULES[name](bars, **params))
    for name, (factory, spec) in fitted_grid.items():
        for params in grid(spec):
            record(name, params, walk_forward_fit(lambda f=factory, p=params: f(**p), bars, oos_start))
    return out


@dataclass(frozen=True)
class Funnel:
    table: pd.DataFrame
    counts: dict[str, int]


def funnel(trials: list[Trial], *, alpha: float = 0.05, method: str = "holm",
           dsr_threshold: float = 0.95, ledger: Ledger | None = None, study: str = "") -> Funnel:
    rows = []
    for t in trials:
        mg, mn = moments(t.gross), moments(t.net)
        rows.append({"signal": t.name, "params": t.params, "gross_sr": mg.sr, "net_sr": mn.sr,
                     "p_net": 1 - probabilistic_sharpe_ratio(t.net), "turnover": t.turnover,
                     "n": mn.n})
        if ledger is not None:
            ledger.log(study, t.name, t.params, t.net, gross_returns=t.gross,
                       extra={"turnover": t.turnover})
    df = pd.DataFrame(rows)
    df["p_adj"] = adjust(df["p_net"].to_numpy(), method)
    n_trials = len(trials)
    sr_var = float(df["net_sr"].var(ddof=1)) if n_trials > 1 else None
    df["dsr"] = [deflated_sharpe_ratio(t.net, n_trials, sr_var) for t in trials]
    method_label = {"holm": "Holm", "bh": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}
    counts = {
        "tested": n_trials,
        "positive gross": int((df["gross_sr"] > 0).sum()),
        "positive net of costs": int((df["net_sr"] > 0).sum()),
        f"significant after {method_label.get(method, method)}": int(((df["net_sr"] > 0) & (df["p_adj"] < alpha)).sum()),
        "survives deflation": int(((df["net_sr"] > 0) & (df["p_adj"] < alpha)
                                   & (df["dsr"] > dsr_threshold)).sum()),
    }
    return Funnel(df, counts)
