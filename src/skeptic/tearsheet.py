"""One-page HTML tearsheet with an honest, plain-language verdict.

The tearsheet will not render without a ledger study, because a Sharpe ratio
without the number of trials behind it cannot be interpreted.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from . import boot as bs
from .ledger import Ledger
from .stats import (
    annualise,
    deflated_sharpe_ratio,
    max_drawdown,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    sharpe_ci,
)


@dataclass(frozen=True)
class Verdict:
    label: str
    text: str


def verdict(psr: float, dsr: float, net_sr: float, gross_sr: float, n_trials: int,
            pbo: float | None = None) -> Verdict:
    caveats = []
    if gross_sr > 0 and net_sr <= 0:
        return Verdict("Dies on costs", "Positive before costs, not after. Not tradeable as specified.")
    if pbo is not None and pbo > 0.5:
        caveats.append(f"the probability of backtest overfitting is {pbo:.0%}")
    if dsr >= 0.95 and not caveats:
        return Verdict("Survives", f"Deflated Sharpe {dsr:.2f} after {n_trials} trials. "
                       "Worth testing on fresh data before believing.")
    if psr >= 0.95:
        caveats.insert(0, f"significant alone (PSR {psr:.2f}) but not after deflating for "
                       f"{n_trials} trials (DSR {dsr:.2f})")
        return Verdict("Probably noise", "It is " + "; ".join(caveats) + ".")
    return Verdict("Noise", f"Not distinguishable from zero skill (PSR {psr:.2f}).")


def _png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render(net: ArrayLike, *, ledger: Ledger, study: str, gross: ArrayLike | None = None,
           periods_per_year: float = 252, title: str = "Strategy", pbo: float | None = None,
           out: str | Path | None = None) -> str:
    n_trials = ledger.n_trials(study)
    if n_trials == 0:
        raise RuntimeError(f"refusing to render: no trials logged for study {study!r}. "
                           "Log every trial you ran, including the failures.")
    summ = ledger.summary(study)
    net_a = np.asarray(net, dtype=float)
    gross_a = np.asarray(gross, dtype=float) if gross is not None else net_a
    m_net, m_gross = moments(net_a), moments(gross_a)
    sr, lo, hi = sharpe_ci(net_a)
    psr = probabilistic_sharpe_ratio(net_a)
    dsr = deflated_sharpe_ratio(net_a, n_trials, summ.sr_variance if n_trials > 1 else None)
    boot = bs.bootstrap(net_a, bs.SHARPE, n_boot=1000)
    mdd = bs.bootstrap(net_a, bs.MAX_DD, n_boot=1000)
    mtrl = min_track_record_length(net_a)
    v = verdict(psr, dsr, m_net.sr, m_gross.sr, n_trials, pbo)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.cumsum(gross_a), label="gross", color="#8a8a8a", lw=1)
    ax.plot(np.cumsum(net_a), label="net of costs", color="#1f5fa8", lw=1.4)
    ax.set_ylabel("cumulative return")
    ax.set_xlabel("period")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    eq_png = _png(fig)

    a = lambda x: annualise(x, periods_per_year)  # noqa: E731
    rows = [
        ("Sharpe (annualised), gross", f"{a(m_gross.sr):.2f}"),
        ("Sharpe (annualised), net", f"{a(sr):.2f}  [{a(lo):.2f}, {a(hi):.2f}] (Lo/Mertens 95%)"),
        ("Sharpe, block bootstrap 95%", f"[{a(boot.lower):.2f}, {a(boot.upper):.2f}]"),
        ("Probabilistic Sharpe (vs 0)", f"{psr:.3f}"),
        ("Trials logged in this study", f"{n_trials}"),
        ("Deflated Sharpe", f"{dsr:.3f}"),
        ("Probability of backtest overfitting", "not computed" if pbo is None else f"{pbo:.2f}"),
        ("Max drawdown (net), bootstrap 95%", f"{mdd.estimate:.3f}  [{mdd.lower:.3f}, {mdd.upper:.3f}]"),
        ("Min track record length (periods)", "inf" if not np.isfinite(mtrl) else f"{mtrl:,.0f}"),
        ("Observations", f"{m_net.n:,}"),
        ("Skew / kurtosis (net)", f"{m_net.skew:.2f} / {m_net.kurt:.2f}"),
    ]
    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(val)}</td></tr>" for k, val in rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#1d1d1f}}
h1{{font-size:20px;margin:0 0 4px}} .v{{padding:10px 14px;border-left:4px solid #1f5fa8;background:#f3f6fa;margin:12px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #e5e5e5}}
th{{font-weight:500;color:#555;width:45%}} img{{max-width:100%}} .c{{color:#666;font-size:12px}}</style></head>
<body><h1>{html.escape(title)}</h1><div class="c">Study: {html.escape(study)}</div>
<div class="v"><b>{v.label}.</b> {html.escape(v.text)}</div>
<img src="data:image/png;base64,{eq_png}" alt="equity curve">
<table>{table}</table>
<p class="c">Caveats: Sharpe intervals assume the return process is stationary. Deflation uses the
trials logged in the ledger; trials run outside it are not counted. Costs are only as good as the cost model.</p>
</body></html>"""
    if out:
        Path(out).write_text(page, encoding="utf-8")
    return page
