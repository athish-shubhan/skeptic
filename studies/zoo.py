"""Signal zoo study, exactly as pre-registered in preregistration.md.

    python studies/zoo.py --data "../data/binance/*-1m-*.zip"
    python studies/zoo.py --sample            # tiny checked-in sample, for CI only
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skeptic.io import read_binance_klines, resample_bars
from skeptic.ledger import Ledger
from skeptic.pbo import pbo
from skeptic.signals import TimeOfDay, XGBDirection
from skeptic.zoo import Trial, funnel, run_grid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"

RULES = {
    "tsmom": {"lookback": [4, 16, 48, 96]},
    "zscore_mr": {"lookback": [16, 48, 96], "entry": [1, 2]},
    "bollinger": {"lookback": [16, 48], "k": [1.5, 2]},
    "rsi": {"lookback": [14], "band": [15, 25]},
    "vol_breakout": {"lookback": [48, 96], "k": [2, 3]},
    "flow_imbalance": {"lookback": [1, 4], "threshold": [0.1, -0.1, 0.2, -0.2]},
}
FITTED = {"time_of_day": (TimeOfDay, {"t_min": [1, 2]}),
          "xgb_direction": (XGBDirection, {"margin": [0.02, 0.05]})}
OOS_START = pd.Timestamp("2026-06-01", tz="UTC")
PRIMARY_BPS, SECONDARY_BPS = 10.0, 2.0


def load(pattern: str) -> dict[str, pd.DataFrame]:
    files = sorted(glob.glob(pattern))
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        fs = [f for f in files if Path(f).name.startswith(sym)]
        if fs:
            out[sym] = read_binance_klines(fs)
    if not out:
        raise SystemExit(f"no kline files match {pattern}")
    return out


def run(data: dict[str, pd.DataFrame], cost: float, oos: pd.Timestamp) -> tuple[list[Trial], dict]:
    trials: list[Trial] = []
    pbos = {}
    for sym, bars in data.items():
        for rule in ("15min", "60min"):
            b = resample_bars(bars, rule)
            t = run_grid(b, RULES, FITTED, oos, cost, tag=f"{sym}_{rule}")
            m = np.column_stack([x.net for x in t])
            pbos[f"{sym}_{rule}"] = pbo(m, n_blocks=16).pbo
            trials += t
    return trials, pbos


def chart(counts: dict[str, int], counts2: dict[str, int], path: Path) -> None:
    labels = list(counts)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y + 0.18, list(counts.values()), height=0.36, color="#1f5fa8",
            label=f"{PRIMARY_BPS:g} bps per turnover (primary)")
    ax.barh(y - 0.18, list(counts2.values()), height=0.36, color="#b9c9de",
            label=f"{SECONDARY_BPS:g} bps (sensitivity)")
    for yy, v in zip(y, counts.values(), strict=True):
        ax.text(v + 0.8, yy + 0.18, str(v), va="center", fontsize=9)
    for yy, v in zip(y, counts2.values(), strict=True):
        ax.text(v + 0.8, yy - 0.18, str(v), va="center", fontsize=9, color="#555")
    ax.set_yticks(y, labels)
    ax.set_xlabel("configurations")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Signal zoo funnel: BTCUSDT and ETHUSDT, Jun to Sep 2026 out of sample", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data" / "binance" / "*-1m-*.zip"))
    ap.add_argument("--sample", action="store_true", help="run on the checked-in sample")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.sample:
        df = pd.read_csv(ROOT / "sample" / "btcusdt_15m_sample.csv", index_col=0, parse_dates=True)
        data = {"BTCUSDT": df}
        oos = df.index[len(df) // 3]
        suffix = "_sample"
    else:
        data, oos, suffix = load(a.data), OOS_START, ""
    ledger = Ledger(OUT / "ledger.sqlite")
    results = {}
    for cost, study in ((PRIMARY_BPS, "zoo_primary"), (SECONDARY_BPS, "zoo_sensitivity")):
        trials, pbos = run(data, cost, oos)
        ledger.clear(study + suffix)
        fn = funnel(trials, method="holm", ledger=ledger, study=study + suffix)
        fn.table.assign(params=fn.table["params"].map(json.dumps)).to_csv(
            OUT / f"{study}{suffix}_trials.csv", index=False)
        results[study] = {"cost_bps": cost, "counts": fn.counts, "pbo": pbos,
                          "best": fn.table.sort_values("net_sr", ascending=False).head(5)
                          .assign(params=lambda d: d["params"].map(json.dumps))
                          .to_dict(orient="records")}
    chart(results["zoo_primary"]["counts"], results["zoo_sensitivity"]["counts"],
          OUT / f"zoo_funnel{suffix}.png")
    (OUT / f"zoo{suffix}.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: (v["counts"], v["pbo"]) for k, v in results.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
