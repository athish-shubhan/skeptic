"""Command line: ``skeptic run config.yaml`` and ``skeptic canaries``."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import canaries
from .io import read_binance_klines, resample_bars
from .ledger import Ledger
from .signals import TimeOfDay, XGBDirection
from .tearsheet import render
from .zoo import funnel, run_grid

FITTED = {"time_of_day": TimeOfDay, "xgb_direction": XGBDirection}


def _load(cfg: dict[str, Any]) -> pd.DataFrame:
    src = cfg["data"]
    if src.get("format", "binance_klines") == "binance_klines":
        bars = read_binance_klines(sorted(glob.glob(src["glob"])))
    else:
        bars = pd.read_csv(src["path"], parse_dates=[src.get("time_col", "time")])
        bars = bars.set_index(src.get("time_col", "time")).tz_localize("UTC")
    if src.get("resample"):
        bars = resample_bars(bars, src["resample"])
    return bars


def cmd_run(path: str) -> int:
    cfg = yaml.safe_load(Path(path).read_text())
    bars = _load(cfg)
    oos = pd.Timestamp(cfg["oos_start"], tz="UTC")
    fitted = {k: (FITTED[k], v) for k, v in cfg.get("fitted", {}).items()}
    trials = run_grid(bars, cfg.get("rules", {}), fitted, oos, cfg["cost_bps"], cfg.get("tag", ""))
    ledger = Ledger(cfg.get("ledger", "ledger.sqlite"))
    ledger.clear(cfg["study"])
    fn = funnel(trials, method=cfg.get("correction", "holm"), ledger=ledger, study=cfg["study"])
    print(json.dumps(fn.counts, indent=2))
    out = Path(cfg.get("out", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    fn.table.to_csv(out / f"{cfg['study']}_trials.csv", index=False)
    best = int(fn.table["net_sr"].idxmax())
    render(trials[best].net, gross=trials[best].gross, ledger=ledger, study=cfg["study"],
           periods_per_year=cfg.get("periods_per_year", 252),
           title=f"Best of {len(trials)}: {trials[best].name} {trials[best].params}",
           out=out / f"{cfg['study']}_tearsheet.html")
    print(f"wrote {out / (cfg['study'] + '_tearsheet.html')}")
    return 0


def cmd_canaries() -> int:
    r = canaries.noise_canary()
    print("noise canary pass rates:", {k: f"{v:.1%}" for k, v in r.pass_rate.items()})
    print(f"best of {r.n_strategies}: PSR {r.best_psr:.3f}, DSR {r.best_dsr:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="skeptic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a signal grid from a YAML config")
    r.add_argument("config")
    sub.add_parser("canaries", help="run the quick noise canary")
    a = ap.parse_args(argv)
    return cmd_run(a.config) if a.cmd == "run" else cmd_canaries()


if __name__ == "__main__":
    raise SystemExit(main())
