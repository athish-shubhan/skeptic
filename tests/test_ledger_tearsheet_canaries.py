import numpy as np
import pandas as pd
import pytest

from skeptic import canaries
from skeptic.costs import FixedBps, backtest
from skeptic.ledger import Ledger
from skeptic.signals import RULES, tsmom
from skeptic.tearsheet import render


def test_ledger_counts_and_variance(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    rng = np.random.default_rng(0)
    for i in range(5):
        led.log("s", "noise", {"i": i}, rng.standard_normal(100))
    s = led.summary("s")
    assert s.n_trials == 5 and s.sr_variance > 0
    assert led.n_trials("other") == 0


def test_tearsheet_refuses_without_trials(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    with pytest.raises(RuntimeError, match="refusing"):
        render(np.random.default_rng(0).standard_normal(300), ledger=led, study="empty")


def test_tearsheet_renders(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    r = np.random.default_rng(0).standard_normal(300) * 0.01
    led.log("s", "x", {}, r)
    page = render(r, ledger=led, study="s", out=tmp_path / "t.html")
    assert "Deflated Sharpe" in page and (tmp_path / "t.html").exists()


def test_backtest_timing_and_costs():
    pos = np.array([0, 1, 1, 0])
    ret = np.array([0.0, 0.01, 0.02, 0.03])
    bt = backtest(pos, ret, FixedBps(10))
    # position set at t=1 earns ret[2] and ret[3]; costs are paid when trading at t=1 and t=3
    assert np.allclose(bt["gross"], [0, 0, 0.02, 0.03])
    assert np.allclose(bt["cost"], [0, 0.001, 0, 0.001])


def test_signals_do_not_look_ahead():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=600, freq="min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.standard_normal(600) * 0.001))
    v = rng.uniform(1, 2, 600)
    bars = pd.DataFrame({"close": c, "high": c, "low": c, "volume": v,
                         "taker_buy_volume": v * rng.uniform(0, 1, 600)}, index=idx)
    params = {"tsmom": {"lookback": 20}, "zscore_mr": {"lookback": 20, "entry": 1},
              "bollinger": {"lookback": 20, "k": 1}, "rsi": {"lookback": 14, "band": 10},
              "vol_breakout": {"lookback": 30, "k": 1}, "flow_imbalance": {"lookback": 5, "threshold": 0.1}}
    for name, fn in RULES.items():
        full = fn(bars, **params[name])
        cut = fn(bars.iloc[:400], **params[name])
        assert np.allclose(full.iloc[:400].fillna(0), cut.fillna(0)), name
    assert tsmom(bars, 5).abs().max() <= 1


def test_noise_canary_rates():
    r = canaries.noise_canary(n_strategies=500, n_obs=500)
    assert 0.02 < r.pass_rate["none"] < 0.09
    assert r.pass_rate["holm"] <= 0.004
    assert r.best_psr > 0.95 and r.best_dsr < 0.95


def test_planted_power_increases_with_length():
    assert canaries.planted_power(1.0, 1) < canaries.planted_power(1.0, 10)
