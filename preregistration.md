# Signal zoo pre-registration

Committed before any signal in this file was run on the data below. The git
history shows this file landing in a commit before `reports/zoo_*`.

## Question

How many of the usual short-horizon signals survive transaction costs and honest
statistics on liquid crypto spot pairs?

## Data (Part A: crypto)

- Binance spot 1-minute klines for BTCUSDT and ETHUSDT, April to September 2026
  (data.binance.vision monthly files), resampled to **15-minute** and **60-minute** bars.
- In-sample / warm-up: 1 April to 31 May 2026. Used only to fit the two fitted models
  and to warm up rolling windows.
- Out-of-sample evaluation: **1 June to 30 September 2026**. All reported metrics use only
  these bars. Fitted models are refit at the start of each month on an expanding window.

Part B (AUDUSD, AUDJPY quotes) will be run with the identical grid once the FX data is
available. It is not part of this registration's results.

## Protocol

- Position at bar t uses information up to the close of bar t and earns the log return of bar t+1.
- Positions are in {-1, 0, +1}. No leverage, no compounding.
- **Primary cost: 10 bps per unit of turnover** (Binance spot base-tier taker fee of 0.1%;
  the BTCUSDT and ETHUSDT quoted spread is a fraction of a basis point and is ignored).
- Secondary sensitivity (reported, never used to claim a result): 2 bps per unit turnover,
  roughly a high-volume fee tier.

## Grid (bars are counted at the bar frequency)

| Signal | Parameters | Configs |
|---|---|---|
| Time-series momentum | lookback in {4, 16, 48, 96} | 4 |
| Z-score mean reversion | lookback in {16, 48, 96} x entry in {1, 2} | 6 |
| Bollinger breakout | lookback in {16, 48} x k in {1.5, 2} | 4 |
| RSI contrarian | lookback 14 x band in {15, 25} | 2 |
| Volatility breakout | lookback in {48, 96} x k in {2, 3} | 4 |
| Trade-flow imbalance (taker buy share) | lookback in {1, 4} x threshold in {+0.1, -0.1, +0.2, -0.2} | 8 |
| Time-of-day seasonality (fitted) | t threshold in {1, 2} | 2 |
| XGBoost direction (fitted) | margin in {0.02, 0.05} | 2 |

32 configurations x 2 symbols x 2 bar frequencies = **128 trials**, all logged to the ledger.

Order book imbalance and microprice reversal need L2 data and cannot be computed from
klines. They are tested in ExecLab on the recorded book instead, and are excluded here.

## Success criteria (all must hold)

1. Net-of-cost Sharpe ratio > 0 on the out-of-sample window.
2. Holm-adjusted one-sided p-value < 0.05 across all 128 trials (p from the
   skew- and kurtosis-adjusted PSR against zero).
3. Deflated Sharpe Ratio > 0.95 with N = 128 and the cross-trial Sharpe variance.

The funnel reports counts at each stage: tested, positive gross, positive net, significant
after Holm, survives deflation. Also reported: Probability of Backtest Overfitting (CSCV,
16 blocks) per dataset, computed on the net returns of all 32 configurations.

## What would change my mind

If more than a handful of configurations survive at the primary cost, I will look for a
bug before believing it: re-run the no-lookahead test, check bar alignment, and check
the cost charge timing.
