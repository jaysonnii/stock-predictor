# Research

The experiments behind the dashboard, kept as a record of what was tested and what it showed. Nothing in this folder is used by the app except as the source of the fixed settings in `ranges.py` and `portfolio.py`, and `universe.py`, which is also the stock list for the ideas screen.

Every test here is walk forward: each forecast uses only data available on its date and is scored against what happened next. Numbers are percentage points unless noted.

Run any script from the project folder as a module, for example:

```powershell
python -m research.evaluate_calibration --vol
```

Downloaded prices are cached in `data_cache/` and `data_long_cache/`, and results are written to `reports/`. All three are ignored by Git.

## 1. Point forecasts (stopped)

The original goal was a price target. Neither version beat simple baselines.

| Script | What it tested | Result |
|---|---|---|
| `model.py`, `evaluate.py` | Single-stock Random Forest on 24 features, AMZN at 63, 126 and 252 days | Beat the average-return baseline in 0 of 4 periods at every horizon |
| `pooled.py`, `evaluate_pooled.py` | One Random Forest across 70 stocks and 23 ETFs (`universe.py`), target is return minus SPY, 252 days | Worse than "same as SPY" in 4 of 4 periods (period 1: 19.32 vs 17.01 error); beat the average-excess baseline in 1 of 4 |
| `backtest.py` | Earlier walk-forward trading backtest of the single-stock model (drawdown, return-to-volatility ratio), saved as JSON to `reports/backtests/` | Kept for reference; `evaluate.py` is the baseline comparison |

Decision: stop chasing price targets. The dashboard shows ranges instead.

## 2. Ranking and the volatility rule (promising, not steady)

| Script | What it tested | Result |
|---|---|---|
| `ranking.py`, `evaluate_ranking.py` | Do the pooled model's top 20% beat the universe average? With a random-group control | Spread 2.74, 25.55, 6.88, 15.46 by period, 12.67 overall; random control 0.07 |
| `factor_check.py` | The same ranking using one feature at a time | `Volatility_20` alone: 12.34, close to the model's 12.67. `Ret_60` 5.11, `Ret_30` 4.79, `Price_to_MA200` 5.24, `RSI` 1.90 |
| `long_vol_test.py` | "Buy the most volatile 20%" year by year, 1993 to 2025 | Positive in 27 of 33 years, +14.20 overall, but -19.78 in 2000 and -10.11 in 2001 |

The model's ranking edge is mostly one rule: higher volatility. That rule is a bull-market bet, not a steady factor, and survivorship bias (see below) likely inflates it.

## 3. Calibrated ranges (adopted, used by `ranges.py`)

Score: share of real outcomes that landed inside the forecast 80% range. Target 80, with misses split 10 below and 10 above.

| Script | What it tested | Result |
|---|---|---|
| `evaluate_calibration.py` | Plain block bootstrap of each stock's past returns, 93 tickers | 77.3 / 77.9 / 75.6 at 63 / 126 / 252 days. Misses cluster by year; worst years 61 to 65 |
| `widening.py` | A fixed walk-forward widening factor on the plain bootstrap | Rejected. Overshot (84 to 89) at 15 to 31% more width and barely helped bad years |
| `evaluate_calibration.py --vol`, `compare_methods.py` | Volatility-scaled bootstrap (EWMA 0.94, 63-day half-life, 21-day blocks; fixed in advance) | Adopted. 84.9 / 84.7 / 83.2, worst years 80 / 75 / 70. Wider only in turbulent years |
| `widening.py --vol` | Walk-forward shrink on the vol-scaled ranges, with the adoption rule set before running | Adopted at 63 days (k 0.88: 80.0, worst 76) and 126 days (k 0.87: 83.3, worst 77). No shrink at 252 days |

At 252 days, misses fall mostly below the range when a downturn starts from calm conditions (forecasts made in 2019 and 2021).

## 4. Portfolio ranges (adopted, used by `portfolio.py`)

| Script | What it tested | Result |
|---|---|---|
| `evaluate_portfolio.py` | 40 fixed random equal-weight portfolios of 6 tickers. Joint draws (every holding gets the same historical blocks) vs independent draws | Joint: 82.3 / 82.2 / 85.8, worst years 68 / 72 / 66. Independent: 66.2 / 67.3 / 70.3, worst years 46 / 59 / 45 |

Ignoring how stocks move together makes portfolio ranges about 30% too narrow. Small caps outside the universe and concentrated weights were not part of this test.

## 5. Ideas worth a look (signals used by `ideas.py`)

The "Keep an eye on" tab lists ideas from four rules. The two price rules live in `signals.py`, shared by the site and this test, so the site shows exactly what was tested. Settings were fixed before testing.

| Script | What it tested | Result |
|---|---|---|
| `signal_backtest.py` | Momentum (top 20% by 12 to 1 month return, above the 200 day average) and pullback in uptrend (above the 200 day average, 50 above 200, last month down at least one normal month). Picks scored every 21 days over 63 and 252 days vs the average stock and SPY. Luck check: 500 runs with picks shuffled across tickers | Run Nov 4, 2026 on 70 stocks, 1994 to 2026. **Momentum held up:** +3.83 pts a year vs the average stock at 252 days (+1.23 at 63), shuffled picks as good 2.4% of the time (0.0% at 63), ahead in 19 of 32 years, worst 2000 (-24.05). **Pullback failed:** -2.04 at 252 days (-0.99 at 63), shuffled picks as good 87.2% (89.8%), ahead in 10 of 32 years. On made-up random prices the luck check gave false alarms in about 5% of runs, as it should |

Decision: momentum stays and is labeled "held up in testing". Pullback no longer picks ideas; its record is still shown as "Dropped". The "vs SPY" numbers (momentum +10.96 at 252 days) are not used: the equal weight average of today's survivors beat SPY by about 7 pts a year on its own, which is survivorship bias, not skill.

A signal is labeled "held up in testing" only when, at 252 days, it beat the average stock, beat 95% of shuffled runs, and was ahead in at least half the years. This rule was set before any real results were seen. Analyst targets and value/growth cannot be tested with yfinance and are labeled untested.

## Known limits

- **Survivorship bias.** Every universe here is made of companies that still trade today. Companies that failed are missing, so real downside misses are likely worse than measured. Fixing this needs price history that includes delisted stocks, which Yahoo does not provide.
- **Small effective sample.** Monthly forecasts with overlapping windows, across stocks that move together, carry far less independent information than the forecast counts suggest.
- **Stop tuning on this history.** Each further adjustment tested on the same years raises the chance of fitting noise.
