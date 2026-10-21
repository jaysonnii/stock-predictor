"""
Long-history test of the volatility rule.

Question: does "buy the most volatile stocks" beat the average stock across
regimes, including 2000 to 2002 and 2008, or only in the recent bull market?

Method, per trading day:
- Volatility = 20-day standard deviation of daily returns.
- Target = each stock's 252-day return minus SPY's 252-day return.
- Top 20% by volatility; spread = their average target minus the group average.

Results are grouped by calendar year, so you can see which years the rule
worked in and which it didn't.

Usage (run from the project folder):
    python -m research.long_vol_test
"""
import os
import time

import numpy as np
import pandas as pd

from data import _naive_dates, fetch_stock_data
from research.universe import _STOCKS

HORIZON = 252
TOP_FRACTION = 0.2
MIN_TICKERS_PER_DATE = 10
BENCHMARK = "SPY"
CACHE_DIR = "data_long_cache"
DOWNLOAD_DELAY_SECONDS = 0.5

os.makedirs(CACHE_DIR, exist_ok=True)


def _load_long(ticker: str) -> pd.Series:
    """Full available daily closes for one ticker, cached on disk."""
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0).iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True).tz_localize(None).normalize()
        return s

    try:
        df = fetch_stock_data(ticker, period="max")
    except Exception:
        time.sleep(3)
        df = fetch_stock_data(ticker, period="max")

    close = pd.Series(df["Close"].values, index=_naive_dates(df.index), name=ticker)
    close = close[~close.index.duplicated()]
    close.to_frame().to_csv(path)
    time.sleep(DOWNLOAD_DELAY_SECONDS)
    return close


def build_wide():
    """Wide tables: rows are dates, columns are tickers."""
    spy = _load_long(BENCHMARK)
    spy_fwd = spy.shift(-HORIZON) / spy - 1

    vols, fwds = {}, {}
    for ticker in _STOCKS:
        try:
            close = _load_long(ticker)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue
        if len(close) < 300:
            print(f"Skipping {ticker}: only {len(close)} days")
            continue
        daily = close.pct_change()
        vols[ticker] = daily.rolling(20).std()
        fwds[ticker] = close.shift(-HORIZON) / close - 1

    vol = pd.DataFrame(vols)
    fwd = pd.DataFrame(fwds)
    excess = fwd.sub(spy_fwd.reindex(fwd.index), axis=0)
    return vol, excess


def daily_spreads(vol: pd.DataFrame, excess: pd.DataFrame) -> pd.Series:
    dates = vol.index.intersection(excess.index)
    V = vol.reindex(dates).values
    E = excess.reindex(dates).values

    out_dates, out_spreads = [], []
    for i in range(len(dates)):
        v, e = V[i], E[i]
        m = ~np.isnan(v) & ~np.isnan(e)
        if m.sum() < MIN_TICKERS_PER_DATE:
            continue
        vi, ei = v[m], e[m]
        k = max(1, int(round(len(ei) * TOP_FRACTION)))
        top = np.argsort(vi)[-k:]
        out_dates.append(dates[i])
        out_spreads.append(ei[top].mean() - ei.mean())

    return pd.Series(out_spreads, index=pd.DatetimeIndex(out_dates), name="spread")


def main():
    vol, excess = build_wide()
    print(f"\nStocks with usable history: {vol.shape[1]}")

    spreads = daily_spreads(vol, excess)
    by_year = spreads.groupby(spreads.index.year).agg(["mean", "count"])
    positive = spreads.groupby(spreads.index.year).apply(lambda s: (s > 0).mean() * 100)

    print(f"\nVolatility rule, top {int(TOP_FRACTION * 100)}% vs universe average, 252-day horizon")
    print(f"{'Year':<8}{'Avg spread (pp)':>17}{'Days positive %':>18}{'Days':>8}")
    for year in by_year.index:
        print(
            f"{year:<8}{by_year.loc[year, 'mean'] * 100:>17.2f}"
            f"{positive.loc[year]:>18.1f}{int(by_year.loc[year, 'count']):>8}"
        )
    print("-" * 51)
    print(f"Overall average spread: {spreads.mean() * 100:.2f} percentage points")
    print(f"Years with positive average spread: {(by_year['mean'] > 0).sum()} of {len(by_year)}")
    print(
        "\nLook at 2000 to 2002 and 2008 in particular. If the rule loses there, "
        "it is a bull-market bet, not a durable factor."
    )


if __name__ == "__main__":
    main()
