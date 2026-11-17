"""
Price signals for the "Ideas worth a look" list.

The same functions are used by the live list (ideas.py) and by the walk
forward test (research/signal_backtest.py), so what the site shows is
exactly what was tested. Settings were fixed before testing; do not tune
them on the test results.

Every function takes a wide table of daily closes (rows are dates, columns
are tickers) and returns tables of the same shape, using only data up to
each row's date.
"""
import numpy as np
import pandas as pd

# Momentum: return from 12 months ago to 1 month ago (the last month is
# skipped because very recent winners tend to give a little back), top 20%
# of the group, and still above the 200 day average.
MOM_LOOKBACK = 252
MOM_SKIP = 21
MOM_TOP_FRACTION = 0.20

# Pullback in an uptrend: long term trend up (price above the 200 day
# average, 50 day above 200 day) but the last month fell at least one
# "normal month" of this stock's own recent volatility.
MA_SHORT = 50
MA_LONG = 200
PULL_WINDOW = 21
PULL_VOL_WINDOW = 63
PULL_Z = -1.0


def momentum_signal(close: pd.DataFrame, eligible: pd.DataFrame = None):
    """
    Returns (picked, ret_12_1). picked is True where the momentum rule fires.
    eligible (same shape, optional) limits the ranking to stocks that could be
    picked that day, for example S&P 500 members on that date.
    """
    ret = close.shift(MOM_SKIP) / close.shift(MOM_LOOKBACK) - 1
    ma_long = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    ranked = ret if eligible is None else ret.where(eligible)
    rank = ranked.rank(axis=1, pct=True)
    picked = (rank > 1 - MOM_TOP_FRACTION) & (close > ma_long) & ranked.notna()
    return picked, ret


def pullback_signal(close: pd.DataFrame, eligible: pd.DataFrame = None):
    """Returns (picked, ret_1m, z). z is the last month's move in units of a normal month."""
    ma_long = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    ma_short = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    log_close = np.log(close)
    daily_sd = log_close.diff().rolling(PULL_VOL_WINDOW, min_periods=PULL_VOL_WINDOW).std()
    move = log_close - log_close.shift(PULL_WINDOW)
    z = move / (daily_sd * np.sqrt(PULL_WINDOW))
    picked = (close > ma_long) & (ma_short > ma_long) & (z <= PULL_Z)
    if eligible is not None:
        picked &= eligible
    ret_1m = close / close.shift(PULL_WINDOW) - 1
    return picked, ret_1m, z


SIGNALS = {
    "momentum": "Momentum",
    "pullback": "Pullback in uptrend",
    "analyst": "Analyst targets",
    "value": "Value and growth",
}
