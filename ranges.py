"""
Price ranges and risk figures for one ticker, using the method chosen in the
calibration work (see claude/range-calibration-findings.md in the project).

Method: volatility-scaled block bootstrap (calibration.vol_scaled_horizon_returns),
then a fixed walk-forward shrink factor k per horizon. k was chosen in walk-forward
tests on 93 tickers (Oct 20, 2026) and is fixed here, not refitted per request.

Center: since Nov 18, 2026 the middle of each range grows at the market's
long-run rate (SPY's average daily log return over all its history), not the
stock's own past average. Each stock keeps its own swings and volatility. This
passed the rule set before testing (research/evaluate_drift_pit.py): misses
became more balanced and the median forecast matched real outcomes better
(1 year: +10.5% vs +11.1% actual, where own growth gave +13.1%), with no worse
worst years.

Hit rates are the share of real outcomes that landed inside each range, from
the point in time test: the same method scored on every stock while it was in
the S&P 500, 1996 to 2026, which includes 2008. That test is still a little
kind, because Yahoo is missing most members that later failed.
"""
import time

import numpy as np
import pandas as pd

from calibration import (
    BLOCK_DAYS,
    EWMA_WARMUP,
    MIN_HISTORY_DAYS,
    _log_returns,
    ewma_volatility,
    vol_scaled_horizon_returns,
)
from data import HISTORY_PERIOD, fetch_stock_data

MARKET_TICKER = "SPY"
MARKET_CACHE_SECONDS = 24 * 60 * 60
RANGE_SIMS = 5000
RANGE_SEED = 42
CACHE_SECONDS = 60 * 60
_cache = {}

# k from the Oct 20, 2026 shrink tests. Hit rates and worst years from the
# Nov 18, 2026 point in time S&P 500 test with market growth (1996 to 2026).
HORIZONS = {
    63: {
        "label": "3 months", "k": 0.88,
        "hit_rate": {50: 53.6, 80: 82.4, 90: 90.7},
        "worst_year_80": 67.7, "worst_year": 1999,
        "note": "When the price lands outside the range, it is below it about 1.4 times as often as above.",
    },
    126: {
        "label": "6 months", "k": 0.87,
        "hit_rate": {50: 56.3, 80: 84.2, 90: 91.6},
        "worst_year_80": 59.3, "worst_year": 2008,
        "note": "In a crash like 2008 the range missed about 4 times in 10. "
                "When the price lands outside the range, it is below it about 1.6 times as often as above.",
    },
    252: {
        "label": "1 year", "k": 1.00,
        "hit_rate": {50: 63.3, 80: 88.7, 90: 94.1},
        "worst_year_80": 63.8, "worst_year": 1999,
        "note": "When the price lands outside the range, it is below it about 2.4 times as often as above, "
                "especially when a downturn starts from calm conditions, as in 2007 and 2008. "
                "The top of the range is optimistic: big gains past it are rarer than it suggests.",
    },
}
RANGES = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}


def shrink(sim_returns: np.ndarray, k: float) -> np.ndarray:
    """Stretch (k > 1) or shrink (k < 1) simulated returns around their median, in log space."""
    log_r = np.log1p(sim_returns)
    m = np.median(log_r)
    return np.expm1(m + k * (log_r - m))


def forecast_ranges(close: pd.Series, horizons=tuple(HORIZONS), seed: int = RANGE_SEED,
                    center: float = None) -> dict:
    """
    Ranges and risk figures for every horizon, from today's close.
    center is the daily log growth for the middle of the range (the market's
    long-run rate in the live app); None uses the stock's own past average.
    """
    log_ret = _log_returns(close)
    values = log_ret.values
    if len(values) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_DAYS} trading days of history for ranges, "
            f"found {len(values)}."
        )

    sigma_before, sigma_after = ewma_volatility(values)
    current_price = float(close.iloc[-1])
    rng = np.random.default_rng(seed)
    out = []

    for horizon in horizons:
        settings = HORIZONS[horizon]
        sims = vol_scaled_horizon_returns(
            values[EWMA_WARMUP:], sigma_before[EWMA_WARMUP:], sigma_after[-1],
            horizon, rng, n_sims=RANGE_SIMS, block_days=BLOCK_DAYS, drift=center,
        )
        sims = shrink(sims, settings["k"])
        worst_cut = np.quantile(sims, 0.10)

        ranges = []
        for level, (lo, hi) in RANGES.items():
            lo_r, hi_r = np.quantile(sims, [lo, hi])
            ranges.append({
                "level": level,
                "low_price": round(current_price * (1 + lo_r), 2),
                "high_price": round(current_price * (1 + hi_r), 2),
                "low_return_pct": round(lo_r * 100, 1),
                "high_return_pct": round(hi_r * 100, 1),
                "historical_hit_rate_pct": settings["hit_rate"][level],
            })

        median_r = float(np.median(sims))
        out.append({
            "horizon": settings["label"],
            "days": horizon,
            "median_price": round(current_price * (1 + median_r), 2),
            "median_return_pct": round(median_r * 100, 1),
            "ranges": ranges,
            "chance_of_loss_pct": round(float((sims < 0).mean()) * 100, 1),
            "bad_case_return_pct": round(float(worst_cut) * 100, 1),
            "avg_worst_10pct_return_pct": round(float(sims[sims <= worst_cut].mean()) * 100, 1),
            "worst_year_hit_rate_80_pct": settings["worst_year_80"],
            "worst_year": settings["worst_year"],
            "shrink_factor": settings["k"],
            "note": settings["note"],
        })

    return {
        "current_price": round(current_price, 2),
        "as_of": str(log_ret.index[-1].date()),
        "current_volatility_annual_pct": round(float(sigma_after[-1]) * np.sqrt(252) * 100, 1),
        "growth_basis": "market" if center is not None else "own",
        "growth_annual_pct": round(float(np.expm1(center * 252)) * 100, 1) if center is not None else None,
        "horizons": out,
    }


_market_cache = {}


def market_center() -> float:
    """
    The market's long-run daily log growth: SPY's average daily log return over
    all its history, as in the test. Refreshed daily; if Yahoo fails, the last
    good value is reused.
    """
    hit = _market_cache.get("center")
    if hit and time.time() - hit[0] < MARKET_CACHE_SECONDS:
        return hit[1]
    try:
        close = fetch_stock_data(MARKET_TICKER, period="max")["Close"]
        lr = _log_returns(close).values
        if len(lr) < 252 * 10:
            raise ValueError(f"only {len(lr)} days of {MARKET_TICKER} history")
        value = float(lr.mean())
    except Exception:
        if hit:
            return hit[1]
        raise
    _market_cache["center"] = (time.time(), value)
    return value


_close_cache = {}


def get_close(ticker: str) -> pd.Series:
    """Daily closes for the ranges, cached for an hour per ticker."""
    ticker = ticker.upper().strip()
    hit = _close_cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    close = fetch_stock_data(ticker, period=HISTORY_PERIOD)["Close"]
    _close_cache[ticker] = (time.time(), close)
    return close


def ranges_for_ticker(ticker: str) -> dict:
    """Fetch history and compute ranges, cached for an hour per ticker."""
    ticker = ticker.upper().strip()
    hit = _cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    result = {"ticker": ticker, **forecast_ranges(get_close(ticker), center=market_center())}
    _cache[ticker] = (time.time(), result)
    return result
