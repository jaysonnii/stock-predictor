"""
Price ranges and risk figures for one ticker, using the method chosen in the
calibration work (see claude/range-calibration-findings.md in the project).

Method: volatility-scaled block bootstrap (calibration.vol_scaled_horizon_returns),
then a fixed walk-forward shrink factor k per horizon. k and the hit rates
below come from walk-forward tests on 93 tickers and are fixed here, not
refitted per request.

Hit rates are the share of real outcomes that landed inside each range in the
walk-forward test years. They are what the dashboard shows next to each range.
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

RANGE_SIMS = 5000
RANGE_SEED = 42
CACHE_SECONDS = 60 * 60
_cache = {}

# From the Oct 20, 2026 calibration and shrink tests.
HORIZONS = {
    63: {
        "label": "3 months", "k": 0.88,
        "hit_rate": {50: 50.5, 80: 80.0, 90: 89.6},
        "worst_year_80": 76.2, "worst_year": 2020,
        "note": None,
    },
    126: {
        "label": "6 months", "k": 0.87,
        "hit_rate": {50: 55.4, 80: 83.3, 90: 91.3},
        "worst_year_80": 76.8, "worst_year": 2020,
        "note": None,
    },
    252: {
        "label": "1 year", "k": 1.00,
        "hit_rate": {50: 57.0, 80: 84.6, 90: 92.2},
        "worst_year_80": 70.2, "worst_year": 2021,
        "note": "Misses at this horizon fall mostly below the range, "
                "especially when a downturn starts from calm conditions.",
    },
}
RANGES = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}


def shrink(sim_returns: np.ndarray, k: float) -> np.ndarray:
    """Stretch (k > 1) or shrink (k < 1) simulated returns around their median, in log space."""
    log_r = np.log1p(sim_returns)
    m = np.median(log_r)
    return np.expm1(m + k * (log_r - m))


def forecast_ranges(close: pd.Series, horizons=tuple(HORIZONS), seed: int = RANGE_SEED) -> dict:
    """Ranges and risk figures for every horizon, from today's close."""
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
            horizon, rng, n_sims=RANGE_SIMS, block_days=BLOCK_DAYS,
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
        "horizons": out,
    }


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

    result = {"ticker": ticker, **forecast_ranges(get_close(ticker))}
    _cache[ticker] = (time.time(), result)
    return result
