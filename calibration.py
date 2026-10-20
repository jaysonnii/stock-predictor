"""
Calibration test for historical-bootstrap return ranges.

Question: when we say "80% of the time the 252-day return lands between A and
B", does it actually land there about 80% of the time?

Method, for each ticker and each forecast date t (every STEP_DAYS):
- Use only daily log returns up to and including t (no lookahead).
- Block bootstrap: glue together random 21-day blocks of past returns until
  the horizon is filled, many times, to get a distribution of H-day returns.
  Blocks keep short-term patterns like volatility clustering intact.
- Read off quantiles (5, 10, 25, 50, 75, 90, 95).
- Score against the return that actually happened from t to t + H.

A forecast is well calibrated when:
- the 50% range holds about 50% of outcomes, 80% holds about 80%, 90% about 90%
- misses are split evenly above and below
- the PIT values (where each outcome fell inside its own forecast, 0 to 1) are
  spread evenly, about 10% in each decile

Every forecast here is out of sample by construction, because the bootstrap
at date t only sees data up to t.
"""
import numpy as np
import pandas as pd

from data import _naive_dates

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
INTERVALS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}

BLOCK_DAYS = 21          # one trading month per block
N_SIMS = 2000            # bootstrap paths per forecast
STEP_DAYS = 21           # one forecast per month per ticker
MIN_HISTORY_DAYS = 504   # two years before the first forecast
SEED = 42


def _log_returns(close: pd.Series) -> pd.Series:
    close = close.copy()
    # Plain calendar dates: fresh Yahoo downloads carry daylight-saving offsets.
    close.index = _naive_dates(close.index)
    close = close[~close.index.duplicated()].sort_index()
    close = close[close > 0]
    return np.log(close).diff().dropna()


def bootstrap_horizon_returns(
    past_log_returns: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    n_sims: int = N_SIMS,
    block_days: int = BLOCK_DAYS,
) -> np.ndarray:
    """
    Simulated simple returns over `horizon` days, built from blocks of
    `past_log_returns`. Returns an array of length n_sims.
    """
    r = np.asarray(past_log_returns, dtype=float)
    n = len(r)
    if n < block_days:
        raise ValueError("Not enough history for one bootstrap block.")

    # Prefix sums make each block's total return a single subtraction.
    cs = np.concatenate([[0.0], np.cumsum(r)])

    full_blocks, remainder = divmod(horizon, block_days)
    total = np.zeros(n_sims)

    if full_blocks:
        starts = rng.integers(0, n - block_days + 1, size=(n_sims, full_blocks))
        total += (cs[starts + block_days] - cs[starts]).sum(axis=1)
    if remainder:
        starts = rng.integers(0, n - remainder + 1, size=n_sims)
        total += cs[starts + remainder] - cs[starts]

    return np.expm1(total)


def calibrate_ticker(
    close: pd.Series,
    horizon: int = 252,
    step_days: int = STEP_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
    n_sims: int = N_SIMS,
    block_days: int = BLOCK_DAYS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    One row per forecast date: the forecast quantiles, the actual return,
    its PIT value, and whether it fell inside each range.
    """
    log_ret = _log_returns(close)
    values = log_ret.values
    dates = log_ret.index
    rng = np.random.default_rng(seed)

    rows = []
    # Position i means "forecast made at the close of dates[i]".
    last_origin = len(values) - horizon
    for i in range(min_history_days - 1, last_origin, step_days):
        past = values[: i + 1]
        sims = bootstrap_horizon_returns(past, horizon, rng, n_sims, block_days)
        actual = float(np.expm1(values[i + 1 : i + 1 + horizon].sum()))

        qs = np.quantile(sims, QUANTILES)
        row = {"Date": dates[i], "Actual": actual, "PIT": float((sims < actual).mean())}
        for q, v in zip(QUANTILES, qs):
            row[f"Q{int(q * 100):02d}"] = float(v)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    for level, (lo, hi) in INTERVALS.items():
        lo_col, hi_col = f"Q{int(lo * 100):02d}", f"Q{int(hi * 100):02d}"
        out[f"In_{level}"] = (out["Actual"] >= out[lo_col]) & (out["Actual"] <= out[hi_col])
        out[f"Below_{level}"] = out["Actual"] < out[lo_col]
        out[f"Above_{level}"] = out["Actual"] > out[hi_col]
        out[f"Width_{level}"] = out[hi_col] - out[lo_col]
    return out


def summarize(results: pd.DataFrame) -> dict:
    """Coverage, misses above and below, and average width for each range."""
    summary = {"forecasts": int(len(results))}
    for level in INTERVALS:
        summary[f"in_{level}_pct"] = round(results[f"In_{level}"].mean() * 100, 1)
        summary[f"below_{level}_pct"] = round(results[f"Below_{level}"].mean() * 100, 1)
        summary[f"above_{level}_pct"] = round(results[f"Above_{level}"].mean() * 100, 1)
        summary[f"width_{level}_pp"] = round(results[f"Width_{level}"].mean() * 100, 1)
    return summary


def pit_deciles(results: pd.DataFrame) -> list:
    """Percent of outcomes in each PIT decile. Calibrated means about 10 each."""
    counts, _ = np.histogram(results["PIT"].values, bins=np.linspace(0, 1, 11))
    return [round(c / max(len(results), 1) * 100, 1) for c in counts]
