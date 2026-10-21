"""
Portfolio Monte Carlo: joint ranges for several holdings at once.

Same method as the single-stock ranges (volatility-scaled block bootstrap),
with one change: every holding draws the SAME blocks of historical dates.
If a simulated path pulls the weeks of March 2020, every stock gets its own
March 2020. Co-movement between holdings is kept that way, without
estimating a correlation matrix.

Each holding keeps its own volatility scaling and the per-horizon shrink
factor k from ranges.py, so within each path a holding's range matches its
own single-stock range. Only the joint behaviour is new.

"independent" mode draws separate blocks per holding. It ignores how stocks
move together and exists only as a control in the walk-forward test.
"""
import numpy as np
import pandas as pd

from calibration import (
    BLOCK_DAYS,
    EWMA_WARMUP,
    VOL_HALF_LIFE,
    _log_returns,
    ewma_volatility,
)

N_SIMS = 4000
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def aligned_log_returns(closes: dict) -> pd.DataFrame:
    """Daily log returns for each ticker on the dates they all traded."""
    frame = pd.concat({t: _log_returns(c) for t, c in closes.items()}, axis=1, join="inner")
    return frame.dropna()


def ewma_matrix(log_returns: np.ndarray):
    """ewma_volatility for each column: (sigma_before, sigma_after), both (days, tickers)."""
    r = np.asarray(log_returns, dtype=float)
    before = np.empty_like(r)
    after = np.empty_like(r)
    for j in range(r.shape[1]):
        before[:, j], after[:, j] = ewma_volatility(r[:, j])
    return before, after


def simulate_joint_returns(
    log_returns: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    ks=None,
    n_sims: int = N_SIMS,
    block_days: int = BLOCK_DAYS,
    half_life: float = VOL_HALF_LIFE,
    joint: bool = True,
    sigma_before=None,
    sigma_now=None,
) -> np.ndarray:
    """
    Simulated simple returns over `horizon` days, shape (n_sims, n_tickers).
    `log_returns` is (days, tickers) of past data only; the last row is today.

    sigma_before / sigma_now can be passed in to skip the EWMA loop (the
    walk-forward test computes them once and slices). They must come from
    ewma_volatility on the same rows, so they use no data past today.
    """
    r = np.asarray(log_returns, dtype=float)
    n_days, n_tickers = r.shape
    ks = np.ones(n_tickers) if ks is None else np.asarray(ks, dtype=float)

    if sigma_before is None:
        sigma_before, sigma_now = ewma_matrix(r)
        sigma_now = sigma_now[-1]
    sig_before = np.asarray(sigma_before, dtype=float)
    sig_now = np.asarray(sigma_now, dtype=float)

    # Drop the EWMA warm-up rows, as in the single-stock method.
    r = r[EWMA_WARMUP:]
    sig_before = sig_before[EWMA_WARMUP:]
    n = len(r)
    if n < block_days:
        raise ValueError("Not enough common history for one bootstrap block.")

    mu = r.mean(axis=0)
    z = (r - mu) / np.maximum(sig_before, 1e-6)
    long_run_var = ((r - mu) ** 2).mean(axis=0)
    phi = 0.5 ** (1.0 / half_life)
    steps = np.arange(1, horizon + 1)[:, None]
    var_path = long_run_var + (sig_now ** 2 - long_run_var) * phi ** steps  # (horizon, tickers)

    cz = np.vstack([np.zeros(n_tickers), np.cumsum(z, axis=0)])
    total = np.tile(mu * horizon, (n_sims, 1))
    start_day = 0
    while start_day < horizon:
        length = min(block_days, horizon - start_day)
        block_sigma = np.sqrt(var_path[start_day : start_day + length].mean(axis=0))
        if joint:
            starts = rng.integers(0, n - length + 1, size=n_sims)
            block = cz[starts + length] - cz[starts]                      # (sims, tickers)
        else:
            starts = rng.integers(0, n - length + 1, size=(n_sims, n_tickers))
            cols = np.arange(n_tickers)
            block = cz[starts + length, cols] - cz[starts, cols]
        total += block * block_sigma
        start_day += length

    # Per-holding shrink around each holding's own median, same k as ranges.py.
    med = np.median(total, axis=0)
    total = med + ks * (total - med)
    return np.expm1(total)


def portfolio_returns(sim_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Buy-and-hold portfolio return for each path. Weights are today's value shares."""
    w = np.asarray(weights, dtype=float)
    return sim_returns @ (w / w.sum())


# --- Dashboard: joint ranges for the user's holdings ---

# From the Oct 20, 2026 walk-forward portfolio test (40 random portfolios of 6).
PORTFOLIO_TESTS = {
    63: {"hit_rate": {50: 53.6, 80: 82.3, 90: 90.3}, "worst_year_80": 68.3, "worst_year": 2018},
    126: {"hit_rate": {50: 56.0, 80: 82.2, 90: 90.0}, "worst_year_80": 72.1, "worst_year": 2019},
    252: {"hit_rate": {50: 58.3, 80: 85.8, 90: 92.8}, "worst_year_80": 66.0, "worst_year": 2021},
}
DASHBOARD_SEED = 42


def forecast_portfolio(closes: dict, shares: dict, seed: int = DASHBOARD_SEED) -> dict:
    """Joint value ranges and risk figures for a set of holdings, from today's prices."""
    from calibration import MIN_HISTORY_DAYS
    from ranges import HORIZONS, RANGES

    tickers = list(closes)
    log_ret = aligned_log_returns(closes)
    if len(log_ret) < MIN_HISTORY_DAYS:
        shortest = min(tickers, key=lambda t: len(closes[t]))
        raise ValueError(
            f"Your holdings share only {len(log_ret)} trading days of history; "
            f"{MIN_HISTORY_DAYS} are needed. {shortest} has the shortest history."
        )

    prices = np.array([float(closes[t].iloc[-1]) for t in tickers])
    values = prices * np.array([float(shares[t]) for t in tickers])
    total_value = float(values.sum())
    rng = np.random.default_rng(seed)

    horizons = []
    for horizon, settings in HORIZONS.items():
        ks = np.full(len(tickers), settings["k"])
        sims = simulate_joint_returns(log_ret.values, horizon, rng, ks=ks)
        port = portfolio_returns(sims, values)
        worst_cut = np.quantile(port, 0.10)
        test = PORTFOLIO_TESTS[horizon]

        ranges = []
        for level, (lo, hi) in RANGES.items():
            lo_r, hi_r = np.quantile(port, [lo, hi])
            ranges.append({
                "level": level,
                "low_value": round(total_value * (1 + lo_r), 2),
                "high_value": round(total_value * (1 + hi_r), 2),
                "low_return_pct": round(lo_r * 100, 1),
                "high_return_pct": round(hi_r * 100, 1),
                "historical_hit_rate_pct": test["hit_rate"][level],
            })
        median_r = float(np.median(port))
        horizons.append({
            "horizon": settings["label"],
            "days": horizon,
            "median_value": round(total_value * (1 + median_r), 2),
            "median_return_pct": round(median_r * 100, 1),
            "ranges": ranges,
            "chance_of_loss_pct": round(float((port < 0).mean()) * 100, 1),
            "bad_case_return_pct": round(float(worst_cut) * 100, 1),
            "bad_case_value": round(total_value * (1 + float(worst_cut)), 2),
            "avg_worst_10pct_return_pct": round(float(port[port <= worst_cut].mean()) * 100, 1),
            "worst_year_hit_rate_80_pct": test["worst_year_80"],
            "worst_year": test["worst_year"],
            "note": settings["note"],
        })

    return {
        "current_value": round(total_value, 2),
        "tickers": tickers,
        "common_history_from": str(log_ret.index[0].date()),
        "as_of": str(log_ret.index[-1].date()),
        "horizons": horizons,
    }


def portfolio_ranges(holdings: list) -> dict:
    """holdings: [{"ticker": "AMZN", "shares": 2.1}, ...]. Fetches prices and forecasts."""
    from ranges import get_close

    shares = {}
    for h in holdings:
        t = str(h["ticker"]).upper().strip()
        if t and float(h["shares"]) > 0:
            shares[t] = shares.get(t, 0.0) + float(h["shares"])
    if not shares:
        raise ValueError("Add at least one holding with shares above zero.")
    closes = {t: get_close(t) for t in shares}
    return forecast_portfolio(closes, shares)
