"""
Walk-forward calibration test for portfolio ranges.

Builds random equal-dollar portfolios from universe.py, makes a forecast
once a month using only data up to that day, and scores it against what the
portfolio actually returned. Two methods are scored on the same forecasts:

- joint: every holding draws the same historical blocks (the method used
  on the dashboard). Keeps how stocks move together.
- independent: separate blocks per holding. Ignores co-movement. A control.

Usage (run from the project folder; uses the data_cache from pooled.py):
    python -m research.evaluate_portfolio              # 63, 126 and 252 days
    python -m research.evaluate_portfolio 252
Results are saved to reports/portfolio_calibration_<H>d.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

from calibration import (
    INTERVALS,
    MIN_HISTORY_DAYS,
    STEP_DAYS,
    add_interval_columns,
    pit_deciles,
    summarize,
)
from research.pooled import _load_raw
from portfolio import QUANTILES, aligned_log_returns, ewma_matrix, portfolio_returns, simulate_joint_returns
from ranges import HORIZONS
from research.universe import TICKERS

REPORTS_DIR = "reports"
DEFAULT_HORIZONS = (63, 126, 252)
N_PORTFOLIOS = 40
HOLDINGS_PER_PORTFOLIO = 6
TEST_SIMS = 2000
SEED = 2026
METHODS = {"joint": True, "independent": False}


def make_portfolios(available, n=N_PORTFOLIOS, size=HOLDINGS_PER_PORTFOLIO, seed=SEED):
    """Fixed random draws of tickers, so every run tests the same portfolios."""
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(available, size=size, replace=False)) for _ in range(n)]


def score_portfolio(log_ret: pd.DataFrame, horizon: int, rng, n_sims=TEST_SIMS,
                    step_days=STEP_DAYS, k=None):
    """One row per forecast date and method. k defaults to the dashboard's factor."""
    r = log_ret.values
    dates = log_ret.index
    n_tickers = r.shape[1]
    weights = np.full(n_tickers, 1.0 / n_tickers)  # equal dollars at each forecast date
    k = np.full(n_tickers, HORIZONS[horizon]["k"] if k is None else k)
    sig_before, sig_after = ewma_matrix(r)

    rows = []
    for i in range(MIN_HISTORY_DAYS - 1, len(r) - horizon, step_days):
        future = r[i + 1 : i + 1 + horizon].sum(axis=0)
        actual = float(np.expm1(future) @ weights)
        for method, joint in METHODS.items():
            sims = simulate_joint_returns(
                r[: i + 1], horizon, rng, ks=k, n_sims=n_sims, joint=joint,
                sigma_before=sig_before[: i + 1], sigma_now=sig_after[i],
            )
            port = portfolio_returns(sims, weights)
            row = {"Date": dates[i], "Method": method, "Actual": actual,
                   "PIT": float((port < actual).mean())}
            for q, v in zip(QUANTILES, np.quantile(port, QUANTILES)):
                row[f"Q{int(q * 100):02d}"] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def run_horizon(horizon: int, closes: dict) -> pd.DataFrame:
    portfolios = make_portfolios(sorted(closes))
    rng = np.random.default_rng(SEED + horizon)
    frames = []
    for n, tickers in enumerate(portfolios, 1):
        log_ret = aligned_log_returns({t: closes[t] for t in tickers})
        if len(log_ret) < MIN_HISTORY_DAYS + horizon:
            print(f"Skipping portfolio {n}: only {len(log_ret)} common days")
            continue
        res = score_portfolio(log_ret, horizon, rng)
        res["Portfolio"] = n
        res["Tickers"] = " ".join(tickers)
        frames.append(res)
        print(f"  portfolio {n}/{len(portfolios)} done", end="\r")
    print()
    return add_interval_columns(pd.concat(frames, ignore_index=True))


def report(results: pd.DataFrame, horizon: int):
    joint = results[results["Method"] == "joint"]
    indep = results[results["Method"] == "independent"]
    print("\n" + "=" * 90)
    print(f"Portfolio range calibration, {horizon}-day horizon, "
          f"{joint['Portfolio'].nunique()} portfolios of {HOLDINGS_PER_PORTFOLIO}, "
          f"{len(joint)} forecasts each method, k = {HORIZONS[horizon]['k']}")

    print("\nOverall")
    print(f"{'Method':<13}" + "".join(
        f"{f'In {lvl}%':>9}{'Below':>7}{'Above':>7}{'Width':>8}" for lvl in INTERVALS))
    for name, df in (("Joint", joint), ("Independent", indep)):
        s = summarize(df)
        print(f"{name:<13}" + "".join(
            f"{s[f'in_{lvl}_pct']:>9.1f}{s[f'below_{lvl}_pct']:>7.1f}"
            f"{s[f'above_{lvl}_pct']:>7.1f}{s[f'width_{lvl}_pp']:>8.1f}" for lvl in INTERVALS))

    print("\nBy forecast year, 80% range (target 80, misses 10 / 10)")
    print(f"{'Year':<6}{'Joint in80':>12}{'below/above':>14}{'Indep in80':>13}{'below/above':>14}")
    worst = {}
    for year in sorted(joint["Date"].dt.year.unique()):
        line = f"{year:<6}"
        for name, df in (("joint", joint), ("indep", indep)):
            s = summarize(df[df["Date"].dt.year == year])
            worst[name] = min(worst.get(name, (101, 0)), (s["in_80_pct"], year))
            width = 12 if name == "joint" else 13
            line += f"{s['in_80_pct']:>{width}.1f}{s['below_80_pct']:>8.1f} / {s['above_80_pct']:<4.1f}"
        print(line)

    print(f"\nWorst year inside the 80% range: joint {worst['joint'][0]:.1f} ({worst['joint'][1]}), "
          f"independent {worst['indep'][0]:.1f} ({worst['indep'][1]})")
    print(f"PIT deciles, joint: {pit_deciles(joint)}")
    print(f"PIT deciles, independent: {pit_deciles(indep)}")
    print("Widths are in percentage points of portfolio return. The joint method should land "
          "near the targets; independent should come out too narrow, most of all in sell-offs.")


def main():
    horizons = [int(a) for a in sys.argv[1:]] or list(DEFAULT_HORIZONS)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    closes = {}
    for t in TICKERS:
        try:
            closes[t] = _load_raw(t)["Close"]
        except Exception as e:
            print(f"Skipping {t}: {e}")

    for horizon in horizons:
        print(f"\nRunning {horizon}-day horizon...")
        results = run_horizon(horizon, closes)
        path = os.path.join(REPORTS_DIR, f"portfolio_calibration_{horizon}d.csv")
        results.to_csv(path, index=False)
        report(results, horizon)
        print(f"\nSaved per-forecast results to {path}")


if __name__ == "__main__":
    main()
