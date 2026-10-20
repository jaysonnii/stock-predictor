"""
Calibration test for bootstrap return ranges across the universe.

Usage (run from the project folder):
    python evaluate_calibration.py              # 63, 126 and 252 day horizons
    python evaluate_calibration.py 252          # one horizon
    python evaluate_calibration.py 63 252       # several

Uses the same price cache as pooled.py (data_cache/). Per-forecast results
are saved to reports/calibration_<H>d.csv.

How to read it: in a calibrated forecast the "Inside" column matches the
range (50, 80, 90) and misses split evenly between "Below" and "Above".
Inside well under target means the ranges are too narrow (overconfident).
"""
import os
import sys

import pandas as pd

from calibration import INTERVALS, calibrate_ticker, pit_deciles, summarize
from pooled import _load_raw
from universe import _ETFS, TICKERS

REPORTS_DIR = "reports"
DEFAULT_HORIZONS = (63, 126, 252)


def run_horizon(horizon: int) -> pd.DataFrame:
    frames = []
    for i, ticker in enumerate(TICKERS):
        try:
            raw = _load_raw(ticker)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue
        res = calibrate_ticker(raw["Close"], horizon=horizon, seed=1000 + i)
        if res.empty:
            print(f"Skipping {ticker}: not enough history")
            continue
        res["Ticker"] = ticker
        res["Type"] = "ETF" if ticker in _ETFS else "Stock"
        frames.append(res)

    if not frames:
        raise ValueError("No tickers had enough history.")
    return pd.concat(frames, ignore_index=True)


def _print_table(title: str, groups: dict):
    print(f"\n{title}")
    header = f"{'Group':<12}{'Forecasts':>10}"
    for level in INTERVALS:
        header += f"{f'In {level}%':>9}{'Below':>7}{'Above':>7}"
    print(header)
    for name, df in groups.items():
        s = summarize(df)
        line = f"{str(name):<12}{s['forecasts']:>10}"
        for level in INTERVALS:
            line += (
                f"{s[f'in_{level}_pct']:>9.1f}"
                f"{s[f'below_{level}_pct']:>7.1f}"
                f"{s[f'above_{level}_pct']:>7.1f}"
            )
        print(line)


def report(results: pd.DataFrame, horizon: int):
    print("\n" + "=" * 78)
    print(f"Bootstrap range calibration, {horizon}-day horizon")
    print(f"{results['Ticker'].nunique()} tickers, {len(results)} forecasts")
    print("Targets: In 50% = 50, In 80% = 80, In 90% = 90. Below and Above should match.")

    _print_table("Overall", {"All": results})
    _print_table("By asset type", dict(tuple(results.groupby("Type"))))
    years = results["Date"].dt.year
    _print_table("By forecast year", dict(tuple(results.groupby(years))))

    s = summarize(results)
    print(
        f"\nAverage range width (percentage points): "
        + ", ".join(f"{lvl}% = {s[f'width_{lvl}_pp']}" for lvl in INTERVALS)
    )
    print(f"PIT deciles (about 10 each if calibrated): {pit_deciles(results)}")
    print(
        "\nCaveat: forecasts a month apart share most of their outcome window, and "
        "stocks move together, so the effective sample is far smaller than the "
        "forecast count. Year-to-year swings are expected; look for a consistent lean."
    )


def main():
    horizons = [int(a) for a in sys.argv[1:]] or list(DEFAULT_HORIZONS)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    for horizon in horizons:
        results = run_horizon(horizon)
        path = os.path.join(REPORTS_DIR, f"calibration_{horizon}d.csv")
        results.to_csv(path, index=False)
        report(results, horizon)
        print(f"\nSaved per-forecast results to {path}")


if __name__ == "__main__":
    main()
