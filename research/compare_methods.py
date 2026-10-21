"""
Side-by-side calibration: plain bootstrap vs volatility-scaled bootstrap.

Both methods are scored on the same forecasts (same tickers and dates), and
both are out of sample by construction, so every year counts.

Usage (run from the project folder, after both calibration runs):
    python -m research.evaluate_calibration          # plain bootstrap CSVs
    python -m research.evaluate_calibration --vol    # vol-scaled CSVs
    python -m research.compare_methods               # 63, 126 and 252 days
    python -m research.compare_methods 252
"""
import os
import sys

import pandas as pd

from calibration import INTERVALS, pit_deciles, summarize
from research.evaluate_calibration import DEFAULT_HORIZONS, results_path
from research.widening import parse_dates

METHOD_LABELS = {"bootstrap": "Plain", "vol": "Vol-scaled"}


def load_aligned(horizon: int):
    frames = {}
    for method in METHOD_LABELS:
        path = results_path(horizon, method)
        if not os.path.exists(path):
            flag = " --vol" if method == "vol" else ""
            print(f"\nMissing {path}. Run: python -m research.evaluate_calibration {horizon}{flag}")
            return None
        df = pd.read_csv(path)
        df["Date"] = parse_dates(df["Date"])
        frames[method] = df

    keys = ["Ticker", "Date"]
    common = frames["bootstrap"][keys].merge(frames["vol"][keys], on=keys)
    return {m: df.merge(common, on=keys) for m, df in frames.items()}


def report(horizon: int):
    data = load_aligned(horizon)
    if data is None:
        return

    print("\n" + "=" * 90)
    print(f"Plain vs vol-scaled bootstrap, {horizon}-day horizon, "
          f"{len(data['bootstrap'])} matched forecasts")

    print(f"\nOverall")
    print(f"{'Method':<12}" + "".join(
        f"{f'In {lvl}%':>9}{'Below':>7}{'Above':>7}{'Width':>8}" for lvl in INTERVALS))
    for method, label in METHOD_LABELS.items():
        s = summarize(data[method])
        print(f"{label:<12}" + "".join(
            f"{s[f'in_{lvl}_pct']:>9.1f}{s[f'below_{lvl}_pct']:>7.1f}"
            f"{s[f'above_{lvl}_pct']:>7.1f}{s[f'width_{lvl}_pp']:>8.1f}" for lvl in INTERVALS))

    print(f"\nBy forecast year, 80% range (target 80, misses 10 / 10)")
    print(f"{'Year':<6}{'Plain in80':>12}{'below/above':>14}{'width':>8}"
          f"{'Vol in80':>11}{'below/above':>14}{'width':>8}")
    worst = {}
    for year in sorted(data["bootstrap"]["Date"].dt.year.unique()):
        line = f"{year:<6}"
        for method in METHOD_LABELS:
            df = data[method]
            s = summarize(df[df["Date"].dt.year == year])
            worst[method] = min(worst.get(method, 101), s["in_80_pct"])
            line += (f"{s['in_80_pct']:>{12 if method == 'bootstrap' else 11}.1f}"
                     f"{s['below_80_pct']:>8.1f} / {s['above_80_pct']:<4.1f}"
                     f"{s['width_80_pp']:>7.1f}")
        print(line)

    print("\nWorst year inside the 80% range: " + ", ".join(
        f"{label} {worst[m]:.1f}" for m, label in METHOD_LABELS.items()))
    for method, label in METHOD_LABELS.items():
        print(f"PIT deciles, {label}: {pit_deciles(data[method])}")
    print("Widths are in percentage points of return. Better means: in80 near 80 "
          "in more years, a higher worst year, and misses split evenly.")


def main():
    horizons = [int(a) for a in sys.argv[1:]] or list(DEFAULT_HORIZONS)
    for horizon in horizons:
        report(horizon)


if __name__ == "__main__":
    main()
