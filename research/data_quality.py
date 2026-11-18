"""
Data quality filter for the research tests (set Nov 18, 2026, before rerunning).

A few Yahoo price histories are broken, most often a split recorded wrong, so the
price jumps 10x or more overnight. One such day makes a stock's simulated swings
enormous: in a dry run, a single fake x12 day made the 1 year range 60,000 times
too wide. That is where the "10^53" average widths and overflow warnings came from.

A stock is dropped from a test, with its reason listed, if its history has:
  1. any zero, negative or missing price;
  2. any single day up more than 900% or down more than 90% (log move beyond
     ln 10), which real large companies essentially never do in one day;
  3. 20 or more trading days in a row at exactly the same price (stale data).

These rules were fixed before looking at real results and are not tuned.
"""
import numpy as np
import pandas as pd

MAX_ABS_LOG_MOVE = float(np.log(10.0))
MAX_STALE_RUN = 20


def problems(close: pd.Series) -> list:
    """Reasons a price history is unusable; empty if it looks fine. NaN gaps at the ends are ignored."""
    s = pd.Series(close).dropna()
    out = []
    if s.empty:
        return ["no prices"]
    values = s.values.astype(float)
    if not np.all(np.isfinite(values)) or (values <= 0).any():
        out.append("zero, negative or non-numeric price")
        return out
    lr = np.diff(np.log(values))
    if len(lr):
        i = int(np.argmax(np.abs(lr)))
        if abs(lr[i]) > MAX_ABS_LOG_MOVE:
            out.append(f"one-day move of {np.expm1(lr[i]) * 100:+.0f}% on {s.index[i + 1].date()}")
    same = np.r_[False, values[1:] == values[:-1]]
    run, longest, end = 0, 0, None
    for k, flag in enumerate(same):
        run = run + 1 if flag else 0
        if run > longest:
            longest, end = run, k
    if longest + 1 >= MAX_STALE_RUN:
        out.append(f"price unchanged for {longest + 1} trading days (to {s.index[end].date()})")
    return out


def filter_wide(close: pd.DataFrame, *masks: pd.DataFrame):
    """
    Drop broken stocks from a wide table of closes (dates x tickers) and from any
    matching masks. Returns (close, masks..., report) where report maps
    ticker -> list of reasons for every dropped stock.
    """
    report = {}
    for t in close.columns:
        p = problems(close[t])
        if p:
            report[t] = p
    keep = [t for t in close.columns if t not in report]
    return (close[keep], *[m[[c for c in keep if c in m.columns]] for m in masks], report)


def print_report(report: dict, total: int, limit: int = 25):
    print(f"Data quality: dropped {len(report)} of {total} stocks with broken price data.")
    for t, reasons in sorted(report.items())[:limit]:
        print(f"  {t}: {'; '.join(reasons)}")
    if len(report) > limit:
        print(f"  ... and {len(report) - limit} more")
