"""
Walk forward test of the two price signals behind "Ideas worth a look".

Question: when a signal picked stocks on a given day, did those stocks go on
to beat (a) the average stock in the group and (b) SPY?

Method:
- Group: the 70 stocks in universe.py, full Yahoo history (cached in
  data_long_cache/, shared with long_vol_test.py).
- Every 21 trading days, apply the signal using only prices up to that day.
- Score the picks' equal weight return over the next 63 and 252 trading
  days, minus the average stock's return and minus SPY's.
- Only days with at least 30 stocks that have enough history, and at least
  3 picks, are scored.
- Luck check: 500 runs where the signal's picks are moved to other stocks by a
  random shuffle of tickers (one shuffle per run). That keeps how many stocks
  it picks and how long it holds them, but breaks any link to which stocks.
  The record says how often the shuffled picks did at least as well.

Analyst targets and value/growth cannot be tested: yfinance only has today's
analyst numbers, and its fundamentals are not recorded as they were known on
each past date.

Survivorship bias: the group is today's survivors, so all results are
likely better than reality, the average stock included.

Usage (from the project folder):
    python -m research.signal_backtest
Writes signal_record.json in the project folder, which the site reads.
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from research.long_vol_test import _load_long
from research.universe import _STOCKS
from signals import (
    MOM_LOOKBACK, MOM_SKIP, MOM_TOP_FRACTION, PULL_Z, PULL_WINDOW,
    momentum_signal, pullback_signal,
)

BENCHMARK = "SPY"
HORIZONS = (63, 252)
STEP_DAYS = 21
MIN_TICKERS = 30
MIN_PICKS = 3
LUCK_RUNS = 500
LUCK_SEED = 2026
OUT_PATH = "signal_record.json"


def load_closes():
    spy = _load_long(BENCHMARK)
    cols = {}
    for t in _STOCKS:
        try:
            cols[t] = _load_long(t)
        except Exception as e:
            print(f"Skipping {t}: {e}")
    close = pd.DataFrame(cols).reindex(spy.index)
    return close, spy


def score(picked: pd.DataFrame, close: pd.DataFrame, spy: pd.Series, horizon: int) -> pd.DataFrame:
    fwd = close.shift(-horizon) / close - 1
    spy_fwd = spy.shift(-horizon) / spy - 1
    enough = close.rolling(MOM_LOOKBACK + 1).count().ge(MOM_LOOKBACK + 1)
    rows, keep = [], []
    for i in range(MOM_LOOKBACK, len(close.index) - horizon, STEP_DAYS):
        d = close.index[i]
        valid = enough.iloc[i] & fwd.iloc[i].notna()
        if valid.sum() < MIN_TICKERS or np.isnan(spy_fwd.iloc[i]):
            continue
        picks = picked.iloc[i] & valid
        if picks.sum() < MIN_PICKS:
            continue
        r = fwd.iloc[i]
        pick_ret = r[picks].mean()
        avg_ret = r[valid].mean()
        keep.append(i)
        rows.append({
            "date": d, "picks": int(picks.sum()), "group": int(valid.sum()),
            "vs_average": pick_ret - avg_ret, "vs_spy": pick_ret - spy_fwd.iloc[i],
        })
    res = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if keep:
        V = (enough.values & fwd.notna().values)[keep]
        res.attrs["luck"] = luck_runs(picked.fillna(False).values[keep], V, fwd.values[keep])
    return res


def luck_runs(P, V, R) -> np.ndarray:
    """Average 'vs the average stock' when the picks are shuffled across tickers."""
    rng = np.random.default_rng(LUCK_SEED)
    R0 = np.where(V, R, 0.0)
    avg = R0.sum(1) / V.sum(1)
    out = np.empty(LUCK_RUNS)
    for b in range(LUCK_RUNS):
        Pp = P[:, rng.permutation(P.shape[1])] & V
        cnt = Pp.sum(1)
        m = cnt > 0
        out[b] = ((R0 * Pp).sum(1)[m] / cnt[m] - avg[m]).mean()
    return out


def summarize(res: pd.DataFrame) -> dict:
    by_year = res.groupby(res.index.year)["vs_average"].mean()
    by_year_spy = res.groupby(res.index.year)["vs_spy"].mean()
    worst = by_year.idxmin()
    luck = res.attrs.get("luck", np.array([]))
    actual = res["vs_average"].mean()
    return {
        "random_as_good_pct": round(float((luck >= actual).mean() * 100), 1) if len(luck) else None,
        "random_95th_pct": round(float(np.percentile(luck, 95) * 100), 2) if len(luck) else None,
        "from": str(res.index.min().date()), "to": str(res.index.max().date()),
        "test_dates": int(len(res)), "avg_picks": round(float(res["picks"].mean()), 1),
        "avg_vs_average_pct": round(float(res["vs_average"].mean() * 100), 2),
        "avg_vs_spy_pct": round(float(res["vs_spy"].mean() * 100), 2),
        "dates_ahead_of_average_pct": round(float((res["vs_average"] > 0).mean() * 100), 1),
        "years_ahead_of_average": int((by_year > 0).sum()), "years": int(len(by_year)),
        "years_ahead_of_spy": int((by_year_spy > 0).sum()),
        "worst_year": int(worst), "worst_year_vs_average_pct": round(float(by_year[worst] * 100), 2),
        "by_year_vs_average_pct": {int(y): round(float(v * 100), 2) for y, v in by_year.items()},
    }


def main():
    close, spy = load_closes()
    print(f"Stocks loaded: {close.shape[1]}, dates {close.index.min().date()} to {close.index.max().date()}")

    mom, _ = momentum_signal(close)
    pull, _, _ = pullback_signal(close)
    record = {
        "tested_on": str(date.today()),
        "group": f"{close.shape[1]} large US stocks in research/universe.py",
        "caveat": "Survivorship bias: the group is today's survivors, so real results are likely worse. "
                  "Overlapping windows mean far fewer independent results than test dates.",
        "signals": {
            "momentum": {"rule": f"Top {int(MOM_TOP_FRACTION * 100)}% by return from 12 to 1 months ago, "
                                 f"and above the 200 day average (skips the last {MOM_SKIP} days)"},
            "pullback": {"rule": f"Above the 200 day average, 50 day above 200 day, and the last "
                                 f"{PULL_WINDOW} days fell at least {abs(PULL_Z):g} normal month"},
            "analyst": {"testable": False, "why": "yfinance only has today's analyst targets"},
            "value": {"testable": False, "why": "yfinance fundamentals are not recorded as known on past dates"},
        },
    }
    for name, picked in (("momentum", mom), ("pullback", pull)):
        record["signals"][name]["testable"] = True
        for h in HORIZONS:
            res = score(picked, close, spy, h)
            if res.empty:
                print(f"{name} {h}d: no scorable dates")
                continue
            s = summarize(res)
            record["signals"][name][f"{h}d"] = s
            print(f"\n{name}, {h} days, {s['from']} to {s['to']}, {s['test_dates']} dates, ~{s['avg_picks']} picks")
            print(f"  vs average stock: {s['avg_vs_average_pct']:+.2f} pts, ahead on {s['dates_ahead_of_average_pct']}% of dates, "
                  f"{s['years_ahead_of_average']} of {s['years']} years; worst {s['worst_year']} ({s['worst_year_vs_average_pct']:+.2f})")
            print(f"  vs SPY: {s['avg_vs_spy_pct']:+.2f} pts, ahead in {s['years_ahead_of_spy']} of {s['years']} years")
            print(f"  luck check: random picks did as well {s['random_as_good_pct']}% of the time (95th percentile {s['random_95th_pct']:+.2f})")

    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
