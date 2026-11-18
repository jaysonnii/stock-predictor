"""
Range calibration on point in time S&P 500 members (the free partial fix for
survivorship bias).

The original calibration test (evaluate_calibration.py) used 93 tickers that
are alive and large today. This one scores the exact method the site uses
(volatility-scaled bootstrap plus the fixed shrink k from ranges.py) on every
stock while it was in the S&P 500, from 1996 on, using only forecasts made on
dates when the stock was a member.

Same caveat as signal_backtest.py --pit: Yahoo is missing most companies that
later failed or were bought out, so this is closer to the truth but still a
little kind.

Usage (from the project folder):
    python -m research.evaluate_calibration_pit            # 63, 126 and 252 days
    python -m research.evaluate_calibration_pit 252
Saves reports/calibration_pit_<H>d.csv and reports/calibration_pit_summary.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from calibration import INTERVALS, add_interval_columns, calibrate_ticker, pit_deciles, summarize
from ranges import HORIZONS as SITE
from research import pit_universe
from research.long_vol_test import _load_long

REPORTS_DIR = "reports"
DEFAULT_HORIZONS = (63, 126, 252)
SEED = 3000


def apply_shrink(res: pd.DataFrame, k: float) -> pd.DataFrame:
    """The site's shrink k, applied to the forecast quantiles in log space around the median."""
    if k == 1.0 or res.empty:
        return res
    out = res.copy()
    m = np.log1p(out["Q50"])
    for col in [c for c in out.columns if c.startswith("Q") and c[1:].isdigit()]:
        out[col] = np.expm1(m + k * (np.log1p(out[col]) - m))
    return add_interval_columns(out)


def run_horizon(horizon: int, close: pd.DataFrame, usable: pd.DataFrame,
                drift: str = "own", market: pd.Series = None, label: str = None) -> pd.DataFrame:
    k = SITE[horizon]["k"]
    frames = []
    cols = [t for t in close.columns if usable[t].any()]
    for i, t in enumerate(cols):
        s = close[t].dropna()
        res = calibrate_ticker(s, horizon=horizon, seed=SEED + i, method="vol", drift=drift, market_close=market)
        if res.empty:
            continue
        member_then = usable[t].reindex(pd.DatetimeIndex(res["Date"])).fillna(False).values
        res = res[member_then]
        if res.empty:
            continue
        res = apply_shrink(res.reset_index(drop=True), k)
        res["Ticker"] = t
        frames.append(res)
        if (i + 1) % 100 == 0:
            print(f"  {label or ''}{horizon} days: {i + 1} of {len(cols)} stocks scored")
    return pd.concat(frames, ignore_index=True)


def report(results: pd.DataFrame, horizon: int) -> dict:
    site = SITE[horizon]
    s = summarize(results)
    years = results["Date"].dt.year
    by_year = results.groupby(years)["In_80"].mean() * 100
    worst = int(by_year.idxmin())
    print("\n" + "=" * 78)
    print(f"Point in time S&P 500 members, {horizon} day ranges (site method, k = {site['k']})")
    print(f"{results['Ticker'].nunique()} stocks, {len(results)} forecasts, {years.min()} to {years.max()}")
    for lvl in INTERVALS:
        print(f"  {lvl}% range: inside {s[f'in_{lvl}_pct']:.1f}% (site says {site['hit_rate'][lvl]}), "
              f"below {s[f'below_{lvl}_pct']:.1f}, above {s[f'above_{lvl}_pct']:.1f}")
    print(f"  80% range by year: worst {worst} ({by_year[worst]:.1f}); site says worst {site['worst_year']} ({site['worst_year_80']})")
    print("  " + ", ".join(f"{y}: {v:.0f}" for y, v in by_year.items()))
    print(f"  PIT deciles (about 10 each if calibrated): {pit_deciles(results)}")
    return {
        "horizon": horizon, "stocks": int(results["Ticker"].nunique()), "forecasts": int(len(results)),
        "from": int(years.min()), "to": int(years.max()),
        **{f"in_{lvl}_pct": s[f"in_{lvl}_pct"] for lvl in INTERVALS},
        "below_80_pct": s["below_80_pct"], "above_80_pct": s["above_80_pct"],
        "site_in_80_pct": site["hit_rate"][80],
        "worst_year": worst, "worst_year_in_80_pct": round(float(by_year[worst]), 1),
        "in_80_by_year": {int(y): round(float(v), 1) for y, v in by_year.items()},
    }


def main():
    horizons = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(DEFAULT_HORIZONS)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    spy = _load_long("SPY")
    close, member, usable = pit_universe.build(spy.index)
    cov = pit_universe.coverage_by_year(member, usable)
    print(f"\nEver in the S&P 500 since 1996: {member.shape[1]}; with usable Yahoo prices: {int(usable.any().sum())}")
    print(f"Average share of members on a date with usable prices: {cov['covered_pct'].mean():.0f}%")

    summary = {"coverage_by_year": {int(y): float(v) for y, v in cov["covered_pct"].items()}, "horizons": {}}
    for h in horizons:
        results = run_horizon(h, close, usable)
        results.to_csv(os.path.join(REPORTS_DIR, f"calibration_pit_{h}d.csv"), index=False)
        summary["horizons"][h] = report(results, h)
    with open(os.path.join(REPORTS_DIR, "calibration_pit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nCaveat: Yahoo is missing most members that later failed or were bought out, and those are the "
          "ones most likely to have fallen below their ranges. Treat these as a little better than reality.")


if __name__ == "__main__":
    main()
