"""
Neutral growth experiment (set up Nov 17, 2026, before any results).

Problem: the live ranges center each stock's future on its own past average
growth. The point in time test showed the result: misses land below the range
1.6 to 3 times as often as above, and at 1 year only 3.2% of outcomes reach the
top tenth of the forecast. A stock with a huge past run (Nvidia) gets a very
optimistic middle.

Change tested: center the range on the market's (SPY's) long-run growth: its
average daily log return over all its history up to the forecast date (from
1993), using only data available on that date. Each stock keeps its own swings
and volatility; the shrink k is unchanged.

Design note (before any real results): a first version averaged the market only
over the stock's own history window. A dry run on made-up prices showed that a
window as short as 2 years makes the market average swing wildly (from -0.12%
to +0.01% a day), so it was changed to the market's full history.

Rule, written before running. Adopt "market" growth for the live site only if
ALL hold on the point in time S&P 500 test:
  1. Misses get more balanced: |below_80 - above_80| shrinks at 126 and 252 days.
  2. 80% coverage stays between 78 and 90 at 63, 126 and 252 days.
  3. No worst year (80% coverage) is more than 2 points lower than with own growth.
  4. At 252 days the PIT top decile moves closer to 10.
"zero" growth can be run with --with-zero for context only. It cannot be
adopted: choosing whichever of several variants looks best would fit this history.

Usage (from the project folder, after evaluate_calibration_pit has cached prices):
    python -m research.evaluate_drift_pit
    python -m research.evaluate_drift_pit --with-zero
Saves reports/drift_pit_summary.json.
"""
import json
import os
import sys

import numpy as np

from calibration import INTERVALS, summarize
from research import pit_universe
from research.evaluate_calibration_pit import REPORTS_DIR, run_horizon
from research.long_vol_test import _load_long

HORIZONS = (63, 126, 252)
COVERAGE_BAND = (78.0, 90.0)
MAX_WORST_YEAR_DROP = 2.0


def stats(results) -> dict:
    s = summarize(results)
    by_year = results.groupby(results["Date"].dt.year)["In_80"].mean() * 100
    worst = int(by_year.idxmin())
    counts, _ = np.histogram(results["PIT"].values, bins=np.linspace(0, 1, 11))
    top_decile = float(counts[-1] / max(len(results), 1) * 100)
    return {
        "in_80": s["in_80_pct"], "below_80": s["below_80_pct"], "above_80": s["above_80_pct"],
        "in_50": s["in_50_pct"], "in_90": s["in_90_pct"], "width_80_pp": s["width_80_pp"],
        "median_forecast_pct": round(float(results["Q50"].median() * 100), 2),
        "median_actual_pct": round(float(results["Actual"].median() * 100), 2),
        "worst_year": worst, "worst_year_in_80": round(float(by_year[worst]), 1),
        "pit_top_decile": round(top_decile, 1),
        "in_80_by_year": {int(y): round(float(v), 1) for y, v in by_year.items()},
    }


def verdict(own: dict, mkt: dict) -> tuple:
    checks = []
    for h in (126, 252):
        o, m = own[h], mkt[h]
        gap_o, gap_m = abs(o["below_80"] - o["above_80"]), abs(m["below_80"] - m["above_80"])
        checks.append((f"1. misses more balanced at {h}d (gap {gap_o:.1f} -> {gap_m:.1f})", gap_m < gap_o))
    for h in HORIZONS:
        v = mkt[h]["in_80"]
        checks.append((f"2. 80% coverage at {h}d in {COVERAGE_BAND[0]:.0f} to {COVERAGE_BAND[1]:.0f} ({v:.1f})",
                       COVERAGE_BAND[0] <= v <= COVERAGE_BAND[1]))
    for h in HORIZONS:
        o, m = own[h]["worst_year_in_80"], mkt[h]["worst_year_in_80"]
        checks.append((f"3. worst year at {h}d not more than {MAX_WORST_YEAR_DROP:.0f} pts worse ({o:.1f} -> {m:.1f})",
                       m >= o - MAX_WORST_YEAR_DROP))
    o, m = own[252]["pit_top_decile"], mkt[252]["pit_top_decile"]
    checks.append((f"4. 252d PIT top decile closer to 10 ({o:.1f} -> {m:.1f})", abs(m - 10) < abs(o - 10)))
    return all(ok for _, ok in checks), checks


def main():
    with_zero = "--with-zero" in sys.argv
    os.makedirs(REPORTS_DIR, exist_ok=True)
    spy = _load_long("SPY")
    close, member, usable = pit_universe.build(spy.index, clean="--no-clean" not in sys.argv)
    variants = ["own", "market"] + (["zero"] if with_zero else [])
    out = {v: {} for v in variants}
    for h in HORIZONS:
        for v in variants:
            res = run_horizon(h, close, usable, drift=v, market=spy if v == "market" else None, label=f"[{v}] ")
            out[v][h] = stats(res)

    print("\n" + "=" * 78)
    print("Neutral growth experiment, point in time S&P 500 members (80% range unless noted)")
    head = f"{'':22}" + "".join(f"{v + ' ' + str(h) + 'd':>14}" for h in HORIZONS for v in variants)
    print(head)
    for key, name in [("in_80", "inside 80%"), ("below_80", "below"), ("above_80", "above"),
                      ("width_80_pp", "avg width (pts)"), ("median_forecast_pct", "median forecast %"),
                      ("median_actual_pct", "median actual %"), ("worst_year_in_80", "worst year"),
                      ("pit_top_decile", "PIT top decile"), ("in_50", "inside 50%"), ("in_90", "inside 90%")]:
        print(f"{name:22}" + "".join(f"{out[v][h][key]:>14}" for h in HORIZONS for v in variants))
    print(f"{'worst year (which)':22}" + "".join(f"{out[v][h]['worst_year']:>14}" for h in HORIZONS for v in variants))

    adopt, checks = verdict(out["own"], out["market"])
    print("\nRule (set before running):")
    for text, ok in checks:
        print(f"  [{'pass' if ok else 'FAIL'}] {text}")
    print(f"\nVerdict: {'ADOPT market growth for the live ranges' if adopt else 'KEEP own growth (rule not met)'}")
    if with_zero:
        print("Zero growth is shown for context only and cannot be adopted.")

    with open(os.path.join(REPORTS_DIR, "drift_pit_summary.json"), "w") as f:
        json.dump({"data_quality": {"clean": "--no-clean" not in sys.argv, "dropped": pit_universe.LAST_QUALITY_REPORT},
                   "variants": {v: {str(h): out[v][h] for h in HORIZONS} for v in variants},
                   "adopt_market": bool(adopt), "checks": [{"check": t, "pass": bool(ok)} for t, ok in checks]}, f, indent=2)
    print(f"Saved {os.path.join(REPORTS_DIR, 'drift_pit_summary.json')}")


if __name__ == "__main__":
    main()
