"""
Walk-forward widening factor for the bootstrap ranges.

The calibration test showed the bootstrap ranges are a little too narrow,
with more misses on the downside. This fixes that by stretching each range
around its median, in log-return space:

    widened = median + k * (quantile - median)

Two versions:
- Symmetric: one k for both sides, fitted so the 80% range holds 80%.
- Asymmetric: k_low for the lower side and k_high for the upper side, fitted
  so 10% of outcomes fall below the 80% range and 10% above.

Walk-forward: for each test year, k is fitted only on forecasts whose
outcome was already known before that year began. The test year is then
scored with that k. No outcome is used to fit the factor that scores it.

Usage (run from the project folder, after evaluate_calibration.py):
    python widening.py              # 63, 126 and 252 day horizons
    python widening.py 252          # one horizon
"""
import os
import sys

import numpy as np
import pandas as pd

from calibration import INTERVALS

REPORTS_DIR = "reports"
DEFAULT_HORIZONS = (63, 126, 252)
K_GRID = np.round(np.arange(0.80, 2.501, 0.01), 2)
MIN_TRAIN_FORECASTS = 500
FIT_LEVEL = 80  # the range the factor is tuned on


def _col(q: float) -> str:
    return f"Q{int(q * 100):02d}"


def widen(quantile: np.ndarray, median: np.ndarray, k: float) -> np.ndarray:
    """Stretch simple-return quantiles around the median by k, in log space."""
    m = np.log1p(median)
    return np.expm1(m + k * (np.log1p(quantile) - m))


def score(results: pd.DataFrame, k_low: float = 1.0, k_high: float = 1.0) -> dict:
    """Percent inside, below and above each range after widening, plus widths."""
    actual = results["Actual"].values
    median = results["Q50"].values
    out = {"forecasts": int(len(results))}
    for level, (lo_q, hi_q) in INTERVALS.items():
        lo = widen(results[_col(lo_q)].values, median, k_low)
        hi = widen(results[_col(hi_q)].values, median, k_high)
        out[f"in_{level}"] = float(np.mean((actual >= lo) & (actual <= hi)) * 100)
        out[f"below_{level}"] = float(np.mean(actual < lo) * 100)
        out[f"above_{level}"] = float(np.mean(actual > hi) * 100)
        out[f"width_{level}"] = float(np.mean(hi - lo) * 100)
    return out


def _best_k(values: list, target: float) -> float:
    errors = np.abs(np.asarray(values) - target)
    return float(K_GRID[int(np.argmin(errors))])


def fit_symmetric(results: pd.DataFrame, level: int = FIT_LEVEL) -> float:
    """One k so the range holds `level` percent of outcomes."""
    return _best_k([score(results, k, k)[f"in_{level}"] for k in K_GRID], level)


def fit_asymmetric(results: pd.DataFrame, level: int = FIT_LEVEL):
    """Separate k for each side so each tail holds (100 - level) / 2 percent."""
    tail = (100 - level) / 2
    k_low = _best_k([score(results, k, 1.0)[f"below_{level}"] for k in K_GRID], tail)
    k_high = _best_k([score(results, 1.0, k)[f"above_{level}"] for k in K_GRID], tail)
    return k_low, k_high


def parse_dates(values) -> pd.Series:
    """
    Calendar dates with no time zone. Yahoo data can carry offsets that switch
    between -05:00 and -04:00 with daylight saving, so the date part is taken
    as written instead of converting through UTC.
    """
    values = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(values):
        if values.dt.tz is not None:
            values = values.dt.tz_localize(None)
        return values.dt.normalize()
    return pd.to_datetime(values.astype(str).str[:10])


def resolve_dates(results: pd.DataFrame, horizon: int) -> pd.Series:
    """Approximate date each forecast's outcome became known."""
    return parse_dates(results["Date"]) + pd.offsets.BDay(horizon)


def walk_forward(results: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per test year: the k values fitted before it, and how they scored."""
    results = results.copy()
    results["Date"] = parse_dates(results["Date"])
    resolved = resolve_dates(results, horizon)
    years = results["Date"].dt.year

    rows = []
    for year in sorted(years.unique()):
        train = results[resolved < pd.Timestamp(year=year, month=1, day=1)]
        test = results[years == year]
        if len(train) < MIN_TRAIN_FORECASTS or test.empty:
            continue

        k_sym = fit_symmetric(train)
        k_low, k_high = fit_asymmetric(train)
        raw = score(test)
        sym = score(test, k_sym, k_sym)
        asym = score(test, k_low, k_high)

        row = {"year": year, "train": len(train), "test": len(test),
               "k_sym": k_sym, "k_low": k_low, "k_high": k_high}
        for name, s in (("raw", raw), ("sym", sym), ("asym", asym)):
            for key, value in s.items():
                if key != "forecasts":
                    row[f"{name}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_test_scores(results: pd.DataFrame, by_year: pd.DataFrame, horizon: int) -> dict:
    """Score every test-year forecast with the k fitted for its own year."""
    results = results.copy()
    results["Date"] = parse_dates(results["Date"])
    years = results["Date"].dt.year
    parts = {"raw": [], "sym": [], "asym": []}

    for _, r in by_year.iterrows():
        test = results[years == r["year"]]
        actual, median = test["Actual"].values, test["Q50"].values
        for name, (kl, kh) in (
            ("raw", (1.0, 1.0)),
            ("sym", (r["k_sym"], r["k_sym"])),
            ("asym", (r["k_low"], r["k_high"])),
        ):
            frame = pd.DataFrame({"Actual": actual, "Q50": median})
            for level, (lo_q, hi_q) in INTERVALS.items():
                frame[_col(lo_q)] = widen(test[_col(lo_q)].values, median, kl)
                frame[_col(hi_q)] = widen(test[_col(hi_q)].values, median, kh)
            parts[name].append(frame)

    return {name: score(pd.concat(frames, ignore_index=True))
            for name, frames in parts.items() if frames}


def report(results: pd.DataFrame, horizon: int):
    by_year = walk_forward(results, horizon)
    print("\n" + "=" * 86)
    print(f"Walk-forward widening, {horizon}-day horizon, tuned on the {FIT_LEVEL}% range")
    if by_year.empty:
        print("Not enough resolved forecasts to fit a factor for any test year.")
        return

    print(f"\nBy test year (k fitted only on outcomes known before that year)")
    print(f"{'Year':<6}{'Train':>7}{'k sym':>7}{'k low':>7}{'k high':>8}"
          f"{'Raw in80':>10}{'Sym in80':>10}{'Asym in80':>11}{'Asym below/above':>18}")
    for _, r in by_year.iterrows():
        print(f"{int(r['year']):<6}{int(r['train']):>7}{r['k_sym']:>7.2f}{r['k_low']:>7.2f}"
              f"{r['k_high']:>8.2f}{r['raw_in_80']:>10.1f}{r['sym_in_80']:>10.1f}"
              f"{r['asym_in_80']:>11.1f}{r['asym_below_80']:>10.1f} / {r['asym_above_80']:.1f}")

    pooled = pooled_test_scores(results, by_year, horizon)
    print(f"\nAll test years pooled ({pooled['raw']['forecasts']} forecasts)")
    print(f"{'Version':<12}" + "".join(
        f"{f'In {lvl}%':>9}{'Below':>7}{'Above':>7}{'Width':>8}" for lvl in INTERVALS))
    for name, label in (("raw", "Raw"), ("sym", "Symmetric"), ("asym", "Asymmetric")):
        s = pooled[name]
        print(f"{label:<12}" + "".join(
            f"{s[f'in_{lvl}']:>9.1f}{s[f'below_{lvl}']:>7.1f}"
            f"{s[f'above_{lvl}']:>7.1f}{s[f'width_{lvl}']:>8.1f}" for lvl in INTERVALS))

    worst = by_year.loc[by_year["asym_in_80"].idxmin()]
    print(f"\nWorst test year, asymmetric: {int(worst['year'])} with "
          f"{worst['asym_in_80']:.1f}% inside the 80% range.")

    # Factor to use from today on: fitted on every outcome known so far.
    results = results.copy()
    known = results[resolve_dates(results, horizon) <= pd.Timestamp.today()]
    k_sym = fit_symmetric(known)
    k_low, k_high = fit_asymmetric(known)
    print(f"Current factors (all {len(known)} resolved forecasts): "
          f"k sym = {k_sym:.2f}, k low = {k_low:.2f}, k high = {k_high:.2f}")
    print("Widths are in percentage points of return.")


def main():
    horizons = [int(a) for a in sys.argv[1:]] or list(DEFAULT_HORIZONS)
    for horizon in horizons:
        path = os.path.join(REPORTS_DIR, f"calibration_{horizon}d.csv")
        if not os.path.exists(path):
            print(f"\nMissing {path}. Run: python evaluate_calibration.py {horizon}")
            continue
        report(pd.read_csv(path), horizon)


if __name__ == "__main__":
    main()
