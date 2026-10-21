"""
Factor check: how well do simple one-line rules rank stocks on the same test
periods as the ranking test? If a simple rule matches the model, the model is
mostly rediscovering that rule.

Usage (run from the project folder, after the pooled data is cached):
    python -m research.factor_check
"""
import numpy as np
import pandas as pd

from research.pooled import build_panel
from research.ranking import MIN_TICKERS_PER_DATE, TOP_FRACTION

# Single features to rank by. "high" = top group by that value.
SIMPLE_FEATURES = ["Ret_60", "Ret_30", "Volatility_20", "Price_to_MA200", "RSI"]


def factor_spreads(horizon: int = 252, n_folds: int = 4):
    panel = build_panel(horizon=horizon)
    dates = np.sort(panel["Date"].unique())
    n = len(dates)
    initial = int(n * 0.5)
    fold_size = (n - initial) // n_folds

    rows = []
    for k in range(n_folds):
        s = initial + k * fold_size
        e = n if k == n_folds - 1 else s + fold_size
        if s - horizon < 100:
            continue
        te = panel[(panel["Date"] >= dates[s]) & (panel["Date"] <= dates[e - 1])]

        result = {"test_from": str(pd.Timestamp(dates[s]).date()),
                  "test_to": str(pd.Timestamp(dates[e - 1]).date())}
        for feat in SIMPLE_FEATURES:
            spreads = []
            for _, g in te.groupby("Date"):
                if len(g) < MIN_TICKERS_PER_DATE:
                    continue
                k_top = max(1, int(round(len(g) * TOP_FRACTION)))
                top = g.nlargest(k_top, feat)
                spreads.append(top["Target"].mean() - g["Target"].mean())
            result[feat] = round(float(np.mean(spreads)) * 100, 2)
        rows.append(result)
    return rows


def main():
    rows = factor_spreads()
    cols = ["test_from", "test_to"] + SIMPLE_FEATURES

    print("\nAverage spread of the TOP 20% by each single feature (percentage points)")
    print("Compare with the model's spread: 2.74, 25.55, 6.88, 15.46 (average 12.67)\n")
    header = f"{'Test period':<26}" + "".join(f"{f:>16}" for f in SIMPLE_FEATURES)
    print(header)
    for r in rows:
        period = f"{r['test_from']} to {r['test_to']}"
        print(f"{period:<26}" + "".join(f"{r[f]:>16.2f}" for f in SIMPLE_FEATURES))

    print("-" * len(header))
    avg = {f: np.mean([r[f] for r in rows]) for f in SIMPLE_FEATURES}
    print(f"{'Average':<26}" + "".join(f"{avg[f]:>16.2f}" for f in SIMPLE_FEATURES))
    print("\nRead it this way: a simple feature with an average close to 12.67 is doing")
    print("the same job as the model. Also check that each period is positive, not just the average.")


if __name__ == "__main__":
    main()
