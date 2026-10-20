"""
Walk-forward test for the pooled model across all tickers in universe.py.

Usage:
    python evaluate_pooled.py          # 252-day horizon
    python evaluate_pooled.py 126      # custom horizon
"""
import sys

from pooled import build_panel, walk_forward_pooled


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 252
    panel = build_panel(horizon=horizon)
    print(f"\nPanel: {panel['Ticker'].nunique()} tickers, {len(panel)} rows\n")

    result = walk_forward_pooled(horizon=horizon, panel=panel)

    print(f"Pooled walk-forward, {horizon}-day horizon, errors vs SPY (percentage points)")
    print(f"{'Test period':<26}{'RF MAE':>9}{'Same-as-SPY':>13}{'Avg excess':>12}")
    for f in result["folds"]:
        period = f"{f['test_from']} to {f['test_to']}"
        print(
            f"{period:<26}"
            f"{f['rf_mae_pct_points']:>9.2f}"
            f"{f['zero_baseline_mae_pct_points']:>13.2f}"
            f"{f['mean_baseline_mae_pct_points']:>12.2f}"
        )
    print("-" * 60)
    print(
        f"Model beat the average-excess baseline in {result['folds_beating_mean_baseline']} "
        f"of {result['folds_evaluated']} folds."
    )
    print(
        f"Model beat the same-as-SPY baseline in {result['folds_beating_zero_baseline']} "
        f"of {result['folds_evaluated']} folds."
    )
    print("Gate from the plan: most folds must beat both baselines.")


if __name__ == "__main__":
    main()
