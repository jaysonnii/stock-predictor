"""
Walk-forward accuracy check for one ticker.

Usage:
    python -m research.evaluate AMZN
    python -m research.evaluate AMZN 126     # custom horizon in trading days
"""
import sys

from research.model import walk_forward_evaluate


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m research.evaluate TICKER [HORIZON_DAYS]")
        sys.exit(1)

    ticker = sys.argv[1]
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 252
    result = walk_forward_evaluate(ticker, prediction_days=horizon)

    print(f"\nWalk-forward results: {result['ticker']}, {horizon}-day horizon")
    print(f"{'Test period':<26}{'RF MAE':>9}{'Zero MAE':>10}{'Mean MAE':>10}{'RF dir%':>9}")
    for f in result["folds"]:
        period = f"{f['test_from']} to {f['test_to']}"
        print(
            f"{period:<26}"
            f"{f['rf_mae_pct_points']:>9.2f}"
            f"{f['zero_baseline_mae_pct_points']:>10.2f}"
            f"{f['mean_baseline_mae_pct_points']:>10.2f}"
            f"{f['rf_directional_accuracy_pct']:>9.1f}"
        )
    print("-" * 64)
    print(
        f"{'Average':<26}"
        f"{result['avg_rf_mae_pct_points']:>9.2f}"
        f"{result['avg_zero_baseline_mae_pct_points']:>10.2f}"
        f"{result['avg_mean_baseline_mae_pct_points']:>10.2f}"
        f"{result['avg_rf_directional_accuracy_pct']:>9.1f}"
    )
    print(
        f"\nModel beat the mean-return baseline in "
        f"{result['folds_rf_beats_mean_baseline']} of {result['folds_evaluated']} folds."
    )
    print("Lower MAE is better. If the RF column is not below both baselines, the model adds nothing.")


if __name__ == "__main__":
    main()
