"""
Ranking test for the pooled model. Run from the project folder:

    python evaluate_ranking.py          # 252-day horizon
    python evaluate_ranking.py 126      # custom horizon
"""
import sys

from pooled import build_panel
from ranking import walk_forward_ranking


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 252
    panel = build_panel(horizon=horizon)
    print(f"\nPanel: {panel['Ticker'].nunique()} tickers, {len(panel)} rows\n")

    result = walk_forward_ranking(horizon=horizon, panel=panel)

    print(f"Ranking test, {horizon}-day horizon, top {int(result['top_fraction'] * 100)}% of tickers")
    print("Spread = top-ranked group's excess return minus the universe average (percentage points)\n")
    print(f"{'Test period':<26}{'Avg spread':>12}{'Dates positive %':>18}")
    for f in result["folds"]:
        period = f"{f['test_from']} to {f['test_to']}"
        print(f"{period:<26}{f['avg_spread_pct_points']:>12.2f}{f['share_of_dates_positive_pct']:>18.1f}")
    print("-" * 56)
    print(f"Overall average spread: {result['overall_avg_spread_pct_points']:.2f} percentage points")
    print(f"Dates with positive spread: {result['overall_share_positive_pct']:.1f}%")
    print(f"Random-group control, overall average spread: {result['random_control_overall_avg_spread_pct_points']:.2f} percentage points")
    print(
        f"Folds with positive average spread: {result['folds_with_positive_spread']} "
        f"of {result['folds_evaluated']}"
    )
    print("\nA model with no skill gives a spread near zero and about 50% positive dates.")


if __name__ == "__main__":
    main()
