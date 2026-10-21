from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import engineer_features, fetch_stock_data, get_feature_columns


def _build_model() -> RandomForestRegressor:
    """Create the same Random Forest configuration used by the main model."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )


def _annualized_return_volatility_ratio(
    returns: np.ndarray,
    horizon_days: int,
) -> float | None:
    """
    Calculate a Sharpe-style consistency ratio.

    This uses a 0% risk-free rate and non-overlapping window returns.
    It is not presented as a formal investment-performance Sharpe ratio.
    """
    if len(returns) < 2:
        return None

    volatility = np.std(returns, ddof=1)

    if np.isclose(volatility, 0.0):
        return None

    periods_per_year = 252 / horizon_days

    return float(
        (np.mean(returns) / volatility)
        * np.sqrt(periods_per_year)
    )


def _maximum_drawdown(growth_values: list[float]) -> float:
    """
    Return the worst percentage decline from a prior cumulative peak.

    Growth values are measured only at backtest window endpoints.
    """
    growth = np.asarray(growth_values, dtype=float)
    running_peak = np.maximum.accumulate(growth)
    drawdowns = growth / running_peak - 1.0

    return float(drawdowns.min())


def _round_optional(
    value: float | None,
    digits: int = 2,
) -> float | None:
    """Round an optional numeric value."""
    if value is None:
        return None

    return round(float(value), digits)


def run_walk_forward_backtest(
    ticker: str,
    prediction_days: int = 63,
    step_days: int | None = None,
    evaluation_years: int = 10,
    initial_training_samples: int = 756,
) -> dict:
    """
    Run expanding-window, rolling-origin backtesting.

    At every prediction origin:

    1. Use only information available before that origin.
    2. Leave a purge gap equal to the prediction horizon.
    3. Fit preprocessing only on the training observations.
    4. Retrain the Random Forest.
    5. Predict the return for the next historical window.
    6. Move forward by one non-overlapping horizon.

    The default horizon is 63 trading days, approximately one quarter.
    """
    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    if prediction_days <= 0:
        raise ValueError("prediction_days must be positive.")

    if evaluation_years <= 0:
        raise ValueError("evaluation_years must be positive.")

    if initial_training_samples <= 0:
        raise ValueError("initial_training_samples must be positive.")

    if step_days is None:
        step_days = prediction_days

    # Overlapping future-return windows cannot be compounded honestly
    # because the same market days would be counted more than once.
    if step_days != prediction_days:
        raise ValueError(
            "step_days must equal prediction_days for cumulative "
            "performance. Overlapping windows would double-count returns."
        )

    print(f"Downloading historical data for {ticker}...")

    # Earlier observations remain available for training so the evaluation
    # itself can cover approximately the most recent requested number of years.
    raw_data = fetch_stock_data(ticker, period="max").sort_index()

    print("Engineering features and future-return targets...")

    data = engineer_features(
        raw_data,
        prediction_days=prediction_days,
    ).sort_index()

    feature_columns = get_feature_columns()
    required_columns = feature_columns + ["Target"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Backtest data is missing columns: {missing_columns}"
        )

    evaluation_cutoff = (
        data.index.max()
        - pd.DateOffset(years=evaluation_years)
    )

    first_evaluation_position = int(
        data.index.searchsorted(evaluation_cutoff)
    )

    # Training ends prediction_days rows before the prediction origin.
    # The first origin therefore needs the requested training history
    # plus a full horizon-sized purge gap.
    first_origin = max(
        first_evaluation_position,
        initial_training_samples + prediction_days,
    )

    if first_origin >= len(data):
        raise ValueError(
            "Not enough historical data for the requested training "
            "period, horizon, and evaluation length."
        )

    predicted_returns: list[float] = []
    actual_returns: list[float] = []
    strategy_returns: list[float] = []
    buy_hold_returns: list[float] = []

    strategy_growth = 1.0
    buy_hold_growth = 1.0
    cash_growth = 1.0

    strategy_growth_values = [strategy_growth]
    buy_hold_growth_values = [buy_hold_growth]

    windows: list[dict] = []

    print("Running walk-forward windows...")

    for origin in range(
        first_origin,
        len(data),
        step_days,
    ):
        train_end = origin - prediction_days

        if train_end < initial_training_samples:
            continue

        origin_date = data.index[origin]

        raw_origin_position = raw_data.index.get_indexer(
            [origin_date]
        )[0]

        if raw_origin_position < 0:
            continue

        target_position = raw_origin_position + prediction_days

        if target_position >= len(raw_data):
            break

        target_date = raw_data.index[target_position]

        X_train = data[feature_columns].iloc[:train_end]
        y_train = data["Target"].iloc[:train_end]

        X_origin = data[feature_columns].iloc[
            origin:origin + 1
        ]

        if X_train.empty or X_origin.empty:
            continue

        # Fit preprocessing only on the historical training observations.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_origin_scaled = scaler.transform(X_origin)

        model = _build_model()
        model.fit(X_train_scaled, y_train)

        predicted_return = float(
            model.predict(X_origin_scaled)[0]
        )

        actual_return = float(
            data["Target"].iloc[origin]
        )

        # Simple prediction-driven strategy:
        # hold the stock when the forecast is positive, otherwise hold cash.
        position = 1 if predicted_return > 0 else 0
        strategy_return = (
            actual_return
            if position == 1
            else 0.0
        )

        strategy_growth *= 1.0 + strategy_return
        buy_hold_growth *= 1.0 + actual_return

        strategy_growth_values.append(strategy_growth)
        buy_hold_growth_values.append(buy_hold_growth)

        predicted_returns.append(predicted_return)
        actual_returns.append(actual_return)
        strategy_returns.append(strategy_return)
        buy_hold_returns.append(actual_return)

        windows.append(
            {
                "origin_date": str(origin_date.date()),
                "target_date": str(target_date.date()),
                "training_samples": int(len(X_train)),
                "purge_gap_samples": prediction_days,
                "predicted_return_pct": round(
                    predicted_return * 100,
                    2,
                ),
                "actual_return_pct": round(
                    actual_return * 100,
                    2,
                ),
                "position": (
                    "long"
                    if position == 1
                    else "cash"
                ),
                "strategy_return_pct": round(
                    strategy_return * 100,
                    2,
                ),
                "strategy_growth": round(
                    strategy_growth,
                    4,
                ),
                "buy_hold_growth": round(
                    buy_hold_growth,
                    4,
                ),
                "cash_growth": round(
                    cash_growth,
                    4,
                ),
            }
        )

    if len(windows) < 2:
        raise ValueError(
            "The selected configuration produced fewer than two "
            "backtesting windows."
        )

    predictions = np.asarray(
        predicted_returns,
        dtype=float,
    )
    actuals = np.asarray(
        actual_returns,
        dtype=float,
    )

    # Forecast baselines:
    # - zero return for MAE/RMSE comparison
    # - always positive for directional comparison
    baseline_predictions = np.zeros_like(actuals)
    positive_window_rate = float(
        np.mean(actuals > 0)
    )

    strategy_array = np.asarray(
        strategy_returns,
        dtype=float,
    )
    buy_hold_array = np.asarray(
        buy_hold_returns,
        dtype=float,
    )

    model_mae = mean_absolute_error(
        actuals,
        predictions,
    )
    model_rmse = np.sqrt(
        mean_squared_error(
            actuals,
            predictions,
        )
    )

    baseline_mae = mean_absolute_error(
        actuals,
        baseline_predictions,
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(
            actuals,
            baseline_predictions,
        )
    )

    model_directional_accuracy = float(
        np.mean(
            (predictions > 0)
            == (actuals > 0)
        )
    )

    zero_return_directional_accuracy = float(
        np.mean(
            (baseline_predictions > 0)
            == (actuals > 0)
        )
    )

    mae_improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
        if baseline_mae > 0
        else 0.0
    )

    rmse_improvement = (
        (baseline_rmse - model_rmse)
        / baseline_rmse
        if baseline_rmse > 0
        else 0.0
    )

    strategy_consistency = (
        _annualized_return_volatility_ratio(
            strategy_array,
            prediction_days,
        )
    )

    buy_hold_consistency = (
        _annualized_return_volatility_ratio(
            buy_hold_array,
            prediction_days,
        )
    )

    long_mask = predictions > 0

    positive_signal_hit_rate = (
        float(
            np.mean(
                actuals[long_mask] > 0
            )
        )
        if np.any(long_mask)
        else None
    )

    result = {
        "ticker": ticker,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_start_date": str(
            raw_data.index.min().date()
        ),
        "data_through_date": str(
            raw_data.index.max().date()
        ),
        "evaluation_start_date": windows[0]["origin_date"],
        "evaluation_end_date": windows[-1]["target_date"],
        "methodology": {
            "validation": "expanding-window rolling origin",
            "historical_source_period": "maximum available",
            "evaluation_years_requested": evaluation_years,
            "prediction_horizon_trading_days": prediction_days,
            "step_trading_days": step_days,
            "initial_training_samples": initial_training_samples,
            "purge_gap_samples": prediction_days,
            "overlapping_windows": False,
            "forecast_error_baseline": (
                "no-change return forecast"
            ),
            "directional_baseline": (
                "always-up forecast"
            ),
            "strategy": (
                "long when prediction is positive, "
                "otherwise cash"
            ),
            "cumulative_comparisons": [
                "prediction-driven long-or-cash strategy",
                "buy and hold",
                "cash",
            ],
            "transaction_costs_included": False,
            "risk_free_rate_assumption_pct": 0.0,
            "drawdown_frequency": (
                "evaluated at window endpoints"
            ),
        },
        "summary": {
            "window_count": len(windows),

            "model_mae_pct_points": round(
                model_mae * 100,
                2,
            ),
            "model_rmse_pct_points": round(
                model_rmse * 100,
                2,
            ),
            "model_directional_accuracy_pct": round(
                model_directional_accuracy * 100,
                2,
            ),

            "positive_window_rate_pct": round(
                positive_window_rate * 100,
                2,
            ),
            "always_up_directional_accuracy_pct": round(
                positive_window_rate * 100,
                2,
            ),

            "baseline_mae_pct_points": round(
                baseline_mae * 100,
                2,
            ),
            "baseline_rmse_pct_points": round(
                baseline_rmse * 100,
                2,
            ),
            "baseline_directional_accuracy_pct": round(
                zero_return_directional_accuracy * 100,
                2,
            ),

            "mae_improvement_over_baseline_pct": round(
                mae_improvement * 100,
                2,
            ),
            "rmse_improvement_over_baseline_pct": round(
                rmse_improvement * 100,
                2,
            ),

            "strategy_cumulative_return_pct": round(
                (strategy_growth - 1.0) * 100,
                2,
            ),
            "buy_hold_cumulative_return_pct": round(
                (buy_hold_growth - 1.0) * 100,
                2,
            ),
            "cash_cumulative_return_pct": 0.0,

            "strategy_annualized_return_volatility_ratio":
                _round_optional(strategy_consistency),

            "buy_hold_annualized_return_volatility_ratio":
                _round_optional(buy_hold_consistency),

            "strategy_max_drawdown_pct": round(
                _maximum_drawdown(
                    strategy_growth_values
                ) * 100,
                2,
            ),
            "buy_hold_max_drawdown_pct": round(
                _maximum_drawdown(
                    buy_hold_growth_values
                ) * 100,
                2,
            ),

            "strategy_market_exposure_pct": round(
                np.mean(long_mask) * 100,
                2,
            ),
            "positive_signal_hit_rate_pct": (
                round(
                    positive_signal_hit_rate * 100,
                    2,
                )
                if positive_signal_hit_rate is not None
                else None
            ),
        },
        "windows": windows,
    }

    return result


def main() -> None:
    """Run a backtest from the command line and save its JSON report."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward stock-model backtesting."
    )

    parser.add_argument(
        "ticker",
        help="Ticker symbol, such as AAPL.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=63,
        help="Prediction horizon in trading days.",
    )
    parser.add_argument(
        "--evaluation-years",
        type=int,
        default=10,
        help="Approximate number of years to evaluate.",
    )
    parser.add_argument(
        "--initial-training-samples",
        type=int,
        default=756,
        help="Minimum initial training observations.",
    )

    args = parser.parse_args()

    result = run_walk_forward_backtest(
        ticker=args.ticker,
        prediction_days=args.horizon_days,
        evaluation_years=args.evaluation_years,
        initial_training_samples=args.initial_training_samples,
    )

    output_directory = Path("reports/backtests")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / (
        f"{result['ticker']}_"
        f"{args.horizon_days}d_"
        f"{args.evaluation_years}y.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Backtest completed.")
    print(f"Report saved to: {output_path}")
    print()
    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
