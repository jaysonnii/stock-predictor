import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import fetch_stock_data, engineer_features, get_feature_columns

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(ticker: str, prediction_days: int = 252) -> dict:
    """
    Train and evaluate a Random Forest model for a given ticker.

    Evaluation uses:
    - Chronological train/test ordering
    - A purge gap equal to the prediction horizon
    - A naive no-change baseline that predicts a 0% future return
    """
    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period="10y")

    print("Engineering features...")
    df = engineer_features(df, prediction_days=prediction_days)

    features = get_feature_columns()
    X = df[features].values
    y = df["Target"].values

    # First 80% establishes the chronological split point.
    split_index = int(len(X) * 0.8)

    # Remove rows immediately before the test set because their
    # prediction targets use future prices that overlap the test period.
    train_end = split_index - prediction_days

    if train_end <= 0 or split_index >= len(X):
        raise ValueError(
            "Not enough historical data for the selected prediction horizon."
        )

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_test = X[split_index:]
    y_test = y[split_index:]

    # Fit preprocessing only on training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Random Forest predictions.
    rf_predictions = model.predict(X_test_scaled)

    # Naive baseline:
    # Predict that the future return will be exactly zero.
    baseline_predictions = np.zeros_like(y_test)

    # Random Forest error metrics.
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

    # Baseline error metrics.
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    # Direction means positive future return versus zero/negative return.
    actual_direction = y_test > 0
    rf_direction = rf_predictions > 0
    baseline_direction = baseline_predictions > 0

    rf_directional_accuracy = (
        np.mean(actual_direction == rf_direction) * 100
    )
    baseline_directional_accuracy = (
        np.mean(actual_direction == baseline_direction) * 100
    )

    # Positive values mean the Random Forest improved over the baseline.
    mae_improvement_pct = (
        ((baseline_mae - rf_mae) / baseline_mae) * 100
        if baseline_mae > 0
        else 0.0
    )

    rmse_improvement_pct = (
        ((baseline_rmse - rf_rmse) / baseline_rmse) * 100
        if baseline_rmse > 0
        else 0.0
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"{ticker}_{prediction_days}d_model.pkl",
    )
    scaler_path = os.path.join(
        MODELS_DIR,
        f"{ticker}_{prediction_days}d_scaler.pkl",
    )

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")

    metrics = {
        "ticker": ticker.upper(),

        # Errors are expressed in percentage points of future return.
        "rf_mae_pct_points": round(rf_mae * 100, 2),
        "rf_rmse_pct_points": round(rf_rmse * 100, 2),
        "rf_directional_accuracy_pct": round(
            rf_directional_accuracy,
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
            baseline_directional_accuracy,
            2,
        ),

        "mae_improvement_over_baseline_pct": round(
            mae_improvement_pct,
            2,
        ),
        "rmse_improvement_over_baseline_pct": round(
            rmse_improvement_pct,
            2,
        ),

        "training_samples": len(X_train),
        "purge_gap_samples": prediction_days,
        "test_samples": len(X_test),
    }

    print(f"Metrics: {metrics}")
    return metrics


def predict_price(ticker: str, prediction_days: int = 252) -> dict:
    """
    Load a saved model and predict the price ~1 year from now.
    Trains a new model if one doesn't exist yet.
    """
    model_path = os.path.join(MODELS_DIR, f"{ticker}_{prediction_days}d_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{prediction_days}d_scaler.pkl")

    # Auto-train if model doesn't exist for this specific horizon
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"No model found for {ticker} ({prediction_days}d), training now...")
        train_model(ticker, prediction_days)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Get latest data — use fresh 5d fetch for current price to avoid stale cache
    df = fetch_stock_data(ticker, period="3y")
    df = engineer_features(df, prediction_days=prediction_days)

    features = get_feature_columns()
    latest_row = df[features].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_row)

    predicted_return = model.predict(latest_scaled)[0]  # % return as decimal

    # Get truly fresh current price from a separate short-period fetch
    fresh_df = fetch_stock_data(ticker, period="5d")
    current_price = fresh_df["Close"].iloc[-1]

    # Convert predicted % return back to a price
    predicted_price = current_price * (1 + predicted_return)
    expected_return_pct = predicted_return * 100

    return {
        "ticker": ticker.upper(),
        "current_price": round(float(current_price), 2),
        "predicted_price_1yr": round(float(predicted_price), 2),
        "expected_return_pct": round(float(expected_return_pct), 2),
        "prediction_horizon_days": prediction_days,
    }


if __name__ == "__main__":
    # Quick test
    metrics = train_model("AAPL")
    print(metrics)
    result = predict_price("AAPL")
    print(result)
