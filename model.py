import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data import (
    HISTORY_PERIOD,
    engineer_features,
    fetch_stock_data,
    get_feature_columns,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Bump this when features change. Old saved models are then ignored
# instead of failing on a feature-count mismatch.
MODEL_VERSION = "v2"

# Shorter horizons to fall back to when history is too short.
FALLBACK_HORIZONS = [252, 126, 63, 21]


def _model_paths(ticker: str, prediction_days: int):
    base = f"{ticker.upper()}_{prediction_days}d_{MODEL_VERSION}"
    return (
        os.path.join(MODELS_DIR, f"{base}_model.pkl"),
        os.path.join(MODELS_DIR, f"{base}_scaler.pkl"),
    )


def _make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )


def _score(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (MAE, RMSE, directional accuracy %) for predicted returns."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    direction = np.mean((y_true > 0) == (y_pred > 0)) * 100
    return mae, rmse, direction


def build_dataset(ticker: str, prediction_days: int = 252) -> pd.DataFrame:
    """Fetch history and build features plus the future-return target."""
    df = fetch_stock_data(ticker, period=HISTORY_PERIOD)
    return engineer_features(df, prediction_days=prediction_days, with_target=True)


def _relative_improvement(baseline: float, model: float) -> float:
    """Positive means the model has lower error than the baseline."""
    return ((baseline - model) / baseline) * 100 if baseline > 0 else 0.0


def train_model(ticker: str, prediction_days: int = 252) -> dict:
    """
    Train and evaluate a Random Forest model for a given ticker.

    Evaluation uses:
    - Chronological train/test ordering
    - A purge gap equal to the prediction horizon
    - Two naive baselines: a 0% return, and the average training return
    """
    print(f"Fetching data for {ticker}...")
    print("Engineering features...")
    df = build_dataset(ticker, prediction_days)

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
    model = _make_model()
    model.fit(X_train_scaled, y_train)

    rf_pred = model.predict(X_test_scaled)
    zero_pred = np.zeros_like(y_test)
    mean_pred = np.full_like(y_test, y_train.mean())

    rf_mae, rf_rmse, rf_dir = _score(y_test, rf_pred)
    zero_mae, zero_rmse, zero_dir = _score(y_test, zero_pred)
    mean_mae, mean_rmse, _ = _score(y_test, mean_pred)

    model_path, scaler_path = _model_paths(ticker, prediction_days)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")

    metrics = {
        "ticker": ticker.upper(),
        "model_version": MODEL_VERSION,

        # Errors are in percentage points of future return.
        "rf_mae_pct_points": round(rf_mae * 100, 2),
        "rf_rmse_pct_points": round(rf_rmse * 100, 2),
        "rf_directional_accuracy_pct": round(rf_dir, 2),

        "baseline_mae_pct_points": round(zero_mae * 100, 2),
        "baseline_rmse_pct_points": round(zero_rmse * 100, 2),
        "baseline_directional_accuracy_pct": round(zero_dir, 2),

        "mean_baseline_mae_pct_points": round(mean_mae * 100, 2),
        "mean_baseline_rmse_pct_points": round(mean_rmse * 100, 2),

        "mae_improvement_over_baseline_pct": round(
            _relative_improvement(zero_mae, rf_mae), 2
        ),
        "rmse_improvement_over_baseline_pct": round(
            _relative_improvement(zero_rmse, rf_rmse), 2
        ),
        "mae_improvement_over_mean_baseline_pct": round(
            _relative_improvement(mean_mae, rf_mae), 2
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
    model_path, scaler_path = _model_paths(ticker, prediction_days)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"No model found for {ticker} ({prediction_days}d), training now...")
        train_model(ticker, prediction_days)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Same history length as training. with_target=False keeps today's row.
    df = fetch_stock_data(ticker, period=HISTORY_PERIOD)
    df = engineer_features(df, prediction_days=prediction_days, with_target=False)
    if df.empty:
        raise ValueError(f"Not enough history to compute features for {ticker}.")

    features = get_feature_columns()
    latest_row = df[features].iloc[[-1]].values
    features_as_of = pd.Timestamp(df.index[-1]).date()
    latest_scaled = scaler.transform(latest_row)

    predicted_return = float(model.predict(latest_scaled)[0])  # decimal, e.g. -0.47

    # Current price from a short, fresh fetch.
    fresh_df = fetch_stock_data(ticker, period="5d")
    current_price = float(fresh_df["Close"].iloc[-1])

    predicted_price = current_price * (1 + predicted_return)
    expected_return_pct = predicted_return * 100

    return {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "predicted_price_1yr": round(float(predicted_price), 2),
        "expected_return_pct": round(float(expected_return_pct), 2),
        "prediction_horizon_days": prediction_days,
        "features_as_of": str(features_as_of),
        "model_version": MODEL_VERSION,
    }


def predict_price_with_fallback(ticker: str, prediction_days: int = 252) -> dict:
    """
    Try the requested horizon first. If there isn't enough history,
    step down to shorter horizons and report which one was used.
    """
    last_error = None
    for days in [h for h in FALLBACK_HORIZONS if h <= prediction_days]:
        try:
            result = predict_price(ticker, days)
            result["requested_horizon_days"] = prediction_days
            result["fallback_used"] = days != prediction_days
            return result
        except ValueError as e:
            last_error = e
    raise ValueError(
        f"Not enough history for {ticker} even at the shortest horizon (21 days)."
    ) from last_error


def walk_forward_evaluate(
    ticker: str,
    prediction_days: int = 252,
    n_folds: int = 4,
) -> dict:
    """
    Walk-forward validation. The model is retrained on everything before each
    test block, with a purge gap, and scored on the next block. This tests the
    model over several periods instead of one split.
    """
    df = build_dataset(ticker, prediction_days)
    features = get_feature_columns()
    X = df[features].values
    y = df["Target"].values
    dates = df.index

    n = len(X)
    initial = int(n * 0.5)
    fold_size = (n - initial) // n_folds

    folds = []
    for k in range(n_folds):
        test_start = initial + k * fold_size
        test_end = n if k == n_folds - 1 else test_start + fold_size
        train_end = test_start - prediction_days
        if train_end < 100 or test_end <= test_start:
            continue

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        scaler = StandardScaler().fit(X_tr)
        model = _make_model().fit(scaler.transform(X_tr), y_tr)
        rf_pred = model.predict(scaler.transform(X_te))

        rf_mae, _, rf_dir = _score(y_te, rf_pred)
        zero_mae, _, zero_dir = _score(y_te, np.zeros_like(y_te))
        mean_mae, _, _ = _score(y_te, np.full_like(y_te, y_tr.mean()))

        folds.append({
            "test_from": str(pd.Timestamp(dates[test_start]).date()),
            "test_to": str(pd.Timestamp(dates[test_end - 1]).date()),
            "train_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
            "rf_mae_pct_points": round(rf_mae * 100, 2),
            "zero_baseline_mae_pct_points": round(zero_mae * 100, 2),
            "mean_baseline_mae_pct_points": round(mean_mae * 100, 2),
            "rf_directional_accuracy_pct": round(rf_dir, 2),
            "zero_baseline_directional_accuracy_pct": round(zero_dir, 2),
        })

    if not folds:
        raise ValueError("Not enough history for walk-forward evaluation.")

    def avg(key):
        return round(float(np.mean([f[key] for f in folds])), 2)

    return {
        "ticker": ticker.upper(),
        "prediction_horizon_days": prediction_days,
        "folds_evaluated": len(folds),
        "avg_rf_mae_pct_points": avg("rf_mae_pct_points"),
        "avg_zero_baseline_mae_pct_points": avg("zero_baseline_mae_pct_points"),
        "avg_mean_baseline_mae_pct_points": avg("mean_baseline_mae_pct_points"),
        "avg_rf_directional_accuracy_pct": avg("rf_directional_accuracy_pct"),
        "avg_zero_baseline_directional_accuracy_pct": avg(
            "zero_baseline_directional_accuracy_pct"
        ),
        "folds_rf_beats_mean_baseline": int(sum(
            f["rf_mae_pct_points"] < f["mean_baseline_mae_pct_points"] for f in folds
        )),
        "folds": folds,
    }
