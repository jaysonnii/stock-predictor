"""
Pooled model: one Random Forest trained across many stocks and ETFs.

Target: each stock's future return MINUS the SPY future return over the same
window ("excess return"). Subtracting the market removes most of the drift
that dominates single-stock returns.

Usage:
    python -m research.pooled train 252            # train and save the model
    python -m research.pooled predict AMZN 252     # predicted excess return for one ticker
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data import (
    HISTORY_PERIOD,
    _naive_dates,
    engineer_features,
    fetch_stock_data,
    get_feature_columns,
)
from research.universe import BENCHMARK, TICKERS

CACHE_DIR = "data_cache"
CACHE_SECONDS = 24 * 60 * 60
DOWNLOAD_DELAY_SECONDS = 0.5
MODELS_DIR = "models"
POOLED_VERSION = "v3"
MIN_ROWS = 300  # skip tickers with less usable history than this

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def _load_raw(ticker: str) -> pd.DataFrame:
    """Daily OHLCV, cached on disk so repeated runs don't hit Yahoo again."""
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_SECONDS:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
        return df

    try:
        df = fetch_stock_data(ticker, period=HISTORY_PERIOD)
    except Exception:
        # Yahoo sometimes rate-limits. Wait once and retry.
        time.sleep(3)
        df = fetch_stock_data(ticker, period=HISTORY_PERIOD)

    df.to_csv(path)
    time.sleep(DOWNLOAD_DELAY_SECONDS)
    return df


def _make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        max_samples=0.5,  # faster, and reduces overlap between trees
        random_state=42,
        n_jobs=-1,
    )


def _spy_forward_return(horizon: int) -> pd.Series:
    spy = _load_raw(BENCHMARK)
    close = pd.Series(spy["Close"].values, index=_naive_dates(spy.index))
    close = close[~close.index.duplicated()]
    return close.shift(-horizon) / close - 1


def build_panel(tickers=TICKERS, horizon: int = 252) -> pd.DataFrame:
    """Stack features and excess-return targets for all tickers, sorted by date."""
    spy_fwd = _spy_forward_return(horizon)
    features = get_feature_columns()
    frames = []

    for ticker in tickers:
        try:
            raw = _load_raw(ticker)
            df = engineer_features(raw, prediction_days=horizon, with_target=True)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

        if len(df) < MIN_ROWS:
            print(f"Skipping {ticker}: only {len(df)} usable rows")
            continue

        dates = _naive_dates(df.index)
        bench = spy_fwd.reindex(dates).values

        part = df[features].copy()
        part["Target"] = df["Target"].values - bench
        part["Date"] = dates
        part["Ticker"] = ticker
        part = part.dropna(subset=["Target"])
        frames.append(part)
        print(f"Loaded {ticker}: {len(part)} rows")

    if not frames:
        raise ValueError("No tickers had enough data to build a panel.")

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values("Date").reset_index(drop=True)


def walk_forward_pooled(horizon: int = 252, n_folds: int = 4, panel=None) -> dict:
    """
    Date-based walk-forward test on the pooled panel. Each fold trains on all
    tickers' rows before a cutoff date, purges one horizon, and tests on the
    following block of dates. Errors are for excess return vs SPY.
    """
    if panel is None:
        panel = build_panel(horizon=horizon)

    features = get_feature_columns()
    dates = np.sort(panel["Date"].unique())
    n = len(dates)
    initial = int(n * 0.5)
    fold_size = (n - initial) // n_folds

    folds = []
    for k in range(n_folds):
        s = initial + k * fold_size
        e = n if k == n_folds - 1 else s + fold_size
        if s - horizon < 100:
            continue

        train_cut = dates[s - horizon]  # purge: training targets end before the test block
        tr = panel[panel["Date"] < train_cut]
        te = panel[(panel["Date"] >= dates[s]) & (panel["Date"] <= dates[e - 1])]
        if tr.empty or te.empty:
            continue

        scaler = StandardScaler().fit(tr[features].values)
        model = _make_model().fit(scaler.transform(tr[features].values), tr["Target"].values)
        pred = model.predict(scaler.transform(te[features].values))
        y = te["Target"].values

        rf_mae = mean_absolute_error(y, pred)
        zero_mae = np.mean(np.abs(y))                      # "same as SPY" baseline
        mean_mae = np.mean(np.abs(y - tr["Target"].mean()))  # average past excess return
        folds.append({
            "test_from": str(pd.Timestamp(dates[s]).date()),
            "test_to": str(pd.Timestamp(dates[e - 1]).date()),
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "rf_mae_pct_points": round(rf_mae * 100, 2),
            "zero_baseline_mae_pct_points": round(zero_mae * 100, 2),
            "mean_baseline_mae_pct_points": round(mean_mae * 100, 2),
        })

    if not folds:
        raise ValueError("Not enough dates for pooled walk-forward evaluation.")

    beat_mean = sum(f["rf_mae_pct_points"] < f["mean_baseline_mae_pct_points"] for f in folds)
    beat_zero = sum(f["rf_mae_pct_points"] < f["zero_baseline_mae_pct_points"] for f in folds)
    return {
        "horizon_days": horizon,
        "tickers": int(panel["Ticker"].nunique()),
        "folds_evaluated": len(folds),
        "folds_beating_mean_baseline": int(beat_mean),
        "folds_beating_zero_baseline": int(beat_zero),
        "folds": folds,
    }


def _paths(horizon: int):
    base = f"POOLED_{horizon}d_{POOLED_VERSION}"
    return (
        os.path.join(MODELS_DIR, f"{base}_model.pkl"),
        os.path.join(MODELS_DIR, f"{base}_scaler.pkl"),
    )


def train_pooled(horizon: int = 252, panel=None) -> dict:
    """Train on all data whose targets are fully known, and save the model."""
    if panel is None:
        panel = build_panel(horizon=horizon)

    features = get_feature_columns()
    dates = np.sort(panel["Date"].unique())
    # Drop the most recent horizon's worth of dates. Their targets overlap
    # the latest data we would predict from.
    cut = dates[-horizon] if len(dates) > horizon else dates[-1]
    tr = panel[panel["Date"] < cut]

    scaler = StandardScaler().fit(tr[features].values)
    model = _make_model().fit(scaler.transform(tr[features].values), tr["Target"].values)

    model_path, scaler_path = _paths(horizon)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved {model_path} on {len(tr)} rows from {panel['Ticker'].nunique()} tickers")
    return {"rows": int(len(tr)), "tickers": int(panel["Ticker"].nunique()), "model_path": model_path}


def predict_relative(ticker: str, horizon: int = 252) -> dict:
    """Predicted return of one ticker relative to SPY over the horizon."""
    model_path, scaler_path = _paths(horizon)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No pooled model for {horizon} days. Run: python -m research.pooled train {horizon}"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    raw = _load_raw(ticker)
    df = engineer_features(raw, prediction_days=horizon, with_target=False)
    if df.empty:
        raise ValueError(f"Not enough history to compute features for {ticker}.")

    row = df[get_feature_columns()].iloc[[-1]].values
    excess = float(model.predict(scaler.transform(row))[0])
    return {
        "ticker": ticker.upper(),
        "prediction_horizon_days": horizon,
        "predicted_excess_return_pct": round(excess * 100, 2),
        "features_as_of": str(pd.Timestamp(df.index[-1]).date()),
        "model_version": POOLED_VERSION,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "train":
        train_pooled(int(sys.argv[2]))
    elif command == "predict":
        print(predict_relative(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 252))
    else:
        print(__doc__)
        sys.exit(1)
