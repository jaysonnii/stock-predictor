import numpy as np
import pandas as pd

import model


class DummyRandomForest:
    """Small test substitute for RandomForestRegressor."""

    def fit(self, X, y):
        self.training_row_count = len(X)
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_training_uses_chronological_purge_gap(monkeypatch):
    row_count = 1000
    prediction_days = 50

    synthetic_data = pd.DataFrame(
        {
            "Close": np.linspace(100.0, 200.0, row_count),
            "Target": np.linspace(-0.20, 0.30, row_count),
        }
    )

    monkeypatch.setattr(
        model,
        "fetch_stock_data",
        lambda ticker, period: synthetic_data.copy(),
    )

    monkeypatch.setattr(
        model,
        "engineer_features",
        lambda dataframe, prediction_days: dataframe,
    )

    monkeypatch.setattr(
        model,
        "get_feature_columns",
        lambda: ["Close"],
    )

    monkeypatch.setattr(
        model,
        "RandomForestRegressor",
        lambda **kwargs: DummyRandomForest(),
    )

    monkeypatch.setattr(
        model.joblib,
        "dump",
        lambda obj, path: None,
    )

    metrics = model.train_model(
        ticker="TEST",
        prediction_days=prediction_days,
    )

    # 80% chronological split of 1,000 rows.
    split_index = 800

    # Training stops 50 rows before the test set.
    expected_training_rows = split_index - prediction_days
    expected_test_rows = row_count - split_index

    assert metrics["training_samples"] == expected_training_rows
    assert metrics["purge_gap_samples"] == prediction_days
    assert metrics["test_samples"] == expected_test_rows

    assert "rf_mae_pct_points" in metrics
    assert "baseline_mae_pct_points" in metrics
    assert "mae_improvement_over_baseline_pct" in metrics
    