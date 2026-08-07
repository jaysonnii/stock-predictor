import numpy as np
import pandas as pd
import pytest

import backtest


class DummyModel:
    """Deterministic substitute for RandomForestRegressor."""

    def fit(self, X, y):
        self.training_rows = len(X)
        return self

    def predict(self, X):
        return np.full(len(X), 0.10)


def make_backtest_frame(
    row_count: int,
    target_return: float = 0.20,
) -> pd.DataFrame:
    index = pd.bdate_range(
        "2010-01-01",
        periods=row_count,
    )

    return pd.DataFrame(
        {
            "Close": np.linspace(
                100.0,
                300.0,
                row_count,
            ),
            "Target": np.full(
                row_count,
                target_return,
            ),
        },
        index=index,
    )


def configure_test_dependencies(
    monkeypatch,
    frame: pd.DataFrame,
):
    monkeypatch.setattr(
        backtest,
        "fetch_stock_data",
        lambda ticker, period: frame.copy(),
    )

    monkeypatch.setattr(
        backtest,
        "engineer_features",
        lambda dataframe, prediction_days:
            dataframe.copy(),
    )

    monkeypatch.setattr(
        backtest,
        "get_feature_columns",
        lambda: ["Close"],
    )

    monkeypatch.setattr(
        backtest,
        "_build_model",
        lambda: DummyModel(),
    )


def test_walk_forward_uses_purge_gap(
    monkeypatch,
):
    prediction_days = 50
    initial_training_samples = 300

    frame = make_backtest_frame(1600)

    configure_test_dependencies(
        monkeypatch,
        frame,
    )

    result = backtest.run_walk_forward_backtest(
        ticker="TEST",
        prediction_days=prediction_days,
        evaluation_years=10,
        initial_training_samples=initial_training_samples,
    )

    first_window = result["windows"][0]

    assert (
        first_window["training_samples"]
        >= initial_training_samples
    )

    assert (
        first_window["purge_gap_samples"]
        == prediction_days
    )

    assert (
        result["methodology"]["overlapping_windows"]
        is False
    )


def test_walk_forward_origins_do_not_overlap(
    monkeypatch,
):
    prediction_days = 60

    frame = make_backtest_frame(1800)

    configure_test_dependencies(
        monkeypatch,
        frame,
    )

    result = backtest.run_walk_forward_backtest(
        ticker="TEST",
        prediction_days=prediction_days,
        initial_training_samples=300,
    )

    origin_dates = [
        pd.Timestamp(window["origin_date"])
        for window in result["windows"]
    ]

    origin_positions = [
        frame.index.get_loc(date)
        for date in origin_dates
    ]

    differences = np.diff(origin_positions)

    assert np.all(
        differences == prediction_days
    )


def test_walk_forward_reports_expected_metrics(
    monkeypatch,
):
    frame = make_backtest_frame(
        row_count=1800,
        target_return=0.20,
    )

    configure_test_dependencies(
        monkeypatch,
        frame,
    )

    result = backtest.run_walk_forward_backtest(
        ticker="TEST",
        prediction_days=63,
        initial_training_samples=300,
    )

    summary = result["summary"]

    # Dummy model predicts +10%; actual is always +20%.
    assert summary["model_mae_pct_points"] == 10.0
    assert summary["baseline_mae_pct_points"] == 20.0

    assert (
        summary["mae_improvement_over_baseline_pct"]
        == 50.0
    )

    assert (
        summary["model_directional_accuracy_pct"]
        == 100.0
    )

    assert (
        summary["baseline_directional_accuracy_pct"]
        == 0.0
    )

    assert summary["strategy_cumulative_return_pct"] > 0
    assert summary["cash_cumulative_return_pct"] == 0.0


def test_backtest_rejects_overlapping_windows():
    with pytest.raises(
        ValueError,
        match="double-count",
    ):
        backtest.run_walk_forward_backtest(
            ticker="TEST",
            prediction_days=63,
            step_days=21,
        )
