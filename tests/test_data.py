import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import data
from data import compute_rsi, engineer_features, get_feature_columns


@pytest.fixture(autouse=True)
def offline_market_context(monkeypatch):
    """Replace the SPY/VIX download with synthetic values so tests stay offline."""

    def fake_market_context(period=data.HISTORY_PERIOD):
        dates = pd.bdate_range("2023-01-01", "2026-12-31")
        count = len(dates)
        return pd.DataFrame(
            {
                "SPY_Ret_20": np.full(count, 0.01),
                "SPY_Price_to_MA200": np.full(count, 1.05),
                "VIX_Level": np.full(count, 18.0),
                "VIX_Change_20": np.full(count, -0.02),
            },
            index=dates,
        )

    monkeypatch.setattr(data, "fetch_market_context", fake_market_context)


def make_market_data(row_count: int = 320) -> pd.DataFrame:
    """Create predictable synthetic market data for offline tests."""
    dates = pd.bdate_range("2024-01-01", periods=row_count)
    # Upward trend plus a small wiggle, so there are down days too.
    # A straight line has no losing days, which leaves MFI undefined.
    trend = np.linspace(100.0, 200.0, row_count)
    close = trend + 2.0 * np.sin(np.arange(row_count) / 3.0)

    return pd.DataFrame(
        {
            "Open": close - 0.50,
            "High": close + 1.00,
            "Low": close - 1.00,
            "Close": close,
            "Volume": np.arange(
                1_000_000,
                1_000_000 + row_count,
            ),
        },
        index=dates,
    )


def test_feature_columns_match_training_schema():
    expected_columns = [
        "Price_to_MA20", "Price_to_MA50", "Price_to_MA200", "MA50_to_MA200",
        "Daily_Return", "Volatility_20", "ATR_pct",
        "RSI", "MACD_pct", "MACD_Signal_pct", "Stoch_K", "Stoch_D",
        "CCI_20", "MFI_14",
        "Ret_7", "Ret_14", "Ret_30", "Ret_60",
        "Volume_Change", "Volume_Ratio",
        "SPY_Ret_20", "SPY_Price_to_MA200", "VIX_Level", "VIX_Change_20",
    ]

    assert get_feature_columns() == expected_columns
    assert "Target" not in get_feature_columns()
    # Raw price levels must stay out of the features (they drift over time).
    for raw in ("Close", "Volume", "MA_20", "MA_50", "MA_200"):
        assert raw not in get_feature_columns()


def test_engineer_features_creates_complete_dataset():
    raw_data = make_market_data()
    original_data = raw_data.copy(deep=True)

    result = engineer_features(
        raw_data,
        prediction_days=5,
    )

    # Feature engineering should not modify the source DataFrame.
    assert_frame_equal(raw_data, original_data)

    required_columns = get_feature_columns() + ["Target"]

    assert not result.empty
    assert set(required_columns).issubset(result.columns)
    assert not result[required_columns].isna().any().any()


def test_return_and_target_values_match_source_prices():
    raw_data = make_market_data()
    prediction_days = 5

    result = engineer_features(
        raw_data,
        prediction_days=prediction_days,
    )

    test_date = result.index[0]
    source_position = raw_data.index.get_loc(test_date)

    current_price = raw_data.iloc[source_position]["Close"]
    future_price = raw_data.iloc[
        source_position + prediction_days
    ]["Close"]
    lagged_price = raw_data.iloc[source_position - 7]["Close"]
    expected_ret_7 = current_price / lagged_price - 1

    expected_target = (
        future_price - current_price
    ) / current_price

    assert np.isclose(
        result.loc[test_date, "Ret_7"],
        expected_ret_7,
    )
    assert np.isclose(
        result.loc[test_date, "Target"],
        expected_target,
    )


def test_rsi_reaches_100_for_consistently_rising_prices():
    rising_prices = pd.Series(
        np.arange(1.0, 31.0),
    )

    rsi = compute_rsi(
        rising_prices,
        window=14,
    ).dropna()

    assert not rsi.empty
    assert np.allclose(rsi, 100.0)

def test_prediction_mode_keeps_newest_row():
    raw_data = make_market_data()

    result = engineer_features(
        raw_data,
        prediction_days=5,
        with_target=False,
    )

    # Prediction must use the latest bar, not one that is prediction_days old.
    assert result.index[-1] == raw_data.index[-1]
    assert "Target" not in result.columns
    assert not result[get_feature_columns()].isna().any().any()
