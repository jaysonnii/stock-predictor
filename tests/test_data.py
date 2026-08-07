import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data import compute_rsi, engineer_features, get_feature_columns


def make_market_data(row_count: int = 320) -> pd.DataFrame:
    """Create predictable synthetic market data for offline tests."""
    dates = pd.bdate_range("2024-01-01", periods=row_count)
    close = np.linspace(100.0, 200.0, row_count)

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
        "Close",
        "Volume",
        "MA_20",
        "MA_50",
        "MA_200",
        "Price_to_MA50",
        "Price_to_MA200",
        "Daily_Return",
        "Volatility_20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Lag_7",
        "Lag_14",
        "Lag_30",
        "Lag_60",
        "Volume_Change",
    ]

    assert get_feature_columns() == expected_columns
    assert "Target" not in get_feature_columns()


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


def test_lag_and_target_values_match_source_prices():
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

    expected_target = (
        future_price - current_price
    ) / current_price

    assert np.isclose(
        result.loc[test_date, "Lag_7"],
        lagged_price,
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