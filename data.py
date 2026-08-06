import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Pull historical stock data from yfinance."""
    stock = yf.Ticker(ticker)
    # prepost=False, repair=True forces fresh data, no cache
    df = stock.history(
        period=period,
        interval="1d",
        auto_adjust=True,
        repair=False,
        timeout=30,
    )

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df


def engineer_features(df: pd.DataFrame, prediction_days: int = 252) -> pd.DataFrame:
    """
    Add technical indicators and create the target variable
    shifted 252 trading days (~1 year) into the future.
    """
    df = df.copy()

    # --- Moving Averages ---
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()

    # --- Price relative to moving averages ---
    df["Price_to_MA50"] = df["Close"] / df["MA_50"]
    df["Price_to_MA200"] = df["Close"] / df["MA_200"]

    # --- Daily Returns ---
    df["Daily_Return"] = df["Close"].pct_change()

    # --- Volatility (rolling std of returns) ---
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    # --- RSI (Relative Strength Index) ---
    df["RSI"] = compute_rsi(df["Close"], window=14)

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- Lag features (past prices as features) ---
    for lag in [7, 14, 30, 60]:
        df[f"Lag_{lag}"] = df["Close"].shift(lag)

    # --- Volume change ---
    df["Volume_Change"] = df["Volume"].pct_change()

    # --- Target: % return N trading days from now (more stable than raw price) ---
    future_price = df["Close"].shift(-prediction_days)
    df["Target"] = (future_price - df["Close"]) / df["Close"]

    # Drop rows with NaN values (from rolling windows + shift)
    df.dropna(inplace=True)

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_feature_columns() -> list:
    """Return the list of feature column names used for training."""
    return [
        "Close", "Volume",
        "MA_20", "MA_50", "MA_200",
        "Price_to_MA50", "Price_to_MA200",
        "Daily_Return", "Volatility_20",
        "RSI", "MACD", "MACD_Signal",
        "Lag_7", "Lag_14", "Lag_30", "Lag_60",
        "Volume_Change",
    ]
