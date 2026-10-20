import time

import numpy as np
import pandas as pd
import yfinance as yf

# History length used for BOTH training and prediction.
HISTORY_PERIOD = "10y"

# Market context: S&P 500 ETF and the VIX volatility index.
MARKET_TICKERS = ("SPY", "^VIX")
MARKET_COLUMNS = ["SPY_Ret_20", "SPY_Price_to_MA200", "VIX_Level", "VIX_Change_20"]
_MARKET_CACHE_SECONDS = 60 * 60
_market_cache = {}


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


def _naive_dates(index) -> pd.DatetimeIndex:
    """Strip timezone and time-of-day so different sources line up by date."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def fetch_market_context(period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """
    Market-wide features indexed by calendar date.
    Cached for an hour so the dashboard does not refetch on every request.
    """
    hit = _market_cache.get(period)
    if hit and time.time() - hit[0] < _MARKET_CACHE_SECONDS:
        return hit[1].copy()

    spy = fetch_stock_data(MARKET_TICKERS[0], period=period)
    vix = fetch_stock_data(MARKET_TICKERS[1], period=period)

    spy_close = pd.Series(spy["Close"].values, index=_naive_dates(spy.index))
    vix_close = pd.Series(vix["Close"].values, index=_naive_dates(vix.index))
    spy_close = spy_close[~spy_close.index.duplicated()]
    vix_close = vix_close[~vix_close.index.duplicated()]

    market = pd.DataFrame(
        {
            "SPY_Ret_20": spy_close.pct_change(20),
            "SPY_Price_to_MA200": spy_close / spy_close.rolling(200).mean(),
            "VIX_Level": vix_close,
            "VIX_Change_20": vix_close.pct_change(20),
        }
    )
    market = market.sort_index().ffill(limit=5)

    _market_cache[period] = (time.time(), market)
    return market.copy()


def _add_market_features(df: pd.DataFrame) -> None:
    market = fetch_market_context(HISTORY_PERIOD)
    aligned = market.reindex(_naive_dates(df.index))
    for col in MARKET_COLUMNS:
        df[col] = aligned[col].values


def engineer_features(
    df: pd.DataFrame,
    prediction_days: int = 252,
    with_target: bool = True,
) -> pd.DataFrame:
    """
    Add features built from price ratios and returns, not raw price levels.

    Raw prices (Close, moving averages, lagged prices) drift over time, so a
    model trained on them cannot extrapolate to prices it never saw. Every
    feature here is scale-free.

    with_target=True  adds the future return target and drops the last
                      prediction_days rows (no known future). Used for training.
    with_target=False keeps the most recent row. Used for prediction.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Trend: price relative to moving averages ---
    df["MA_20"] = close.rolling(window=20).mean()
    df["MA_50"] = close.rolling(window=50).mean()
    df["MA_200"] = close.rolling(window=200).mean()
    df["Price_to_MA20"] = close / df["MA_20"]
    df["Price_to_MA50"] = close / df["MA_50"]
    df["Price_to_MA200"] = close / df["MA_200"]
    df["MA50_to_MA200"] = df["MA_50"] / df["MA_200"]

    # --- Returns and volatility ---
    df["Daily_Return"] = close.pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()
    df["ATR_pct"] = compute_atr(high, low, close, window=14) / close

    # --- Momentum oscillators ---
    df["RSI"] = compute_rsi(close, window=14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_pct"] = macd / close
    df["MACD_Signal_pct"] = macd_signal / close
    stoch_k, stoch_d = compute_stochastic(high, low, close, window=14, smooth=3)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d
    df["CCI_20"] = compute_cci(high, low, close, window=20)

    # --- Past returns over several horizons (replaces raw lagged prices) ---
    for lag in [7, 14, 30, 60]:
        df[f"Ret_{lag}"] = close / close.shift(lag) - 1

    # --- Volume ---
    df["Volume_Change"] = volume.pct_change()
    df["Volume_Ratio"] = volume / volume.rolling(window=20).mean()
    df["MFI_14"] = compute_mfi(high, low, close, volume, window=14)

    # --- Market context ---
    _add_market_features(df)

    # --- Target: % return N trading days from now ---
    if with_target:
        future_price = close.shift(-prediction_days)
        df["Target"] = (future_price - close) / close

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if with_target:
        df.dropna(inplace=True)
    else:
        # Only warm-up rows are dropped. The newest row is kept.
        df.dropna(subset=get_feature_columns(), inplace=True)

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


def compute_atr(high, low, close, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window).mean()


def compute_stochastic(high, low, close, window: int = 14, smooth: int = 3):
    """Stochastic oscillator: %K (0-100) and its smoothed %D."""
    lowest = low.rolling(window=window).min()
    highest = high.rolling(window=window).max()
    span = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / span
    d = k.rolling(window=smooth).mean()
    return k, d


def compute_cci(high, low, close, window: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical = (high + low + close) / 3
    sma = typical.rolling(window=window).mean()
    mean_dev = typical.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical - sma) / (0.015 * mean_dev.replace(0, np.nan))


def compute_mfi(high, low, close, volume, window: int = 14) -> pd.Series:
    """Money Flow Index: volume-weighted RSI (0-100)."""
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0).rolling(window=window).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(window=window).sum()
    ratio = positive / negative.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def get_feature_columns() -> list:
    """Return the feature column names used for training and prediction."""
    return [
        # Trend
        "Price_to_MA20", "Price_to_MA50", "Price_to_MA200", "MA50_to_MA200",
        # Returns and volatility
        "Daily_Return", "Volatility_20", "ATR_pct",
        # Oscillators
        "RSI", "MACD_pct", "MACD_Signal_pct", "Stoch_K", "Stoch_D",
        "CCI_20", "MFI_14",
        # Past returns
        "Ret_7", "Ret_14", "Ret_30", "Ret_60",
        # Volume
        "Volume_Change", "Volume_Ratio",
        # Market
        "SPY_Ret_20", "SPY_Price_to_MA200", "VIX_Level", "VIX_Change_20",
    ]
