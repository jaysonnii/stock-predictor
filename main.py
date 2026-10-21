from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor

from model import predict_price, predict_price_with_fallback, train_model
from data import fetch_stock_data
from ranges import ranges_for_ticker
from portfolio import portfolio_ranges

app = FastAPI(
    title="Stock Price Predictor API",
    description="Predicts stock prices into the future using ML trained on real historical data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class HorizonPrediction(BaseModel):
    horizon: str
    days: int
    predicted_price: float
    expected_return_pct: float

class MultiPredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str
    current_price: float
    predictions: list[HorizonPrediction]
    requested_horizon_days: int
    model_horizon_days: int
    fallback_used: bool
    timestamp: str

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price_1yr: float
    expected_return_pct: float
    prediction_horizon_days: int
    timestamp: str

class TrainResponse(BaseModel):
    ticker: str

    rf_mae_pct_points: float
    rf_rmse_pct_points: float
    rf_directional_accuracy_pct: float

    baseline_mae_pct_points: float
    baseline_rmse_pct_points: float
    baseline_directional_accuracy_pct: float

    mae_improvement_over_baseline_pct: float
    rmse_improvement_over_baseline_pct: float

    training_samples: int
    purge_gap_samples: int
    test_samples: int

    message: str

class PriceRange(BaseModel):
    level: int
    low_price: float
    high_price: float
    low_return_pct: float
    high_return_pct: float
    historical_hit_rate_pct: float

class HorizonRange(BaseModel):
    horizon: str
    days: int
    median_price: float
    median_return_pct: float
    ranges: list[PriceRange]
    chance_of_loss_pct: float
    bad_case_return_pct: float
    avg_worst_10pct_return_pct: float
    worst_year_hit_rate_80_pct: float
    worst_year: int
    shrink_factor: float
    note: Optional[str]

class RangesResponse(BaseModel):
    ticker: str
    current_price: float
    as_of: str
    current_volatility_annual_pct: float
    horizons: list[HorizonRange]
    method: str

class HoldingIn(BaseModel):
    ticker: str
    shares: float

class PortfolioRequest(BaseModel):
    holdings: list[HoldingIn]

class PortfolioRange(BaseModel):
    level: int
    low_value: float
    high_value: float
    low_return_pct: float
    high_return_pct: float
    historical_hit_rate_pct: float

class PortfolioHorizon(BaseModel):
    horizon: str
    days: int
    median_value: float
    median_return_pct: float
    ranges: list[PortfolioRange]
    chance_of_loss_pct: float
    bad_case_return_pct: float
    bad_case_value: float
    avg_worst_10pct_return_pct: float
    worst_year_hit_rate_80_pct: float
    worst_year: int
    note: Optional[str]

class PortfolioRangesResponse(BaseModel):
    current_value: float
    tickers: list[str]
    common_history_from: str
    as_of: str
    horizons: list[PortfolioHorizon]
    method: str

class HistoryResponse(BaseModel):
    ticker: str
    dates: list[str]
    prices: list[float]
    period: str

class StockInfoResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    fifty_two_week_high: float
    fifty_two_week_low: float

@app.get("/health", tags=["system"])
def health_check():
    """Return the current API health status."""
    return {"status": "ok"}

@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Stock Predictor API", "docs": "/docs"}


@app.get("/predict/all", response_model=MultiPredictionResponse)
def get_all_predictions(ticker: str):
    """
    Get 1-month, 6-month, and 1-year predictions.
    The ML model runs at the longest horizon the stock's history supports
    (falls back to 6 months, 3 months, then 1 month for newer tickers).
    Shorter horizons are scaled proportionally from that trend.
    Check fallback_used and model_horizon_days to see which horizon was used.
    """
    ticker = ticker.upper().strip()
    try:
        result = predict_price_with_fallback(ticker, prediction_days=252)
        model_days = result["prediction_horizon_days"]
        current_price = result["current_price"]
        model_price = result["predicted_price_1yr"]

        # Annual return rate implied by the model's own horizon
        model_return = (model_price - current_price) / current_price
        annual_return = model_return * (252 / model_days)

        # Scale proportionally for shorter horizons
        def scale(trading_days):
            fraction = trading_days / 252
            scaled_return = annual_return * fraction
            predicted = round(current_price * (1 + scaled_return), 2)
            pct = round(scaled_return * 100, 2)
            return predicted, pct

        price_1m, pct_1m = scale(21)
        price_6m, pct_6m = scale(126)

        # If the model ran on a shorter horizon, label the top row honestly
        if result["fallback_used"]:
            top_label = f"{model_days} Days (model)"
        else:
            top_label = "1 Year"

        predictions = [
            HorizonPrediction(horizon="1 Month",  days=21,  predicted_price=price_1m,  expected_return_pct=pct_1m),
            HorizonPrediction(horizon="6 Months", days=126, predicted_price=price_6m,  expected_return_pct=pct_6m),
            HorizonPrediction(horizon=top_label,  days=model_days, predicted_price=model_price, expected_return_pct=result["expected_return_pct"]),
        ]

        return MultiPredictionResponse(
            ticker=ticker,
            current_price=current_price,
            predictions=predictions,
            requested_horizon_days=result["requested_horizon_days"],
            model_horizon_days=model_days,
            fallback_used=result["fallback_used"],
            timestamp=datetime.utcnow().isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict", response_model=PredictionResponse)
def get_prediction(ticker: str, days: int = 252):
    """Predict a stock's price for a given horizon."""
    ticker = ticker.upper().strip()
    try:
        result = predict_price(ticker, prediction_days=days)
        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/ranges", response_model=RangesResponse)
def get_ranges(ticker: str):
    """
    Price ranges (50%, 80%, 90%) and risk figures at 3, 6 and 12 months.
    Uses a volatility-scaled bootstrap of past returns. Each range comes with
    the share of real outcomes that landed inside it in walk-forward tests,
    including the worst year, so the range is not read as a promise.
    """
    ticker = ticker.upper().strip()
    try:
        result = ranges_for_ticker(ticker)
        return {**result, "method": "Volatility-scaled block bootstrap, walk-forward calibrated"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranges failed: {str(e)}")


@app.post("/portfolio/ranges", response_model=PortfolioRangesResponse)
def get_portfolio_ranges(request: PortfolioRequest):
    """
    Value ranges and risk figures for a whole portfolio at 3, 6 and 12 months.
    All holdings are simulated together from the same historical blocks, so
    the ranges account for stocks falling at the same time.
    """
    if len(request.holdings) > 50:
        raise HTTPException(status_code=400, detail="Up to 50 holdings at a time.")
    try:
        result = portfolio_ranges([h.model_dump() for h in request.holdings])
        return {**result, "method": "Joint volatility-scaled block bootstrap, walk-forward tested"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio ranges failed: {str(e)}")


@app.post("/train", response_model=TrainResponse)
def retrain_model(ticker: str):
    """Train or retrain the ML model for a given ticker."""
    ticker = ticker.upper().strip()
    try:
        metrics = train_model(ticker)
        metrics["message"] = f"Model trained successfully for {ticker}"
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/history", response_model=HistoryResponse)
def get_history(ticker: str, period: str = "1y"):
    """Return historical closing prices for a ticker."""
    ticker = ticker.upper().strip()
    valid_periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Choose from: {valid_periods}")
    try:
        df = fetch_stock_data(ticker, period=period)
        return {
            "ticker": ticker,
            "dates": [str(d.date()) for d in df.index],
            "prices": [round(p, 2) for p in df["Close"].tolist()],
            "period": period,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


# High-dividend ETFs. The page keeps a list of tickers you choose, and this
# endpoint checks the live yield for only those tickers. Each ticker's result
# is cached for a day.
DIVIDEND_CACHE_SECONDS = 24 * 60 * 60
DIVIDEND_MAX_TICKERS = 100
DIVIDEND_WORKERS = 2
DIVIDEND_DELAY_SECONDS = 0.5

_div_cache = {}  # ticker -> {"at": timestamp, "row": dict}


def _yield_for(symbol: str):
    """Return a yield row for one ETF, or None if Yahoo has no yield for it."""
    import pandas as pd
    import yfinance as yf
    time.sleep(DIVIDEND_DELAY_SECONDS)  # pace requests so Yahoo does not rate-limit
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get("shortName", "N/A")

        # Preferred: actual cash paid in the last 12 months divided by price.
        # Yahoo's dividendYield field is wrong for some funds (QQQI shows 0.11).
        yield_pct = None
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        try:
            divs = ticker.dividends
            if price and len(divs) > 0:
                cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
                paid = float(divs[divs.index >= cutoff].sum())
                if paid > 0:
                    yield_pct = paid / float(price) * 100
        except Exception:
            pass

        # Fallback: Yahoo's dividendYield, already a percent (SPY 0.99 = 0.99%).
        if yield_pct is None:
            raw_yield = info.get("dividendYield")
            if raw_yield is None:
                return None
            yield_pct = float(raw_yield)

        return {
            "ticker": symbol,
            "name": name,
            "dividend_yield_pct": round(yield_pct, 2),
        }
    except Exception:
        return None


def _cached_yield(symbol: str):
    """Yield row for one ticker, reusing a result from the last day if there is one."""
    hit = _div_cache.get(symbol)
    if hit and time.time() - hit["at"] < DIVIDEND_CACHE_SECONDS:
        return hit["row"]
    row = _yield_for(symbol)
    if row:  # do not cache failures, so a rate-limited ticker gets retried
        _div_cache[symbol] = {"at": time.time(), "row": row}
    return row


@app.get("/dividend-etfs")
def get_dividend_etfs(min_yield_pct: float = 10.0, tickers: str = ""):
    """
    Live dividend yield for each ticker in the comma-separated tickers list.
    etfs holds the ones at or above min_yield_pct. checked holds every ticker
    that returned a yield. Both are sorted highest yield first.
    """
    symbols = []
    for t in tickers.split(","):
        t = t.upper().strip()
        if t and t not in symbols:
            symbols.append(t)
    symbols = symbols[:DIVIDEND_MAX_TICKERS]

    with ThreadPoolExecutor(max_workers=DIVIDEND_WORKERS) as pool:
        rows = [r for r in pool.map(_cached_yield, symbols) if r]
    checked = sorted(rows, key=lambda r: r["dividend_yield_pct"], reverse=True)
    matches = [r for r in checked if r["dividend_yield_pct"] >= min_yield_pct]
    return {
        "min_yield_pct": min_yield_pct,
        "status": "ready",
        "scanned": len(symbols),
        "total": len(symbols),
        "etfs": matches,
        "checked": checked,
    }


@app.get("/info", response_model=StockInfoResponse)
def get_stock_info(ticker: str):
    """Return general info about a stock."""
    import yfinance as yf
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "shortName" not in info:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
        return {
            "ticker": ticker,
            "company_name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch info: {str(e)}")
