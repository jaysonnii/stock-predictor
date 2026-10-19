from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from model import predict_price, predict_price_with_fallback, train_model
from data import fetch_stock_data

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


# Yahoo's ETF screener has no dividend-yield filter, so the scan works in two
# steps: (1) list US ETFs in income-oriented categories with enough assets,
# (2) look up each one's dividendYield. Step 2 is slow, so it runs in a
# background thread and the endpoint returns progress until it finishes.
DIVIDEND_MIN_FUND_ASSETS = 50_000_000
DIVIDEND_MAX_UNIVERSE = 2000
DIVIDEND_CACHE_SECONDS = 24 * 60 * 60
DIVIDEND_INCOME_CATEGORIES = [
    "Large Value", "Large Blend", "Mid-Cap Value", "Small Value",
    "Option Writing", "Preferred Stock", "High Yield Bond", "Multisector Bond",
    "Long-Short Credit", "Multialternative", "Tactical Allocation",
    "Convertibles", "Bank Loan", "Real Estate", "Financial", "Infrastructure",
    "Equity Energy", "Natural Resources", "Other Allocation", "Other",
]

_div_lock = threading.Lock()
_div_state = {"status": "idle", "scanned": 0, "total": 0, "rows": [], "fetched_at": 0.0, "error": None}


def _list_etf_universe() -> list[str]:
    """Pull US ETF tickers in income-oriented categories from Yahoo's screener."""
    import yfinance as yf
    from yfinance import ETFQuery

    query = ETFQuery("and", [
        ETFQuery("eq", ["region", "us"]),
        ETFQuery("is-in", ["categoryname", *DIVIDEND_INCOME_CATEGORIES]),
        ETFQuery("gt", ["fundnetassets", DIVIDEND_MIN_FUND_ASSETS]),
    ])
    symbols, offset = [], 0
    while offset < DIVIDEND_MAX_UNIVERSE:
        data = yf.screen(query, offset=offset, size=250)
        quotes = data.get("quotes", [])
        symbols += [q["symbol"] for q in quotes if q.get("symbol")]
        if len(quotes) < 250:
            break
        offset += 250
    return list(dict.fromkeys(symbols))  # de-duplicate, keep order


def _yield_for(symbol: str):
    """Return a yield row for one ETF, or None if Yahoo has no yield for it."""
    import yfinance as yf
    try:
        info = yf.Ticker(symbol).info
        raw_yield = info.get("dividendYield")
        if raw_yield is None:
            return None
        # Yahoo's dividendYield is already a percent (SPY ~0.99 means 0.99%).
        # Do not rescale it.
        return {
            "ticker": symbol,
            "name": info.get("shortName", "N/A"),
            "dividend_yield_pct": round(float(raw_yield), 2),
        }
    except Exception:
        return None


def _run_dividend_scan():
    try:
        universe = _list_etf_universe()
        with _div_lock:
            _div_state.update(status="scanning", scanned=0, total=len(universe), rows=[], error=None)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row in pool.map(_yield_for, universe):
                with _div_lock:
                    _div_state["scanned"] += 1
                    if row:
                        _div_state["rows"].append(row)
        with _div_lock:
            _div_state.update(status="ready", fetched_at=time.time())
    except Exception as e:
        with _div_lock:
            _div_state.update(status="error", error=str(e))


def _ensure_dividend_scan():
    """Start a scan if there is no fresh result and none is running."""
    with _div_lock:
        if _div_state["status"] == "scanning":
            return
        if _div_state["status"] == "ready" and time.time() - _div_state["fetched_at"] < DIVIDEND_CACHE_SECONDS:
            return
        _div_state["status"] = "scanning"
    threading.Thread(target=_run_dividend_scan, daemon=True).start()


@app.get("/dividend-etfs")
def get_dividend_etfs(min_yield_pct: float = 4.0):
    """
    High-dividend US ETFs at or above min_yield_pct, highest yield first.
    The first call starts a full scan (several minutes). Until status is
    'ready', etfs holds the matches found so far and scanned/total show progress.
    """
    _ensure_dividend_scan()
    with _div_lock:
        matches = [r for r in _div_state["rows"] if r["dividend_yield_pct"] >= min_yield_pct]
        matches = sorted(matches, key=lambda r: r["dividend_yield_pct"], reverse=True)
        return {
            "min_yield_pct": min_yield_pct,
            "status": _div_state["status"],
            "scanned": _div_state["scanned"],
            "total": _div_state["total"],
            "etfs": matches,
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
