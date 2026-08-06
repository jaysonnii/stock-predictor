from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os

from model import predict_price, train_model
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
    ticker: str
    current_price: float
    predictions: list[HorizonPrediction]
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
    Only the 1-year model is trained via ML.
    Shorter horizons are scaled proportionally from the 1-year trend,
    which keeps all three numbers logically consistent.
    """
    ticker = ticker.upper().strip()
    try:
        # Only train/use the reliable 1-year model
        result = predict_price(ticker, prediction_days=252)
        current_price = result["current_price"]
        price_1yr = result["predicted_price_1yr"]

        # Annual return rate implied by the model
        annual_return = (price_1yr - current_price) / current_price

        # Scale proportionally for shorter horizons
        def scale(trading_days):
            fraction = trading_days / 252
            scaled_return = annual_return * fraction
            predicted = round(current_price * (1 + scaled_return), 2)
            pct = round(scaled_return * 100, 2)
            return predicted, pct

        price_1m, pct_1m = scale(21)
        price_6m, pct_6m = scale(126)

        predictions = [
            HorizonPrediction(horizon="1 Month",  days=21,  predicted_price=price_1m,  expected_return_pct=pct_1m),
            HorizonPrediction(horizon="6 Months", days=126, predicted_price=price_6m,  expected_return_pct=pct_6m),
            HorizonPrediction(horizon="1 Year",   days=252, predicted_price=price_1yr, expected_return_pct=result["expected_return_pct"]),
        ]

        return MultiPredictionResponse(
            ticker=ticker,
            current_price=current_price,
            predictions=predictions,
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
