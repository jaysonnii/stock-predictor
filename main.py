from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional
import os
import time
from concurrent.futures import ThreadPoolExecutor

from data import fetch_stock_data
from ranges import ranges_for_ticker
from portfolio import portfolio_ranges
from news import news_for_ticker
import json
import ideas as ideas_mod
import goal_log

app = FastAPI(
    title="Stock Predictor API",
    description="Calibrated price ranges and risk figures for stocks and portfolios, "
                "tested walk forward on historical data. No price targets.",
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

class NewsItem(BaseModel):
    title: str
    publisher: str
    url: str
    published: Optional[str]
    summary: Optional[str]

class NewsResponse(BaseModel):
    ticker: str
    items: list[NewsItem]
    source: str
    note: str

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


@app.get("/news", response_model=NewsResponse)
def get_news(ticker: str, limit: int = 10):
    """
    Latest Yahoo Finance headlines for a ticker, newest first. Context only:
    the headlines do not change the ranges.
    """
    try:
        items = news_for_ticker(ticker, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"News unavailable: {str(e)}")
    return {
        "ticker": ticker.upper().strip(),
        "items": items,
        "source": "Yahoo Finance",
        "note": "Headlines are shown for context and do not change the ranges.",
    }


@app.get("/ideas")
def get_ideas():
    """
    Ideas worth a look: a rule based screen of the stocks in research/universe.py
    using momentum, pullback in uptrend, analyst targets, and value and growth.
    Each idea lists why it was picked, and each signal carries its walk forward
    record when it can be tested. Ideas to research, not predictions.
    """
    try:
        return ideas_mod.ideas()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ideas unavailable: {str(e)}")


PERSONAL_PATH = "personal.json"


@app.get("/personal.js", include_in_schema=False)
def personal_js():
    """
    Your own holdings, goal and core beliefs, read from personal.json on this
    computer. personal.json is in .gitignore, so it never goes to GitHub; the
    public code starts blank. See personal.example.json for the format.
    """
    try:
        with open(PERSONAL_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    return Response(f"window.PERSONAL = {json.dumps(data)};", media_type="application/javascript",
                    headers={"Cache-Control": "no-store"})


GOAL_LOG_OFF = ("The check in log file is only kept on your own computer (it needs a personal.json "
                "next to main.py). On a shared server it is off, so no one's net worth is stored or shown.")


def goal_log_enabled() -> bool:
    """
    The check in log is one shared file with no accounts, so it only runs on a
    personal computer: where personal.json exists, or GOAL_LOG=1 is set. On the
    hosted demo neither is true, so visitors' net worth is never stored or served.
    """
    return os.path.exists(PERSONAL_PATH) or os.environ.get("GOAL_LOG") == "1"


def _require_goal_log():
    if not goal_log_enabled():
        raise HTTPException(status_code=404, detail=GOAL_LOG_OFF)


@app.get("/goal/checkins")
def get_goal_checkins():
    """Every daily $1M check in saved on this computer, oldest first. Off on shared servers."""
    _require_goal_log()
    entries = goal_log.read_all()
    return {"count": len(entries), "checkins": entries}


@app.post("/goal/checkins")
def add_goal_checkins(entries: list[dict]):
    """Save check ins. Ones already saved (same date and same words) are skipped. Off on shared servers."""
    _require_goal_log()
    if len(entries) > 5000:
        raise HTTPException(status_code=400, detail="Too many check ins at once.")
    try:
        return goal_log.append(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/goal/checkins.csv")
def download_goal_checkins():
    """The full check in log as a spreadsheet file. Off on shared servers."""
    _require_goal_log()
    return Response(
        content=goal_log.to_csv(goal_log.read_all()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="million-goal-checkins.csv"'},
    )


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
