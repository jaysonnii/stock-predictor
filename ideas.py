"""
"Ideas worth a look": a rule based screen of the stocks in research/universe.py.

Signals, each shown with its reason and, where it can be tested, its
walk forward record from signal_record.json (written by
research/signal_backtest.py). These are ideas to research, not predictions.

Pullback in uptrend was tested (Nov 4, 2026) and did worse than the average
stock (-1.0 pts at 63 days, -2.0 at 252; shuffled random picks did as well
about 88% of the time), so it no longer picks ideas. Its record is still
shown so the result stays visible.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from research.universe import _STOCKS
from signals import SIGNALS, momentum_signal

PRICE_CACHE_SECONDS = 6 * 60 * 60
INFO_CACHE_SECONDS = 24 * 60 * 60
INFO_CACHE_PATH = os.path.join("data_cache", "ideas_info.json")
RECORD_PATH = "signal_record.json"
INFO_WORKERS = 2
INFO_DELAY_SECONDS = 0.3
MAX_IDEAS = 15
# Signals that can put a stock on the list. Pullback is tested but not used.
USED_SIGNALS = ("momentum", "analyst", "value")
DROPPED_NOTE = {"pullback": "Dropped: in testing its picks did worse than the average stock, "
                            "and random picks did as well about 88% of the time. Not used to pick ideas."}

# Analyst targets: at least 5 analysts, average rating buy or better (Yahoo's
# 1 = strong buy to 5 = sell scale, 2.5 or lower), target 15%+ above price.
ANALYST_MIN_COUNT = 5
ANALYST_MAX_REC = 2.5
ANALYST_MIN_UPSIDE = 0.15
# Value and growth: earnings up 10%+ and sales up from a year ago, and a P/E
# below the median of its sector in the group (forward P/E when available).
VALUE_MIN_EARNINGS_GROWTH = 0.10
VALUE_MIN_SECTOR_PEERS = 3

RULES = {
    "momentum": "Top 20% by return from 12 to 1 months ago, and above the 200 day average",
    "pullback": "Above the 200 day average, 50 day above 200 day, and the last month fell at least 1 normal month",
    "analyst": f"At least {ANALYST_MIN_COUNT} analysts, average rating buy or better, average target "
               f"{int(ANALYST_MIN_UPSIDE * 100)}%+ above today's price",
    "value": f"Earnings up {int(VALUE_MIN_EARNINGS_GROWTH * 100)}%+ and sales up from a year ago, "
             f"with a P/E below its sector's median in the group",
}

_price_cache = {}   # "close" -> (timestamp, DataFrame)
_ideas_cache = {}   # "ideas" -> (timestamp, dict)


def _download_closes(tickers) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(list(tickers), period="2y", interval="1d", auto_adjust=True,
                     progress=False, threads=True, group_by="column")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close.dropna(how="all")


def get_closes() -> pd.DataFrame:
    hit = _price_cache.get("close")
    if hit and time.time() - hit[0] < PRICE_CACHE_SECONDS:
        return hit[1]
    close = _download_closes(_STOCKS)
    if close.empty:
        raise RuntimeError("Yahoo returned no prices")
    _price_cache["close"] = (time.time(), close)
    return close


INFO_FIELDS = ("shortName", "sector", "targetMeanPrice", "numberOfAnalystOpinions",
               "recommendationMean", "recommendationKey", "forwardPE", "trailingPE",
               "earningsGrowth", "revenueGrowth")


def _load_info_cache() -> dict:
    try:
        with open(INFO_CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_info_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(INFO_CACHE_PATH), exist_ok=True)
        with open(INFO_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def _fetch_info(ticker: str):
    import yfinance as yf
    time.sleep(INFO_DELAY_SECONDS)  # pace requests so Yahoo does not rate limit
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    return {k: info.get(k) for k in INFO_FIELDS}


def get_infos(tickers) -> dict:
    """Analyst and fundamental fields per ticker, cached on disk for a day."""
    cache = _load_info_cache()
    now = time.time()
    stale = [t for t in tickers if not cache.get(t) or now - cache[t].get("_at", 0) > INFO_CACHE_SECONDS]
    if stale:
        with ThreadPoolExecutor(max_workers=INFO_WORKERS) as pool:
            for t, info in zip(stale, pool.map(_fetch_info, stale)):
                if info:  # do not cache failures, so they get retried
                    cache[t] = {**info, "_at": now}
        _save_info_cache(cache)
    return {t: cache.get(t) or {} for t in tickers}


def load_record() -> dict:
    try:
        with open(RECORD_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _pe(info):
    for k in ("forwardPE", "trailingPE"):
        v = _num(info.get(k))
        if v is not None and 0 < v < 500:
            return v, "forward" if k == "forwardPE" else "trailing"
    return None, None


def sector_pe_medians(infos: dict) -> dict:
    by_sector = {}
    for info in infos.values():
        pe, _ = _pe(info)
        if pe is not None and info.get("sector"):
            by_sector.setdefault(info["sector"], []).append(pe)
    return {s: (float(np.median(v)), len(v)) for s, v in by_sector.items()}


def analyst_reason(info, price):
    target, n, rec = _num(info.get("targetMeanPrice")), _num(info.get("numberOfAnalystOpinions")), _num(info.get("recommendationMean"))
    if not (target and n and rec and price):
        return None
    upside = target / price - 1
    if n < ANALYST_MIN_COUNT or rec > ANALYST_MAX_REC or upside < ANALYST_MIN_UPSIDE:
        return None
    rating = (info.get("recommendationKey") or "buy").replace("_", " ")
    return {"upside_pct": round(upside * 100, 1),
            "text": f"{int(n)} analysts average a ${target:,.2f} target, {upside * 100:.0f}% above today, rated {rating}"}


def value_reason(info, sector_medians):
    eg, rg = _num(info.get("earningsGrowth")), _num(info.get("revenueGrowth"))
    pe, kind = _pe(info)
    sector = info.get("sector")
    if eg is None or rg is None or pe is None or not sector or sector not in sector_medians:
        return None
    med, peers = sector_medians[sector]
    if peers < VALUE_MIN_SECTOR_PEERS or eg < VALUE_MIN_EARNINGS_GROWTH or rg <= 0 or pe >= med:
        return None
    return {"text": f"Earnings up {eg * 100:.0f}% and sales up {rg * 100:.0f}% from a year ago, "
                    f"{kind} P/E {pe:.1f} vs {med:.1f} for its sector here"}


def build_ideas(close: pd.DataFrame, infos: dict, record: dict) -> dict:
    mom, ret_12_1 = momentum_signal(close)
    last = close.index[-1]
    medians = sector_pe_medians(infos)

    ideas = []
    for t in close.columns:
        price = _num(close[t].iloc[-1])
        if price is None:
            continue
        info = infos.get(t) or {}
        reasons = []
        if bool(mom.loc[last, t]):
            reasons.append({"signal": "momentum", "text": f"Up {ret_12_1.loc[last, t] * 100:.0f}% from 12 months to 1 month ago, "
                                                          f"top 20% of the group, and above its 200 day average"})
        a = analyst_reason(info, price)
        if a:
            reasons.append({"signal": "analyst", **a})
        v = value_reason(info, medians)
        if v:
            reasons.append({"signal": "value", **v})
        if reasons:
            ideas.append({"ticker": t, "name": info.get("shortName") or t, "sector": info.get("sector"),
                          "price": round(price, 2), "reasons": reasons})

    def sort_key(i):
        upside = next((r.get("upside_pct", 0) for r in i["reasons"] if r["signal"] == "analyst"), 0)
        return (-len(i["reasons"]), -upside, i["ticker"])
    ideas.sort(key=sort_key)

    signals = record.get("signals", {})
    return {
        "as_of": str(last.date()),
        "screened": int(close.shape[1]),
        "ideas": ideas[:MAX_IDEAS],
        "signals": {k: {"label": label, "rule": RULES[k], "used": k in USED_SIGNALS,
                        **({"note": DROPPED_NOTE[k]} if k in DROPPED_NOTE else {}), **signals.get(k, {})}
                    for k, label in SIGNALS.items()},
        "signals_used": len(USED_SIGNALS),
        "tested_on": record.get("tested_on"),
        "caveat": record.get("caveat"),
    }


def ideas() -> dict:
    hit = _ideas_cache.get("ideas")
    if hit and time.time() - hit[0] < PRICE_CACHE_SECONDS:
        return hit[1]
    close = get_closes()
    result = build_ideas(close, get_infos(list(close.columns)), load_record())
    _ideas_cache["ideas"] = (time.time(), result)
    return result
