"""
Recent Yahoo Finance headlines for a ticker.

Shown next to the ranges as context only. Yahoo returns just the latest
handful of articles with no history, so news cannot be walk-forward tested
and is never fed into the range math.
"""
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

NEWS_CACHE_SECONDS = 15 * 60
NEWS_MAX_ITEMS = 20
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")

_news_cache = {}  # (ticker, limit) -> {"at": timestamp, "items": list}


def _iso(value):
    """Turn a Yahoo date (ISO string or unix seconds) into an ISO UTC string."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(text)  # RSS style: "Tue, 03 Nov 2026 14:00:00 +0000"
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _url(obj):
    if isinstance(obj, dict):
        return obj.get("url")
    return obj or None


def parse_article(raw):
    """
    Normalise one Yahoo article. Handles the current format (fields under
    "content") and the older flat one. Returns None for items with no title
    or link, and for videos.
    """
    if not isinstance(raw, dict):
        return None
    c = raw.get("content") if isinstance(raw.get("content"), dict) else raw

    if str(c.get("contentType") or c.get("type") or "").upper() == "VIDEO":
        return None
    title = (c.get("title") or "").strip()
    url = (_url(c.get("clickThroughUrl")) or _url(c.get("canonicalUrl"))
           or c.get("link") or c.get("url"))
    if not title or not url or not str(url).startswith(("http://", "https://")):
        return None

    provider = c.get("provider")
    publisher = (provider.get("displayName") if isinstance(provider, dict) else None) or c.get("publisher")
    published = _iso(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"))
    summary = (c.get("summary") or c.get("description") or "").strip()
    if len(summary) > 300:
        summary = summary[:297].rstrip() + "..."

    return {
        "title": title,
        "publisher": publisher or "Yahoo Finance",
        "url": url,
        "published": published,
        "summary": summary or None,
    }


RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def _from_ticker_news(ticker, count):
    """Yahoo's quote page news feed (yfinance Ticker.get_news)."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        return t.get_news(count=count, tab="news") or []
    except AttributeError:  # older yfinance
        return t.news or []


def _from_search(ticker, count):
    """Yahoo's search API, which returns news in the older flat format."""
    import yfinance as yf
    items = yf.Search(ticker, max_results=1, news_count=count, lists_count=0,
                      recommended=0, include_cb=False).news or []
    related = [i for i in items if ticker in (i.get("relatedTickers") or [])]
    return related or items


def parse_rss(xml_text):
    """Turn Yahoo's headline RSS feed into flat article dicts."""
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        out.append({
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "summary": (item.findtext("description") or "").strip(),
            "pubDate": (item.findtext("pubDate") or "").strip(),
            "publisher": "Yahoo Finance",
        })
    return out


def _from_rss(ticker, count):
    """Yahoo's plain RSS headline feed. Needs no cookies, so it often works when the others are throttled."""
    import requests
    r = requests.get(RSS_URL, params={"s": ticker, "region": "US", "lang": "en-US"},
                     headers=_UA, timeout=15)
    r.raise_for_status()
    return parse_rss(r.text)[:count]


SOURCES = [_from_ticker_news, _from_search, _from_rss]


def _fetch_raw(ticker, count):
    """Try each Yahoo source in turn and return the first one that gives articles."""
    errors = []
    for source in SOURCES:
        try:
            items = source(ticker, count)
            if items:
                return items
        except Exception as e:
            errors.append(f"{source.__name__.lstrip('_')}: {e}")
    if errors and len(errors) == len(SOURCES):
        raise RuntimeError("all Yahoo news sources failed (" + "; ".join(errors) + ")")
    return []


def news_for_ticker(ticker, limit=10):
    """Latest headlines for a ticker, newest first, cached for 15 minutes."""
    ticker = ticker.upper().strip()
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker '{ticker}'.")
    limit = max(1, min(int(limit), NEWS_MAX_ITEMS))

    key = (ticker, limit)
    hit = _news_cache.get(key)
    if hit and time.time() - hit["at"] < NEWS_CACHE_SECONDS:
        return hit["items"]

    seen, items = set(), []
    for raw in _fetch_raw(ticker, limit):
        a = parse_article(raw)
        if a and a["url"] not in seen:
            seen.add(a["url"])
            items.append(a)
    items.sort(key=lambda a: a["published"] or "", reverse=True)
    items = items[:limit]

    if items:  # do not cache empty results, so a Yahoo hiccup gets retried
        _news_cache[key] = {"at": time.time(), "items": items}
    return items
