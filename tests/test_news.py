from fastapi.testclient import TestClient

import main
import news


client = TestClient(main.app)

NEW_FORMAT = {
    "id": "a1",
    "content": {
        "title": "Apple beats on services",
        "summary": "Services revenue rose.",
        "pubDate": "2026-11-03T14:00:00Z",
        "contentType": "STORY",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://finance.yahoo.com/news/apple-1.html"},
        "clickThroughUrl": {"url": "https://www.reuters.com/apple-1"},
    },
}
OLD_FORMAT = {
    "title": "Apple supplier update",
    "publisher": "Bloomberg",
    "link": "https://finance.yahoo.com/news/apple-2.html",
    "providerPublishTime": 1762300800,  # 2025-11-05
}


def setup_function():
    news._news_cache.clear()


def test_parse_new_format():
    a = news.parse_article(NEW_FORMAT)
    assert a == {
        "title": "Apple beats on services",
        "publisher": "Reuters",
        "url": "https://www.reuters.com/apple-1",
        "published": "2026-11-03T14:00:00Z",
        "summary": "Services revenue rose.",
    }


def test_parse_old_format():
    a = news.parse_article(OLD_FORMAT)
    assert a["publisher"] == "Bloomberg"
    assert a["url"] == "https://finance.yahoo.com/news/apple-2.html"
    assert a["published"].startswith("2025-11-05")


def test_parse_skips_videos_bad_links_and_junk():
    video = {"content": {**NEW_FORMAT["content"], "contentType": "VIDEO"}}
    bad_link = {"title": "x", "link": "javascript:alert(1)"}
    assert news.parse_article(video) is None
    assert news.parse_article(bad_link) is None
    assert news.parse_article({"content": {}}) is None
    assert news.parse_article("nope") is None


def test_news_sorted_deduped_and_cached(monkeypatch):
    calls = []

    def fake(ticker, count):
        calls.append(ticker)
        return [OLD_FORMAT, NEW_FORMAT, NEW_FORMAT]

    monkeypatch.setattr(news, "_fetch_raw", fake)
    items = news.news_for_ticker("aapl", 10)
    assert [a["title"] for a in items] == ["Apple beats on services", "Apple supplier update"]
    news.news_for_ticker("AAPL", 10)
    assert calls == ["AAPL"]


def test_empty_results_are_not_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(news, "_fetch_raw", lambda t, c: calls.append(t) or [])
    news.news_for_ticker("ZZZ")
    news.news_for_ticker("ZZZ")
    assert len(calls) == 2


def test_news_endpoint(monkeypatch):
    monkeypatch.setattr(news, "_fetch_raw", lambda t, c: [NEW_FORMAT])
    r = client.get("/news", params={"ticker": "aapl"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["items"][0]["publisher"] == "Reuters"
    assert "do not change the ranges" in body["note"]


def test_news_endpoint_rejects_bad_ticker():
    r = client.get("/news", params={"ticker": "AA PL;"})
    assert r.status_code == 400


def test_news_endpoint_reports_yahoo_failure(monkeypatch):
    def boom(t, c):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(news, "_fetch_raw", boom)
    r = client.get("/news", params={"ticker": "AAPL"})
    assert r.status_code == 502
    assert "News unavailable" in r.json()["detail"]


RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Yahoo</title>
<item><title>Apple RSS headline</title><link>https://finance.yahoo.com/news/apple-3.html</link>
<description>From the feed.</description><pubDate>Tue, 03 Nov 2026 16:30:00 +0000</pubDate></item>
</channel></rss>"""


def test_parse_rss():
    a = news.parse_article(news.parse_rss(RSS)[0])
    assert a["title"] == "Apple RSS headline"
    assert a["published"] == "2026-11-03T16:30:00Z"
    assert a["summary"] == "From the feed."


def _fail(name):
    def f(t, c):
        raise RuntimeError(f"{name} down")
    f.__name__ = name
    return f


def test_falls_back_to_next_source(monkeypatch):
    rss = lambda t, c: news.parse_rss(RSS)
    monkeypatch.setattr(news, "SOURCES", [_fail("_a"), lambda t, c: [], rss])
    items = news.news_for_ticker("AAPL")
    assert [a["title"] for a in items] == ["Apple RSS headline"]


def test_all_sources_failing_raises(monkeypatch):
    monkeypatch.setattr(news, "SOURCES", [_fail("_a"), _fail("_b")])
    try:
        news.news_for_ticker("AAPL")
        assert False, "expected an error"
    except RuntimeError as e:
        assert "a: _a down" in str(e) and "b: _b down" in str(e)


def test_search_prefers_related_tickers(monkeypatch):
    import yfinance as yf

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [
                {"title": "Other", "link": "https://x.com/1", "relatedTickers": ["MSFT"]},
                {"title": "Mine", "link": "https://x.com/2", "relatedTickers": ["AAPL"]},
            ]

    monkeypatch.setattr(yf, "Search", FakeSearch)
    assert [i["title"] for i in news._from_search("AAPL", 10)] == ["Mine"]
