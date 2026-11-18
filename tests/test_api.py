from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_returns_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "stock predictor" in response.text.lower()


def test_history_rejects_invalid_period():
    response = client.get(
        "/history",
        params={
            "ticker": "AAPL",
            "period": "10y",
        },
    )

    assert response.status_code == 400
    assert "Invalid period" in response.json()["detail"]


def test_point_forecast_endpoints_are_gone():
    # The point forecast model had no tested skill and was moved to research/.
    assert client.get("/predict", params={"ticker": "AAPL"}).status_code == 404
    assert client.post("/train", params={"ticker": "AAPL"}).status_code == 404


def test_history_is_sliced_from_cached_closes(monkeypatch):
    import numpy as np
    import pandas as pd
    import ranges
    idx = pd.date_range("2016-01-01", "2026-11-17", freq="D")   # every day, like crypto
    close = pd.Series(np.linspace(10, 70, len(idx)), index=idx)
    monkeypatch.setattr(ranges, "get_close", lambda t: close)
    r = client.get("/history", params={"ticker": "sol-usd", "period": "1mo"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "SOL-USD"
    assert body["dates"][0] == "2026-10-17" and body["dates"][-1] == "2026-11-17"
    assert len(body["dates"]) == len(body["prices"]) == 32


def test_get_close_reuses_last_good_copy_when_yahoo_fails(monkeypatch):
    import time as _t
    import pandas as pd
    import ranges
    old = pd.Series([1.0, 2.0], index=pd.date_range("2026-01-01", periods=2))
    ranges._close_cache["ZZZ"] = (_t.time() - 10 * 3600, old)   # stale
    def boom(ticker, period):
        raise RuntimeError("Too Many Requests. Rate limited.")
    monkeypatch.setattr(ranges, "fetch_stock_data", boom)
    assert ranges.get_close("ZZZ") is old
    ranges._close_cache.pop("ZZZ", None)
