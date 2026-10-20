import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import main
import ranges
from ranges import HORIZONS, forecast_ranges, shrink


def make_close(days=1500, daily_sd=0.012, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, daily_sd, size=days))), index=dates)


def test_shrink_narrows_around_the_median():
    sims = np.random.default_rng(1).normal(0.05, 0.2, size=10_000)
    narrowed = shrink(sims, 0.8)
    assert np.isclose(np.median(np.log1p(narrowed)), np.median(np.log1p(sims)))
    assert np.std(np.log1p(narrowed)) < np.std(np.log1p(sims))
    assert np.allclose(shrink(sims, 1.0), sims)


def test_ranges_are_ordered_and_nested():
    result = forecast_ranges(make_close())
    assert [h["days"] for h in result["horizons"]] == [63, 126, 252]

    for h in result["horizons"]:
        g = {r["level"]: r for r in h["ranges"]}
        assert g[90]["low_price"] <= g[80]["low_price"] <= g[50]["low_price"]
        assert g[50]["low_price"] <= h["median_price"] <= g[50]["high_price"]
        assert g[50]["high_price"] <= g[80]["high_price"] <= g[90]["high_price"]
        assert 0 <= h["chance_of_loss_pct"] <= 100
        assert h["avg_worst_10pct_return_pct"] <= h["bad_case_return_pct"]
        assert g[80]["historical_hit_rate_pct"] == HORIZONS[h["days"]]["hit_rate"][80]


def test_longer_horizons_give_wider_ranges():
    result = forecast_ranges(make_close())
    widths = [
        next(r for r in h["ranges"] if r["level"] == 80) for h in result["horizons"]
    ]
    spans = [w["high_return_pct"] - w["low_return_pct"] for w in widths]
    assert spans[0] < spans[1] < spans[2]


def test_a_volatile_stock_gets_wider_ranges_than_a_calm_one():
    calm = forecast_ranges(make_close(daily_sd=0.008, seed=2))["horizons"][0]
    wild = forecast_ranges(make_close(daily_sd=0.03, seed=2))["horizons"][0]
    width = lambda h: next(r for r in h["ranges"] if r["level"] == 80)
    assert (width(wild)["high_return_pct"] - width(wild)["low_return_pct"]) > \
        2 * (width(calm)["high_return_pct"] - width(calm)["low_return_pct"])


def test_same_input_gives_same_ranges():
    close = make_close()
    assert forecast_ranges(close) == forecast_ranges(close)


def test_ranges_endpoint(monkeypatch):
    close = make_close()
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1})
    monkeypatch.setattr(ranges, "fetch_stock_data", lambda ticker, period: frame)
    ranges._cache.clear()

    response = TestClient(main.app).get("/ranges", params={"ticker": "test"})
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "TEST"
    assert len(body["horizons"]) == 3
    assert body["horizons"][2]["note"]  # the 1 year downside warning


def test_ranges_endpoint_rejects_short_history(monkeypatch):
    close = make_close(days=200)
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1})
    monkeypatch.setattr(ranges, "fetch_stock_data", lambda ticker, period: frame)
    ranges._cache.clear()

    response = TestClient(main.app).get("/ranges", params={"ticker": "NEW"})
    assert response.status_code == 404
    assert "history" in response.json()["detail"]
