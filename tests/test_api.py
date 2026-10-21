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
