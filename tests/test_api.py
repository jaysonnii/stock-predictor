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


def test_prediction_response_schema(monkeypatch):
    def fake_predict_price(ticker: str, prediction_days: int = 252):
        return {
            "ticker": ticker.upper(),
            "current_price": 200.00,
            "predicted_price_1yr": 220.00,
            "expected_return_pct": 10.00,
            "prediction_horizon_days": prediction_days,
        }

    monkeypatch.setattr(main, "predict_price", fake_predict_price)

    response = client.get(
        "/predict",
        params={
            "ticker": "aapl",
            "days": 252,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["ticker"] == "AAPL"
    assert body["current_price"] == 200.00
    assert body["predicted_price_1yr"] == 220.00
    assert body["expected_return_pct"] == 10.00
    assert body["prediction_horizon_days"] == 252
    assert "timestamp" in body


def test_training_response_includes_baseline_metrics(monkeypatch):
    fake_metrics = {
        "ticker": "AAPL",
        "rf_mae_pct_points": 17.75,
        "rf_rmse_pct_points": 21.21,
        "rf_directional_accuracy_pct": 77.24,
        "baseline_mae_pct_points": 21.70,
        "baseline_rmse_pct_points": 26.75,
        "baseline_directional_accuracy_pct": 8.72,
        "mae_improvement_over_baseline_pct": 18.21,
        "rmse_improvement_over_baseline_pct": 20.72,
        "training_samples": 1397,
        "purge_gap_samples": 252,
        "test_samples": 413,
    }

    monkeypatch.setattr(
        main,
        "train_model",
        lambda ticker: fake_metrics.copy(),
    )

    response = client.post(
        "/train",
        params={"ticker": "aapl"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["ticker"] == "AAPL"
    assert body["rf_mae_pct_points"] == 17.75
    assert body["baseline_mae_pct_points"] == 21.70
    assert body["purge_gap_samples"] == 252
    assert body["mae_improvement_over_baseline_pct"] == 18.21
    assert body["message"] == "Model trained successfully for AAPL"
    