import numpy as np
import pandas as pd

from calibration import add_interval_columns, summarize
from evaluate_portfolio import make_portfolios, score_portfolio
from portfolio import aligned_log_returns, ewma_matrix, portfolio_returns, simulate_joint_returns


def correlated_returns(days=2500, n=6, rho=0.7, sd=0.012, seed=0):
    """Daily log returns with a common market factor, correlation rho between any two."""
    rng = np.random.default_rng(seed)
    market = rng.normal(size=(days, 1))
    own = rng.normal(size=(days, n))
    z = np.sqrt(rho) * market + np.sqrt(1 - rho) * own
    return 0.0003 + sd * z


def test_joint_draws_keep_correlation_and_independent_draws_lose_it():
    r = correlated_returns(n=2, rho=0.8)
    rng = np.random.default_rng(1)
    joint = np.log1p(simulate_joint_returns(r, 63, rng, n_sims=4000, joint=True))
    indep = np.log1p(simulate_joint_returns(r, 63, rng, n_sims=4000, joint=False))
    assert np.corrcoef(joint.T)[0, 1] > 0.65
    assert abs(np.corrcoef(indep.T)[0, 1]) < 0.1


def test_shrink_factor_applies_to_each_holding():
    r = correlated_returns(n=3)
    a = np.log1p(simulate_joint_returns(r, 63, np.random.default_rng(2), ks=[1, 1, 1]))
    b = np.log1p(simulate_joint_returns(r, 63, np.random.default_rng(2), ks=[0.5, 1, 1]))
    assert np.isclose(b[:, 0].std(), 0.5 * a[:, 0].std(), rtol=1e-6)
    assert np.allclose(a[:, 1:], b[:, 1:])


def test_precomputed_volatility_matches_computing_it_on_the_slice():
    r = correlated_returns(n=3, days=900)
    before, after = ewma_matrix(r)
    i = 700
    a = simulate_joint_returns(r[: i + 1], 21, np.random.default_rng(3), n_sims=500)
    b = simulate_joint_returns(r[: i + 1], 21, np.random.default_rng(3), n_sims=500,
                               sigma_before=before[: i + 1], sigma_now=after[i])
    assert np.allclose(a, b)


def test_portfolio_return_uses_value_weights():
    sims = np.array([[0.10, -0.10], [0.20, 0.00]])
    assert np.allclose(portfolio_returns(sims, [3, 1]), [0.05, 0.15])


def test_aligned_returns_keep_only_common_dates():
    a = pd.Series(np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30))
    b = pd.Series(np.linspace(50, 55, 20), index=pd.bdate_range("2024-01-15", periods=20))
    out = aligned_log_returns({"A": a, "B": b})
    assert list(out.columns) == ["A", "B"]
    assert out.index.min() > pd.Timestamp("2024-01-15")
    assert not out.isna().any().any()


def test_walk_forward_joint_is_calibrated_and_independent_is_too_narrow():
    frames = []
    for seed in range(3):
        r = correlated_returns(days=3000, n=6, rho=0.7, seed=10 + seed)
        log_ret = pd.DataFrame(r, index=pd.bdate_range("2014-01-01", periods=len(r)))
        # Forecasts one horizon apart, so outcomes do not overlap. k = 1: synthetic data
        # needs no correction, unlike the real stocks the dashboard factor was fitted on.
        frames.append(score_portfolio(log_ret, 63, np.random.default_rng(seed),
                                      n_sims=1000, step_days=63, k=1.0))
    res = add_interval_columns(pd.concat(frames, ignore_index=True))
    joint = summarize(res[res["Method"] == "joint"])
    indep = summarize(res[res["Method"] == "independent"])
    assert 70 <= joint["in_80_pct"] <= 90
    assert indep["in_80_pct"] < joint["in_80_pct"] - 15


def test_portfolios_are_the_same_every_run():
    tickers = [f"T{i}" for i in range(30)]
    assert make_portfolios(tickers) == make_portfolios(tickers)
    assert all(len(set(p)) == 6 for p in make_portfolios(tickers))


# --- Dashboard forecast and endpoint ---

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import ranges  # noqa: E402
from portfolio import forecast_portfolio  # noqa: E402


def make_closes(n=3, days=1500, rho=0.5, seed=0):
    r = correlated_returns(days=days, n=n, rho=rho, seed=seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    return {f"T{j}": pd.Series(100 * np.exp(np.cumsum(r[:, j])), index=dates) for j in range(n)}


def test_portfolio_forecast_is_nested_and_valued_correctly():
    closes = make_closes()
    shares = {"T0": 10, "T1": 5, "T2": 2}
    out = forecast_portfolio(closes, shares)
    expected_value = sum(float(closes[t].iloc[-1]) * s for t, s in shares.items())
    assert np.isclose(out["current_value"], expected_value, atol=0.01)
    assert [h["days"] for h in out["horizons"]] == [63, 126, 252]
    for h in out["horizons"]:
        g = {x["level"]: x for x in h["ranges"]}
        assert g[90]["low_value"] <= g[80]["low_value"] <= g[50]["low_value"] <= h["median_value"]
        assert h["median_value"] <= g[50]["high_value"] <= g[80]["high_value"] <= g[90]["high_value"]
        assert h["bad_case_value"] < out["current_value"] or h["bad_case_return_pct"] >= 0


def test_diversified_portfolio_is_narrower_than_one_stock():
    closes = make_closes(n=6, rho=0.3, seed=3)
    one = forecast_portfolio({"T0": closes["T0"]}, {"T0": 1})["horizons"][0]
    six = forecast_portfolio(closes, {t: 1 for t in closes})["horizons"][0]
    width = lambda h: next(x for x in h["ranges"] if x["level"] == 80)
    assert (width(six)["high_return_pct"] - width(six)["low_return_pct"]) < \
        0.8 * (width(one)["high_return_pct"] - width(one)["low_return_pct"])


def test_portfolio_endpoint(monkeypatch):
    closes = make_closes()
    monkeypatch.setattr(ranges, "get_close", lambda t: closes[t])
    body = {"holdings": [{"ticker": "t0", "shares": 3}, {"ticker": "T1", "shares": 1},
                         {"ticker": "T0", "shares": 2}]}
    response = TestClient(main.app).post("/portfolio/ranges", json=body)
    assert response.status_code == 200
    out = response.json()
    assert out["tickers"] == ["T0", "T1"]  # duplicate T0 rows are merged
    expected = 5 * float(closes["T0"].iloc[-1]) + float(closes["T1"].iloc[-1])
    assert np.isclose(out["current_value"], expected, atol=0.01)


def test_portfolio_endpoint_rejects_empty_and_short_history(monkeypatch):
    client = TestClient(main.app)
    assert client.post("/portfolio/ranges", json={"holdings": []}).status_code == 400

    short = make_closes(days=300)
    monkeypatch.setattr(ranges, "get_close", lambda t: short[t])
    response = client.post("/portfolio/ranges", json={"holdings": [{"ticker": "T0", "shares": 1}]})
    assert response.status_code == 400
    assert "history" in response.json()["detail"]
