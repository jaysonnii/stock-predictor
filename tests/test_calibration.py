import numpy as np
import pandas as pd

from calibration import bootstrap_horizon_returns, calibrate_ticker, pit_deciles, summarize


def make_prices(days: int = 2000, daily_sd: float = 0.01, seed: int = 0) -> pd.Series:
    """Random-walk prices with a constant volatility, for offline tests."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, daily_sd, size=days)
    dates = pd.bdate_range("2015-01-01", periods=days)
    return pd.Series(100 * np.exp(np.cumsum(log_returns)), index=dates)


def test_bootstrap_matches_known_distribution():
    rng = np.random.default_rng(1)
    past = rng.normal(0.0, 0.01, size=20_000)
    sims = bootstrap_horizon_returns(past, horizon=63, rng=rng, n_sims=20_000)

    log_sims = np.log1p(sims)
    # Sum of 63 daily returns with sd 0.01 has sd 0.01 * sqrt(63).
    assert abs(log_sims.std() - 0.01 * np.sqrt(63)) < 0.004
    # The bootstrap resamples this exact history, so its mean is 63x the sample mean.
    assert abs(log_sims.mean() - 63 * past.mean()) < 0.002


def test_horizon_that_is_not_a_multiple_of_block_length():
    rng = np.random.default_rng(2)
    past = np.full(500, 0.001)
    sims = bootstrap_horizon_returns(past, horizon=50, rng=rng, n_sims=100, block_days=21)
    # Constant 0.1% a day for 50 days, whatever blocks were drawn.
    assert np.allclose(sims, np.expm1(0.05))


def test_actual_return_matches_prices():
    close = make_prices(days=800)
    res = calibrate_ticker(close, horizon=21, step_days=50, min_history_days=300, n_sims=200)

    row = res.iloc[0]
    start = close.index.get_loc(row["Date"])
    expected = close.iloc[start + 21] / close.iloc[start] - 1
    assert np.isclose(row["Actual"], expected)


def test_forecast_uses_no_future_data():
    close = make_prices(days=900, seed=3)
    changed = close.copy()
    changed.iloc[600:] *= 3.0  # wild move after the first forecast window

    kwargs = dict(horizon=21, step_days=1000, min_history_days=400, n_sims=500, seed=7)
    a = calibrate_ticker(close, **kwargs).iloc[0]
    b = calibrate_ticker(changed, **kwargs).iloc[0]

    quantile_cols = [c for c in a.index if c.startswith("Q")]
    assert a["Date"] == b["Date"]
    assert np.allclose(a[quantile_cols].astype(float), b[quantile_cols].astype(float))


def test_constant_volatility_data_is_roughly_calibrated():
    frames = []
    for seed in range(15):
        close = make_prices(days=2500, seed=100 + seed)
        # One forecast per horizon, so outcomes within a series don't overlap.
        frames.append(
            calibrate_ticker(close, horizon=63, step_days=63, n_sims=1000, seed=seed)
        )
    results = pd.concat(frames, ignore_index=True)
    s = summarize(results)

    assert s["forecasts"] > 300
    assert 72 <= s["in_80_pct"] <= 88
    assert 43 <= s["in_50_pct"] <= 57
    assert all(4 <= d <= 16 for d in pit_deciles(results))


def test_time_zone_aware_prices_give_plain_dates():
    close = make_prices(days=800)
    close.index = close.index.tz_localize("America/New_York")
    res = calibrate_ticker(close, horizon=21, step_days=100, min_history_days=300, n_sims=100)
    assert res["Date"].dt.tz is None


# --- Volatility-scaled method ---

from calibration import ewma_volatility  # noqa: E402


def test_ewma_volatility_is_causal():
    r = np.random.default_rng(5).normal(0, 0.01, size=300)
    before_a, _ = ewma_volatility(r)
    r2 = r.copy()
    r2[200] = 0.5  # huge shock on day 200
    before_b, after_b = ewma_volatility(r2)
    # The vol "known before day 200" cannot see day 200's shock.
    assert np.allclose(before_a[: 201], before_b[: 201])
    assert after_b[200] > 5 * before_b[200]


def test_vol_ranges_widen_after_a_volatility_jump():
    rng = np.random.default_rng(6)
    calm = rng.normal(0.0003, 0.005, size=1000)
    wild = rng.normal(0.0003, 0.03, size=300)
    dates = pd.bdate_range("2015-01-01", periods=1300)
    close = pd.Series(100 * np.exp(np.cumsum(np.concatenate([calm, wild]))), index=dates)

    kwargs = dict(horizon=21, step_days=1000, min_history_days=1060, n_sims=2000, seed=1)
    plain = calibrate_ticker(close, method="bootstrap", **kwargs).iloc[0]
    vol = calibrate_ticker(close, method="vol", **kwargs).iloc[0]
    # 60 days into the wild regime, the plain bootstrap is still mostly calm history.
    assert vol["Q90"] - vol["Q10"] > 2 * (plain["Q90"] - plain["Q10"])


def test_vol_method_uses_no_future_data():
    close = make_prices(days=900, seed=8)
    changed = close.copy()
    changed.iloc[600:] *= np.exp(np.random.default_rng(9).normal(0, 0.1, size=300).cumsum())

    kwargs = dict(horizon=21, step_days=1000, min_history_days=400, n_sims=500, seed=7, method="vol")
    a = calibrate_ticker(close, **kwargs).iloc[0]
    b = calibrate_ticker(changed, **kwargs).iloc[0]
    quantile_cols = [c for c in a.index if c.startswith("Q")]
    assert np.allclose(a[quantile_cols].astype(float), b[quantile_cols].astype(float))


def test_vol_method_is_roughly_calibrated_on_constant_volatility():
    frames = []
    for seed in range(15):
        close = make_prices(days=2500, seed=200 + seed)
        frames.append(calibrate_ticker(close, horizon=63, step_days=63, n_sims=1000,
                                       seed=seed, method="vol"))
    s = summarize(pd.concat(frames, ignore_index=True))
    assert 72 <= s["in_80_pct"] <= 88


def test_unknown_method_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        calibrate_ticker(make_prices(days=800), method="magic")
