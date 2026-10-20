import numpy as np
import pandas as pd
from scipy.stats import norm

import widening
from widening import fit_asymmetric, fit_symmetric, parse_dates, score, walk_forward, widen

QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def make_results(n=4000, true_sd_low=0.10, true_sd_high=0.10, forecast_sd=0.10, seed=0):
    """
    Synthetic forecasts with log-normal quantiles of sd `forecast_sd`, scored
    against outcomes whose real sd differs below and above the median.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    log_actual = np.where(z < 0, z * true_sd_low, z * true_sd_high)
    data = {
        "Date": pd.bdate_range("2015-01-01", periods=n, freq="B"),
        "Actual": np.expm1(log_actual),
    }
    for q in QS:
        data[f"Q{int(q * 100):02d}"] = np.full(n, np.expm1(norm.ppf(q) * forecast_sd))
    return pd.DataFrame(data)


def test_k_of_one_changes_nothing():
    q = np.array([-0.2, 0.0, 0.3])
    assert np.allclose(widen(q, np.zeros(3), 1.0), q)


def test_well_calibrated_forecasts_need_no_widening():
    res = make_results()
    assert abs(fit_symmetric(res) - 1.0) <= 0.05
    assert abs(score(res)["in_80"] - 80) < 2.5


def test_too_narrow_forecasts_get_widened_by_the_right_amount():
    res = make_results(true_sd_low=0.13, true_sd_high=0.13)
    assert abs(fit_symmetric(res) - 1.3) <= 0.06


def test_asymmetric_fit_widens_the_fatter_side_more():
    res = make_results(true_sd_low=0.14, true_sd_high=0.10)
    k_low, k_high = fit_asymmetric(res)
    assert k_low > 1.25
    assert abs(k_high - 1.0) <= 0.06


def test_walk_forward_only_trains_on_resolved_outcomes(monkeypatch):
    res = make_results(n=3000)
    seen = []
    original = widening.fit_symmetric

    def spy(train, level=80):
        seen.append(pd.to_datetime(train["Date"]).max())
        return original(train, level)

    monkeypatch.setattr(widening, "fit_symmetric", spy)
    by_year = walk_forward(res, horizon=63)

    assert not by_year.empty
    for year, latest_train_date in zip(by_year["year"], seen):
        resolved = latest_train_date + pd.offsets.BDay(63)
        assert resolved < pd.Timestamp(year=int(year), month=1, day=1)


def test_dates_with_daylight_saving_offsets_parse(tmp_path):
    # Fresh Yahoo downloads write offsets that switch between -05:00 and -04:00.
    res = make_results(n=3000)
    winter = res["Date"].dt.strftime("%Y-%m-%d 00:00:00-05:00")
    summer = res["Date"].dt.strftime("%Y-%m-%d 00:00:00-04:00")
    res["Date"] = np.where(res.index % 2 == 0, winter, summer)
    path = tmp_path / "calibration.csv"
    res.to_csv(path, index=False)

    loaded = pd.read_csv(path)
    dates = parse_dates(loaded["Date"])
    assert dates.dt.tz is None
    assert dates.iloc[0] == pd.Timestamp("2015-01-01")
    assert not walk_forward(loaded, horizon=63).empty


def test_too_wide_forecasts_get_narrowed():
    # k below 1: the vol-scaled ranges overshoot, so the fit must be able to shrink.
    res = make_results(true_sd_low=0.088, true_sd_high=0.088)
    assert abs(fit_symmetric(res) - 0.88) <= 0.05
