import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import ideas
import main
import signals

client = TestClient(main.app)
DATES = pd.bdate_range("2024-01-01", periods=320)


def _wide(**series):
    return pd.DataFrame(series, index=DATES)


def _trend(daily, n=320, start=100.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r = np.full(n, daily) + rng.normal(0, noise, n)
    return start * np.exp(np.cumsum(r))


def test_momentum_picks_the_strongest_uptrend_only():
    close = _wide(UP=_trend(0.002), FLAT=_trend(0.0, noise=0.01, seed=1),
                  DOWN=_trend(-0.001), A=_trend(0.0005), B=_trend(0.0003))
    picked, ret = signals.momentum_signal(close)
    last = picked.iloc[-1]
    assert bool(last["UP"]) and not last[["FLAT", "DOWN", "A", "B"]].any()
    # uses only past data: the first year has no momentum value
    assert ret.iloc[: signals.MOM_LOOKBACK].isna().all().all()


def test_momentum_needs_price_above_200_day_average():
    p = _trend(0.003)
    p[-30:] = p[-31] * np.exp(np.cumsum(np.full(30, -0.02)))  # crash in the last month
    close = _wide(X=p, A=_trend(0.0), B=_trend(0.0001), C=_trend(-0.0001), D=_trend(0.00005))
    picked, ret = signals.momentum_signal(close)
    assert ret.iloc[-1]["X"] > 0.3  # still strong 12-1 return, the last month is skipped
    assert not bool(picked.iloc[-1]["X"])  # but now below the 200 day average


def test_pullback_fires_on_a_dip_in_an_uptrend():
    up = _trend(0.002, noise=0.005, seed=2)
    dip = up.copy()
    dip[-21:] = dip[-22] * np.exp(np.cumsum(np.full(21, -0.003)))  # ~6% down in a month
    crash = _trend(0.002, noise=0.005, seed=3)
    crash[-80:] = crash[-81] * np.exp(np.cumsum(np.full(80, -0.006)))  # broke the long trend
    picked, ret_1m, z = signals.pullback_signal(_wide(UP=up, DIP=dip, CRASH=crash))
    last = picked.iloc[-1]
    assert bool(last["DIP"]) and not bool(last["UP"]) and not bool(last["CRASH"])
    assert z.iloc[-1]["DIP"] <= signals.PULL_Z


def test_analyst_and_value_rules():
    good = {"targetMeanPrice": 120, "numberOfAnalystOpinions": 20, "recommendationMean": 1.8, "recommendationKey": "buy"}
    assert ideas.analyst_reason(good, 100)["upside_pct"] == 20.0
    assert ideas.analyst_reason({**good, "numberOfAnalystOpinions": 3}, 100) is None
    assert ideas.analyst_reason({**good, "recommendationMean": 3.1}, 100) is None
    assert ideas.analyst_reason({**good, "targetMeanPrice": 110}, 100) is None
    assert ideas.analyst_reason({}, 100) is None

    infos = {f"T{i}": {"sector": "Tech", "forwardPE": pe} for i, pe in enumerate([20, 30, 40])}
    med = ideas.sector_pe_medians(infos)
    assert med["Tech"] == (30.0, 3)
    cheap = {"sector": "Tech", "forwardPE": 22, "earningsGrowth": 0.25, "revenueGrowth": 0.08}
    assert "P/E 22.0 vs 30.0" in ideas.value_reason(cheap, med)["text"]
    assert ideas.value_reason({**cheap, "forwardPE": 35}, med) is None
    assert ideas.value_reason({**cheap, "earningsGrowth": 0.05}, med) is None
    assert ideas.value_reason({**cheap, "revenueGrowth": -0.01}, med) is None
    assert ideas.value_reason({**cheap, "sector": "Energy"}, med) is None


def test_build_ideas_ranks_by_number_of_reasons():
    close = _wide(UP=_trend(0.002), B=_trend(0.0), C=_trend(-0.0005), D=_trend(0.0001), E=_trend(0.00005))
    up_price = close["UP"].iloc[-1]
    infos = {
        "UP": {"shortName": "Up Co", "targetMeanPrice": up_price * 1.3, "numberOfAnalystOpinions": 12, "recommendationMean": 2.0},
        "B": {"shortName": "B Co", "targetMeanPrice": close["B"].iloc[-1] * 1.5, "numberOfAnalystOpinions": 12, "recommendationMean": 2.0},
    }
    record = {"tested_on": "2026-11-04", "signals": {"momentum": {"testable": True, "63d": {"avg_vs_average_pct": 1.2}}}}
    out = ideas.build_ideas(close, infos, record)
    assert [i["ticker"] for i in out["ideas"]] == ["UP", "B"]
    assert {r["signal"] for r in out["ideas"][0]["reasons"]} == {"momentum", "analyst"}
    assert out["signals"]["momentum"]["63d"]["avg_vs_average_pct"] == 1.2
    assert out["signals"]["analyst"]["label"] == "Analyst targets"
    assert out["screened"] == 5
    assert out["signals"]["pullback"]["used"] is False and "Dropped" in out["signals"]["pullback"]["note"]
    assert out["signals"]["momentum"]["used"] is True and out["signals_used"] == 3


def test_pullback_no_longer_picks_ideas():
    up = _trend(0.002, noise=0.005, seed=2)
    up[-21:] = up[-22] * np.exp(np.cumsum(np.full(21, -0.003)))  # a dip that the pullback rule catches
    close = _wide(DIP=up)
    assert bool(signals.pullback_signal(close)[0].iloc[-1]["DIP"])
    out = ideas.build_ideas(close, {}, {})
    assert all(r["signal"] != "pullback" for i in out["ideas"] for r in i["reasons"])


def test_ideas_endpoint(monkeypatch):
    monkeypatch.setattr(ideas, "ideas", lambda: {"ideas": [], "screened": 0})
    assert client.get("/ideas").json()["screened"] == 0

    def boom():
        raise RuntimeError("rate limited")
    monkeypatch.setattr(ideas, "ideas", boom)
    r = client.get("/ideas")
    assert r.status_code == 502 and "Ideas unavailable" in r.json()["detail"]
