import numpy as np
import pandas as pd

from research import data_quality as dq

IDX = pd.bdate_range("2005-01-03", periods=1500)


def _stock(seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, vol, len(IDX)))), index=IDX)


def test_clean_and_volatile_stocks_pass():
    assert dq.problems(_stock()) == []
    wild = _stock(seed=1, vol=0.06)                      # very volatile but real-looking
    assert dq.problems(wild) == []
    crash = _stock(seed=2)
    crash.iloc[800:] *= 0.35                              # a real 65% one-day crash is allowed
    assert dq.problems(crash) == []


def test_broken_split_is_caught():
    s = _stock()
    s.iloc[900:] *= 12                                    # x12 overnight
    p = dq.problems(s)
    assert len(p) == 1 and str(IDX[900].date()) in p[0]
    move = float(p[0].split("move of ")[1].split("%")[0])
    assert 1000 < move < 1200                             # about +1100%, plus that day's normal move
    down = _stock()
    down.iloc[700:] *= 0.05                               # -95% overnight
    p = dq.problems(down)
    assert len(p) == 1 and float(p[0].split("move of ")[1].split("%")[0]) < -90


def test_stale_prices_are_caught_but_short_pauses_are_not():
    s = _stock()
    s.iloc[300:330] = s.iloc[299]                         # 31 identical closes
    assert any("unchanged for 31" in p for p in dq.problems(s))
    short = _stock()
    short.iloc[300:310] = short.iloc[299]                 # 11 identical closes: fine
    assert dq.problems(short) == []


def test_bad_values_are_caught():
    s = _stock()
    s.iloc[50] = 0.0
    assert dq.problems(s) == ["zero, negative or non-numeric price"]
    assert dq.problems(pd.Series([], dtype=float)) == ["no prices"]


def test_filter_wide_drops_only_broken_and_keeps_masks_aligned():
    broken = _stock(seed=3)
    broken.iloc[1000:] *= 20
    close = pd.DataFrame({"AAA": _stock(4), "BAD": broken, "CCC": _stock(5)})
    close.iloc[:200, 2] = np.nan                          # a later listing: leading gap is fine
    mask = pd.DataFrame(True, index=IDX, columns=close.columns)
    c, m, report = dq.filter_wide(close, mask)
    assert list(c.columns) == ["AAA", "CCC"] and list(m.columns) == ["AAA", "CCC"]
    assert list(report) == ["BAD"]
