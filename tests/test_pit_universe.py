import pandas as pd

from research import pit_universe as pu

CSV = """date,tickers
2000-01-03,"AAA,BBB,CC.B"
2000-06-01,"AAA,CC.B,DDD"
2001-01-02,"AAA,BBB,CC.B,DDD"
"""


def _setup(tmp_path, monkeypatch):
    path = tmp_path / "m.csv"
    path.write_text(CSV)
    monkeypatch.setattr(pu, "MEMBERSHIP_CACHE", str(path))


def test_membership_matrix_follows_changes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    idx = pd.bdate_range("1999-12-01", "2001-03-01")
    m = pu.membership_matrix(idx)
    assert set(m.columns) == {"AAA", "BBB", "CC-B", "DDD"}  # dots become Yahoo dashes
    assert not m.loc["1999-12-15"].any()  # nothing known before the first list
    assert m.loc["2000-03-01", "BBB"] and not m.loc["2000-03-01", "DDD"]
    assert not m.loc["2000-09-01", "BBB"] and m.loc["2000-09-01", "DDD"]  # BBB left, DDD joined
    assert m.loc["2001-02-01", "BBB"]  # BBB came back


def test_usable_mask_guards_reused_tickers(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    idx = pd.bdate_range("1999-01-01", "2001-03-01")
    m = pu.membership_matrix(idx)
    full = pd.Series(1.0, index=idx)
    prices = {
        "AAA": full,                                   # covers everything
        "BBB": full["2000-10-01":],                    # starts after its first period: first period unusable, second fine
        "CC-B": full,
        # DDD missing on Yahoo entirely
    }
    u = pu.usable_mask(m, prices)
    assert u["AAA"].equals(m["AAA"])
    assert not u.loc["2000-03-01", "BBB"]   # this period predates Yahoo's history: may be another company
    assert u.loc["2001-02-01", "BBB"]       # the later period is covered from its start
    assert not u["DDD"].any()
    cov = pu.coverage_by_year(m, u)
    assert cov.loc[2001, "covered_pct"] < 100 and cov.loc[2000, "covered_pct"] < 100
