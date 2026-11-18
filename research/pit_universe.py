"""
Point in time S&P 500 universe, the free partial fix for survivorship bias.

The earlier tests used 70 stocks that are large and alive today, which quietly
picks winners in advance. This module instead uses the stocks that were in the
S&P 500 on each past date.

Sources (both free):
- Membership: daily S&P 500 components since 1996 from github.com/fja05680/sp500
  (built from Andreas Clenow's list plus index changes). Cached in data_cache/.
- Prices: Yahoo, cached in data_long_cache/ like long_vol_test.py.

What this fixes and what it does not:
- Fixed: the group on each date is the companies that were actually large then,
  not today's winners looked at backwards.
- Not fixed: Yahoo only has tickers that still trade, so most companies that
  went bankrupt or were bought out are missing. Every test reports how many
  members on each date have usable prices, so the size of the gap is visible.
- Guard against reused tickers: Yahoo gives a ticker's current company. A
  membership period is only used when Yahoo's prices for that ticker start
  before the period does; otherwise it may be a different company and is
  counted as missing.
"""
import io
import json
import os
import time

import numpy as np
import pandas as pd

MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv")
MEMBERSHIP_CACHE = os.path.join("data_cache", "sp500_membership.csv")
PRICE_DIR = "data_long_cache"
MISSING_FILE = os.path.join(PRICE_DIR, "_missing_on_yahoo.json")
BATCH = 40
BATCH_DELAY_SECONDS = 1.0
START_SLACK_DAYS = 10  # Yahoo history may start up to this many days after a period starts


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def load_membership(refresh: bool = False) -> pd.DataFrame:
    """Rows are change dates, one column 'tickers' with a comma separated list."""
    if refresh or not os.path.exists(MEMBERSHIP_CACHE):
        import requests
        r = requests.get(MEMBERSHIP_URL, timeout=60)
        if not (r.ok and r.text.startswith("date")):
            raise RuntimeError(f"Could not download the S&P 500 membership file from GitHub (HTTP {r.status_code}).")
        text = r.text
        os.makedirs(os.path.dirname(MEMBERSHIP_CACHE), exist_ok=True)
        with open(MEMBERSHIP_CACHE, "w", encoding="utf-8") as f:
            f.write(text)
    df = pd.read_csv(MEMBERSHIP_CACHE, parse_dates=["date"]).set_index("date").sort_index()
    return df


def membership_matrix(index: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Boolean table (dates x tickers): True where the ticker was in the S&P 500 that day."""
    m = load_membership(refresh)
    tickers = sorted({t for row in m["tickers"] for t in row.split(",") if t})
    rows = np.zeros((len(m), len(tickers)), dtype=bool)
    pos = {t: j for j, t in enumerate(tickers)}
    for i, row in enumerate(m["tickers"]):
        for t in row.split(","):
            if t:
                rows[i, pos[t]] = True
    changes = pd.DataFrame(rows, index=m.index, columns=[yahoo_symbol(t) for t in tickers])
    changes = changes.T.groupby(level=0).any().T  # merge tickers that map to the same Yahoo symbol
    out = changes.astype(float).reindex(index.union(changes.index)).ffill().reindex(index)
    out.loc[out.index < changes.index[0]] = 0.0  # nothing known before the first list
    return out.fillna(0.0).astype(bool)


def _read_cached(t):
    path = os.path.join(PRICE_DIR, f"{t}.csv")
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None).normalize()
    return s[~s.index.duplicated()].dropna()


def load_prices(tickers, verbose=True) -> dict:
    """Full daily closes from Yahoo for every ticker it has. Cached; known misses are skipped."""
    import yfinance as yf
    os.makedirs(PRICE_DIR, exist_ok=True)
    try:
        with open(MISSING_FILE) as f:
            missing = set(json.load(f))
    except (OSError, ValueError):
        missing = set()

    out, todo = {}, []
    for t in tickers:
        s = _read_cached(t)
        if s is not None:
            out[t] = s
        elif t not in missing:
            todo.append(t)

    if todo and verbose:
        print(f"Downloading {len(todo)} tickers from Yahoo in batches of {BATCH} (cached for next time)...")
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        try:
            df = yf.download(chunk, period="max", interval="1d", auto_adjust=True,
                             progress=False, threads=True, group_by="column")
            close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": chunk[0]})
        except Exception as e:
            if verbose:
                print(f"  batch {b // BATCH + 1} failed ({e}); will retry next run")
            continue
        for t in chunk:
            s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
            if len(s) < 30:
                missing.add(t)
                continue
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated()]
            s.rename(t).to_frame().to_csv(os.path.join(PRICE_DIR, f"{t}.csv"))
            out[t] = s
        if verbose:
            print(f"  {min(b + BATCH, len(todo))} of {len(todo)} done")
        time.sleep(BATCH_DELAY_SECONDS)

    with open(MISSING_FILE, "w") as f:
        json.dump(sorted(missing), f)
    return out


def usable_mask(member: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """
    member, limited to membership periods that Yahoo's prices actually cover
    from the start (guards against a reused ticker belonging to another company).
    """
    usable = pd.DataFrame(False, index=member.index, columns=member.columns)
    slack = pd.Timedelta(days=START_SLACK_DAYS)
    for t in member.columns:
        s = prices.get(t)
        col = member[t].values
        if s is None or not col.any():
            continue
        first_price = s.index[0]
        # contiguous membership periods
        edges = np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0]))
        for start_i, end_i in zip(edges[::2], edges[1::2]):
            start = member.index[start_i]
            if first_price <= start + slack:
                usable.iloc[start_i:end_i, usable.columns.get_loc(t)] = True
    return usable


def coverage_by_year(member: pd.DataFrame, usable: pd.DataFrame) -> pd.DataFrame:
    """Average members per day and the share with usable prices, by year."""
    total = member.sum(axis=1)
    covered = usable.sum(axis=1)
    keep = total > 0
    df = pd.DataFrame({"members": total[keep], "covered": covered[keep]})
    g = df.groupby(df.index.year).mean()
    g["covered_pct"] = (g["covered"] / g["members"] * 100).round(1)
    return g.round(1)


LAST_QUALITY_REPORT = {}


def build(index: pd.DatetimeIndex, refresh: bool = False, verbose: bool = True, clean: bool = True):
    """
    Returns (close, member, usable): wide closes for all ever-members and the two masks.
    clean=True (the default since Nov 18, 2026) drops stocks with broken price data
    (research/data_quality.py); dropped stocks still count as members, so coverage
    figures include them as missing.
    """
    global LAST_QUALITY_REPORT
    member = membership_matrix(index, refresh)
    prices = load_prices(list(member.columns), verbose)
    close = pd.DataFrame({t: s for t, s in prices.items()}).reindex(index)
    usable = usable_mask(member, prices)
    cols = [c for c in member.columns if c in close.columns]
    close, usable = close[cols], usable[cols]
    LAST_QUALITY_REPORT = {}
    if clean:
        from research import data_quality
        total = close.shape[1]
        close, usable, LAST_QUALITY_REPORT = data_quality.filter_wide(close, usable)
        if verbose:
            data_quality.print_report(LAST_QUALITY_REPORT, total)
    elif verbose:
        print("Data quality filter OFF (--no-clean): broken price histories are included.")
    return close, member, usable
