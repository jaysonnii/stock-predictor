"""
Ranking test for the pooled model.

Question: each day, does the model's top-ranked group of stocks beat the
average stock in the universe, even if its exact numbers are off?

Per test date:
- Rank every ticker by predicted excess return.
- Take the top TOP_FRACTION of tickers.
- Spread = their average realized excess return minus the universe average.

A positive average spread means the ranking has some skill. A model with no
skill gives a spread near zero.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data import get_feature_columns
from pooled import _make_model, build_panel

MIN_TICKERS_PER_DATE = 10
TOP_FRACTION = 0.2


def walk_forward_ranking(horizon: int = 252, n_folds: int = 4,
                         top_fraction: float = TOP_FRACTION, panel=None) -> dict:
    if panel is None:
        panel = build_panel(horizon=horizon)

    features = get_feature_columns()
    dates = np.sort(panel["Date"].unique())
    n = len(dates)
    initial = int(n * 0.5)
    fold_size = (n - initial) // n_folds

    rng = np.random.default_rng(0)
    folds = []
    all_spreads = []
    all_control = []
    for k in range(n_folds):
        s = initial + k * fold_size
        e = n if k == n_folds - 1 else s + fold_size
        if s - horizon < 100:
            continue

        tr = panel[panel["Date"] < dates[s - horizon]]
        te = panel[(panel["Date"] >= dates[s]) & (panel["Date"] <= dates[e - 1])].copy()
        if tr.empty or te.empty:
            continue

        scaler = StandardScaler().fit(tr[features].values)
        model = _make_model().fit(scaler.transform(tr[features].values), tr["Target"].values)
        te["pred"] = model.predict(scaler.transform(te[features].values))

        spreads = []
        control = []
        for _, g in te.groupby("Date"):
            if len(g) < MIN_TICKERS_PER_DATE:
                continue
            k_top = max(1, int(round(len(g) * top_fraction)))
            top = g.nlargest(k_top, "pred")
            spreads.append(top["Target"].mean() - g["Target"].mean())
            # Control: a random group of the same size, chosen without the model.
            pick = rng.choice(len(g), size=k_top, replace=False)
            control.append(g["Target"].values[pick].mean() - g["Target"].mean())

        if not spreads:
            continue
        all_spreads.extend(spreads)
        all_control.extend(control)
        folds.append({
            "test_from": str(pd.Timestamp(dates[s]).date()),
            "test_to": str(pd.Timestamp(dates[e - 1]).date()),
            "dates_tested": len(spreads),
            "avg_spread_pct_points": round(float(np.mean(spreads)) * 100, 2),
            "share_of_dates_positive_pct": round(float(np.mean(np.array(spreads) > 0)) * 100, 1),
            "random_control_avg_spread_pct_points": round(float(np.mean(control)) * 100, 2),
        })

    if not folds:
        raise ValueError("Not enough data for ranking evaluation.")

    return {
        "horizon_days": horizon,
        "top_fraction": top_fraction,
        "tickers": int(panel["Ticker"].nunique()),
        "folds": folds,
        "folds_with_positive_spread": int(sum(f["avg_spread_pct_points"] > 0 for f in folds)),
        "folds_evaluated": len(folds),
        "overall_avg_spread_pct_points": round(float(np.mean(all_spreads)) * 100, 2),
        "overall_share_positive_pct": round(float(np.mean(np.array(all_spreads) > 0)) * 100, 1),
        "random_control_overall_avg_spread_pct_points": round(float(np.mean(all_control)) * 100, 2),
    }
