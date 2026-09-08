"""Shared configuration and helpers for the experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    breakeven_cost, cross_sectional_ic, hit_ratio, mae, r2_score, rmse,
    sharpe_per_period, zero_target_share,
)

PANEL = Path(__file__).resolve().parent / "panel.pkl"
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "report" / "figures"

BARS_PER_DAY = 38
PERIODS_PER_YEAR = 252 * BARS_PER_DAY
COST = 0.0005                      # 5 bps per unit of turnover
QUANTILE = 0.10
TRAIN_FRACTION, VALIDATION_FRACTION = 0.70, 0.15

FEATURES = [
    "RETURN_CURRENT", "SUM_DELTA_CURRENT", "SUM_DELTA_ZSCORE", "SUM_DELTA_CS_RANK",
    "RETURN_CS_RANK", "MINUTES_FROM_OPEN", "INTRADAY_TIME_FRACTION",
    "TIME_SIN", "TIME_COS",
    "RETURN_LAG_1", "RETURN_LAG_2", "RETURN_LAG_3",
    "SUM_DELTA_LAG_1", "SUM_DELTA_LAG_2", "SUM_DELTA_LAG_3",
    "ROLLING_RETURN_MEAN_3", "ROLLING_RETURN_MEAN_5",
    "ROLLING_RETURN_STD_3", "ROLLING_RETURN_STD_5",
    "ROLLING_SUM_DELTA_MEAN_3", "ROLLING_SUM_DELTA_MEAN_5",
]


# --------------------------------------------------------------------- data
def splits(df):
    bars = np.sort(df["BAR"].unique())
    n = len(bars)
    train = set(bars[: int(n * TRAIN_FRACTION)])
    validation = set(bars[int(n * TRAIN_FRACTION): int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))])
    test = set(bars[int(n * (TRAIN_FRACTION + VALIDATION_FRACTION)):])
    return train, validation, test


def build_features(df, train_bars, horizon=1):
    """Add the derived features. All fitted quantities use TRAIN rows only."""
    d = df.copy()
    train_rows = d["BAR"].isin(train_bars)

    bounds = d.loc[train_rows, "TARGET_RETURN_RAW"].dropna().quantile([0.01, 0.99])
    lo, hi = float(bounds.iloc[0]), float(bounds.iloc[1])

    moments = d.loc[train_rows].groupby("SYMBOL")["SUM_DELTA_CURRENT"].agg(["mean", "std"])
    d = d.join(moments, on="SYMBOL")
    d["SUM_DELTA_ZSCORE"] = ((d["SUM_DELTA_CURRENT"] - d["mean"]) / (d["std"] + 1e-8)).astype("float32")
    d = d.drop(columns=["mean", "std"])

    d["SUM_DELTA_CS_RANK"] = d.groupby("DATETIME")["SUM_DELTA_CURRENT"].rank(pct=True).astype("float32")
    d["RETURN_CS_RANK"] = d.groupby("DATETIME")["RETURN_CURRENT"].rank(pct=True).astype("float32")

    for c in ["RETURN_CURRENT", "RETURN_LAG_1", "RETURN_LAG_2", "RETURN_LAG_3"]:
        d[c] = d[c].clip(lo, hi)

    d = d.sort_values(["SYMBOL", "BAR"])
    g = d.groupby("SYMBOL", sort=False)
    lp = np.log(d["MID_OPEN"])
    ok = (g["BAR"].shift(-horizon) - d["BAR"] == horizon) & (g["DAY"].shift(-horizon) == d["DAY"])
    d["FORWARD_RETURN"] = np.where(ok, np.log(g["MID_OPEN"].shift(-horizon)) - lp, np.nan).astype("float32")
    d["TARGET"] = d["FORWARD_RETURN"].clip(lo * horizon, hi * horizon)

    return d.dropna(subset=FEATURES + ["FORWARD_RETURN", "TARGET"]), (lo, hi)


# ------------------------------------------------------------------ metrics
def prediction_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(
        rmse=float(rmse(y, p)),
        mae=float(mae(y, p)),
        r2=float(r2_score(y, p)),
        ic_pooled=float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 else np.nan,
        hit=float(hit_ratio(y, p)),
    )


# ---------------------------------------------------------------- portfolio
def target_weights(group, pred_col, q=QUANTILE, no_trade_zone=0.0, weights=None):
    """Dollar-neutral decile book at one timestamp.

    weights: optional per-symbol multiplier (e.g. inverse volatility), rescaled
    so each side still sums to 0.5 in absolute value.
    """
    g = group
    if no_trade_zone > 0:
        g = g[g[pred_col].abs() >= g[pred_col].abs().quantile(no_trade_zone)]
    if len(g) < 20 or g[pred_col].nunique() < 2:
        return None

    hi, lo = g[pred_col].quantile(1 - q), g[pred_col].quantile(q)
    w = pd.Series(0.0, index=g["SYMBOL"].values)
    for mask, sign in ((g[pred_col] >= hi, 1.0), (g[pred_col] <= lo, -1.0)):
        names = g.loc[mask, "SYMBOL"].values
        if len(names) == 0:
            continue
        raw = pd.Series(1.0, index=names) if weights is None else \
            pd.Series(weights.reindex(names).fillna(weights.median()).values, index=names)
        raw = raw.clip(lower=0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=names)
        w.loc[names] = sign * 0.5 * raw.values / raw.sum()
    return w


def run_backtest(groups, pred_col, ret_col="FORWARD_RETURN", every=1, q=QUANTILE,
                 no_trade_zone=0.0, rho=1.0, weight_col=None):
    """Rebalance every `every` bars, holding in between. rho = partial adjustment."""
    rows, prev = [], None
    for bar in sorted(groups)[::every]:
        g = groups[bar]
        wts = None if weight_col is None else pd.Series(g[weight_col].values, index=g["SYMBOL"].values)
        target = target_weights(g, pred_col, q=q, no_trade_zone=no_trade_zone, weights=wts)
        if target is None:
            continue
        if prev is None:
            w, turnover = target, float(target.abs().sum())
        else:
            a = pd.concat([prev, target], axis=1).fillna(0.0)
            w = (1 - rho) * a.iloc[:, 0] + rho * a.iloc[:, 1]
            turnover = float((w - a.iloc[:, 0]).abs().sum())
        r = pd.Series(g[ret_col].values, index=g["SYMBOL"].values).reindex(w.index).fillna(0.0)
        rows.append((bar, float((w * r).sum()), turnover))
        prev = w
    return pd.DataFrame(rows, columns=["BAR", "gross", "turnover"])


def portfolio_metrics(p, cost=COST, bars_held=1):
    net = p["gross"] - cost * p["turnover"]
    wealth = (1 + net).cumprod()
    sharpe = sharpe_per_period(net)
    return dict(
        periods=int(len(p)),
        gross_bps=float(p["gross"].mean() * 1e4),
        net_bps=float(net.mean() * 1e4),
        turnover=float(p["turnover"].mean()),
        sharpe_per_bar=sharpe,
        sharpe_annual=float(sharpe * np.sqrt(PERIODS_PER_YEAR / bars_held)),
        t_stat_net=float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))),
        cumulative_net=float((1 + net).prod() - 1),
        max_drawdown=float((wealth / wealth.cummax() - 1).min()),
        losing_bars=float((net < 0).mean()),
        breakeven_bps=float(1e4 * breakeven_cost(p["gross"], p["turnover"])),
    )


def make_groups(df, pred_col, ret_col="FORWARD_RETURN"):
    d = df.dropna(subset=[pred_col, ret_col])
    return {b: g for b, g in d.groupby("BAR", sort=True)}


def write(table, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    table.to_csv(path)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
