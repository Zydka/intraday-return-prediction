"""Lightweight robustness tools for prediction-based long-short portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PERIODS_PER_YEAR = 252 * 38


@dataclass(frozen=True)
class BacktestConfig:
    """Small container used when recording robustness experiments."""

    cost: float = 0.001
    quantile: float = 0.10
    rebalance_every: int = 1
    signal_threshold: float | str | None = None
    smoothing_rho: float = 1.0
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute one-way turnover as the sum of absolute weight changes."""
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame.")
    if weights.empty:
        return pd.Series(dtype=float, name="turnover")

    weights = weights.sort_index().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    return turnover.rename("turnover")


def _validate_prediction_frame(
    pred_df: pd.DataFrame,
    time_col: str,
    symbol_col: str,
    actual_col: str | None,
    pred_col: str,
) -> None:
    required = {time_col, symbol_col, pred_col}
    if actual_col is not None:
        required.add(actual_col)
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction dataframe is missing columns: {sorted(missing)}")


def _normalise_prediction_frame(
    pred_df: pd.DataFrame,
    time_col: str,
    symbol_col: str,
    pred_col: str,
) -> pd.DataFrame:
    out = pred_df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.dropna(subset=[time_col, symbol_col, pred_col])
    return out.sort_values([time_col, symbol_col])


def _threshold_value(
    predictions: pd.Series,
    signal_threshold: float | str | None,
) -> float | None:
    if signal_threshold is None:
        return None
    if isinstance(signal_threshold, str):
        if not signal_threshold.startswith("abs_q"):
            raise ValueError("String signal_threshold must look like 'abs_q0.25'.")
        q = float(signal_threshold.replace("abs_q", ""))
        return float(predictions.abs().quantile(q))
    return float(signal_threshold)


def long_short_weights(
    predictions: pd.DataFrame,
    timestamps: str = "DATETIME",
    symbols: str = "SYMBOL",
    quantile: float = 0.10,
    signal_threshold: float | str | None = None,
    pred_col: str = "prediction",
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Build equal-weight long-short weights from cross-sectional predictions.

    At each timestamp the top quantile receives positive weights and the
    bottom quantile receives negative weights. By default the total gross
    exposure is 1.0, split equally between the long and short sides.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5].")
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive.")

    _validate_prediction_frame(predictions, timestamps, symbols, None, pred_col)
    df = _normalise_prediction_frame(predictions, timestamps, symbols, pred_col)
    side_exposure = gross_exposure / 2.0

    frames: list[pd.Series] = []
    for timestamp, group in df.groupby(timestamps, sort=True):
        g = group[[symbols, pred_col]].dropna().copy()
        threshold = _threshold_value(g[pred_col], signal_threshold)
        if threshold is not None:
            g = g[g[pred_col].abs() >= threshold]

        row = pd.Series(0.0, index=group[symbols].astype(str).unique(), name=timestamp)
        if g.empty:
            frames.append(row)
            continue

        lower_cut = g[pred_col].quantile(quantile)
        upper_cut = g[pred_col].quantile(1.0 - quantile)

        longs = g[g[pred_col] >= upper_cut][symbols].astype(str).unique()
        shorts = g[g[pred_col] <= lower_cut][symbols].astype(str).unique()

        if len(longs) > 0:
            row.loc[longs] = side_exposure / len(longs)
        if len(shorts) > 0:
            row.loc[shorts] = -side_exposure / len(shorts)

        frames.append(row)

    if not frames:
        return pd.DataFrame()

    weights = pd.DataFrame(frames).fillna(0.0).sort_index()
    weights.index = pd.to_datetime(weights.index)
    return weights


def _apply_rebalance_frequency(weights: pd.DataFrame, rebalance_every: int) -> pd.DataFrame:
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be >= 1.")
    weights = weights.sort_index().fillna(0.0)
    if rebalance_every == 1 or weights.empty:
        return weights

    rebalanced = weights.copy()
    keep = np.arange(len(weights)) % rebalance_every == 0
    rebalanced.loc[~keep, :] = np.nan
    return rebalanced.ffill().fillna(0.0)


def apply_position_smoothing(weights: pd.DataFrame, rho: float = 1.0) -> pd.DataFrame:
    """
    Partially adjust current weights toward target weights.

    rho=1.0 gives the original full rebalance. Lower values reduce turnover by
    moving only part of the way from previous weights to target weights.
    """
    if not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1].")
    weights = weights.sort_index().fillna(0.0)
    if weights.empty or rho == 1.0:
        return weights

    smoothed_rows = []
    previous = pd.Series(0.0, index=weights.columns)
    for timestamp, target in weights.iterrows():
        current = (1.0 - rho) * previous + rho * target
        current.name = timestamp
        smoothed_rows.append(current)
        previous = current
    return pd.DataFrame(smoothed_rows).fillna(0.0)


def _actual_returns_wide(
    pred_df: pd.DataFrame,
    actual_col: str,
    time_col: str,
    symbol_col: str,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    actual = pred_df.pivot_table(
        index=time_col,
        columns=symbol_col,
        values=actual_col,
        aggfunc="mean",
    )
    actual.index = pd.to_datetime(actual.index)
    actual = actual.sort_index()
    if columns is not None:
        actual = actual.reindex(columns=list(columns))
    return actual.fillna(0.0)


def _summarise_returns(
    gross_return: pd.Series,
    net_return: pd.Series,
    turnover: pd.Series,
    config: BacktestConfig,
) -> dict[str, float]:
    net_vol = float(net_return.std())
    sharpe = np.nan
    if net_vol > 0:
        sharpe = float(net_return.mean() / net_vol * np.sqrt(config.periods_per_year))

    return {
        "mean_gross_return": float(gross_return.mean()),
        "mean_net_return": float(net_return.mean()),
        "cumulative_gross_return": float((1.0 + gross_return).prod() - 1.0),
        "cumulative_net_return": float((1.0 + net_return).prod() - 1.0),
        "average_turnover": float(turnover.mean()),
        "volatility": net_vol,
        "sharpe_ratio": sharpe,
        "number_of_periods": int(len(net_return)),
        "transaction_cost": float(config.cost),
        "quantile": float(config.quantile),
        "rebalance_every": int(config.rebalance_every),
        "signal_threshold": (
            np.nan
            if config.signal_threshold is None or isinstance(config.signal_threshold, str)
            else float(config.signal_threshold)
        ),
        "periods_per_year": int(config.periods_per_year),
    }


def backtest_long_short(
    pred_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    cost: float = 0.001,
    quantile: float = 0.10,
    rebalance_every: int = 1,
    signal_threshold: float | str | None = None,
    gross_exposure: float = 1.0,
    smoothing_rho: float = 1.0,
    time_col: str = "DATETIME",
    symbol_col: str = "SYMBOL",
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    """
    Backtest a prediction-ranked equal-weight long-short portfolio.

    The input target column is assumed to be the realized next-period return
    aligned with the prediction timestamp, as in the project baseline outputs.
    """
    _validate_prediction_frame(pred_df, time_col, symbol_col, actual_col, pred_col)
    df = _normalise_prediction_frame(pred_df, time_col, symbol_col, pred_col)
    df = df.dropna(subset=[actual_col])

    threshold = signal_threshold
    config = BacktestConfig(
        cost=cost,
        quantile=quantile,
        rebalance_every=rebalance_every,
        signal_threshold=threshold,
        smoothing_rho=smoothing_rho,
        periods_per_year=periods_per_year,
    )

    weights = long_short_weights(
        df,
        timestamps=time_col,
        symbols=symbol_col,
        quantile=quantile,
        signal_threshold=threshold,
        pred_col=pred_col,
        gross_exposure=gross_exposure,
    )
    weights = _apply_rebalance_frequency(weights, rebalance_every)
    weights = apply_position_smoothing(weights, rho=smoothing_rho)

    actual = _actual_returns_wide(df, actual_col, time_col, symbol_col, weights.columns)
    actual = actual.reindex(weights.index).fillna(0.0)

    gross_return = (weights * actual).sum(axis=1).rename("gross_return")
    turnover = compute_turnover(weights)
    transaction_cost = (cost * turnover).rename("transaction_cost")
    net_return = (gross_return - transaction_cost).rename("net_return")

    returns = pd.concat([gross_return, turnover, transaction_cost, net_return], axis=1)
    metrics = _summarise_returns(gross_return, net_return, turnover, config)
    return {"metrics": metrics, "returns": returns, "weights": weights}


def run_cost_sweep(
    pred_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    costs: Iterable[float],
    model_name: str,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for cost in costs:
        result = backtest_long_short(pred_df, actual_col, pred_col, cost=cost, **kwargs)
        rows.append({"model": model_name, **result["metrics"]})
    return pd.DataFrame(rows)


def run_quantile_sweep(
    pred_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    quantiles: Iterable[float],
    model_name: str,
    cost: float = 0.001,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for quantile in quantiles:
        result = backtest_long_short(
            pred_df,
            actual_col,
            pred_col,
            cost=cost,
            quantile=quantile,
            **kwargs,
        )
        rows.append({"model": model_name, **result["metrics"]})
    return pd.DataFrame(rows)


def run_rebalance_sweep(
    pred_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    rebalance_values: Iterable[int],
    model_name: str,
    cost: float = 0.001,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for rebalance_every in rebalance_values:
        result = backtest_long_short(
            pred_df,
            actual_col,
            pred_col,
            cost=cost,
            rebalance_every=rebalance_every,
            **kwargs,
        )
        rows.append({"model": model_name, **result["metrics"]})
    return pd.DataFrame(rows)


def run_signal_filter_sweep(
    pred_df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    thresholds: Iterable[float | str | None],
    model_name: str,
    cost: float = 0.001,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        result = backtest_long_short(
            pred_df,
            actual_col,
            pred_col,
            cost=cost,
            signal_threshold=threshold,
            **kwargs,
        )
        rows.append({"model": model_name, "filter_rule": threshold, **result["metrics"]})
    return pd.DataFrame(rows)
