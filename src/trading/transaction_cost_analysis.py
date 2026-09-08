import numpy as np
import pandas as pd

from src.trading.portfolio_construction import build_long_short_backtest


INTRADAY_BARS_PER_DAY = {
    "10min": 38,
    "30min": 13,
    "60min": 7,
}


def _normalize_frequency(rebalance_frequency):
    """Return the canonical frequency label used by the analysis helpers."""
    if isinstance(rebalance_frequency, int):
        rebalance_frequency = f"{rebalance_frequency}min"

    if not isinstance(rebalance_frequency, str):
        raise TypeError("rebalance_frequency must be one of: 10min, 30min, 60min.")

    frequency = rebalance_frequency.strip().lower()
    aliases = {
        "10": "10min",
        "10m": "10min",
        "10min": "10min",
        "30": "30min",
        "30m": "30min",
        "30min": "30min",
        "60": "60min",
        "60m": "60min",
        "60min": "60min",
        "1h": "60min",
    }

    if frequency not in aliases:
        raise ValueError("rebalance_frequency must be one of: 10min, 30min, 60min.")

    return aliases[frequency]


def _periods_per_year(rebalance_frequency, trading_days_per_year=252):
    frequency = _normalize_frequency(rebalance_frequency)
    return trading_days_per_year * INTRADAY_BARS_PER_DAY[frequency]


def _make_timestamp_index(df, time_col):
    timestamps = pd.Series(pd.to_datetime(df[time_col].drop_duplicates()))
    return timestamps.sort_values().reset_index(drop=True)


def _frequency_step(rebalance_frequency):
    frequency = _normalize_frequency(rebalance_frequency)
    return {"10min": 1, "30min": 3, "60min": 6}[frequency]


def _summarize_backtest(returns, metrics):
    return {
        "average_turnover": returns["turnover"].mean(),
        "gross_return": returns["gross_return"].mean(),
        "net_return": returns["net_return"].mean(),
        "sharpe_ratio": metrics["annualized_sharpe"],
        "cumulative_return": metrics["cumulative_return"],
        "max_drawdown": metrics["max_drawdown"],
        "hit_ratio": metrics["hit_ratio"],
    }


def filter_rebalance_frequency(
    df,
    rebalance_frequency="10min",
    time_col="DATETIME",
):
    """
    Keep only timestamps where the portfolio is allowed to rebalance.

    The raw data is 10-minute intraday data. Moving to 30-minute or 60-minute
    rebalancing keeps every third or sixth timestamp respectively, which lowers
    the number of portfolio updates before positions are built.
    """
    frequency = _normalize_frequency(rebalance_frequency)
    step = _frequency_step(frequency)

    df_filtered = df.copy()
    timestamps = _make_timestamp_index(df_filtered, time_col)
    allowed_timestamps = set(timestamps.iloc[::step])

    df_filtered[time_col] = pd.to_datetime(df_filtered[time_col])
    return df_filtered[df_filtered[time_col].isin(allowed_timestamps)].copy()


def filter_signal_threshold(
    df,
    prediction_col="SIGNAL_SCORE",
    threshold=0.0,
    output_col=None,
):
    """
    Set weak model signals to zero before portfolio ranking.

    This keeps the model's strong cross-sectional views intact while removing
    small predictions that are unlikely to survive trading costs.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    output_col = output_col or prediction_col
    df_filtered = df.copy()
    df_filtered[output_col] = df_filtered[prediction_col].where(
        df_filtered[prediction_col].abs() >= threshold,
        0.0,
    )

    return df_filtered


def apply_signal_threshold(
    df,
    signal_col="SIGNAL_SCORE",
    threshold=0.001,
):
    """
    Neutralize weak signals before portfolio construction.

    The raw prediction is left untouched in any other column; only signal_col is
    set to zero when the signal is too small to justify trading.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    df_thresholded = df.copy()
    df_thresholded[signal_col] = df_thresholded[signal_col].where(
        df_thresholded[signal_col].abs() >= threshold,
        0.0,
    )

    return df_thresholded


def _maybe_apply_signal_threshold(
    df,
    signal_col,
    use_signal_threshold,
    signal_threshold,
):
    if not use_signal_threshold:
        return df

    return apply_signal_threshold(
        df,
        signal_col=signal_col,
        threshold=signal_threshold,
    )


def _run_single_backtest(
    df,
    prediction_col,
    target_col,
    time_col,
    symbol_col,
    long_quantile,
    short_quantile,
    cost_per_unit_turnover,
    rebalance_frequency,
    trading_days_per_year,
):
    periods_per_year = _periods_per_year(
        rebalance_frequency,
        trading_days_per_year=trading_days_per_year,
    )

    _, returns, metrics = build_long_short_backtest(
        df=df,
        prediction_col=prediction_col,
        target_col=target_col,
        time_col=time_col,
        symbol_col=symbol_col,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        cost_per_unit_turnover=cost_per_unit_turnover,
        periods_per_year=periods_per_year,
    )

    return returns, metrics


def run_cost_sensitivity_analysis(
    df,
    transaction_costs=(0.0, 0.0005, 0.001, 0.0025, 0.005),
    prediction_col="SIGNAL_SCORE",
    target_col="TARGET_RETURN",
    time_col="DATETIME",
    symbol_col="SYMBOL",
    long_quantile=0.90,
    short_quantile=0.10,
    rebalance_frequency="10min",
    use_signal_threshold=False,
    signal_threshold=0.001,
    trading_days_per_year=252,
):
    """
    Re-run the long-short strategy across a grid of transaction costs.

    Costs are expressed per unit of turnover. A value of 0.0005 corresponds to
    five basis points for a full one-unit change in portfolio weights.
    """
    df_analysis = filter_rebalance_frequency(
        df,
        rebalance_frequency=rebalance_frequency,
        time_col=time_col,
    )

    df_analysis = _maybe_apply_signal_threshold(
        df_analysis,
        signal_col=prediction_col,
        use_signal_threshold=use_signal_threshold,
        signal_threshold=signal_threshold,
    )

    rows = []
    frequency = _normalize_frequency(rebalance_frequency)

    for cost in transaction_costs:
        returns, metrics = _run_single_backtest(
            df=df_analysis,
            prediction_col=prediction_col,
            target_col=target_col,
            time_col=time_col,
            symbol_col=symbol_col,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            cost_per_unit_turnover=cost,
            rebalance_frequency=frequency,
            trading_days_per_year=trading_days_per_year,
        )

        row = {
            "transaction_cost": cost,
            "rebalance_frequency": frequency,
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
            "use_signal_threshold": use_signal_threshold,
            "signal_threshold": signal_threshold if use_signal_threshold else np.nan,
        }
        row.update(_summarize_backtest(returns, metrics))
        rows.append(row)

    return pd.DataFrame(rows)


def run_rebalance_frequency_analysis(
    df,
    rebalance_frequencies=("10min", "30min", "60min"),
    prediction_col="SIGNAL_SCORE",
    target_col="TARGET_RETURN",
    time_col="DATETIME",
    symbol_col="SYMBOL",
    long_quantile=0.90,
    short_quantile=0.10,
    cost_per_unit_turnover=0.0005,
    use_signal_threshold=False,
    signal_threshold=0.001,
    trading_days_per_year=252,
):
    """Compare portfolio results when the signal is traded less frequently."""
    rows = []

    for frequency in rebalance_frequencies:
        frequency = _normalize_frequency(frequency)
        df_analysis = filter_rebalance_frequency(
            df,
            rebalance_frequency=frequency,
            time_col=time_col,
        )

        df_analysis = _maybe_apply_signal_threshold(
            df_analysis,
            signal_col=prediction_col,
            use_signal_threshold=use_signal_threshold,
            signal_threshold=signal_threshold,
        )

        returns, metrics = _run_single_backtest(
            df=df_analysis,
            prediction_col=prediction_col,
            target_col=target_col,
            time_col=time_col,
            symbol_col=symbol_col,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            cost_per_unit_turnover=cost_per_unit_turnover,
            rebalance_frequency=frequency,
            trading_days_per_year=trading_days_per_year,
        )

        row = {
            "rebalance_frequency": frequency,
            "transaction_cost": cost_per_unit_turnover,
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
            "use_signal_threshold": use_signal_threshold,
            "signal_threshold": signal_threshold if use_signal_threshold else np.nan,
        }
        row.update(_summarize_backtest(returns, metrics))
        rows.append(row)

    return pd.DataFrame(rows)


def run_quantile_analysis(
    df,
    quantiles=(0.05, 0.10, 0.20),
    prediction_col="SIGNAL_SCORE",
    target_col="TARGET_RETURN",
    time_col="DATETIME",
    symbol_col="SYMBOL",
    cost_per_unit_turnover=0.0005,
    rebalance_frequency="10min",
    use_signal_threshold=False,
    signal_threshold=0.001,
    trading_days_per_year=252,
):
    """Compare how concentrated the long and short books should be."""
    df_analysis = filter_rebalance_frequency(
        df,
        rebalance_frequency=rebalance_frequency,
        time_col=time_col,
    )

    df_analysis = _maybe_apply_signal_threshold(
        df_analysis,
        signal_col=prediction_col,
        use_signal_threshold=use_signal_threshold,
        signal_threshold=signal_threshold,
    )

    rows = []
    frequency = _normalize_frequency(rebalance_frequency)

    for quantile in quantiles:
        if not 0 < quantile < 0.5:
            raise ValueError("Each quantile must be between 0 and 0.5.")

        short_quantile = quantile
        long_quantile = 1.0 - quantile

        returns, metrics = _run_single_backtest(
            df=df_analysis,
            prediction_col=prediction_col,
            target_col=target_col,
            time_col=time_col,
            symbol_col=symbol_col,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            cost_per_unit_turnover=cost_per_unit_turnover,
            rebalance_frequency=frequency,
            trading_days_per_year=trading_days_per_year,
        )

        row = {
            "portfolio_tail_fraction": quantile,
            "rebalance_frequency": frequency,
            "transaction_cost": cost_per_unit_turnover,
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
            "use_signal_threshold": use_signal_threshold,
            "signal_threshold": signal_threshold if use_signal_threshold else np.nan,
        }
        row.update(_summarize_backtest(returns, metrics))
        rows.append(row)

    return pd.DataFrame(rows)


def run_signal_threshold_analysis(
    df,
    thresholds=(0.0, 0.0005, 0.001, 0.002, 0.005),
    prediction_col="SIGNAL_SCORE",
    target_col="TARGET_RETURN",
    time_col="DATETIME",
    symbol_col="SYMBOL",
    long_quantile=0.90,
    short_quantile=0.10,
    cost_per_unit_turnover=0.0005,
    rebalance_frequency="10min",
    trading_days_per_year=252,
):
    """Compare portfolio results after neutralizing weak signals."""
    df_analysis = filter_rebalance_frequency(
        df,
        rebalance_frequency=rebalance_frequency,
        time_col=time_col,
    )

    rows = []
    frequency = _normalize_frequency(rebalance_frequency)

    for threshold in thresholds:
        thresholded_df = apply_signal_threshold(
            df_analysis,
            signal_col=prediction_col,
            threshold=threshold,
        )

        returns, metrics = _run_single_backtest(
            df=thresholded_df,
            prediction_col=prediction_col,
            target_col=target_col,
            time_col=time_col,
            symbol_col=symbol_col,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            cost_per_unit_turnover=cost_per_unit_turnover,
            rebalance_frequency=frequency,
            trading_days_per_year=trading_days_per_year,
        )

        row = {
            "signal_threshold": threshold,
            "rebalance_frequency": frequency,
            "transaction_cost": cost_per_unit_turnover,
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
        }
        row.update(_summarize_backtest(returns, metrics))
        rows.append(row)

    return pd.DataFrame(rows)
