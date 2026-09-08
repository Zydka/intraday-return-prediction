import numpy as np
import pandas as pd


def add_prediction_ranks(
    df,
    prediction_col="prediction",
    time_col="DATETIME",
    rank_col="prediction_rank",
):
    """
    Cross-sectional ranking of predicted returns at each timestamp.
    Higher rank = better predicted return.
    """
    df = df.copy()

    df[rank_col] = df.groupby(time_col)[prediction_col].rank(
        pct=True,
        method="first"
    )

    return df


def assign_long_short_positions(
    df,
    rank_col="prediction_rank",
    long_quantile=0.9,
    short_quantile=0.1,
    position_col="position",
):
    """
    Assign +1 to top predicted stocks, -1 to bottom predicted stocks, 0 otherwise.
    """
    df = df.copy()

    df[position_col] = 0
    df.loc[df[rank_col] >= long_quantile, position_col] = 1
    df.loc[df[rank_col] <= short_quantile, position_col] = -1

    return df


def compute_equal_weights(
    df,
    time_col="DATETIME",
    symbol_col="SYMBOL",
    position_col="position",
    weight_col="weight",
):
    """
    Equal-weight long-short portfolio.

    Long side sums to +1.
    Short side sums to -1.
    Neutral stocks have weight 0.
    """
    df = df.copy()
    df[weight_col] = 0.0

    long_mask = df[position_col] == 1
    short_mask = df[position_col] == -1

    n_longs = df[long_mask].groupby(time_col)[symbol_col].transform("count")
    n_shorts = df[short_mask].groupby(time_col)[symbol_col].transform("count")

    df.loc[long_mask, weight_col] = 1.0 / n_longs
    df.loc[short_mask, weight_col] = -1.0 / n_shorts

    return df


def compute_portfolio_returns(
    df,
    target_col="TARGET_RETURN",
    time_col="DATETIME",
    weight_col="weight",
):
    """
    Compute gross portfolio return at each timestamp.
    """
    df = df.copy()

    df["weighted_return"] = df[weight_col] * df[target_col]

    portfolio_returns = (
        df.groupby(time_col)["weighted_return"]
        .sum()
        .rename("gross_return")
        .to_frame()
    )

    return portfolio_returns


def compute_turnover(
    df,
    time_col="DATETIME",
    symbol_col="SYMBOL",
    weight_col="weight",
):
    """
    Turnover = sum of absolute changes in portfolio weights.
    """
    weights = (
        df[[time_col, symbol_col, weight_col]]
        .copy()
        .pivot(index=time_col, columns=symbol_col, values=weight_col)
        .fillna(0.0)
        .sort_index()
    )

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    return turnover.rename("turnover").to_frame()


def apply_transaction_costs(
    portfolio_returns,
    turnover,
    cost_per_unit_turnover=0.0005,
):
    """
    Net return = gross return - transaction cost * turnover.
    Example: 0.0005 = 5 basis points.
    """
    results = portfolio_returns.join(turnover, how="left")
    results["turnover"] = results["turnover"].fillna(0.0)

    results["transaction_cost"] = (
        cost_per_unit_turnover * results["turnover"]
    )

    results["net_return"] = (
        results["gross_return"] - results["transaction_cost"]
    )

    return results


def compute_performance_metrics(
    returns,
    return_col="net_return",
    periods_per_year=252 * 38,
):
    """
    Compute standard backtest performance metrics.
    periods_per_year = trading days * intraday periods per day.
    For 10-minute data with ~38 bars/day: 252 * 38.
    """
    r = returns[return_col].dropna()

    cumulative_return = (1 + r).prod() - 1
    mean_return = r.mean()
    volatility = r.std()

    sharpe = np.nan
    if volatility != 0:
        sharpe = mean_return / volatility * np.sqrt(periods_per_year)

    cumulative_curve = (1 + r).cumprod()
    running_max = cumulative_curve.cummax()
    drawdown = cumulative_curve / running_max - 1
    max_drawdown = drawdown.min()

    hit_ratio = (r > 0).mean()

    metrics = {
        "cumulative_return": cumulative_return,
        "mean_return": mean_return,
        "volatility": volatility,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_ratio": hit_ratio,
    }

    return pd.Series(metrics)


def build_long_short_backtest(
    df,
    prediction_col="prediction",
    target_col="TARGET_RETURN",
    time_col="DATETIME",
    symbol_col="SYMBOL",
    long_quantile=0.9,
    short_quantile=0.1,
    cost_per_unit_turnover=0.0005,
    periods_per_year=252 * 38,
):
    """
    Full long-short portfolio construction and backtest pipeline.

    Input dataframe must contain:
    - DATETIME
    - SYMBOL
    - prediction column
    - TARGET_RETURN or realized next-period return
    """

    df_bt = df.copy()

    df_bt = add_prediction_ranks(
        df_bt,
        prediction_col=prediction_col,
        time_col=time_col,
    )

    df_bt = assign_long_short_positions(
        df_bt,
        rank_col="prediction_rank",
        long_quantile=long_quantile,
        short_quantile=short_quantile,
    )

    df_bt = compute_equal_weights(
        df_bt,
        time_col=time_col,
        symbol_col=symbol_col,
    )

    gross_returns = compute_portfolio_returns(
        df_bt,
        target_col=target_col,
        time_col=time_col,
    )

    turnover = compute_turnover(
        df_bt,
        time_col=time_col,
        symbol_col=symbol_col,
    )

    returns = apply_transaction_costs(
        gross_returns,
        turnover,
        cost_per_unit_turnover=cost_per_unit_turnover,
    )

    metrics = compute_performance_metrics(
        returns,
        return_col="net_return",
        periods_per_year=periods_per_year,
    )

    return df_bt, returns, metrics