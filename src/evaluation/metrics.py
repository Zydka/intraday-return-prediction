import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    R-squared score.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def information_coefficient(y_true, y_pred):
    """
    Pearson correlation between predictions and realized returns.

    Commonly used in quantitative finance.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.corrcoef(y_true, y_pred)[0, 1]


def hit_ratio(y_true, y_pred, exclude_zero_targets=True):
    """
    Percentage of correct directional predictions.

    A large share of 10-minute returns in this panel are exactly zero because
    the quote does not refresh (about 14% of the sample). np.sign(0) is 0, so a
    zero target can never be matched by a non-zero prediction and the metric is
    driven by staleness rather than by directional skill. Those observations are
    therefore excluded by default.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if exclude_zero_targets:
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.mean(np.sign(y_true) == np.sign(y_pred))


def zero_target_share(y_true):
    """Share of targets that are exactly zero (stale quotes)."""
    y_true = np.array(y_true)
    return float(np.mean(y_true == 0))


def cross_sectional_ic(df, prediction_col, target_col, time_col="DATETIME",
                       min_names=20, method="spearman"):
    """
    Mean per-timestamp rank correlation between prediction and realised return,
    with a t-statistic across timestamps.

    This is the metric that corresponds to a cross-sectional long-short
    strategy. The pooled Pearson correlation used previously mixes the
    cross-sectional signal with time-series variation in volatility and is not
    a measure of ranking skill.

    Returns (mean_ic, t_stat, n_timestamps).
    """
    ics = (
        df.groupby(time_col)[[prediction_col, target_col]]
        .apply(
            lambda g: g[prediction_col].corr(g[target_col], method=method)
            if len(g) >= min_names else np.nan
        )
        .dropna()
    )

    if len(ics) < 2:
        return np.nan, np.nan, len(ics)

    mean_ic = float(ics.mean())
    t_stat = float(mean_ic / (ics.std(ddof=1) / np.sqrt(len(ics))))

    return mean_ic, t_stat, int(len(ics))


def breakeven_cost(gross_returns, turnover_series):
    """
    Cost per unit of turnover at which the strategy stops making money.

    This is the single most informative economic number for a high-turnover
    intraday strategy: it can be compared directly with a realistic spread.
    """
    gross_returns = np.array(gross_returns)
    turnover_series = np.array(turnover_series)

    mean_turnover = np.mean(turnover_series)
    if mean_turnover <= 0:
        return np.nan

    return float(np.mean(gross_returns) / mean_turnover)


# The intraday panel has 38 ten-minute bars per trading day. Annualising a
# per-bar Sharpe therefore multiplies by sqrt(252 * 38) = 97.9, not sqrt(252).
# Every module in this project must use this one constant.
BARS_PER_DAY = 38
PERIODS_PER_YEAR = 252 * BARS_PER_DAY


def sharpe_ratio(returns, annualization_factor=PERIODS_PER_YEAR):
    """
    Annualized Sharpe ratio.

    Prefer sharpe_per_period() when reporting: over a test window of a few
    hundred bars the annualised number is dominated by the sqrt(9576) factor
    and is not an estimate of anything achievable.
    """

    returns = np.array(returns)

    per_period = sharpe_per_period(returns)
    if np.isnan(per_period):
        return np.nan

    return np.sqrt(annualization_factor) * per_period


def sharpe_per_period(returns):
    """
    Sharpe ratio per bar, with no annualisation. This is the number to sanity
    check: anything above ~0.1 per 10-minute bar deserves a leakage audit.
    """

    returns = np.array(returns)

    std_return = np.std(returns, ddof=1)
    if std_return == 0 or np.isnan(std_return):
        return np.nan

    return float(np.mean(returns) / std_return)


def cumulative_returns(returns):
    """
    Compute cumulative returns series.
    """

    returns = np.array(returns)

    return np.cumprod(1 + returns)


def maximum_drawdown(returns):
    """
    Compute maximum drawdown.
    """

    cumulative = cumulative_returns(returns)

    running_max = np.maximum.accumulate(cumulative)

    drawdown = (
        cumulative - running_max
    ) / running_max

    return np.min(drawdown)


def turnover(weights):
    """
    Portfolio turnover.

    weights shape:
    (n_periods, n_assets)
    """

    weights = np.array(weights)

    return np.mean(
        np.sum(
            np.abs(weights[1:] - weights[:-1]),
            axis=1
        )
    )


def net_returns(
    gross_returns,
    turnover_series,
    transaction_cost
):
    """
    Compute net returns after transaction costs.
    """

    gross_returns = np.array(gross_returns)
    turnover_series = np.array(turnover_series)

    return (
        gross_returns
        - transaction_cost * turnover_series
    )


def evaluate_predictions(
    y_true,
    y_pred
):
    """
    Compute standard prediction metrics.
    """

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "IC": information_coefficient(y_true, y_pred),
        "Hit Ratio": hit_ratio(y_true, y_pred)
    }

    return pd.Series(metrics)


def evaluate_portfolio(
    returns,
    turnover_series=None,
    transaction_cost=None
):
    """
    Compute portfolio performance metrics.
    """

    results = {}

    if turnover_series is not None and transaction_cost is not None:

        returns = net_returns(
            returns,
            turnover_series,
            transaction_cost
        )

    results["Mean Return"] = np.mean(returns)
    results["Volatility"] = np.std(returns)
    results["Sharpe Ratio"] = sharpe_ratio(returns)
    results["Maximum Drawdown"] = maximum_drawdown(returns)

    return pd.Series(results)