import numpy as np
import pandas as pd

# 09:30 to 15:40 inclusive, in 10-minute bars: 38 bars, 370 minutes.
SESSION_MINUTES = 370
BARS_PER_DAY = 38


def add_lagged_features(df, n_lags=3, demand_is_bar_aggregate=True):
    """Create lagged return and demand features stock by stock.

    SUM_DELTA is a 10-minute AGGREGATE of signed order flow. The value stamped
    at time t covers the interval [t, t+10min) and is therefore only known once
    that bar has closed -- the same instant TARGET_RETURN is realised. Verified
    on the raw files:

        corr(SUM_DELTA_t, return over bar t)   = +0.102
        corr(SUM_DELTA_t, return over bar t-1) = +0.003
        corr(SUM_DELTA_{t-1}, return over bar t) = -0.001

    Using SUM_DELTA_t as a predictor of the return over bar t is look-ahead: it
    recovers the contemporaneous price-impact relation, not a forecast. With
    demand_is_bar_aggregate=True the demand series is shifted one bar so that
    every feature is observable at the moment a trade would be placed.
    """
    df = df.copy()
    df = df.sort_values(["SYMBOL", "DATETIME"]).reset_index(drop=True)

    if demand_is_bar_aggregate:
        demand = df.groupby("SYMBOL")["SUM_DELTA"].shift(1)
        if "CONTIGUOUS" in df.columns:
            demand = demand.where(df["CONTIGUOUS"])
        df["SUM_DELTA_OBSERVABLE"] = demand
    else:
        df["SUM_DELTA_OBSERVABLE"] = df["SUM_DELTA"]

    df["RETURN_CURRENT"] = df["RETURN_CLIPPED"]
    df["SUM_DELTA_CURRENT"] = df["SUM_DELTA_OBSERVABLE"]

    for lag in range(1, n_lags + 1):
        df[f"RETURN_LAG_{lag}"] = (
            df.groupby("SYMBOL")["RETURN_CLIPPED"].shift(lag)
        )

        df[f"SUM_DELTA_LAG_{lag}"] = (
            df.groupby("SYMBOL")["SUM_DELTA_OBSERVABLE"].shift(lag)
        )

    return df


def add_rolling_features(df, windows=(3, 5)):
    """Create short rolling statistics using only past and current values."""
    df = df.copy()
    df = df.sort_values(["SYMBOL", "DATETIME"]).reset_index(drop=True)

    grouped_returns = df.groupby("SYMBOL")["RETURN_CLIPPED"]
    grouped_delta = df.groupby("SYMBOL")["SUM_DELTA_OBSERVABLE"]

    for window in windows:
        df[f"ROLLING_RETURN_MEAN_{window}"] = (
            grouped_returns
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"ROLLING_RETURN_STD_{window}"] = (
            grouped_returns
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )

        df[f"ROLLING_SUM_DELTA_MEAN_{window}"] = (
            grouped_delta
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def add_intraday_time_features(df):
    """Add simple time-of-day features."""
    df = df.copy()

    time_as_datetime = pd.to_datetime(df["TIME"], format="%H:%M:%S")

    df["MINUTES_FROM_OPEN"] = (
        (time_as_datetime.dt.hour - 9) * 60
        + time_as_datetime.dt.minute
        - 30
    )

    # A fixed session length, not the sample maximum: using .max() makes the
    # feature depend on which rows happen to be in the frame.
    df["INTRADAY_TIME_FRACTION"] = df["MINUTES_FROM_OPEN"] / SESSION_MINUTES

    df["TIME_SIN"] = np.sin(2 * np.pi * df["INTRADAY_TIME_FRACTION"])
    df["TIME_COS"] = np.cos(2 * np.pi * df["INTRADAY_TIME_FRACTION"])

    return df


def add_cross_sectional_features(df):
    """Create cross-sectional ranks at each timestamp."""
    df = df.copy()

    df["RETURN_CS_RANK"] = (
        df.groupby("DATETIME")["RETURN_CLIPPED"]
        .rank(pct=True)
    )

    df["SUM_DELTA_CS_RANK"] = (
        df.groupby("DATETIME")["SUM_DELTA_OBSERVABLE"]
        .rank(pct=True)
    )

    return df


def add_normalized_demand_feature(df, fit_mask=None):
    """Normalize the observable demand series stock by stock.

    fit_mask selects the rows the per-symbol mean and standard deviation are
    estimated on. Pass the training window: computing them over the full panel,
    as the original code did, leaks test-window information.
    """
    df = df.copy()

    source = "SUM_DELTA_OBSERVABLE" if "SUM_DELTA_OBSERVABLE" in df.columns else "SUM_DELTA"

    if fit_mask is None:
        grouped = df.groupby("SYMBOL")[source]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
    else:
        moments = df.loc[fit_mask].groupby("SYMBOL")[source].agg(["mean", "std"])
        mean = df["SYMBOL"].map(moments["mean"])
        std = df["SYMBOL"].map(moments["std"])

    df["SUM_DELTA_ZSCORE"] = (df[source] - mean) / (std + 1e-8)

    return df


def add_direction_target(df, return_target_col="TARGET_RETURN"):
    """
    Create a binary target for direction prediction.

    TARGET_DIRECTION = 1 if future return is positive
    TARGET_DIRECTION = 0 otherwise
    """

    df = df.copy()

    df["TARGET_DIRECTION"] = (
        df[return_target_col] > 0
    ).astype(int)

    return df


def build_feature_dataset(
    df,
    n_lags=3,
    rolling_windows=(3, 5),
    target_type="return",
    drop_missing=True,
    fit_mask=None,
    demand_is_bar_aggregate=True
):
    """
    Run the full feature engineering pipeline.

    target_type can be:
    - "return" for regression on TARGET_RETURN
    - "direction" for classification on TARGET_DIRECTION
    """

    df = df.copy()

    df = add_lagged_features(
        df, n_lags=n_lags, demand_is_bar_aggregate=demand_is_bar_aggregate
    )
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_intraday_time_features(df)
    df = add_cross_sectional_features(df)
    df = add_normalized_demand_feature(df, fit_mask=fit_mask)
    df = add_direction_target(df)

    feature_cols = [
        "RETURN_CURRENT",
        "SUM_DELTA_CURRENT",
        "SUM_DELTA_ZSCORE",
        "MINUTES_FROM_OPEN",
        "INTRADAY_TIME_FRACTION",
        "TIME_SIN",
        "TIME_COS",
        "RETURN_CS_RANK",
        "SUM_DELTA_CS_RANK",
    ]

    for lag in range(1, n_lags + 1):
        feature_cols.append(f"RETURN_LAG_{lag}")
        feature_cols.append(f"SUM_DELTA_LAG_{lag}")

    for window in rolling_windows:
        feature_cols.append(f"ROLLING_RETURN_MEAN_{window}")
        feature_cols.append(f"ROLLING_RETURN_STD_{window}")
        feature_cols.append(f"ROLLING_SUM_DELTA_MEAN_{window}")

    if target_type == "return":
        target_col = "TARGET_RETURN"

    elif target_type == "direction":
        target_col = "TARGET_DIRECTION"

    else:
        raise ValueError(
            "target_type must be either 'return' or 'direction'"
        )

    if drop_missing:
        df = df.dropna(subset=feature_cols + [target_col]).copy()

    return df, feature_cols, target_col