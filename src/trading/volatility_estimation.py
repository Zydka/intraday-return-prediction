"""
Volatility estimators used by the final portfolio risk overlay.

The project data are 10-minute intraday returns with no high/low/close option
surface. For this setting the useful volatility inputs are simple realized
volatility and EWMA volatility, both lagged so that estimates at time t only
use information available up to t-1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if df.empty:
        raise ValueError(f"{name} is empty.")


def _as_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def long_returns_to_wide(
    df: pd.DataFrame,
    date_col: str = "DATETIME",
    symbol_col: str = "SYMBOL",
    return_col: str = "TARGET_RETURN",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Convert long-format intraday returns to a wide date-by-symbol matrix."""
    _validate_dataframe(df, "df")
    required = {date_col, symbol_col, return_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in returns dataframe: {sorted(missing)}")

    wide = df.pivot_table(
        index=date_col,
        columns=symbol_col,
        values=return_col,
        aggfunc=aggfunc,
    )
    return _as_datetime_index(wide)


def ensure_wide_returns(
    returns: pd.DataFrame | pd.Series,
    date_col: str = "DATETIME",
    symbol_col: str = "SYMBOL",
    return_col: str = "TARGET_RETURN",
) -> pd.DataFrame:
    """Accept either long or wide returns and return a numeric wide matrix."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    _validate_dataframe(returns, "returns")
    if {date_col, symbol_col, return_col}.issubset(returns.columns):
        return long_returns_to_wide(returns, date_col, symbol_col, return_col)

    numeric = returns.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("Wide returns dataframe must contain numeric columns.")
    return _as_datetime_index(numeric)


def align_returns_by_date_symbol(
    returns: pd.DataFrame | pd.Series,
    symbols: list[str] | None = None,
    dates: pd.Index | list[Any] | None = None,
    fill_method: str | None = None,
    date_col: str = "DATETIME",
    symbol_col: str = "SYMBOL",
    return_col: str = "TARGET_RETURN",
) -> pd.DataFrame:
    """
    Align returns to a common date index and symbol set.

    Missing values are kept by default. Allowed fills do not use future data.
    """
    wide = ensure_wide_returns(returns, date_col, symbol_col, return_col)

    if dates is not None:
        wide = wide.reindex(pd.to_datetime(pd.Index(dates)))
    if symbols is not None:
        wide = wide.reindex(columns=symbols)

    if fill_method is None:
        return _as_datetime_index(wide)
    if fill_method == "zero":
        return _as_datetime_index(wide.fillna(0.0))
    if fill_method == "ffill":
        return _as_datetime_index(wide.ffill())
    raise ValueError("fill_method must be None, 'zero', or 'ffill'.")


def rolling_historical_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = 60,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Rolling realized volatility from past intraday returns.

    The returned volatility at time t is computed from returns up to t-1.
    """
    wide = ensure_wide_returns(returns)
    min_periods = min_periods or window
    return wide.shift(1).rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def ewma_volatility(
    returns: pd.DataFrame | pd.Series,
    span: int | None = None,
    halflife: float | None = 20,
    alpha: float | None = None,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    EWMA volatility forecast for intraday returns.

    EWMA reacts faster than a long rolling window, which makes it a useful
    simple estimator for 10-minute data. Returns are shifted by one period to
    avoid look-ahead bias.
    """
    wide = ensure_wide_returns(returns)
    shifted = wide.shift(1)
    return shifted.ewm(
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=False,
    ).std(bias=False)


def diagonal_covariance_from_volatility(
    vol_df: pd.DataFrame,
    regularization: float = 1e-8,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Build diagonal covariance matrices from individual volatility forecasts.

    This keeps the risk model simple and avoids unstable full covariance
    estimates on a short high-frequency sample with many assets.
    """
    _validate_dataframe(vol_df, "vol_df")
    vol_df = _as_datetime_index(vol_df)
    covariances: dict[pd.Timestamp, pd.DataFrame] = {}

    for date, row in vol_df.iterrows():
        row = row.dropna()
        if row.empty:
            continue
        values = np.diag(np.square(row.to_numpy(dtype=float)) + regularization)
        covariances[pd.Timestamp(date)] = pd.DataFrame(values, index=row.index, columns=row.index)

    return covariances


def get_latest_volatility(vol_df: pd.DataFrame, date: Any) -> pd.Series | None:
    """Return the latest volatility vector available on or before a date."""
    _validate_dataframe(vol_df, "vol_df")
    vol_df = _as_datetime_index(vol_df)
    target = pd.Timestamp(date)
    eligible = vol_df.index[vol_df.index <= target]
    if len(eligible) == 0:
        return None
    return vol_df.loc[eligible[-1]].copy()


def regularize_covariance(Sigma: pd.DataFrame | np.ndarray, epsilon: float = 1e-6) -> pd.DataFrame | np.ndarray:
    """Add a small diagonal term to a covariance matrix."""
    if isinstance(Sigma, pd.DataFrame):
        values = Sigma.to_numpy(dtype=float) + epsilon * np.eye(len(Sigma))
        return pd.DataFrame(values, index=Sigma.index, columns=Sigma.columns)
    values = np.asarray(Sigma, dtype=float)
    return values + epsilon * np.eye(values.shape[0])


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=120, freq="10min")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.002, size=(len(dates), 3)),
        index=dates,
        columns=["A", "B", "C"],
    )

    print("Latest rolling volatility:")
    print(get_latest_volatility(rolling_historical_volatility(returns, window=30), dates[-1]))
    print("\nLatest EWMA volatility:")
    print(get_latest_volatility(ewma_volatility(returns, halflife=20), dates[-1]))
