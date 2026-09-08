from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class TimeSeriesSpec:
    name: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)


def default_time_series_specs() -> list[TimeSeriesSpec]:
    return [
        TimeSeriesSpec(name="arma_1_1", order=(1, 0, 1)),
        TimeSeriesSpec(name="arima_1_1_1", order=(1, 1, 1)),
        TimeSeriesSpec(name="sarima_1_0_1_1_0_1_39", order=(1, 0, 1), seasonal_order=(1, 0, 1, 39)),
    ]


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "information_coefficient": safe_corr(y_true, y_pred),
        "hit_ratio": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
    }


def fit_forecast_sarimax(
    train_series: pd.Series | np.ndarray,
    test_steps: int,
    spec: TimeSeriesSpec,
    maxiter: int = 80,
    clip_forecasts: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_array = np.asarray(train_series, dtype=float)
    train_array = train_array[np.isfinite(train_array)]

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_array,
            order=spec.order,
            seasonal_order=spec.seasonal_order,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted = model.fit(disp=False, maxiter=maxiter)
        forecast = fitted.forecast(steps=test_steps)

    elapsed = time.perf_counter() - start
    forecast = np.asarray(forecast, dtype=float)
    n_clipped = 0
    if clip_forecasts and len(train_array) > 0:
        lower = float(np.quantile(train_array, 0.001))
        upper = float(np.quantile(train_array, 0.999))
        finite = np.isfinite(forecast)
        n_clipped = int((~finite).sum())
        forecast = np.where(finite, forecast, 0.0)
        clipped = np.clip(forecast, lower, upper)
        n_clipped += int(np.sum(clipped != forecast))
        forecast = clipped

    info = {
        "aic": float(getattr(fitted, "aic", np.nan)),
        "bic": float(getattr(fitted, "bic", np.nan)),
        "fit_forecast_seconds": elapsed,
        "converged": bool(fitted.mle_retvals.get("converged", False)),
        "n_clipped_forecasts": n_clipped,
    }
    return forecast, info


def evaluate_time_series_specs(
    df: pd.DataFrame,
    specs: list[TimeSeriesSpec],
    symbol_col: str = "SYMBOL",
    datetime_col: str = "DATETIME",
    return_col: str = "RETURN_CURRENT",
    target_col: str = "TARGET_RETURN",
    maxiter: int = 80,
    clip_forecasts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    fit_rows = []

    for symbol, group in df.groupby(symbol_col, sort=True):
        group = group.sort_values(datetime_col).copy()
        train_group = group[group["split"].isin(["train", "validation"])]
        test_group = group[group["split"].eq("test")]
        if len(train_group) < 80 or len(test_group) == 0:
            continue

        train_series = train_group[return_col].fillna(0.0)
        y_test = test_group[target_col].to_numpy(dtype=float)

        for spec in specs:
            try:
                forecast, info = fit_forecast_sarimax(
                    train_series=train_series,
                    test_steps=len(test_group),
                    spec=spec,
                    maxiter=maxiter,
                    clip_forecasts=clip_forecasts,
                )
            except Exception as exc:
                fit_rows.append(
                    {
                        "symbol": symbol,
                        "model": spec.name,
                        "status": "failed",
                        "error": type(exc).__name__,
                        "message": str(exc)[:300],
                        "train_rows": len(train_group),
                        "test_rows": len(test_group),
                    }
                )
                continue

            for dt, actual, pred in zip(test_group[datetime_col], y_test, forecast):
                prediction_rows.append(
                    {
                        "DATETIME": dt,
                        "SYMBOL": symbol,
                        "model": spec.name,
                        "TARGET_RETURN": actual,
                        "prediction": pred,
                    }
                )

            fit_rows.append(
                {
                    "symbol": symbol,
                    "model": spec.name,
                    "status": "ok",
                    "error": "",
                    "message": "",
                    "train_rows": len(train_group),
                    "test_rows": len(test_group),
                    **info,
                }
            )

    return pd.DataFrame(prediction_rows), pd.DataFrame(fit_rows)


def aggregate_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model"):
        rows.append(
            prediction_metrics(
                group["TARGET_RETURN"].to_numpy(dtype=float),
                group["prediction"].to_numpy(dtype=float),
                model_name,
            )
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
