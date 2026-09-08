from __future__ import annotations

import itertools
import ast
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def scaled_linear_model(model: Any) -> Pipeline:
    """Pipeline used for linear models, where feature scaling matters."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def tree_model(model: Any) -> Pipeline:
    """Pipeline used for tree models, which do not require standardization."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def regression_models(
    random_state: int,
    n_jobs: int = 4,
    tuned_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Pipeline]:
    """
    Define the baseline regression models used in the prediction experiment.

    The set deliberately mixes interpretable linear benchmarks with nonlinear
    tree-based methods, so that the final comparison shows whether the signal is
    mostly linear or depends on interactions between intraday features.
    """
    models = {
        # Linear family: useful as transparent benchmarks for weak return signals.
        "linear_regression": scaled_linear_model(LinearRegression()),
        "ridge": scaled_linear_model(Ridge(alpha=1.0, random_state=random_state)),
        "lasso": scaled_linear_model(
            Lasso(alpha=1e-5, max_iter=20_000, random_state=random_state)
        ),
        "elastic_net": scaled_linear_model(
            ElasticNet(alpha=1e-5, l1_ratio=0.30, max_iter=20_000, random_state=random_state)
        ),
        "huber": scaled_linear_model(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)),
        # Nonlinear family: captures interactions between lags, demand, ranks and time of day.
        "random_forest": tree_model(
            RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=50,
                max_features="sqrt",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        ),
        "hist_gradient_boosting": tree_model(
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1e-4,
                random_state=random_state,
            )
        ),
    }

    try:
        from lightgbm import LGBMRegressor

        # LightGBM is optional because it may not be installed on every machine.
        models["lightgbm"] = tree_model(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.80,
                colsample_bytree=0.80,
                objective="regression",
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=-1,
            )
        )
    except ImportError:
        pass

    if tuned_params:
        # Parameters saved from validation tuning are applied before the final fit.
        for name, params in tuned_params.items():
            if name in models:
                models[name].set_params(**params)

    return models


def tuning_grids() -> dict[str, list[dict[str, Any]]]:
    """Small validation grids chosen to keep the full experiment reproducible."""
    return {
        "ridge": [{"model__alpha": alpha} for alpha in [0.01, 0.1, 1.0, 10.0]],
        "lasso": [{"model__alpha": alpha} for alpha in [1e-6, 1e-5, 1e-4]],
        "elastic_net": [
            {"model__alpha": alpha, "model__l1_ratio": l1_ratio}
            for alpha, l1_ratio in itertools.product([1e-6, 1e-5, 1e-4], [0.15, 0.30])
        ],
        "huber": [
            {"model__epsilon": epsilon, "model__alpha": alpha}
            for epsilon, alpha in itertools.product([1.20, 1.35, 1.50], [1e-5, 1e-4])
        ],
        "random_forest": [
            {"model__n_estimators": 80, "model__max_depth": 6, "model__min_samples_leaf": 80},
            {"model__n_estimators": 120, "model__max_depth": 8, "model__min_samples_leaf": 50},
            {"model__n_estimators": 160, "model__max_depth": 10, "model__min_samples_leaf": 80},
        ],
        "hist_gradient_boosting": [
            {"model__max_iter": 150, "model__learning_rate": 0.05, "model__max_leaf_nodes": 15},
            {"model__max_iter": 250, "model__learning_rate": 0.05, "model__max_leaf_nodes": 31},
            {"model__max_iter": 250, "model__learning_rate": 0.03, "model__max_leaf_nodes": 31},
            {"model__max_iter": 350, "model__learning_rate": 0.03, "model__max_leaf_nodes": 31},
        ],
        "lightgbm": [
            {"model__n_estimators": 300, "model__learning_rate": 0.05, "model__num_leaves": 15},
            {"model__n_estimators": 500, "model__learning_rate": 0.03, "model__num_leaves": 31},
            {"model__n_estimators": 700, "model__learning_rate": 0.02, "model__num_leaves": 31},
        ],
    }


def best_params_from_tuning_results(tuning_results_path: str | Any) -> dict[str, dict[str, Any]]:
    """Load the best validation parameters for each model from a previous tuning run."""
    tuning_df = pd.read_csv(tuning_results_path)
    idx = tuning_df.groupby("model")["validation_rmse"].idxmin()
    best_rows = tuning_df.loc[idx, ["model", "params"]]
    return {
        row["model"]: ast.literal_eval(row["params"])
        for _, row in best_rows.iterrows()
    }


def sample_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly cap rows for tuning steps that would otherwise be too expensive."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    sampled_index = X.sample(max_rows, random_state=random_state).index
    return X.loc[sampled_index], pd.Series(y, index=X.index).loc[sampled_index].to_numpy()


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Correlation is undefined when either series has zero variance."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict[str, float]:
    """Compute statistical metrics for next-period return prediction."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # IC measures ranking quality, which is central for cross-sectional trading.
        "information_coefficient": safe_corr(y_true, y_pred),
        "hit_ratio": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
    }


def tune_models_on_validation(
    base_models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    random_state: int,
    train_sample_rows: int | None = 150_000,
    validation_sample_rows: int | None = 50_000,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """
    Tune baseline models on a reduced train/validation sample.

    The final evaluation still uses the chronological test set; the sampling here
    only keeps the grid search runtime manageable on the intraday panel.
    """
    X_train_sample, y_train_sample = sample_rows(
        X_train, y_train, train_sample_rows, random_state=random_state
    )
    X_validation_sample, y_validation_sample = sample_rows(
        X_validation,
        y_validation,
        validation_sample_rows,
        random_state=random_state + 1,
    )

    rows = []
    best_params: dict[str, dict[str, Any]] = {}
    grids = tuning_grids()

    for model_name, grid in grids.items():
        if model_name not in base_models:
            continue

        model_best_rmse = np.inf
        model_best_params: dict[str, Any] = {}

        for params in grid:
            # Each candidate is cloned so validation trials remain independent.
            model = clone(base_models[model_name])
            model.set_params(**params)
            start = time.perf_counter()
            model.fit(X_train_sample, y_train_sample)
            validation_pred = model.predict(X_validation_sample)
            elapsed = time.perf_counter() - start
            rmse = float(np.sqrt(mean_squared_error(y_validation_sample, validation_pred)))
            mae = float(mean_absolute_error(y_validation_sample, validation_pred))
            ic = safe_corr(y_validation_sample, validation_pred)

            rows.append(
                {
                    "model": model_name,
                    "params": str(params),
                    "validation_rmse": rmse,
                    "validation_mae": mae,
                    "validation_ic": ic,
                    "fit_predict_seconds": elapsed,
                    "train_sample_rows": len(X_train_sample),
                    "validation_sample_rows": len(X_validation_sample),
                }
            )

            if rmse < model_best_rmse:
                model_best_rmse = rmse
                model_best_params = params

        best_params[model_name] = model_best_params

    tuning_df = pd.DataFrame(rows)
    return best_params, tuning_df
