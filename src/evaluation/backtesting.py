import numpy as np
import pandas as pd

from src.deep_learning import (
    build_attention_lstm_model,
    fit_lstm_model_from_dataframes,
    predict_lstm_model_from_dataframe,
    validate_lstm_regressor,
)
from src.trading.portfolio_construction import (
    build_long_short_backtest,
    compute_performance_metrics,
)
from src.trading.transaction_cost_analysis import apply_signal_threshold


def get_trading_days(df, datetime_col="DATETIME"):
    """Return sorted unique trading dates from an intraday panel."""
    dates = pd.to_datetime(df[datetime_col]).dt.date
    return sorted(pd.unique(dates))


def create_walk_forward_windows(
    df,
    train_days=10,
    validation_days=3,
    test_days=1,
    step_days=1,
    datetime_col="DATETIME",
):
    """
    Build chronological rolling windows with train, validation, and test blocks.

    Every block is strictly later than the previous one, so the test period never
    leaks into model fitting or validation.
    """
    if min(train_days, validation_days, test_days, step_days) <= 0:
        raise ValueError("Window lengths and step_days must be positive.")

    trading_days = get_trading_days(df, datetime_col=datetime_col)
    total_days = train_days + validation_days + test_days
    windows = []

    for start in range(0, len(trading_days) - total_days + 1, step_days):
        train_start_idx = start
        train_end_idx = start + train_days - 1
        validation_start_idx = train_end_idx + 1
        validation_end_idx = validation_start_idx + validation_days - 1
        test_start_idx = validation_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1

        windows.append({
            "train_start": trading_days[train_start_idx],
            "train_end": trading_days[train_end_idx],
            "validation_start": trading_days[validation_start_idx],
            "validation_end": trading_days[validation_end_idx],
            "test_start": trading_days[test_start_idx],
            "test_end": trading_days[test_end_idx],
        })

    return windows


def slice_by_dates(df, start_date, end_date, datetime_col="DATETIME"):
    """Return rows whose trading date is between start_date and end_date."""
    df_sliced = df.copy()
    dates = pd.to_datetime(df_sliced[datetime_col]).dt.date
    mask = (dates >= start_date) & (dates <= end_date)
    return df_sliced.loc[mask].sort_values(datetime_col).copy()


def _validate_model_type(model_type):
    model_type = model_type.lower()
    if model_type not in {"lstm", "attention_lstm"}:
        raise ValueError("model_type must be either 'lstm' or 'attention_lstm'.")
    return model_type


def train_deep_learning_model_for_window(
    model_type,
    train_df,
    validation_df,
    feature_cols,
    target_col,
    sequence_length=36,
    hidden_size=128,
    num_layers=1,
    dropout=0.20,
    epochs=3,
    batch_size=4096,
    learning_rate=5e-4,
    weight_decay=1e-5,
    patience=2,
    scale_data=True,
    seed=42,
    device=None,
    symbol_col="SYMBOL",
    datetime_col="DATETIME",
):
    """Train one LSTM-family model for one walk-forward window."""
    model_type = _validate_model_type(model_type)

    if model_type == "lstm":
        return validate_lstm_regressor(
            train_df=train_df,
            validation_df=validation_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            scale_data=scale_data,
            seed=seed,
            device=device,
        )

    model = build_attention_lstm_model(
        n_features=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed,
    )

    model = fit_lstm_model_from_dataframes(
        model=model,
        train_df=train_df,
        validation_df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        scale_data=scale_data,
        symbol_col=symbol_col,
        datetime_col=datetime_col,
        device=device,
        verbose=True,
    )

    validation_results = predict_lstm_model_from_dataframe(
        model=model,
        df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        batch_size=batch_size,
        scale_data=scale_data,
        symbol_col=symbol_col,
        datetime_col=datetime_col,
        device=device,
    )

    return model, model.history, validation_results


def summarize_backtest_results(
    portfolio_returns,
    periods_per_year=252 * 38,
):
    """Summarize the concatenated out-of-sample portfolio return series."""
    if portfolio_returns is None or portfolio_returns.empty:
        return pd.Series(dtype=float)

    metrics = compute_performance_metrics(
        portfolio_returns,
        return_col="net_return",
        periods_per_year=periods_per_year,
    )

    summary = {
        "cumulative_return": metrics["cumulative_return"],
        "mean_return": metrics["mean_return"],
        "volatility": metrics["volatility"],
        "sharpe_ratio": metrics["annualized_sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "hit_ratio": metrics["hit_ratio"],
        "average_turnover": portfolio_returns["turnover"].mean(),
    }

    if "transaction_cost" in portfolio_returns.columns:
        summary["average_transaction_cost"] = portfolio_returns["transaction_cost"].mean()

    if "gross_return" in portfolio_returns.columns:
        gross_metrics = compute_performance_metrics(
            portfolio_returns,
            return_col="gross_return",
            periods_per_year=periods_per_year,
        )
        summary["gross_cumulative_return"] = gross_metrics["cumulative_return"]

    return pd.Series(summary)


def _window_to_row(window_number, model_type, window, metrics, history):
    row = {
        "window": window_number,
        "model_type": model_type,
        **window,
        "best_validation_loss": np.nan,
        "final_validation_loss": np.nan,
    }

    validation_loss = history.get("validation_loss", []) if history else []
    if len(validation_loss) > 0:
        row["best_validation_loss"] = min(validation_loss)
        row["final_validation_loss"] = validation_loss[-1]

    for key, value in metrics.items():
        metric_name = "sharpe_ratio" if key == "annualized_sharpe" else key
        row[metric_name] = value

    return row


def walk_forward_deep_learning_backtest(
    df_model,
    feature_cols,
    target_col,
    model_type="lstm",
    sequence_length=36,
    hidden_size=128,
    num_layers=1,
    dropout=0.20,
    epochs=3,
    batch_size=4096,
    learning_rate=5e-4,
    weight_decay=1e-5,
    patience=2,
    train_days=10,
    validation_days=3,
    test_days=1,
    step_days=1,
    max_windows=None,
    long_quantile=0.90,
    short_quantile=0.10,
    cost_per_unit_turnover=0.001,
    periods_per_year=252 * 38,
    symbol_col="SYMBOL",
    datetime_col="DATETIME",
    scale_data=True,
    seed=42,
    device=None,
    use_signal_threshold=False,
    signal_threshold=0.001,
):
    """
    Run a rolling-window deep learning backtest and collect out-of-sample trades.
    """
    model_type = _validate_model_type(model_type)
    df_model = df_model.sort_values([datetime_col, symbol_col]).copy()

    windows = create_walk_forward_windows(
        df_model,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        step_days=step_days,
        datetime_col=datetime_col,
    )

    if max_windows is not None:
        windows = windows[:max_windows]

    all_predictions = []
    all_portfolio_returns = []
    window_metric_rows = []

    for window_number, window in enumerate(windows, start=1):
        print("\n" + "=" * 80)
        print(f"Training {model_type} walk-forward window {window_number}/{len(windows)}")
        print(
            f"Train {window['train_start']} to {window['train_end']} | "
            f"Validation {window['validation_start']} to {window['validation_end']} | "
            f"Test {window['test_start']} to {window['test_end']}"
        )

        train_df = slice_by_dates(
            df_model,
            window["train_start"],
            window["train_end"],
            datetime_col=datetime_col,
        )
        validation_df = slice_by_dates(
            df_model,
            window["validation_start"],
            window["validation_end"],
            datetime_col=datetime_col,
        )
        test_df = slice_by_dates(
            df_model,
            window["test_start"],
            window["test_end"],
            datetime_col=datetime_col,
        )

        print(
            f"Rows: train={len(train_df):,}, "
            f"validation={len(validation_df):,}, test={len(test_df):,}"
        )

        model, history, _ = train_deep_learning_model_for_window(
            model_type=model_type,
            train_df=train_df,
            validation_df=validation_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            scale_data=scale_data,
            seed=seed,
            device=device,
            symbol_col=symbol_col,
            datetime_col=datetime_col,
        )

        predictions = predict_lstm_model_from_dataframe(
            model=model,
            df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=sequence_length,
            batch_size=batch_size,
            scale_data=scale_data,
            symbol_col=symbol_col,
            datetime_col=datetime_col,
            device=device,
        )
        if use_signal_threshold:
            predictions = predictions.copy()
            predictions["SIGNAL_SCORE_RAW"] = predictions["SIGNAL_SCORE"]
            predictions = apply_signal_threshold(
                predictions,
                signal_col="SIGNAL_SCORE",
                threshold=signal_threshold,
            )

        predictions["window"] = window_number
        predictions["model_type"] = model_type
        predictions["use_signal_threshold"] = use_signal_threshold
        predictions["signal_threshold"] = signal_threshold if use_signal_threshold else np.nan
        all_predictions.append(predictions)

        _, portfolio_returns, portfolio_metrics = build_long_short_backtest(
            df=predictions,
            prediction_col="SIGNAL_SCORE",
            target_col=target_col,
            time_col=datetime_col,
            symbol_col=symbol_col,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            cost_per_unit_turnover=cost_per_unit_turnover,
            periods_per_year=periods_per_year,
        )

        portfolio_returns = portfolio_returns.reset_index()
        portfolio_returns["window"] = window_number
        portfolio_returns["model_type"] = model_type
        portfolio_returns["use_signal_threshold"] = use_signal_threshold
        portfolio_returns["signal_threshold"] = signal_threshold if use_signal_threshold else np.nan
        all_portfolio_returns.append(portfolio_returns)

        window_metric_rows.append(
            _window_to_row(
                window_number=window_number,
                model_type=model_type,
                window=window,
                metrics=portfolio_metrics,
                history=history,
            )
        )
        window_metric_rows[-1]["use_signal_threshold"] = use_signal_threshold
        window_metric_rows[-1]["signal_threshold"] = (
            signal_threshold if use_signal_threshold else np.nan
        )

    predictions_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions else pd.DataFrame()
    )
    portfolio_returns_df = (
        pd.concat(all_portfolio_returns, ignore_index=True)
        if all_portfolio_returns else pd.DataFrame()
    )
    window_metrics = pd.DataFrame(window_metric_rows)

    final_metrics = summarize_backtest_results(
        portfolio_returns_df,
        periods_per_year=periods_per_year,
    )
    final_metrics["model_type"] = model_type
    final_metrics["n_windows"] = len(window_metrics)
    final_metrics["use_signal_threshold"] = use_signal_threshold
    final_metrics["signal_threshold"] = signal_threshold if use_signal_threshold else np.nan

    return predictions_df, portfolio_returns_df, window_metrics, final_metrics
