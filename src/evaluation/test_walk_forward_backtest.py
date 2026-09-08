from pathlib import Path

import pandas as pd

from src.evaluation.backtesting import walk_forward_deep_learning_backtest
from src.preprocessing.features import build_feature_dataset
from src.preprocessing.preprocessing import preprocess_intraday_folder
from src.trading.transaction_cost_analysis import run_signal_threshold_analysis


def _save_backtest_outputs(
    model_type,
    predictions,
    portfolio_returns,
    window_metrics,
    final_metrics,
    output_dir,
):
    final_metrics_df = final_metrics.to_frame().T

    predictions.to_csv(
        output_dir / f"{model_type}_walk_forward_predictions.csv",
        index=False,
    )
    portfolio_returns.to_csv(
        output_dir / f"{model_type}_walk_forward_portfolio_returns.csv",
        index=False,
    )
    window_metrics.to_csv(
        output_dir / f"{model_type}_walk_forward_window_metrics.csv",
        index=False,
    )
    final_metrics_df.to_csv(
        output_dir / f"{model_type}_walk_forward_final_metrics.csv",
        index=False,
    )


def _run_one_model(model_type, df_model, feature_cols, target_col, output_dir):
    print("\n" + "#" * 80)
    print(f"Running walk-forward backtest for {model_type}")
    print("#" * 80)

    predictions, portfolio_returns, window_metrics, final_metrics = (
        walk_forward_deep_learning_backtest(
            df_model=df_model,
            feature_cols=feature_cols,
            target_col=target_col,
            model_type=model_type,
            train_days=10,
            validation_days=3,
            test_days=1,
            step_days=1,
            max_windows=8 or None,
            epochs=5,
            sequence_length=36,
            hidden_size=128,
            num_layers=1,
            dropout=0.20,
            batch_size=4096,
            learning_rate=5e-4,
            weight_decay=1e-5,
            patience=2,
            long_quantile=0.90,
            short_quantile=0.10,
            cost_per_unit_turnover=0.001,
            periods_per_year=252 * 38,
            symbol_col="SYMBOL",
            datetime_col="DATETIME",
            device=None,
        )
    )

    print("\nNumber of windows:", len(window_metrics))
    print("\nFinal metrics:")
    print(final_metrics)

    print("\nFirst predictions:")
    print(predictions.head())

    print("\nPortfolio returns head:")
    print(portfolio_returns.head())

    _save_backtest_outputs(
        model_type=model_type,
        predictions=predictions,
        portfolio_returns=portfolio_returns,
        window_metrics=window_metrics,
        final_metrics=final_metrics,
        output_dir=output_dir,
    )

    return final_metrics, predictions


def _run_signal_threshold_experiment(
    model_type,
    predictions,
    target_col,
    output_dir,
):
    threshold_analysis = run_signal_threshold_analysis(
        df=predictions,
        thresholds=(0.0, 0.0005, 0.001, 0.002),
        prediction_col="SIGNAL_SCORE",
        target_col=target_col,
        time_col="DATETIME",
        symbol_col="SYMBOL",
        long_quantile=0.90,
        short_quantile=0.10,
        cost_per_unit_turnover=0.001,
        rebalance_frequency="10min",
        trading_days_per_year=252,
    )

    output_name = f"signal_threshold_analysis_{model_type}.csv"
    threshold_analysis.to_csv(output_dir / output_name, index=False)

    print(f"\nSignal threshold analysis for {model_type}:")
    print(threshold_analysis)

    return threshold_analysis


def main():
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    folder_path = "data/demand_minute10_wct_202112"

    df_clean, preprocessing_info = preprocess_intraday_folder(
        folder_path=folder_path,
        min_timestamps=30,
        lower_q=0.01,
        upper_q=0.99,
        target_horizon=1,
    )

    print("Preprocessing completed.")
    print(preprocessing_info)

    df_model, feature_cols, target_col = build_feature_dataset(
        df_clean,
        n_lags=3,
        rolling_windows=(3, 5),
        drop_missing=True,
    )

    print("\nFeature engineering completed.")
    print("Full model dataset shape:", df_model.shape)

    # top_symbols = (
    #     df_model.groupby("SYMBOL")
    #     .size()
    #     .sort_values(ascending=False)
    #     .head(500)
    #     .index
    # )
    # df_model = df_model[df_model["SYMBOL"].isin(top_symbols)].copy()

    # print("\nUsing 500 symbols for the walk-forward test.")
    # print("Reduced dataset shape:", df_model.shape)
    # print("Number of symbols:", df_model["SYMBOL"].nunique())

    comparison_rows = []
    for model_type in ("lstm", "attention_lstm"):
        final_metrics, predictions = _run_one_model(
            model_type=model_type,
            df_model=df_model,
            feature_cols=feature_cols,
            target_col=target_col,
            output_dir=output_dir,
        )

        _run_signal_threshold_experiment(
            model_type=model_type,
            predictions=predictions,
            target_col=target_col,
            output_dir=output_dir,
        )

        comparison_rows.append({
            "model_type": model_type,
            "cumulative_return": final_metrics.get("cumulative_return"),
            "mean_return": final_metrics.get("mean_return"),
            "volatility": final_metrics.get("volatility"),
            "sharpe_ratio": final_metrics.get("sharpe_ratio"),
            "max_drawdown": final_metrics.get("max_drawdown"),
            "hit_ratio": final_metrics.get("hit_ratio"),
            "average_turnover": final_metrics.get("average_turnover"),
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output_dir / "walk_forward_model_comparison.csv",
        index=False,
    )

    print("\nWalk-forward model comparison:")
    print(comparison)


if __name__ == "__main__":
    main()
