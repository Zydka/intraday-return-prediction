import numpy as np
import pandas as pd

from src.preprocessing.preprocessing import preprocess_intraday_folder
from src.preprocessing.features import build_feature_dataset
from src.preprocessing.split import chronological_train_test_split

from src.models.deep_learning import (
    validate_lstm_regressor,
    predict_lstm_model_from_dataframe,
    evaluate_regression_predictions
)

from src.trading.portfolio_construction import build_long_short_backtest
from src.trading.transaction_cost_analysis import (
    run_cost_sensitivity_analysis,
    run_rebalance_frequency_analysis,
    run_quantile_analysis,
)


def main():

    folder_path = "data/demand_minute10_wct_202112"

    df_clean, preprocessing_info = preprocess_intraday_folder(
        folder_path=folder_path,
        min_timestamps=30,
        lower_q=0.01,
        upper_q=0.99,
        target_horizon=1
    )

    print("Preprocessing completed.")
    print(preprocessing_info)

    df_model, feature_cols, target_col = build_feature_dataset(
        df_clean,
        n_lags=3,
        rolling_windows=(3, 5),
        drop_missing=True
    )

    print("\nFeature engineering completed.")
    print("Full model dataset shape:", df_model.shape)

    # ----------------------------------------------------
    # Keep only 500 symbols for a first portfolio test
    # ----------------------------------------------------
    top_symbols = (
        df_model.groupby("SYMBOL")
        .size()
        .sort_values(ascending=False)
        .head(500)
        .index
    )

    df_model = df_model[df_model["SYMBOL"].isin(top_symbols)].copy()

    print("\nUsing 500 symbols only.")
    print("Reduced dataset shape:", df_model.shape)
    print("Number of symbols:", df_model["SYMBOL"].nunique())

    # ----------------------------------------------------
    # Chronological split
    # ----------------------------------------------------
    train_df, validation_df, test_df = chronological_train_test_split(
        df_model,
        train_size=0.70,
        validation_size=0.15,
        datetime_col="DATETIME"
    )

    print("\nSplit sizes:")
    print("Train:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Test:", test_df.shape)

    # ----------------------------------------------------
    # Train LSTM
    # ----------------------------------------------------
    model, history, validation_results = validate_lstm_regressor(
        train_df=train_df,
        validation_df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=36,
        hidden_size=128,
        num_layers=1,
        dropout=0.20,
        epochs=5,
        batch_size=4096,
        learning_rate=5e-4,
        weight_decay=1e-5,
        patience=3,
        scale_data=True,
        seed=42,
        device=None
    )

    print("\nLSTM training completed.")
    print("Train loss:", history["train_loss"])
    print("Validation loss:", history["validation_loss"])

    # ----------------------------------------------------
    # Predict on test set
    # ----------------------------------------------------
    test_predictions = predict_lstm_model_from_dataframe(
        model=model,
        df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=36,
        batch_size=4096,
        scale_data=True,
        device=None
    )

    print("\nFirst test predictions:")
    print(test_predictions.head())

    regression_metrics = evaluate_regression_predictions(
        y_true=test_predictions[target_col],
        y_pred=test_predictions["PREDICTION"]
    )

    print("\nTest regression metrics:")
    for key, value in regression_metrics.items():
        print(f"{key}: {value}")

    # ----------------------------------------------------
    # Portfolio construction using LSTM signal score
    # ----------------------------------------------------
    df_bt, portfolio_returns, portfolio_metrics = build_long_short_backtest(
        df=test_predictions,
        prediction_col="SIGNAL_SCORE",
        target_col=target_col,
        time_col="DATETIME",
        symbol_col="SYMBOL",
        long_quantile=0.90,
        short_quantile=0.10,
        cost_per_unit_turnover=0.001,
        periods_per_year=252 * 38
    )

    print("\nPortfolio returns:")
    print(portfolio_returns.head())

    print("\nPortfolio performance metrics:")
    print(portfolio_metrics)

    print("Addional info:")
    print(portfolio_returns.shape)
    print(portfolio_returns.index.min())
    print(portfolio_returns.index.max())
    print(portfolio_returns["net_return"].describe())
    print(portfolio_returns["turnover"].describe())

    # ----------------------------------------------------
    # Transaction-cost and portfolio realism analysis
    # ----------------------------------------------------
    cost_sensitivity = run_cost_sensitivity_analysis(
        df=test_predictions,
        transaction_costs=(0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01),
        prediction_col="SIGNAL_SCORE",
        target_col=target_col,
        time_col="DATETIME",
        symbol_col="SYMBOL",
        long_quantile=0.90,
        short_quantile=0.10,
        rebalance_frequency="10min"
    )

    rebalance_frequency_analysis = run_rebalance_frequency_analysis(
        df=test_predictions,
        rebalance_frequencies=("10min", "30min", "60min"),
        prediction_col="SIGNAL_SCORE",
        target_col=target_col,
        time_col="DATETIME",
        symbol_col="SYMBOL",
        long_quantile=0.90,
        short_quantile=0.10,
        cost_per_unit_turnover=0.005
    )

    quantile_analysis = run_quantile_analysis(
        df=test_predictions,
        quantiles=(0.05, 0.10, 0.20),
        prediction_col="SIGNAL_SCORE",
        target_col=target_col,
        time_col="DATETIME",
        symbol_col="SYMBOL",
        cost_per_unit_turnover=0.005,
        rebalance_frequency="10min"
    )

    cost_sensitivity.to_csv(
        "models/transaction_cost_sensitivity.csv",
        index=False
    )
    rebalance_frequency_analysis.to_csv(
        "models/rebalance_frequency_analysis.csv",
        index=False
    )
    quantile_analysis.to_csv(
        "models/portfolio_quantile_analysis.csv",
        index=False
    )

    print("\nTransaction-cost sensitivity analysis:")
    print(cost_sensitivity)

    print("\nRebalance frequency analysis:")
    print(rebalance_frequency_analysis)

    print("\nPortfolio quantile analysis:")
    print(quantile_analysis)






if __name__ == "__main__":
    main()
