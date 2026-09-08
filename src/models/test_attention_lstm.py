import numpy as np

from src.preprocessing.preprocessing import preprocess_intraday_folder
from src.preprocessing.features import build_feature_dataset
from src.preprocessing.split import chronological_train_test_split

from src.models.deep_learning import (
    build_attention_lstm_model,
    fit_lstm_model_from_dataframes,
    predict_lstm_model_from_dataframe,
    evaluate_regression_predictions
)


def main():
    """
    We train and evaluate the Attention-LSTM on the full prepared dataset.

    This script mirrors the plain LSTM test, but uses the attention model so we
    can compare whether learned timestep weighting improves validation/test
    behavior.
    """

    folder_path = "data/demand_minute10_wct_202112"

    # preprocessing
    df_clean, preprocessing_info = preprocess_intraday_folder(
        folder_path=folder_path,
        min_timestamps=30,
        lower_q=0.01,
        upper_q=0.99,
        target_horizon=1
    )

    print("Preprocessing completed.")
    print(preprocessing_info)

    # adding features
    df_model, feature_cols, target_col = build_feature_dataset(
        df_clean,
        n_lags=3,
        rolling_windows=(3, 5),
        drop_missing=True
    )

    print("\nFeature engineering completed.")
    print("Full model dataset shape:", df_model.shape)
    print("Number of symbols:", df_model["SYMBOL"].nunique())
    print("Number of features:", len(feature_cols))
    print("Return target column:", target_col)


    # split
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

    print("\nTraining full-data Attention-LSTM regression model...")

    # We build the attention model
    model = build_attention_lstm_model(
        n_features=len(feature_cols),
        hidden_size=128,
        num_layers=1,
        dropout=0.10,
        seed=42
    )

    # These hyperparameters come from the Attention-LSTM tuning run.
    model = fit_lstm_model_from_dataframes(
        model=model,
        train_df=train_df,
        validation_df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=12,
        batch_size=4096,
        epochs=8,
        learning_rate=5e-4,
        weight_decay=1e-4,
        patience=3,
        scale_data=True,
        symbol_col="SYMBOL",
        datetime_col="DATETIME",
        device=None,
        verbose=True
    )

    history = model.history

    # We predict on validation data first so we can inspect out-of-sample
    # behavior before touching the final test period.
    validation_results = predict_lstm_model_from_dataframe(
        model=model,
        df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=12,
        batch_size=4096,
        scale_data=True,
        symbol_col="SYMBOL",
        datetime_col="DATETIME",
        device=None
    )

    print("\nTraining history:")
    print("Train loss:", history["train_loss"])
    print("Validation loss:", history["validation_loss"])

    print("\nFirst validation predictions:")
    print(validation_results.head())

    print("\nPrediction summary:")
    print(validation_results["PREDICTION"].describe())

    print("\nTarget summary:")
    print(validation_results[target_col].describe())

    # We look at both regression error and signal direction. 
    metrics = evaluate_regression_predictions(
        y_true=validation_results[target_col],
        y_pred=validation_results["PREDICTION"]
    )

    print("\nValidation regression metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nCorrelation:")
    print(validation_results[[target_col, "PREDICTION"]].corr().iloc[0, 1])

    print("\nDirectional accuracy:")
    directional_accuracy = (
        np.sign(validation_results[target_col])
        == np.sign(validation_results["PREDICTION"])
    ).mean()
    print(directional_accuracy)

    print("\nSignal score summary:")
    print(validation_results["SIGNAL_SCORE"].describe())

    validation_output_path = "models/attention_lstm_full_validation_predictions.csv"
    validation_results.to_csv(validation_output_path, index=False)

    print(f"\nValidation predictions saved to: {validation_output_path}")

    # We keep the test period separate until the end.
    test_results = predict_lstm_model_from_dataframe(
        model=model,
        df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=12,
        batch_size=4096,
        scale_data=True,
        symbol_col="SYMBOL",
        datetime_col="DATETIME",
        device=None
    )

    print("\nFirst test predictions:")
    print(test_results.head())

    print("\nTest prediction summary:")
    print(test_results["PREDICTION"].describe())

    print("\nTest target summary:")
    print(test_results[target_col].describe())

    test_metrics = evaluate_regression_predictions(
        y_true=test_results[target_col],
        y_pred=test_results["PREDICTION"]
    )

    print("\nTest regression metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\nTest correlation:")
    print(test_results[[target_col, "PREDICTION"]].corr().iloc[0, 1])

    print("\nTest directional accuracy:")
    test_directional_accuracy = (
        np.sign(test_results[target_col])
        == np.sign(test_results["PREDICTION"])
    ).mean()
    print(test_directional_accuracy)

    print("\nTest signal score summary:")
    print(test_results["SIGNAL_SCORE"].describe())

    test_output_path = "models/attention_lstm_full_test_predictions.csv"
    test_results.to_csv(test_output_path, index=False)

    print(f"\nTest predictions saved to: {test_output_path}")

    # These checks make sure the full pipeline produced usable prediction output
    # instead of silently returning empty frames or NaN signals.
    assert len(validation_results) > 0
    assert len(test_results) > 0
    assert "PREDICTION" in validation_results.columns
    assert "SIGNAL_SCORE" in validation_results.columns
    assert "PREDICTION" in test_results.columns
    assert "SIGNAL_SCORE" in test_results.columns
    assert not np.isnan(validation_results["PREDICTION"]).any()
    assert not np.isnan(validation_results["SIGNAL_SCORE"]).any()
    assert not np.isnan(test_results["PREDICTION"]).any()
    assert not np.isnan(test_results["SIGNAL_SCORE"]).any()

    print("\nFull-data Attention-LSTM regression training completed successfully.")


if __name__ == "__main__":
    main()
