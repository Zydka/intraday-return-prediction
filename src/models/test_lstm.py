import numpy as np

from src.preprocessing.preprocessing import preprocess_intraday_folder
from src.preprocessing.features import build_feature_dataset
from src.preprocessing.split import chronological_train_test_split

from src.models.deep_learning import (
    validate_lstm_regressor,
    predict_lstm_model_from_dataframe,
    evaluate_regression_predictions
)


def main():
    """
    We train and evaluate the plain LSTM on the full prepared dataset.

    This script is more like an end-to-end smoke test than a tiny unit test: it
    checks that preprocessing, feature creation, training, prediction, metrics,
    and output saving all work together on the real data.
    """

    folder_path = "data/demand_minute10_wct_202112"

    # load data
    df_clean, preprocessing_info = preprocess_intraday_folder(
        folder_path=folder_path,
        min_timestamps=30,
        lower_q=0.01,
        upper_q=0.99,
        target_horizon=1
    )

    print("Preprocessing completed.")
    print(preprocessing_info)

    # add features
    df_model, feature_cols, target_col = build_feature_dataset(
        df_clean,
        n_lags=3,
        rolling_windows=(3, 5),
        drop_missing=True
    )

    print("\nFeature engineering completed.")
    print("Full model dataset shape:", df_model.shape)
    print("Number of features:", len(feature_cols))
    print("Return target column:", target_col)

    print("\nTraining continuous LSTM regression model on FULL data...")
    df_model = df_model[df_model["SYMBOL"].isin(df_model["SYMBOL"].unique()[:100])].copy()


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

    # These settings come from the LSTM tuning run and are used here for the
    # full-data training 
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

    print("\nTraining history:")
    print("Train loss:", history["train_loss"])
    print("Validation loss:", history["validation_loss"])

    print("\nFirst validation predictions:")
    print(validation_results.head())

    print("\nPrediction summary:")
    print(validation_results["PREDICTION"].describe())

    print("\nTarget summary:")
    print(validation_results[target_col].describe())


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
    print(
        (
            np.sign(validation_results[target_col])
            == np.sign(validation_results["PREDICTION"])
        ).mean()
    )

    print("\nSignal score summary:")
    print(validation_results["SIGNAL_SCORE"].describe())

    # We evaluate on the untouched test period only after training and validation
    # are finished.
    test_results = predict_lstm_model_from_dataframe(
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
    print(
        (
            np.sign(test_results[target_col])
            == np.sign(test_results["PREDICTION"])
        ).mean()
    )

    print("\nTest signal score summary:")
    print(test_results["SIGNAL_SCORE"].describe())

    test_output_path = "models/lstm_full_test_predictions.csv"
    test_results.to_csv(test_output_path, index=False)

    print(f"\nTest predictions saved to: {test_output_path}")

    # We keep a few direct checks at the end so a failed run is obvious: the
    # model must produce non-empty prediction frames with usable signal columns.
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

    print("\nFull-data continuous LSTM regression test passed successfully.")


if __name__ == "__main__":
    main()
