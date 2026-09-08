import pandas as pd

from src.preprocessing.preprocessing import preprocess_intraday_folder
from src.preprocessing.features import build_feature_dataset
from src.preprocessing.split import chronological_train_test_split

from src.models.deep_learning import tune_lstm_regressor


def main():
    """
    We tune the plain LSTM on a smaller but realistic slice of the data.

    The goal here is not to train the final model. We just want a reasonable
    hyperparameter search that is fast enough to run and informative enough to
    guide the full-data experiment.
    """

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
    print("Number of features:", len(feature_cols))
    print("Return target column:", target_col)

    # We tune on a realistic subset first. Full-data tuning would multiply the
    # cost of every hyperparameter combination and would be too slow here.
    n_symbols_for_tuning = 500

    # We take the first symbols after preprocessing so the script stays simple
    # and reproducible.
    selected_symbols = df_model["SYMBOL"].unique()[:n_symbols_for_tuning]

    df_tuning = df_model[
        df_model["SYMBOL"].isin(selected_symbols)
    ].copy()

    print("\nTuning dataset created.")
    print("Number of selected symbols:", len(selected_symbols))
    print("Tuning dataset shape:", df_tuning.shape)


    train_df, validation_df, test_df = chronological_train_test_split(
        df_tuning,
        train_size=0.70,
        validation_size=0.15,
        datetime_col="DATETIME"
    )

    print("\nSplit sizes for tuning:")
    print("Train:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Test:", test_df.shape)

    print("\nStarting continuous LSTM hyperparameter tuning...")

    tuning_results = tune_lstm_regressor(
        train_df=train_df,
        validation_df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,

        sequence_lengths=(12, 24, 36),
        hidden_sizes=(32, 64, 128),
        num_layers_list=(1,),
        dropouts=(0.10, 0.15, 0.20),
        learning_rates=(3e-4, 5e-4),
        weight_decays=(1e-5,),

        epochs=3,
        batch_size=2048,
        patience=2,
        scale_data=True,
        device=None
    )

    print("\nHyperparameter tuning completed.")
    print("\nBest configurations:")
    print(tuning_results.head(10))


    output_path = "models/lstm_regression_hypertuning_results.csv"
    tuning_results.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")

    best_config = tuning_results.iloc[0]

    print("\nBest configuration selected:")
    print(best_config)

    print("\nUse this best configuration later for full-data training:")
    print(f"sequence_length = {int(best_config['sequence_length'])}")
    print(f"hidden_size = {int(best_config['hidden_size'])}")
    print(f"num_layers = {int(best_config['num_layers'])}")
    print(f"dropout = {best_config['dropout']}")
    print(f"learning_rate = {best_config['learning_rate']}")
    print(f"weight_decay = {best_config['weight_decay']}")


if __name__ == "__main__":
    main()
