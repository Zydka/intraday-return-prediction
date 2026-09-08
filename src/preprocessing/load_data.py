import json
import pandas as pd

from src.preprocessing.preprocessing import preprocess_intraday_folder
from src.preprocessing.features import build_feature_dataset


def main():

    folder_path = "data/demand_minute10_wct_202112"

    print("Loading and preprocessing raw data...")

    df_clean, preprocessing_info = preprocess_intraday_folder(
        folder_path=folder_path,
        min_timestamps=30,
        lower_q=0.01,
        upper_q=0.99,
        target_horizon=1
    )

    print("\nPreprocessing completed.")
    print(preprocessing_info)

    print("\nRunning feature engineering...")

    df_model, feature_cols, target_col = build_feature_dataset(
        df_clean,
        n_lags=3,
        rolling_windows=(3, 5),
        drop_missing=True
    )

    print("\nFeature engineering completed.")
    print("Dataset shape:", df_model.shape)
    print("Number of features:", len(feature_cols))

    data_output_path = "data/ready_data.csv.gz"

    print(f"\nSaving dataset to: {data_output_path}")

    df_model.to_csv(
        data_output_path,
        index=False,
        compression="gzip"
    )

    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col
    }

    metadata_output_path = "data/ready_data_metadata.json"

    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_output_path}")

    print("\nDone.")
    print("Ready-to-train dataset saved successfully.")


if __name__ == "__main__":
    main()