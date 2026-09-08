import numpy as np
import pandas as pd


def chronological_train_test_split(
    df,
    train_size=0.70,
    validation_size=0.15,
    datetime_col="DATETIME"
):
    """
    Split the dataframe chronologically into train, validation and test sets.

    This is used for quick model development and debugging.
    It avoids random shuffling, which would create look-ahead bias.
    """

    df = df.sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    train_end = int(train_size * n)
    validation_end = int((train_size + validation_size) * n)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def walk_forward_split(
    df,
    train_window,
    test_window,
    step_size,
    datetime_col="DATETIME"
):
    """
    Create walk-forward train/test splits.

    This is used for final out-of-sample evaluation.
    The model is trained on a past window, tested on the next future window,
    then the window moves forward.
    """

    df = df.sort_values(datetime_col).reset_index(drop=True)

    start = 0

    while start + train_window + test_window <= len(df):

        train_start = start
        train_end = start + train_window

        test_start = train_end
        test_end = train_end + test_window

        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        yield train_df, test_df

        start += step_size


def create_lstm_sequences(
    df,
    feature_cols,
    target_col,
    sequence_length=20,
    symbol_col="SYMBOL",
    datetime_col="DATETIME"
):
    """
    Convert a dataframe into LSTM sequences.

    The output X has shape:
    (n_samples, sequence_length, n_features)

    The output y has shape:
    (n_samples,)
    """

    X = []
    y = []
    index_info = []

    df = df.sort_values([symbol_col, datetime_col])

    for symbol, group in df.groupby(symbol_col):

        group = group.sort_values(datetime_col)

        features = group[feature_cols].values
        targets = group[target_col].values
        datetimes = group[datetime_col].values

        for i in range(sequence_length, len(group)):

            X.append(features[i - sequence_length:i])
            y.append(targets[i])

            index_info.append({
                symbol_col: symbol,
                datetime_col: datetimes[i]
            })

    X = np.array(X)
    y = np.array(y)
    index_info = pd.DataFrame(index_info)

    return X, y, index_info


def print_split_summary(
    train_df,
    validation_df=None,
    test_df=None,
    datetime_col="DATETIME"
):
    """
    Print the size and time range of each split.
    """

    print("Split summary")
    print("-" * 40)

    print(f"Train observations: {len(train_df):,}")
    print(
        f"Train period: "
        f"{train_df[datetime_col].min()} -> {train_df[datetime_col].max()}"
    )

    if validation_df is not None:
        print(f"Validation observations: {len(validation_df):,}")
        print(
            f"Validation period: "
            f"{validation_df[datetime_col].min()} -> "
            f"{validation_df[datetime_col].max()}"
        )

    if test_df is not None:
        print(f"Test observations: {len(test_df):,}")
        print(
            f"Test period: "
            f"{test_df[datetime_col].min()} -> {test_df[datetime_col].max()}"
        )

    print("-" * 40)