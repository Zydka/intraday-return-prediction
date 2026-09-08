import numpy as np
import pandas as pd


def validate_tabular_model(
    model,
    train_df,
    validation_df,
    feature_cols,
    target_col
):
    """
    Train and validate a tabular model.

    This is for models such as:
    - Linear Regression
    - Ridge Regression
    - XGBoost
    - LightGBM
    """

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_validation = validation_df[feature_cols]
    y_validation = validation_df[target_col]

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    results = validation_df[["SYMBOL", "DATETIME", target_col]].copy()
    results["PREDICTION"] = predictions

    return model, results


def walk_forward_validate_tabular_model(
    model_builder,
    splitter,
    feature_cols,
    target_col
):
    """
    Walk-forward validation for tabular models.

    model_builder must be a function that returns a fresh model.

    Example:
    def model_builder():
        return LinearRegression()
    """

    all_results = []

    for fold, (train_df, test_df) in enumerate(splitter, start=1):

        print(f"Running fold {fold}")

        model = model_builder()

        fitted_model, fold_results = validate_tabular_model(
            model=model,
            train_df=train_df,
            validation_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col
        )

        fold_results["FOLD"] = fold
        all_results.append(fold_results)

    results = pd.concat(all_results, ignore_index=True)

    return results


def validate_lstm_model(
    model,
    train_data,
    validation_data,
    fit_function,
    predict_function
):
    """
    Train and validate an LSTM or deep learning model.

    Parameters
    ----------
    model :
        Deep learning model.

    train_data : tuple
        Usually (X_train, y_train).

    validation_data : tuple
        Usually (X_validation, y_validation, validation_index).

    fit_function : callable
        Function that trains the model.

    predict_function : callable
        Function that returns predictions.

    Returns
    -------
    model, results
    """

    X_train, y_train = train_data
    X_validation, y_validation, validation_index = validation_data

    model = fit_function(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    predictions = predict_function(model, X_validation)

    results = validation_index.copy()
    results["TARGET_RETURN"] = y_validation
    results["PREDICTION"] = predictions

    return model, results


def walk_forward_validate_lstm_model(
    model_builder,
    splitter,
    sequence_function,
    feature_cols,
    target_col,
    fit_function,
    predict_function,
    sequence_length=20
):
    """
    Walk-forward validation for LSTM models.

    This function:
    - receives train/test folds from split.py
    - creates LSTM sequences
    - trains a fresh model on each fold
    - stores predictions
    """

    all_results = []

    for fold, (train_df, test_df) in enumerate(splitter, start=1):

        print(f"Running fold {fold}")

        X_train, y_train, _ = sequence_function(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=sequence_length
        )

        X_test, y_test, test_index = sequence_function(
            test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=sequence_length
        )

        model = model_builder()

        fitted_model, fold_results = validate_lstm_model(
            model=model,
            train_data=(X_train, y_train),
            validation_data=(X_test, y_test, test_index),
            fit_function=fit_function,
            predict_function=predict_function
        )

        fold_results["FOLD"] = fold
        all_results.append(fold_results)

    results = pd.concat(all_results, ignore_index=True)

    return results


def train_validation_test_results(
    model,
    train_df,
    validation_df,
    test_df,
    feature_cols,
    target_col
):
    """
    Simple train / validation / test workflow for tabular models.

    This is useful during development before running full walk-forward validation.
    """

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_validation = validation_df[feature_cols]
    y_validation = validation_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    validation_results = validation_df[["SYMBOL", "DATETIME", target_col]].copy()
    validation_results["PREDICTION"] = validation_predictions
    validation_results["SPLIT"] = "validation"

    test_results = test_df[["SYMBOL", "DATETIME", target_col]].copy()
    test_results["PREDICTION"] = test_predictions
    test_results["SPLIT"] = "test"

    results = pd.concat(
        [validation_results, test_results],
        ignore_index=True
    )

    return model, results