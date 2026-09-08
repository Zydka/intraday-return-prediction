from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


class LSTMRegressionDataset(Dataset):
    """
    We use this small wrapper to give PyTorch the data format it expects.

    By the time we reach this class, the feature windows are already prepared:
    X has one sequence per row, and y has one return target per sequence. Here
    we mainly convert everything to tensors so DataLoader can batch it during
    training and prediction.
    """

    def __init__(self, X, y):
        # We keep both inputs and targets as float tensors. The reshape gives y
        # the same (batch_size, 1) shape as the model output.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """
    We use this as the baseline LSTM model for future-return prediction.

    The model reads a short history of market features for one stock. We keep
    the final hidden representation, apply dropout, and map it to one number:
    the predicted return. It stays deliberately simple so we can compare it
    cleanly with the attention version later in the file.
    """

    def __init__(
        self,
        n_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.20
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Here we ask the last timestep to summarize the whole input window.
        # Later, the attention model gives us a less rigid way to do this.
        last_hidden = lstm_out[:, -1, :]

        out = self.dropout(last_hidden)
        out = self.output_layer(out)

        return out


def build_lstm_model(
    n_features,
    hidden_size=64,
    num_layers=1,
    dropout=0.20,
    seed=42
):
    """
    We create a seeded LSTM model with the requested architecture.

    This keeps model construction in one place and makes tuning code easier to
    read, since every configuration goes through the same setup.
    """

    set_seed(seed)

    return LSTMRegressor(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )


def set_seed(seed=42):
    """
    We seed the random number generators used in this module.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_lstm_sequences(
    df,
    feature_cols,
    target_col="TARGET_RETURN",
    sequence_length=20,
    symbol_col="SYMBOL",
    datetime_col="DATETIME"
):
    """
    We turn the panel DataFrame into rolling sequences for the LSTM.

    For each stock, we walk through time and build windows like:
        t-sequence_length+1, ..., t

    Each window ends on the date we want to predict. We also keep the symbol and
    timestamp so we can join predictions back to the original panel later.
    """

    df = df.copy()
    df = df.sort_values([symbol_col, datetime_col])

    X_list = []
    y_list = []
    index_rows = []

    for symbol, symbol_df in tqdm(
        df.groupby(symbol_col),
        desc="Creating LSTM sequences",
        leave=False
    ):
        # We build windows stock by stock so the model never sees a sequence
        # that jumps from one company to another.
        symbol_df = symbol_df.sort_values(datetime_col)

        X_values = symbol_df[feature_cols].values
        y_values = symbol_df[target_col].values
        datetimes = symbol_df[datetime_col].values

        if len(symbol_df) < sequence_length:
            # We skip stocks that do not have enough history for one full window.
            continue

        for i in range(sequence_length - 1, len(symbol_df)):
            # We end the window at i and use the target from the same row, so the
            # sample stays aligned with the date we evaluate later.
            X_list.append(X_values[i - sequence_length + 1:i + 1])
            y_list.append(y_values[i])

            index_rows.append({
                symbol_col: symbol,
                datetime_col: datetimes[i]
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    index_info = pd.DataFrame(index_rows)

    return X, y, index_info


def _transform_lstm_tensor_in_chunks(X, scaler, chunk_size=1_000_000):
    """
    We apply the already-fitted scaler to LSTM sequences while keeping memory
    under control.

    Our LSTM data has shape:
        (samples, timesteps, features)

    but StandardScaler can only scale 2D data:
        (rows, features)

    For large datasets, scaling everything at once can use too much RAM. Here
    we flatten the tensor, scale it in smaller chunks, then restore the original
    LSTM shape.
    """

    n_obs, seq_len, n_features = X.shape

    # We flatten time temporarily because StandardScaler works on 2D arrays.
    X_2d = X.reshape(-1, n_features)

    for start in range(0, len(X_2d), chunk_size):
        # We scale by chunks so memory use stays predictable on long stock panels.
        end = min(start + chunk_size, len(X_2d))
        X_2d[start:end] = scaler.transform(X_2d[start:end])

    return X_2d.reshape(n_obs, seq_len, n_features)


def scale_lstm_data(X_train, X_validation=None, X_test=None, chunk_size=1_000_000):
    """
    We standardize features using only the training data.

    The scaler is fitted on train sequences only, then reused for validation and
    test data. We do this to avoid leakage: even something as simple as a feature
    mean would give away information if it used validation or test rows.
    """

    n_train, seq_len, n_features = X_train.shape

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)

    for start in range(0, len(X_train_2d), chunk_size):
        # We use partial_fit so we do not need every timestep-feature row in
        # memory at once.
        end = min(start + chunk_size, len(X_train_2d))
        scaler.partial_fit(X_train_2d[start:end])

    X_train_scaled = _transform_lstm_tensor_in_chunks(
        X_train,
        scaler,
        chunk_size=chunk_size
    )

    outputs = [X_train_scaled, scaler]

    if X_validation is not None:
        X_val_scaled = _transform_lstm_tensor_in_chunks(
            X_validation,
            scaler,
            chunk_size=chunk_size
        )
        outputs.append(X_val_scaled)

    if X_test is not None:
        X_test_scaled = _transform_lstm_tensor_in_chunks(
            X_test,
            scaler,
            chunk_size=chunk_size
        )
        outputs.append(X_test_scaled)

    return tuple(outputs)


def train_lstm_regressor(
    model,
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=None,
    device=None,
    verbose=True
):
    """
    We train the LSTM to predict continuous future returns.

    The target is a numeric return, so we use MSE loss. When validation data is
    available, we also keep the best validation checkpoint and stop early if the
    model stops improving for several epochs.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("Using device:", device)

    model = model.to(device)

    train_dataset = LSTMRegressionDataset(X_train, y_train)
    # We shuffle training sequences so each mini-batch is less tied to one
    # period or one symbol. Validation stays ordered because we only measure it.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if X_validation is not None and y_validation is not None:
        validation_dataset = LSTMRegressionDataset(X_validation, y_validation)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        validation_loader = None

    criterion = nn.MSELoss()

    # We use AdamW as a stable default optimizer for this regression model.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    best_validation_loss = np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        # We switch to training mode so dropout is active and weights can update.
        model.train()
        train_losses = []

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} - training",
            leave=False
        )

        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # We backpropagate the loss, then let the optimizer update the model.
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_bar.set_postfix(loss=np.mean(train_losses))

        mean_train_loss = float(np.mean(train_losses))
        history["train_loss"].append(mean_train_loss)

        if validation_loader is not None:
            # We switch to eval mode so validation behaves like real inference.
            model.eval()
            validation_losses = []

            validation_bar = tqdm(
                validation_loader,
                desc=f"Epoch {epoch}/{epochs} - validation",
                leave=False
            )

            with torch.no_grad():
                for X_batch, y_batch in validation_bar:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                    validation_losses.append(loss.item())
                    validation_bar.set_postfix(loss=np.mean(validation_losses))

            mean_validation_loss = float(np.mean(validation_losses))
            history["validation_loss"].append(mean_validation_loss)

            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"train loss: {mean_train_loss:.8f} | "
                    f"validation loss: {mean_validation_loss:.8f}"
                )

            if mean_validation_loss < best_validation_loss:
                # We keep a CPU copy of the best model so we can restore it even
                # if later epochs start to overfit.
                best_validation_loss = mean_validation_loss
                best_state = {
                    key: value.cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None:
                if epochs_without_improvement >= patience:
                    # If validation has not improved for a while, we stop before
                    # the model spends more time overfitting.
                    if verbose:
                        print(f"Early stopping at epoch {epoch}.")
                    break

        else:
            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"train loss: {mean_train_loss:.8f}"
                )

    if best_state is not None:
        # We return the best validation model, not just the final epoch.
        model.load_state_dict(best_state)

    return model, history


def predict_lstm_regressor(
    model,
    X,
    batch_size=512,
    device=None
):
    """
    We predict continuous future returns from an already prepared LSTM tensor.

    This mirrors the training DataLoader path, but we switch the model to eval
    mode and disable gradient tracking. That keeps prediction faster and avoids
    storing computation graphs we do not need.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # We reuse the same Dataset class even though targets are not needed here.
    # The dummy array keeps batching code identical to training.
    dummy_y = np.zeros(len(X), dtype=np.float32)

    dataset = LSTMRegressionDataset(X, dummy_y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_predictions = []

    prediction_bar = tqdm(
        loader,
        desc="Predicting",
        leave=False
    )

    with torch.no_grad():
        for X_batch, _ in prediction_bar:
            X_batch = X_batch.to(device)

            predictions = model(X_batch)
            all_predictions.append(predictions.cpu().numpy())

    predictions = np.vstack(all_predictions).reshape(-1)

    return predictions


def fit_lstm_model_from_dataframes(
    model,
    train_df,
    validation_df,
    feature_cols,
    target_col="TARGET_RETURN",
    sequence_length=20,
    batch_size=256,
    epochs=20,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=5,
    scale_data=True,
    symbol_col="SYMBOL",
    datetime_col="DATETIME",
    device=None,
    verbose=True
):
    """
    We fit an LSTM regression model directly from train and validation DataFrames.

    This is the main high-level training helper. We turn raw panel data into
    LSTM windows, optionally standardize the features, train the model, and
    store the preprocessing metadata we need for future predictions.
    """

    X_train, y_train, _ = create_lstm_sequences(
        df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        symbol_col=symbol_col,
        datetime_col=datetime_col
    )

    X_validation, y_validation, _ = create_lstm_sequences(
        df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        symbol_col=symbol_col,
        datetime_col=datetime_col
    )

    if scale_data:
        # We attach the fitted scaler to the model so prediction uses the same
        # feature normalization as training.
        X_train, scaler, X_validation = scale_lstm_data(
            X_train=X_train,
            X_validation=X_validation
        )
        model.scaler = scaler
    else:
        model.scaler = None

    model.sequence_length = sequence_length
    # We save these choices on the model so prediction does not have to repeat
    # every training-time column name and sequence setting.
    model.feature_cols = feature_cols
    model.target_col = target_col
    model.symbol_col = symbol_col
    model.datetime_col = datetime_col

    model, history = train_lstm_regressor(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        verbose=verbose
    )

    model.history = history

    return model


def predict_lstm_model_from_dataframe(
    model,
    df,
    feature_cols=None,
    target_col=None,
    sequence_length=None,
    batch_size=512,
    scale_data=True,
    symbol_col=None,
    datetime_col=None,
    device=None
):
    """
    We predict continuous returns from a DataFrame and return a results DataFrame.

    The returned frame includes the original target, the prediction, and a
    SIGNAL_SCORE column. We keep SIGNAL_SCORE equal to the prediction so the
    portfolio code can rank stocks without caring which model produced the
    signal.
    """

    if feature_cols is None:
        # We fall back to the settings saved during training unless the caller
        # explicitly overrides them.
        feature_cols = model.feature_cols

    if target_col is None:
        target_col = model.target_col

    if sequence_length is None:
        sequence_length = model.sequence_length

    if symbol_col is None:
        symbol_col = model.symbol_col

    if datetime_col is None:
        datetime_col = model.datetime_col

    X, y, index_info = create_lstm_sequences(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        symbol_col=symbol_col,
        datetime_col=datetime_col
    )

    if scale_data and getattr(model, "scaler", None) is not None:
        # We reuse the training scaler. Refitting here would leak information
        # from the prediction period and make results look better than they are.
        X = _transform_lstm_tensor_in_chunks(X, model.scaler)

    predictions = predict_lstm_regressor(
        model=model,
        X=X,
        batch_size=batch_size,
        device=device
    )

    results = index_info.copy()
    results[target_col] = y
    results["PREDICTION"] = predictions

    # We keep a model-agnostic signal column for long-short portfolio ranking.
    results["SIGNAL_SCORE"] = results["PREDICTION"]

    return results


def validate_lstm_regressor(
    train_df,
    validation_df,
    feature_cols,
    target_col="TARGET_RETURN",
    sequence_length=20,
    hidden_size=64,
    num_layers=1,
    dropout=0.20,
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=5,
    scale_data=True,
    seed=42,
    device=None
):
    """
    We run the full train / validation workflow for return prediction.

    This function is mostly orchestration: we build a plain LSTM, fit it on the
    training DataFrame, then run it on validation data so we can inspect both
    the loss history and the out-of-sample predictions.
    """

    model = build_lstm_model(
        n_features=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed
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
        device=device,
        verbose=True
    )

    validation_results = predict_lstm_model_from_dataframe(
        model=model,
        df=validation_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        batch_size=batch_size,
        scale_data=scale_data,
        device=device
    )

    return model, model.history, validation_results


def evaluate_regression_predictions(
    y_true,
    y_pred
):
    """
    We compute simple regression metrics for return prediction.

    MSE, RMSE and MAE measure the size of the error. We also look at correlation
    to see whether predictions move with the target, and directional accuracy to
    see how often we get the sign right.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        # We cannot compute a meaningful correlation if either side is constant.
        correlation = np.nan

    directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "directional_accuracy": directional_accuracy,
        "prediction_mean": np.mean(y_pred),
        "prediction_std": np.std(y_pred),
        "target_mean": np.mean(y_true),
        "target_std": np.std(y_true)
    }


def tune_lstm_regressor(
    train_df,
    validation_df,
    feature_cols,
    target_col="TARGET_RETURN",
    sequence_lengths=(12, 24),
    hidden_sizes=(64, 128),
    num_layers_list=(1, 2),
    dropouts=(0.15, 0.25),
    learning_rates=(5e-4,),
    weight_decays=(1e-5,),
    epochs=3,
    batch_size=1024,
    patience=None,
    scale_data=True,
    device=None
):
    """
    We run a small grid search for the plain LSTM regressor.

    The default grid is intentionally modest because each combination trains a
    real neural network. We return one DataFrame with the validation metrics so
    the best configuration is easy to inspect or reuse.
    """

    tuning_results = []

    grid = list(product(
        sequence_lengths,
        hidden_sizes,
        num_layers_list,
        dropouts,
        learning_rates,
        weight_decays
    ))

    # We loop through the grid explicitly so each configuration is logged and
    # evaluated with the same validation workflow.
    for i, (
        sequence_length,
        hidden_size,
        num_layers,
        dropout,
        learning_rate,
        weight_decay
    ) in enumerate(tqdm(grid, desc="LSTM regression tuning"), start=1):

        print("\n" + "=" * 70)
        print(f"Model {i}/{len(grid)}")
        print(
            f"sequence_length={sequence_length}, "
            f"hidden_size={hidden_size}, "
            f"num_layers={num_layers}, "
            f"dropout={dropout}, "
            f"learning_rate={learning_rate}, "
            f"weight_decay={weight_decay}"
        )

        model, history, results = validate_lstm_regressor(
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
            device=device
        )

        metrics = evaluate_regression_predictions(
            y_true=results[target_col],
            y_pred=results["PREDICTION"]
        )

        tuning_results.append({
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_validation_loss": min(history["validation_loss"]),
            "final_validation_loss": history["validation_loss"][-1],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "correlation": metrics["correlation"],
            "directional_accuracy": metrics["directional_accuracy"],
            "prediction_mean": metrics["prediction_mean"],
            "prediction_std": metrics["prediction_std"],
            "target_mean": metrics["target_mean"],
            "target_std": metrics["target_std"]
        })

    tuning_results = pd.DataFrame(tuning_results)

    tuning_results = tuning_results.sort_values(
        by=["correlation", "directional_accuracy"],
        ascending=False
    ).reset_index(drop=True)

    return tuning_results


# ============================================================
# Attention-LSTM extension
# ============================================================


class AttentionLayer(nn.Module):
    """
    We use this attention layer to decide which timesteps matter most.

    The layer learns one score per timestep. After softmax, those scores become
    weights that tell us how much each timestep contributes to the final context
    vector. This lets the model look back across the whole window instead of
    relying only on the final LSTM output.
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.attention_score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # We score each timestep separately. Shape is still batch x time x 1.
        attention_scores = self.attention_score(lstm_outputs)

        # We apply softmax over time, so the weights for each sequence sum to 1.
        attention_weights = torch.softmax(attention_scores, dim=1)

        # We combine all hidden states into one vector using the learned weights.
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)

        return context_vector, attention_weights


class AttentionLSTMRegressor(nn.Module):
    """
    We extend the plain LSTM with a learned attention pooling step.

    The first half is the same recurrent encoder as LSTMRegressor. The difference
    is how we summarize the sequence: instead of taking only the last hidden
    state, we let the attention layer decide which timesteps matter most.
    """

    def __init__(
        self,
        n_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.20
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = AttentionLayer(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        lstm_outputs, _ = self.lstm(x)

        # We keep both the pooled representation and the weights, because the
        # weights are useful later if we want to inspect what dates mattered.
        context_vector, attention_weights = self.attention(lstm_outputs)

        out = self.dropout(context_vector)
        out = self.output_layer(out)

        if return_attention:
            return out, attention_weights

        return out


def build_attention_lstm_model(
    n_features,
    hidden_size=64,
    num_layers=1,
    dropout=0.20,
    seed=42
):
    """
    We create a seeded Attention-LSTM model.

    This mirrors build_lstm_model so we can swap between the plain and
    attention models without changing the rest of the training pipeline.
    """

    set_seed(seed)

    return AttentionLSTMRegressor(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )


def extract_attention_weights(
    model,
    X,
    batch_size=512,
    device=None
):
    """
    Extract attention weights for each LSTM sequence.

    The output has shape:
        number of sequences x sequence length

    These weights are mainly for interpretation. For each prediction, they show
    which timesteps in the input window the model focused on most.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # We use the same batching trick as prediction: targets are unused, but the
    # Dataset keeps the input pipeline consistent.
    dummy_y = np.zeros(len(X), dtype=np.float32)

    dataset = LSTMRegressionDataset(X, dummy_y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_attention_weights = []

    attention_bar = tqdm(
        loader,
        desc="Extracting attention weights",
        leave=False
    )

    with torch.no_grad():
        for X_batch, _ in attention_bar:
            X_batch = X_batch.to(device)

            _, attention_weights = model(
                X_batch,
                return_attention=True
            )

            # We remove the final singleton dimension: batch x time x 1 -> batch x time.
            attention_weights = attention_weights.squeeze(-1)
            all_attention_weights.append(attention_weights.cpu().numpy())

    attention_weights = np.vstack(all_attention_weights)

    return attention_weights



def tune_attention_lstm_regressor(
    train_df,
    validation_df,
    feature_cols,
    target_col="TARGET_RETURN",
    sequence_lengths=(12, 24),
    hidden_sizes=(64, 128),
    num_layers_list=(1, 2),
    dropouts=(0.15, 0.25),
    learning_rates=(5e-4,),
    weight_decays=(1e-5,),
    epochs=3,
    batch_size=1024,
    patience=None,
    scale_data=True,
    device=None
):
    """
    We run a grid search for the Attention-LSTM regressor.

    This follows the same idea as tune_lstm_regressor, but we build the
    attention variant for every hyperparameter combination. We keep the output
    columns similar so plain LSTM and Attention-LSTM runs are easy to compare.
    """

    tuning_results = []

    grid = list(product(
        sequence_lengths,
        hidden_sizes,
        num_layers_list,
        dropouts,
        learning_rates,
        weight_decays
    ))

    for i, (
        sequence_length,
        hidden_size,
        num_layers,
        dropout,
        learning_rate,
        weight_decay
    ) in enumerate(tqdm(grid, desc="Attention-LSTM tuning"), start=1):

        print("\n" + "=" * 70)
        print(f"Model {i}/{len(grid)}")

        print(
            f"sequence_length={sequence_length}, "
            f"hidden_size={hidden_size}, "
            f"num_layers={num_layers}, "
            f"dropout={dropout}, "
            f"learning_rate={learning_rate}, "
            f"weight_decay={weight_decay}"
        )

        model = build_attention_lstm_model(
            n_features=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seed=42
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
            device=device,
            verbose=True
        )

        history = model.history

        validation_results = predict_lstm_model_from_dataframe(
            model=model,
            df=validation_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=sequence_length,
            batch_size=batch_size,
            scale_data=scale_data,
            device=device
        )

        metrics = evaluate_regression_predictions(
            y_true=validation_results[target_col],
            y_pred=validation_results["PREDICTION"]
        )

        tuning_results.append({
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_validation_loss": min(history["validation_loss"]),
            "final_validation_loss": history["validation_loss"][-1],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "correlation": metrics["correlation"],
            "directional_accuracy": metrics["directional_accuracy"]
        })

    tuning_results = pd.DataFrame(tuning_results)

    tuning_results = tuning_results.sort_values(
        by=["correlation", "directional_accuracy"],
        ascending=False
    ).reset_index(drop=True)

    return tuning_results
