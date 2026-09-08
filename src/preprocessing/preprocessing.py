from pathlib import Path
import pandas as pd
import numpy as np


def load_intraday_file(file_path):
    """Load one compressed intraday CSV file."""
    return pd.read_csv(file_path)

def load_intraday_folder(folder_path):
    """Load all compressed intraday CSV files from a folder."""
    folder_path = Path(folder_path)

    files = sorted(folder_path.glob("*.csv.gz"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .csv.gz files found in {folder_path}")

    df_list = []

    for file in files:
        print(f"Loading {file.name}")
        df_list.append(pd.read_csv(file))

    return pd.concat(df_list, ignore_index=True)


def add_datetime_column(df):
    """Build a proper datetime column from DATE and TIME."""
    df = df.copy()
    df["DATETIME"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"])
    return df


def filter_active_symbols(df, min_timestamps=30):
    """Keep symbols with enough intraday observations."""
    df = df.copy()

    times_per_symbol = df.groupby("SYMBOL")["TIME"].nunique()
    valid_symbols = times_per_symbol[times_per_symbol >= min_timestamps].index

    return df[df["SYMBOL"].isin(valid_symbols)].copy()


def sort_intraday_panel(df):
    """Sort the panel by symbol and time."""
    return df.sort_values(["SYMBOL", "DATETIME"]).reset_index(drop=True)


def compute_log_returns(df, price_col="MID_OPEN", break_on_day=True):
    """Compute log prices and log returns for each symbol.

    A return is only defined between two CONSECUTIVE bars of the SAME trading
    day. Using a plain groupby diff() takes the previous available row, which
    for the first bar of a day is the last bar of the previous day (an
    overnight gap), and for an illiquid symbol can be hours or days earlier.
    Those gaps were previously winsorised down to the intraday clip bound and
    fed to the models as ordinary 10-minute returns.
    """
    df = df.copy()

    df["LOG_MID_OPEN"] = np.log(df[price_col])

    # A dense bar index over the whole sample, used to detect missing bars.
    bars = np.sort(df["DATETIME"].unique())
    df["BAR"] = df["DATETIME"].map(pd.Series(np.arange(len(bars)), index=bars))

    grouped = df.groupby("SYMBOL")
    prev_bar = grouped["BAR"].shift(1)
    contiguous = df["BAR"] - prev_bar == 1

    if break_on_day:
        contiguous &= df["DATE"].values == grouped["DATE"].shift(1).values

    df["CONTIGUOUS"] = contiguous.fillna(False)

    # Returns must be computed stock by stock, not across the full dataframe.
    df["RETURN"] = np.where(
        df["CONTIGUOUS"],
        df["LOG_MID_OPEN"] - grouped["LOG_MID_OPEN"].shift(1),
        np.nan,
    )

    return df


def clip_returns(df, return_col="RETURN", lower_q=0.01, upper_q=0.99,
                 fit_mask=None, bounds=None):
    """Winsorize returns using empirical quantiles.

    fit_mask selects the rows the quantiles are estimated on. Pass the training
    window: estimating the bounds on the full panel, as the original code did,
    leaks test-window information into the training features.
    bounds overrides estimation entirely (e.g. bounds fitted on train, applied
    to validation and test).
    """
    df = df.copy()

    if bounds is not None:
        lower_bound, upper_bound = bounds
    else:
        returns = df.loc[fit_mask, return_col] if fit_mask is not None else df[return_col]
        returns = returns.dropna()

        lower_bound = returns.quantile(lower_q)
        upper_bound = returns.quantile(upper_q)

    # Clipping keeps extreme observations from dominating the ML models.
    df["RETURN_CLIPPED"] = df[return_col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    return df, lower_bound, upper_bound


def create_future_return_target(df, return_col="RETURN_CLIPPED", horizon=1):
    """Create the future return target."""
    df = df.copy()

    # Negative shift means that row t receives the return observed at t+horizon.
    # The shifted value is only a valid target if the row it comes from is the
    # bar immediately after t, in the same session; otherwise the "next-period
    # return" would actually span an overnight gap or a stretch of missing bars.
    grouped = df.groupby("SYMBOL")
    target = grouped[return_col].shift(-horizon)

    if "BAR" in df.columns:
        valid = grouped["BAR"].shift(-horizon) - df["BAR"] == horizon
        if "DATE" in df.columns:
            valid &= grouped["DATE"].shift(-horizon).values == df["DATE"].values
        target = target.where(valid.fillna(False))

    df["TARGET_RETURN"] = target

    return df


def preprocess_intraday_folder(
    folder_path,
    min_timestamps=30,
    lower_q=0.01,
    upper_q=0.99,
    target_horizon=1
):
    """Run the full preprocessing pipeline for a folder of intraday files."""
    df = load_intraday_folder(folder_path)

    df = add_datetime_column(df)
    df = filter_active_symbols(df, min_timestamps=min_timestamps)
    df = sort_intraday_panel(df)
    df = compute_log_returns(df)

    df, lower_bound, upper_bound = clip_returns(
        df,
        lower_q=lower_q,
        upper_q=upper_q
    )

    df = create_future_return_target(
        df,
        horizon=target_horizon
    )

    info = {
        "n_rows": len(df),
        "n_symbols": df["SYMBOL"].nunique(),
        "n_files": len(list(Path(folder_path).glob("*.csv.gz"))),
        "return_lower_bound": lower_bound,
        "return_upper_bound": upper_bound,
        "min_timestamps": min_timestamps,
        "target_horizon": target_horizon,
    }

    return df, info