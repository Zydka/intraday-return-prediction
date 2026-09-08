"""
Step 1 - build the supervised panel from the raw 10-minute files.

Design points that matter for correctness:

 1. SUM_DELTA is a bar AGGREGATE covering [t, t+10min). Measured on the raw
    files:

        corr(SUM_DELTA_t,     return over bar t)   = +0.102
        corr(SUM_DELTA_t,     return over bar t-1) = +0.003
        corr(SUM_DELTA_{t-1}, return over bar t)   = -0.001

    The value stamped at t is therefore realised jointly with the return it
    would be used to predict, and is not observable when a trade would be
    placed. Every demand feature is built from the series shifted one bar.

 2. A return is defined only between two consecutive bars of the same session.
    Taking the previous available row instead turns overnight gaps and stretches
    of missing quotes into apparent 10-minute returns.

 3. A liquidity flag marks symbols quoting in at least 95% of the bars of the
    days they trade, used for the tradeable-universe checks.

Winsorisation bounds and per-symbol normalisation moments are NOT fitted here;
they are fitted on the training window in step 2 and applied forward.

Output: panel.pkl next to this script (~6.2M rows, ~45 s).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data" / "demand_minute10_wct_202112"
OUT = Path(__file__).resolve().parent / "panel.pkl"

N_LAGS = 3
ROLL = (3, 5)
SESSION_MINUTES = 370          # 09:30 to 15:40 inclusive
LIQUID_COVERAGE = 0.95


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"no .csv.gz files in {DATA}")
    df = pd.concat(
        [pd.read_csv(f, usecols=["DATE", "SYMBOL", "TIME", "MID_OPEN", "SUM_DELTA"])
         for f in files],
        ignore_index=True,
    )
    df["DATETIME"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"])
    df["SYMBOL"] = df["SYMBOL"].astype("category")
    df["SUM_DELTA"] = df["SUM_DELTA"].astype("float32")
    return df


def main() -> None:
    df = load_raw()
    print(f"raw rows={len(df):,}  symbols={df.SYMBOL.nunique():,}  bars={df.DATETIME.nunique()}")

    bars = np.sort(df["DATETIME"].unique())
    df["BAR"] = df["DATETIME"].map(pd.Series(np.arange(len(bars)), index=bars)).astype("int32")
    df["DAY"] = df["DATE"].astype("int32")

    df = df.sort_values(["SYMBOL", "BAR"]).reset_index(drop=True)
    g = df.groupby("SYMBOL", observed=True, sort=False)
    df = df[g["BAR"].transform("size") >= 30].copy()
    g = df.groupby("SYMBOL", observed=True, sort=False)

    # -- returns, only across consecutive bars of the same session ----------
    lp = np.log(df["MID_OPEN"])
    prev_bar = g["BAR"].shift(1)
    contiguous = (df["BAR"] - prev_bar == 1) & (df["DAY"] == g["DAY"].shift(1))
    df["RETURN"] = np.where(contiguous, lp - np.log(g["MID_OPEN"].shift(1)), np.nan).astype("float32")
    print(f"rows whose previous observation is not the preceding bar: "
          f"{int((~contiguous & prev_bar.notna()).sum()):,} (returns left undefined)")

    # -- demand, observable at the open of bar t ----------------------------
    df["SUM_DELTA_OBS"] = np.where(contiguous, g["SUM_DELTA"].shift(1), np.nan).astype("float32")
    g = df.groupby("SYMBOL", observed=True, sort=False)

    # -- target: the return over bar t --------------------------------------
    nxt = (g["BAR"].shift(-1) - df["BAR"] == 1) & (g["DAY"].shift(-1) == df["DAY"])
    df["TARGET_RETURN_RAW"] = np.where(nxt, g["RETURN"].shift(-1), np.nan).astype("float32")

    # -- lags and rolling statistics ----------------------------------------
    df["RETURN_CURRENT"] = df["RETURN"]
    df["SUM_DELTA_CURRENT"] = df["SUM_DELTA_OBS"]
    for k in range(1, N_LAGS + 1):
        df[f"RETURN_LAG_{k}"] = g["RETURN"].shift(k).astype("float32")
        df[f"SUM_DELTA_LAG_{k}"] = g["SUM_DELTA_OBS"].shift(k).astype("float32")
    for w in ROLL:
        r, d = g["RETURN"], g["SUM_DELTA_OBS"]
        df[f"ROLLING_RETURN_MEAN_{w}"] = r.rolling(w, min_periods=w).mean().reset_index(level=0, drop=True).astype("float32")
        df[f"ROLLING_RETURN_STD_{w}"] = r.rolling(w, min_periods=w).std().reset_index(level=0, drop=True).astype("float32")
        df[f"ROLLING_SUM_DELTA_MEAN_{w}"] = d.rolling(w, min_periods=w).mean().reset_index(level=0, drop=True).astype("float32")

    # -- intraday seasonality ------------------------------------------------
    tod = pd.to_datetime(df["TIME"], format="%H:%M:%S")
    df["MINUTES_FROM_OPEN"] = ((tod.dt.hour - 9) * 60 + tod.dt.minute - 30).astype("float32")
    frac = (df["MINUTES_FROM_OPEN"] / SESSION_MINUTES).astype("float32")
    df["INTRADAY_TIME_FRACTION"] = frac
    df["TIME_SIN"] = np.sin(2 * np.pi * frac).astype("float32")
    df["TIME_COS"] = np.cos(2 * np.pi * frac).astype("float32")

    # -- liquidity flag ------------------------------------------------------
    per_day = df.groupby("DAY", observed=True)["BAR"].nunique()
    obs = df.groupby(["SYMBOL", "DAY"], observed=True).size().rename("n").reset_index()
    obs["cov"] = obs["n"] / obs["DAY"].map(per_day)
    liquid = set(obs.groupby("SYMBOL", observed=True)["cov"].mean().pipe(lambda s: s[s >= LIQUID_COVERAGE]).index)
    df["LIQUID"] = df["SYMBOL"].isin(liquid)
    print(f"liquid symbols (>= {LIQUID_COVERAGE:.0%} bar coverage): {len(liquid):,} of {df.SYMBOL.nunique():,}")

    keep = ["DATETIME", "BAR", "DAY", "SYMBOL", "LIQUID", "MID_OPEN", "RETURN",
            "TARGET_RETURN_RAW", "RETURN_CURRENT", "SUM_DELTA_CURRENT",
            "MINUTES_FROM_OPEN", "INTRADAY_TIME_FRACTION", "TIME_SIN", "TIME_COS"] \
        + [f"RETURN_LAG_{k}" for k in range(1, N_LAGS + 1)] \
        + [f"SUM_DELTA_LAG_{k}" for k in range(1, N_LAGS + 1)] \
        + [f"ROLLING_RETURN_MEAN_{w}" for w in ROLL] \
        + [f"ROLLING_RETURN_STD_{w}" for w in ROLL] \
        + [f"ROLLING_SUM_DELTA_MEAN_{w}" for w in ROLL]

    out = df[keep].copy()
    out["SYMBOL"] = out["SYMBOL"].astype(str)
    out.to_pickle(OUT)
    print(f"saved {OUT}  rows={len(out):,}")
    print(f"share of targets exactly zero (stale quotes): "
          f"{(out['TARGET_RETURN_RAW'] == 0).mean():.4f}")


if __name__ == "__main__":
    main()
