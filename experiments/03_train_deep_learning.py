"""
Step 3 - LSTM and Attention-LSTM on the corrected features.

Run this on a machine with PyTorch installed:

    python experiments/03_train_deep_learning.py --symbols 500

It writes results/deep_learning.csv and results/deep_learning.json, which the
report picks up. Defaults reproduce the tuned configuration used for the panel
models: sequence length 24 for the LSTM, 12 for the Attention-LSTM, hidden size
128, one layer, AdamW, early stopping on validation loss.

Note on sequence length: a window must sit on consecutive bars inside one
session, and feature warm-up (the 5-bar rolling statistics and the third lag)
removes the first six bars of every day. That leaves 32 usable bars per session,
so 32 is the hard ceiling and anything close to it yields very few windows.

The sequences are built from the same 21 features and the same chronological
split as the panel baselines, so the numbers are directly comparable with
results/prediction_metrics.csv.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import (COST, FEATURES, PANEL, RESULTS, build_features,
                    cross_sectional_ic, make_groups, portfolio_metrics,
                    prediction_metrics, run_backtest, splits, write)
from src.models.deep_learning import (  # noqa: E402  (common sets sys.path)
    build_attention_lstm_model, build_lstm_model, set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=int, default=500,
                   help="most active symbols to use (sequence models are expensive)")
    p.add_argument("--sequence-length", type=int, default=24)
    p.add_argument("--attention-sequence-length", type=int, default=12)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_sequences(frame, length):
    """Rolling windows of `length` bars ending at t; target = return over bar t.

    A window is only valid if all `length` rows belong to the same symbol, sit
    on consecutive bars, and fall inside a single trading session. Feature
    warm-up (the 5-bar rolling statistics and the third return lag) removes the
    first few bars of every day, so the usable run inside one session is shorter
    than the 38 bars the session contains -- roughly 32. A sequence length above
    that yields no windows at all, which is why the default is well below it.

    Returns (x, y, key) with key carrying SYMBOL, BAR and the raw forward return
    needed for the backtest.
    """
    feats = frame[FEATURES].to_numpy(np.float32)
    targ = frame["TARGET"].to_numpy(np.float32)
    fwd = frame["FORWARD_RETURN"].to_numpy(np.float32)
    bar = frame["BAR"].to_numpy(np.int64)
    day = frame["DAY"].to_numpy(np.int64)
    sym = frame["SYMBOL"].to_numpy()

    ends = []
    for positions in frame.groupby("SYMBOL", sort=False).indices.values():
        if len(positions) < length:
            continue
        e = positions[length - 1:]          # candidate end positions
        b = positions[: len(positions) - length + 1]   # matching window starts
        ok = (bar[e] - bar[b] == length - 1) & (day[e] == day[b])
        ends.append(e[ok])

    if not ends or sum(len(e) for e in ends) == 0:
        raise SystemExit(
            f"No sequences of length {length} exist. Feature warm-up leaves about "
            f"32 usable bars per session, so the window must be shorter than that. "
            f"Re-run with a smaller --sequence-length / --attention-sequence-length."
        )

    end = np.concatenate(ends)
    offsets = np.arange(-(length - 1), 1)
    rows = end[:, None] + offsets[None, :]              # (n_windows, length)

    x = feats[rows]
    y = targ[end]
    key = pd.DataFrame({"SYMBOL": sym[end], "BAR": bar[end],
                        "FORWARD_RETURN": fwd[end]})
    return x, y, key


def main():
    args = parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} on {device}")

    df = pd.read_pickle(PANEL)
    train_bars, validation_bars, test_bars = splits(df)
    d, _ = build_features(df, train_bars, horizon=1)
    del df

    active = d["SYMBOL"].value_counts().head(args.symbols).index
    d = d[d["SYMBOL"].isin(active)].sort_values(["SYMBOL", "BAR"]).reset_index(drop=True)
    print(f"{len(d):,} rows over {d.SYMBOL.nunique()} symbols")

    # standardise features on the training window only
    train_rows = d["BAR"].isin(train_bars)
    mu = d.loc[train_rows, FEATURES].mean()
    sd = d.loc[train_rows, FEATURES].std().replace(0, 1.0)
    d[FEATURES] = (d[FEATURES] - mu) / sd

    results, portfolios = [], []
    for name, length, attention in (("lstm", args.sequence_length, False),
                                    ("attention_lstm", args.attention_sequence_length, True)):
        t0 = time.time()
        x, y, key = build_sequences(d, length)
        in_train = key["BAR"].isin(train_bars).to_numpy()
        in_val = key["BAR"].isin(validation_bars).to_numpy()
        in_test = key["BAR"].isin(test_bars).to_numpy()
        print(f"\n{name}: sequences train={in_train.sum():,} "
              f"val={in_val.sum():,} test={in_test.sum():,}")

        if in_train.sum() == 0 or in_val.sum() == 0 or in_test.sum() == 0:
            raise SystemExit(
                f"{name}: one of the splits has no sequences of length {length}. "
                "Use a shorter sequence length.")

        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[in_train]), torch.from_numpy(y[in_train])),
            batch_size=args.batch_size, shuffle=True, drop_last=False)
        xv = torch.from_numpy(x[in_val]).to(device)
        yv = torch.from_numpy(y[in_val]).to(device)

        builder = build_attention_lstm_model if attention else build_lstm_model
        model = builder(n_features=len(FEATURES), hidden_size=args.hidden,
                        num_layers=1, dropout=args.dropout, seed=args.seed).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()

        best, best_state, stale = np.inf, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb.to(device)).squeeze(-1), yb.to(device)).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(xv).squeeze(-1), yv))
            print(f"  epoch {epoch:2d}  validation MSE {vl:.3e}")
            if vl < best - 1e-12:
                best, best_state, stale = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"  early stop at epoch {epoch}")
                    break
        model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pred = np.concatenate([
                model(torch.from_numpy(x[in_test][i:i + 8192]).to(device))
                .squeeze(-1).cpu().numpy()
                for i in range(0, int(in_test.sum()), 8192)])

        te = key[in_test].copy()
        te["prediction"] = pred
        te["TARGET"] = y[in_test]
        te["DATETIME"] = te["BAR"]
        m = prediction_metrics(te["TARGET"].to_numpy(), pred)
        ic, t, n = cross_sectional_ic(te, "prediction", "TARGET")
        m.update(model=name, ic_cross_sectional=ic, ic_t_stat=t, n_timestamps=n,
                 validation_mse=best, seconds=round(time.time() - t0, 1))
        results.append(m)

        pm = portfolio_metrics(run_backtest(make_groups(te, "prediction"), "prediction"))
        pm["model"] = name
        portfolios.append(pm)
        print(f"  IC={ic:+.4f} (t={t:.2f})  hit={m['hit']:.4f}  "
              f"net={pm['net_bps']:+.2f} bps  break-even={pm['breakeven_bps']:.2f} bps")

    prediction = pd.DataFrame(results).set_index("model")[
        ["rmse", "mae", "r2", "ic_pooled", "ic_cross_sectional", "ic_t_stat", "hit",
         "n_timestamps", "validation_mse"]]
    portfolio = pd.DataFrame(portfolios).set_index("model")
    write(prediction, "deep_learning_prediction.csv")
    write(portfolio, "deep_learning_portfolio.csv")
    (RESULTS / "deep_learning.json").write_text(json.dumps(
        {"args": vars(args), "prediction": prediction.to_dict(),
         "portfolio": portfolio.to_dict()}, indent=2))
    print("\n" + prediction.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n" + portfolio[["gross_bps", "net_bps", "turnover", "sharpe_per_bar",
                            "breakeven_bps"]].to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
