"""
Step 2 - the full analysis, in one pass.

  A. panel baselines and prediction metrics
  B. long-short portfolio, transaction-cost sensitivity, robustness sweeps
  C. execution-aware trading: rule selected on VALIDATION, evaluated on TEST
  D. inverse-volatility risk overlay
  E. diagnostics: what the surviving book actually holds
  F. figures for the report

Models are fitted once and their test predictions reused throughout.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import (COST, FEATURES, PANEL, PERIODS_PER_YEAR, QUANTILE, RESULTS,
                    build_features, cross_sectional_ic, make_groups,
                    portfolio_metrics, prediction_metrics, run_backtest, splits, write)
from src.models.baseline_models import regression_models  # noqa: E402  (common sets sys.path)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
EXECUTION_GRID = [dict(q=q, no_trade_zone=z, rho=r)
                  for q in (0.02, 0.05, 0.10)
                  for z in (0.0, 0.5, 0.75, 0.9, 0.95)
                  for r in (1.0, 0.5, 0.25)]


def models():
    """Model definitions live in src/models/baseline_models.py."""
    return regression_models(random_state=RANDOM_STATE, n_jobs=-1)


CAPPED = ("random_forest", "huber")


def fit_model(name, frame):
    """Fit with the same subsampling rule everywhere, so section C matches section A."""
    model = models()[name]
    sub = frame.sample(300_000, random_state=RANDOM_STATE) if name in CAPPED else frame
    model.fit(sub[FEATURES], sub["TARGET"])
    return model


def main():
    df = pd.read_pickle(PANEL)
    train_bars, validation_bars, test_bars = splits(df)
    fit_bars = train_bars | validation_bars
    panel = dict(panel_rows=int(len(df)), panel_symbols=int(df.SYMBOL.nunique()),
                 panel_bars=int(df.BAR.nunique()))
    d, (clip_lo, clip_hi) = build_features(df, train_bars, horizon=1)
    del df

    train = d[d["BAR"].isin(fit_bars)]
    test = d[d["BAR"].isin(test_bars)].copy()
    info = dict(
        rows=int(len(d)), symbols=int(d.SYMBOL.nunique()), bars=int(d.BAR.nunique()),
        train_rows=int(len(train)), test_rows=int(len(test)),
        test_bars=int(test.BAR.nunique()), test_symbols=int(test.SYMBOL.nunique()),
        clip_lower=clip_lo, clip_upper=clip_hi,
        zero_target_share=float((test["FORWARD_RETURN"] == 0).mean()),
        features=len(FEATURES), **panel,
    )
    print(json.dumps(info, indent=2))

    # ---------------------------------- A. model selection, then prediction
    # The model is chosen on VALIDATION, never on the test window. Each model is
    # fitted twice: on training alone to score it, then on training plus
    # validation to produce the reported test predictions.
    print("\nA. model selection on validation")
    train_only = train[train["BAR"].isin(train_bars)]
    val = d[d["BAR"].isin(validation_bars)].copy()
    val_preds, val_ic, timings = {}, {}, {}
    for name in models():
        t0 = time.time()
        val_preds[name] = fit_model(name, train_only).predict(val[FEATURES])
        timings[name] = round(time.time() - t0, 1)
        val["_p"] = val_preds[name]
        ic, t, _ = cross_sectional_ic(val, "_p", "TARGET")
        val_ic[name] = ic
        print(f"  {name:<24s} {timings[name]:6.1f}s   validation IC {ic:+.4f} (t={t:.2f})")
    best = max(val_ic, key=val_ic.get)
    print(f"  selected on validation: {best}")

    print("\n   refitting on training + validation")
    preds = {"zero": np.zeros(len(test)), "lag1_return": test["RETURN_CURRENT"].to_numpy()}
    for name in models():
        preds[name] = fit_model(name, train).predict(test[FEATURES])

    pd.DataFrame({k: v for k, v in preds.items()}).assign(
        BAR=test["BAR"].values, SYMBOL=test["SYMBOL"].values
    ).to_pickle(Path(__file__).resolve().parent / "test_predictions.pkl")

    y = test["TARGET"].to_numpy(float)
    rows = []
    for name, pr in preds.items():
        m = prediction_metrics(y, pr)
        test["_p"] = pr
        ic, t, n = (np.nan, np.nan, 0) if name == "zero" else cross_sectional_ic(test, "_p", "TARGET")
        m.update(model=name, ic_cross_sectional=ic, ic_t_stat=t, n_timestamps=n,
                 validation_ic=val_ic.get(name, float("nan")), selected=(name == best))
        rows.append(m)
    prediction = pd.DataFrame(rows).set_index("model")[
        ["rmse", "mae", "r2", "ic_pooled", "ic_cross_sectional", "ic_t_stat", "hit",
         "validation_ic", "selected"]]
    print(prediction.to_string(float_format=lambda v: f"{v:.4f}"))
    write(prediction, "prediction_metrics.csv")

    test["prediction"] = preds[best]
    val["prediction"] = val_preds[best]
    groups = make_groups(test, "prediction")

    # ------------------------------------ B. portfolio, costs, robustness
    print("\nB. long-short portfolio")
    plain = run_backtest(groups, "prediction")
    cost_rows = []
    for c in (0, 1, 2, 3, 5, 10, 20, 50):
        m = portfolio_metrics(plain, cost=c / 1e4)
        m["cost_bps"] = c
        cost_rows.append(m)
    cost_table = pd.DataFrame(cost_rows).set_index("cost_bps")
    print(cost_table[["gross_bps", "net_bps", "sharpe_per_bar", "t_stat_net",
                      "cumulative_net", "max_drawdown"]].to_string(float_format=lambda v: f"{v:.4f}"))
    write(cost_table, "transaction_cost_sensitivity.csv")

    rob = []
    for q in (0.02, 0.05, 0.10, 0.20, 0.30):
        m = portfolio_metrics(run_backtest(groups, "prediction", q=q))
        m.update(sweep="quantile", setting=f"{q:.0%}")
        rob.append(m)
    robustness = pd.DataFrame(rob).set_index(["sweep", "setting"])
    horizon_rows = []
    for h in (1, 3, 6):
        dh, _ = build_features(pd.read_pickle(PANEL), train_bars, horizon=h)
        trh, teh = dh[dh["BAR"].isin(fit_bars)], dh[dh["BAR"].isin(test_bars)].copy()
        teh["prediction"] = fit_model(best, trh).predict(teh[FEATURES])
        ich, th, nh = cross_sectional_ic(teh, "prediction", "FORWARD_RETURN")
        pm = portfolio_metrics(run_backtest(make_groups(teh, "prediction"), "prediction",
                                            every=h), bars_held=h)
        pm.update(horizon_bars=h, minutes=10 * h, ic=ich, ic_t=th, n_timestamps=nh)
        horizon_rows.append(pm)
        del dh, trh, teh
    horizons = pd.DataFrame(horizon_rows).set_index("minutes")
    print("\n  signal decay by holding horizon")
    print(horizons[["ic", "ic_t", "gross_bps", "net_bps", "turnover", "breakeven_bps"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    write(horizons, "horizon_decay.csv")
    print("\n", robustness[["gross_bps", "net_bps", "turnover", "sharpe_per_bar",
                            "breakeven_bps"]].to_string(float_format=lambda v: f"{v:.4f}"))
    write(robustness, "robustness.csv")

    # --------------------------------- C. execution rule, selected properly
    print("\nC. execution-aware trading (rule selected on validation)")
    gv = make_groups(val, "prediction")

    sel = []
    for cfg in EXECUTION_GRID:
        m = portfolio_metrics(run_backtest(gv, "prediction", **cfg))
        m.update(cfg)
        sel.append(m)
    selection = pd.DataFrame(sel).sort_values("sharpe_per_bar", ascending=False)
    write(selection.set_index(["q", "no_trade_zone", "rho"]), "execution_selection_validation.csv")
    chosen = {k: float(selection.iloc[0][k]) for k in ("q", "no_trade_zone", "rho")}
    print(f"  selected on validation: {chosen}")

    final_rows = []
    m = portfolio_metrics(plain); m["strategy"] = "plain decile, rebalance every bar"
    final_rows.append(m)
    executed = run_backtest(groups, "prediction", **chosen)
    m = portfolio_metrics(executed)
    m["strategy"] = (f"execution-aware (q={chosen['q']:.2f}, "
                     f"no-trade zone={chosen['no_trade_zone']:.2f}, rho={chosen['rho']:.2f})")
    final_rows.append(m)
    final = pd.DataFrame(final_rows).set_index("strategy")
    print(final[["gross_bps", "net_bps", "turnover", "sharpe_per_bar", "t_stat_net",
                 "cumulative_net", "max_drawdown", "losing_bars", "breakeven_bps"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    write(final, "strategy_comparison.csv")

    exec_cost = pd.DataFrame(
        [{**portfolio_metrics(executed, cost=c / 1e4), "cost_bps": c}
         for c in (0, 1, 2, 5, 10, 15, 20, 30)]).set_index("cost_bps")
    write(exec_cost, "execution_cost_curve.csv")

    # ------------------------------------------- D. inverse-volatility overlay
    print("\nD. inverse-volatility risk overlay")
    test_sorted = test.sort_values(["SYMBOL", "BAR"])
    ew = test_sorted.groupby("SYMBOL")["RETURN_CURRENT"].transform(
        lambda s: s.ewm(halflife=80, min_periods=5).std().shift(1))
    test_sorted["INV_VOL"] = (1.0 / (ew + 1e-6)).replace([np.inf, -np.inf], np.nan)
    gv2 = make_groups(test_sorted.dropna(subset=["INV_VOL"]), "prediction")
    overlay_rows = []
    for label, kw in [("equal weight", {}), ("inverse lagged EWMA volatility", {"weight_col": "INV_VOL"})]:
        m = portfolio_metrics(run_backtest(gv2, "prediction", **chosen, **kw))
        m["weighting"] = label
        overlay_rows.append(m)
    overlay = pd.DataFrame(overlay_rows).set_index("weighting")
    print(overlay[["gross_bps", "net_bps", "turnover", "sharpe_per_bar", "cumulative_net"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    write(overlay, "risk_overlay.csv")

    # --------------------------------------------------- E. diagnostics
    print("\nE. what the surviving book holds")
    held = []
    for bar, g in groups.items():
        w = None
        gg = g[g["prediction"].abs() >= g["prediction"].abs().quantile(chosen["no_trade_zone"])]
        if len(gg) < 20:
            continue
        hi, lo = gg["prediction"].quantile(1 - chosen["q"]), gg["prediction"].quantile(chosen["q"])
        held.append(gg[(gg["prediction"] >= hi) | (gg["prediction"] <= lo)]
                    [["SYMBOL", "LIQUID", "RETURN_CURRENT", "prediction"]])
    held = pd.concat(held)
    side = np.sign(held["prediction"])
    diagnostics = pd.Series({
        "names_per_rebalance": len(held) / len(groups),
        "share_positions_liquid": held["LIQUID"].mean(),
        "share_panel_liquid": test["LIQUID"].mean(),
        "corr_prediction_prev_return": test["prediction"].corr(test["RETURN_CURRENT"]),
        "mean_prev_return_longs_bps": held.loc[side > 0, "RETURN_CURRENT"].mean() * 1e4,
        "mean_prev_return_shorts_bps": held.loc[side < 0, "RETURN_CURRENT"].mean() * 1e4,
    })
    print(diagnostics.to_string(float_format=lambda v: f"{v:.4f}"))
    write(diagnostics.to_frame("value"), "book_diagnostics.csv")

    liq_groups = make_groups(test[test["LIQUID"]], "prediction")
    liq = pd.DataFrame(
        [{**portfolio_metrics(run_backtest(liq_groups, "prediction", **chosen), cost=c / 1e4),
          "cost_bps": c} for c in (5, 10, 15, 20, 30)]).set_index("cost_bps")
    print("\n  liquid universe only")
    print(liq[["gross_bps", "net_bps", "sharpe_per_bar", "cumulative_net"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    write(liq, "liquid_universe_cost_curve.csv")

    # ---------------------------------------------------------- F. figures
    make_figures(plain, executed, cost_table, exec_cost, test)

    (RESULTS / "run_info.json").write_text(json.dumps(
        {**info, "best_model": best, "execution_config": chosen,
         "fit_seconds": timings}, indent=2))
    print("\ndone")


def make_figures(plain, executed, cost_table, exec_cost, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import FIGURES

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 200, "savefig.bbox": "tight"})
    INK, ACCENT, MUTED = "#1f3b57", "#b4531f", "#8a94a0"

    # cumulative net wealth
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for p, label, colour in ((plain, "plain decile book", MUTED),
                             (executed, "execution-aware", INK)):
        net = p["gross"] - COST * p["turnover"]
        ax.plot(range(len(net)), ((1 + net).cumprod() - 1) * 100, color=colour, lw=1.6, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("10-minute bar in the test window")
    ax.set_ylabel("cumulative net return (%)")
    ax.set_title("Net of 5 bps per unit of turnover", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "cumulative_returns.pdf")
    plt.close(fig)

    # break-even cost curves
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(cost_table.index, cost_table["net_bps"], "o-", color=MUTED, lw=1.6, ms=3,
            label="plain decile book")
    ax.plot(exec_cost.index, exec_cost["net_bps"], "o-", color=INK, lw=1.6, ms=3,
            label="execution-aware")
    ax.axhline(0, color=ACCENT, lw=1.0, ls="--")
    ax.set_xlabel("transaction cost (bps per unit of turnover)")
    ax.set_ylabel("net return per bar (bps)")
    ax.set_title("Break-even cost", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "cost_curves.pdf")
    plt.close(fig)

    # the signal is reversal
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    s = test.sample(min(200_000, len(test)), random_state=0)
    ax.hexbin(s["RETURN_CURRENT"] * 1e4, s["prediction"] * 1e4, gridsize=45,
              bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.set_xlabel("return over the previous bar (bps)")
    ax.set_ylabel("model prediction (bps)")
    ax.set_title(f"Prediction versus previous-bar return "
                 f"(corr = {test['prediction'].corr(test['RETURN_CURRENT']):.2f})",
                 loc="left", fontsize=9)
    fig.savefig(FIGURES / "reversal.pdf")
    plt.close(fig)
    print(f"  wrote 3 figures to {FIGURES}")


if __name__ == "__main__":
    main()
