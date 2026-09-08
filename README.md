# Machine Learning for Cost-Aware Intraday Trading

Do machine learning models forecast the cross-section of 10-minute equity
returns, and is the forecast worth more than the cost of trading on it?

The short answer this project arrives at: **yes to the first, no to the
second.** The engineered features carry a small but strongly significant
cross-sectional signal. Traded as a decile long–short book it earns a few basis
points per bar before costs against a turnover of roughly 1.5x, which puts
break-even at a level below any realistic spread. Execution filtering improves
that materially, but the profit that survives is short-horizon reversal
concentrated in wide-spread names, where the cost assumption that made it
survive is the least defensible.

The written report is `PROJECT_REPORT.pdf`.

## Data

10-minute bars for December 2021, in `data/demand_minute10_wct_202112/`
(excluded from version control):

| field | meaning |
|---|---|
| `DATE`, `TIME`, `SYMBOL` | bar identifier |
| `MID_OPEN` | mid quote at the **opening** of the bar |
| `SUM_DELTA` | signed order-flow imbalance aggregated **over** the bar |

The session runs 09:30–15:40, so 38 bars per day, 836 bars over 22 trading days,
about 6.2M rows across 9,500 symbols.

### One timing point that governs the whole project

`SUM_DELTA` is a bar *aggregate*. The value stamped at time `t` covers
`[t, t+10min)` — the same interval as the prediction target — and is therefore
not observable when a trade would be placed. Measured on the raw files:

```
corr(SUM_DELTA_t,     return over bar t)   = +0.102
corr(SUM_DELTA_t,     return over bar t-1) = +0.003
corr(SUM_DELTA_{t-1}, return over bar t)   = -0.001
```

It is correlated with the return it spans, through price impact, and carries no
predictive content one bar later. Every demand feature here is built from
`SUM_DELTA` shifted one bar, so the whole feature vector at `t` is observable at
`t`. Using the contemporaneous value recovers the price-impact relation rather
than a forecast and inflates every statistic by roughly an order of magnitude.

Two related choices follow the same principle: a return is defined only between
consecutive bars of the same session (overnight gaps and missing-quote stretches
are not 10-minute returns), and every fitted preprocessing quantity —
winsorisation bounds, per-symbol normalisation moments — is estimated on the
training window and applied forward.

## Layout

```
data/                      raw and derived data, not versioned
experiments/
  01_build_panel.py        raw files -> supervised panel (~45 s)
  02_run_analysis.py       baselines, portfolio, costs, execution, diagnostics
  03_train_deep_learning.py  LSTM and Attention-LSTM (needs PyTorch)
  04_build_report.py       results/*.csv -> report/report.tex -> PDF
  common.py                shared config, metrics and backtest helpers
report/
  report.tex               generated; never edited by hand
  figures/                 generated
results/                   compact result tables, versioned
src/
  config.py                shared paths and constants
  preprocessing/           loading, cleaning, feature engineering, splitting
  models/                  baseline, time-series and sequence models
  evaluation/              metrics, validation, walk-forward backtesting
  trading/                 portfolio construction, costs, volatility
  portfolio/               execution-aware backtest helpers
PROJECT_REPORT.pdf
```

## Reproducing

```bash
pip install -r requirements.txt

python experiments/01_build_panel.py          # writes experiments/panel.pkl
python experiments/02_run_analysis.py         # writes results/ and report/figures/
python experiments/03_train_deep_learning.py --symbols 500   # optional, needs torch
python experiments/04_build_report.py --compile              # writes PROJECT_REPORT.pdf
```

Step 2 takes roughly 25 minutes on two cores, most of it the random forest.
Step 4 reads every number from `results/*.csv`, so the report can never drift
from the run that produced it; re-running step 4 after step 3 fills in the
deep-learning section.

## Method

**Features (21).** Current and three lagged returns; observable demand
imbalance and its three lags; rolling return mean and standard deviation and
rolling demand mean over 3- and 5-bar windows; intraday seasonality (minutes
from open, session fraction, sine and cosine); cross-sectional percentile ranks
of return and demand; per-symbol demand z-score.

**Split.** Chronological by timestamp, 70 / 15 / 15. No test timestamp is seen
during fitting or during model selection.

**Models.** Zero and lag-1 benchmarks; OLS, ridge, lasso, elastic net; Huber;
random forest; histogram gradient boosting; LSTM and Attention-LSTM.

**Strategy.** Dollar-neutral decile long–short, equal weight within each side,
rebalanced every bar. Net return is gross minus cost times turnover. P&L uses
raw realised returns, not the winsorised target.

## Metrics, and why these ones

- **Cross-sectional IC** — mean per-timestamp Spearman correlation, with a
  t-statistic across timestamps. This is what a long–short book monetises. The
  pooled Pearson IC is also reported but mixes ranking skill with time-series
  variation in volatility.
- **Hit ratio excluding zero targets** — about 14% of returns are exactly zero
  because inactive quotes do not refresh. A sign-matching rule that counts those
  measures staleness, not skill.
- **Sharpe per bar**, not annualised. Annualising multiplies by
  `sqrt(252 x 38) = 97.9`; over a test window of ~107 bars that produces numbers
  with no interpretation as achievable performance.
- **Break-even transaction cost** — gross mean over mean turnover. For a
  strategy rebalancing every ten minutes this is the most informative economic
  statistic, because it can be compared directly against a spread.

## Notes

- LightGBM is listed in `requirements.txt` but the results in `results/` were
  produced without it; histogram gradient boosting is the tree baseline used.
- The test window is ~107 usable bars, under three trading days. Every economic
  figure is a small-sample estimate.
- Market impact is not modelled, and costs are a constant per unit of turnover.
  The diagnostics section of the report argues that a name-varying spread model
  would be considerably less favourable.

## Authors

Zouhir Al Baridi (356618), Pierre-Gabriel Meyrignac (358421).
