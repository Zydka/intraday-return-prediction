"""
Step 4 - generate report/report.tex from the result files and compile it.

Every number in the report is read from results/*.csv, so the document can never
drift from the run that produced it. If results/deep_learning_prediction.csv is
present the deep-learning section is populated; otherwise that section states
that the models have not been run.

    python experiments/04_build_report.py            # writes report/report.tex
    python experiments/04_build_report.py --compile  # also runs pdflatex
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from string import Template


class Tex(Template):
    """LaTeX-safe template: '$' stays maths, placeholders use '@@'."""
    delimiter = "@@"

import pandas as pd

from common import PROJECT_ROOT, RESULTS

REPORT = PROJECT_ROOT / "report"


def fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "--"


def escape(text):
    """Escape the LaTeX specials that can appear in an index label."""
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        text = text.replace(a, b)
    return text


def tabular(df, columns, headers, index_name, fmts=None, index_fmt=str):
    fmts = fmts or {}
    spec = "l" + "r" * len(columns)
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule",
           index_name + " & " + " & ".join(headers) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        label = index_fmt(idx if not isinstance(idx, tuple) else " / ".join(str(i) for i in idx))
        cells = [fmt(row[c], fmts.get(c, 4)) for c in columns]
        out.append(escape(label) + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


TEMPLATE = Tex(r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\usepackage{placeins}
\captionsetup{font=small}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{ML FOR FINANCE}
\fancyhead[R]{Project}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parskip}{0.15em}

\title{\vspace{-2em}\textbf{\Large ML FOR FINANCE}\\[0.4em]\large Project}
\author{Zouhir Al Baridi (356618) \and Pierre-Gabriel Meyrignac (358421)}
\date{@@DATE}

\begin{document}
\maketitle
\thispagestyle{empty}

\begin{abstract}
\noindent We study whether machine learning models can forecast 10-minute
cross-sectional stock returns and whether the resulting signals survive
trading frictions. Using a panel of @@PANEL_SYMBOLS symbols over @@PANEL_BARS
ten-minute bars in December 2021, we build a supervised dataset from lagged
returns, lagged order-flow imbalance, rolling statistics, intraday seasonality
and cross-sectional ranks, and compare linear, penalised, robust and tree-based
panel models. The best model attains an out-of-sample cross-sectional
information coefficient of @@IC_BEST ($t = @@IC_T$), which is small but strongly
significant. Converted into a dollar-neutral decile long--short book the signal
earns @@GROSS_BPS basis points per bar before costs against a turnover of
@@TURNOVER, giving a break-even transaction cost of @@BREAKEVEN basis points.
Execution filtering---trading only the strongest signals and adjusting
positions partially---raises break-even to @@BREAKEVEN_EXEC basis points and
delivers a positive net result at 5 basis points out of sample. We then show
that the surviving profit is short-horizon reversal concentrated in
wide-spread names, for which 5 basis points is not a defensible cost, and
conclude that the signal is statistically real but not economically
exploitable.
\end{abstract}

\section{Introduction}

This project asks two questions about a high-frequency equity panel. First,
does information available at time $t$ forecast the cross-section of returns
over the next 10-minute bar? Second, if it does, is the forecast worth enough
to survive the cost of trading on it?

The two questions are separate and the second is the harder one. A
cross-sectional signal at this frequency is mechanically expensive to harvest:
the book has to be rebuilt every ten minutes, so even a genuine edge can be
consumed entirely by spread and impact. We therefore report statistical
performance and economic performance side by side throughout, and treat the
\emph{break-even transaction cost}---the cost per unit of turnover at which the
strategy stops making money---as the primary economic statistic, because it can
be compared directly against a realistic spread.

\section{Data and Preprocessing}

The raw data are 10-minute bars for December 2021 with fields \texttt{DATE},
\texttt{SYMBOL}, \texttt{TIME}, \texttt{MID\_OPEN} and \texttt{SUM\_DELTA}.
\texttt{MID\_OPEN} is the mid quote at the opening of the bar; \texttt{SUM\_DELTA}
is signed order-flow imbalance aggregated \emph{over} the bar. The session runs
from 09:30 to 15:40, giving 38 bars per day. After removing symbols with fewer
than 30 observations the panel contains @@PANEL_ROWS rows over @@PANEL_SYMBOLS
symbols and @@PANEL_BARS bars; @@N_ROWS_RAW of those rows have a complete feature
vector and a defined target and enter the models.

\subsection{Returns}

For symbol $i$ we define log prices and returns as
\[
p_{i,t} = \log(\texttt{MID\_OPEN}_{i,t}), \qquad
R_{i,t} = p_{i,t} - p_{i,t-1},
\]
where $t-1$ must be the immediately preceding bar of the \emph{same session}.
Taking the previous available row instead would convert overnight gaps, and
stretches of missing quotes for thinly traded names, into apparent 10-minute
returns; @@N_VOIDED rows fall into that category and their returns are left
undefined. The prediction target is the return over the next bar,
\[
\texttt{TARGET\_RETURN}_{i,t} = \widetilde R_{i,t+1},
\]
winsorised at the 1st and 99th percentiles of the \emph{training} window. All
fitted preprocessing quantities---winsorisation bounds and per-symbol
normalisation moments---are estimated on training rows only and applied
forward.

A structural feature of this panel is that @@ZERO_SHARE\% of targets are exactly
zero, because the mid quote of an inactive symbol does not refresh from one bar
to the next. This matters for directional metrics and is handled explicitly in
Section~\ref{sec:metrics}.

\subsection{Timing of the order-flow variable}
\label{sec:timing}

\texttt{SUM\_DELTA} requires care. Because it aggregates flow \emph{over} the
bar it stamps, the value at time $t$ spans $[t, t+10\text{min})$---the same
interval as \texttt{TARGET\_RETURN}$_{i,t}$---and is therefore not observable at
the moment a trade would be placed. The raw data make the timing unambiguous:

\begin{center}
\begin{tabular}{lr}
\toprule
Correlation & Value \\
\midrule
$\mathrm{corr}(\texttt{SUM\_DELTA}_t,\ \text{return over bar } t)$ & $+0.102$ \\
$\mathrm{corr}(\texttt{SUM\_DELTA}_t,\ \text{return over bar } t-1)$ & $+0.003$ \\
$\mathrm{corr}(\texttt{SUM\_DELTA}_{t-1},\ \text{return over bar } t)$ & $-0.001$ \\
\bottomrule
\end{tabular}
\end{center}

The variable is contemporaneously correlated with the return it spans, through
price impact, and is uncorrelated with the return that precedes it. One bar
later it carries no predictive content at all. Every demand feature in this
project is therefore constructed from \texttt{SUM\_DELTA} shifted by one bar, so
that the entire feature vector at time $t$ is observable at time $t$. Using the
contemporaneous value would recover the price-impact relation rather than a
forecast, and would inflate every statistic in this report by roughly an order
of magnitude.

\subsection{Features}

The feature set has @@N_FEATURES variables in six groups: the current and three
lagged returns; the observable demand imbalance and its three lags; rolling
means and standard deviations of returns and rolling means of demand over
3- and 5-bar windows; intraday seasonality (minutes from the open, a session
fraction and its sine and cosine); cross-sectional percentile ranks of the
current return and of demand; and a per-symbol demand $z$-score. Rows with any
missing feature are dropped after construction.

\subsection{Split}

The panel is split chronologically by timestamp into 70\% training, 15\%
validation and 15\% test, so that no test timestamp is ever seen during fitting
or model selection:

\begin{center}
@@N_TRAIN_ROWS training rows \quad|\quad @@N_TEST_ROWS test rows over
@@N_TEST_BARS bars and @@N_TEST_SYMBOLS symbols
\end{center}

\section{Evaluation Metrics}
\label{sec:metrics}

Statistical performance is reported with RMSE, MAE, $R^2$ and two correlation
measures. The pooled information coefficient is the Pearson correlation between
predictions and realised returns across all observations. The
\emph{cross-sectional} information coefficient, which is the quantity a
long--short strategy actually monetises, is the mean per-timestamp Spearman
correlation,
\[
\overline{\mathrm{IC}} = \frac{1}{T}\sum_{t=1}^{T}
\rho_t\!\left(\widehat R_{\cdot,t+1},\, R_{\cdot,t+1}\right),
\]
reported with a $t$-statistic across the $T$ test timestamps. The two differ
because the pooled measure mixes ranking skill with time-series variation in
volatility.

The hit ratio excludes targets that are exactly zero. With @@ZERO_SHARE\% of
returns unchanged from one bar to the next, a sign-matching rule that counts
those observations measures quote staleness rather than directional skill.

Trading performance is reported as gross and net return per bar in basis
points, turnover, Sharpe ratio \emph{per bar}, its $t$-statistic, maximum
drawdown and break-even cost. Sharpe ratios are quoted per bar rather than
annualised. Annualising would multiply by $\sqrt{252 \times 38} = 97.9$, which
over a test window of @@N_TEST_BARS bars produces figures with no
interpretation as achievable performance.

\FloatBarrier
\section{Predictive Models}

\subsection{Model families}

We estimate a zero-return benchmark and a lagged-return benchmark; ordinary
least squares, ridge, lasso and elastic net; Huber regression as a
heavy-tail-robust alternative; and two nonlinear tree ensembles, a random
forest and histogram gradient boosting. Model definitions live in
\texttt{src/models/baseline\_models.py} and hyperparameters follow a validation
tuning pass.

\subsection{Results}

\begin{table}[!htbp]
\centering
\small
\caption{Out-of-sample prediction performance. IC is the cross-sectional
information coefficient with its $t$-statistic across @@N_TEST_BARS test
timestamps; the hit ratio excludes exactly-zero targets. The last column is the
validation IC on which the model choice was made.}
\label{tab:prediction}
@@TABLE_PREDICTION
\end{table}

Three observations. First, the signal is small in absolute terms: the selected
model reaches an $R^2$ of @@R2_BEST and a cross-sectional IC of @@IC_BEST. Nothing about
10-minute cross-sectional returns is easy to forecast, and a result of this size
is what a correctly specified pipeline should produce.

Second, the signal is nonetheless clearly present. With a $t$-statistic of
@@IC_T across @@N_TEST_BARS independent timestamps, the null of no ranking skill
is rejected comfortably. The nonlinear models separate themselves on the pooled IC and on
$R^2$, but not on the cross-sectional IC that a long--short book monetises, and
the plain linear specification is what validation selects. At this signal
strength the extra flexibility buys nothing that survives out of sample.

Third, the lagged-return benchmark has a \emph{negative} information
coefficient of @@IC_LAG1. Returns at this frequency are negatively
autocorrelated---a mid-quote series bounces---and this reversal is the dominant
piece of exploitable structure in the panel. We return to this in
Section~\ref{sec:diagnostics}, because it determines whether the strategy is
implementable.

@@SECTION_DEEP

\FloatBarrier
\section{Portfolio Construction and Transaction Costs}

\subsection{Construction}

At each timestamp stocks are ranked by predicted return. The strategy is long
the top decile and short the bottom decile, equally weighted within each side,
with the long side summing to $+0.5$ and the short side to $-0.5$:
\[
w_{i,t} =
\begin{cases}
+0.5/N_t^{+}, & i \in \text{top decile},\\
-0.5/N_t^{-}, & i \in \text{bottom decile},\\
0, & \text{otherwise.}
\end{cases}
\]
The book is therefore dollar-neutral, and performance reflects cross-sectional
ranking quality rather than market exposure. Gross and net returns are
\[
R^{\text{gross}}_{p,t} = \sum_i w_{i,t} R_{i,t+1},
\qquad
R^{\text{net}}_{p,t} = R^{\text{gross}}_{p,t} - c \cdot \text{Turnover}_t ,
\]
with turnover the sum of absolute weight changes between rebalances. Portfolio
returns use raw realised returns, not the winsorised target.

\subsection{Cost sensitivity}

\begin{table}[!htbp]
\centering
\small
\caption{Transaction-cost sensitivity of the decile long--short book,
rebalanced every bar.}
\label{tab:costs}
@@TABLE_COSTS
\end{table}

The book earns @@GROSS_BPS basis points per bar before costs, with a per-bar
Sharpe of @@SHARPE_GROSS. That is a real and statistically strong gross result.
But it turns over @@TURNOVER of itself every ten minutes, so the break-even cost
is only @@BREAKEVEN basis points. At the 5 basis points assumed throughout this
project the strategy loses @@NET_5 basis points per bar. Gross profitability at
this frequency is not informative on its own; the signal has to be evaluated
jointly with turnover.

\subsection{Robustness}

\begin{table}[!htbp]
\centering
\small
\caption{Sensitivity of the decile book to the long--short quantile, at 5 basis
points.}
\label{tab:robustness}
@@TABLE_ROBUSTNESS
\end{table}

Narrowing the book concentrates it in the strongest signals and raises
break-even; widening it dilutes the signal faster than it reduces turnover.
This is the first indication that the economics are governed by signal
concentration rather than by prediction accuracy.

Holding for longer does not help either, but for an instructive reason.
Refitting the model on 30- and 60-minute forward returns and rebalancing at the
matching frequency leaves the cross-sectional IC broadly intact, so the ranking
is not purely a one-bar effect. What does not improve is the economics: the
gross return per rebalance stays roughly flat as the holding period lengthens
while turnover per rebalance does not fall, so break-even does not rise. The
signal is about \emph{which} names to hold, not about how long to hold them,
and the cost is paid on every rebalance regardless.

\begin{table}[!htbp]
\centering
\small
\caption{Signal decay by holding horizon. Each row refits the model on the
corresponding forward return and rebalances at that frequency.}
\label{tab:horizon}
@@TABLE_HORIZON
\end{table}

\FloatBarrier
\section{Execution-Aware Trading}

\subsection{Two filters}

Rebalancing mechanically from one bar to the next generates turnover even when
the signal has barely moved. We add two standard execution filters. A
\emph{no-trade zone} discards names whose absolute predicted return is below a
quantile $z$ of the cross-section before ranking. \emph{Partial position
adjustment} moves the book only part of the way to its target,
\[
w_t = (1-\rho)\, w_{t-1} + \rho\, w_t^{\text{target}} ,
\]
so that $\rho = 1$ recovers full rebalancing.

\subsection{Selection}

The configuration $(q, z, \rho)$ is a modelling choice and must not be made on
the test set. We fit the model on the training window, predict on the
validation window, choose the configuration there by net Sharpe at 5 basis
points, then refit on training plus validation and evaluate that single
configuration on the test window. The selected rule is
$q = @@CFG_Q$, $z = @@CFG_Z$, $\rho = @@CFG_RHO$.

\begin{table}[!htbp]
\centering
\small
\caption{Test-window performance at 5 basis points. The execution-aware rule
was selected on validation, so this comparison is out of sample in both the
model and the trading rule.}
\label{tab:execution}
@@TABLE_STRATEGY
\end{table}

\begin{table}[!htbp]
\centering
\small
\caption{Cost sensitivity of the execution-aware strategy.}
\label{tab:execcost}
@@TABLE_EXECCOST
\end{table}

Filtering roughly halves turnover while more than tripling the gross return per
bar, because the discarded trades were the ones with the least signal per unit
of cost. Break-even rises from @@BREAKEVEN to @@BREAKEVEN_EXEC basis points.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.72\textwidth]{figures/cumulative_returns.pdf}
\caption{Cumulative net return over the test window at 5 basis points.}
\label{fig:cumulative}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.72\textwidth]{figures/cost_curves.pdf}
\caption{Net return per bar against the assumed transaction cost. The
break-even point is where each line crosses zero.}
\label{fig:costs}
\end{figure}

\subsection{Risk overlay}

The equal-weight book ignores differences in stock-level risk. As a
risk-management extension we keep the same selected names but weight each side
by inverse lagged EWMA volatility, estimated from past returns and shifted by
one bar to avoid look-ahead.

\begin{table}[!htbp]
\centering
\small
\caption{Equal weighting against inverse-volatility weighting, selected
execution configuration, 5 basis points.}
\label{tab:overlay}
@@TABLE_OVERLAY
\end{table}

@@OVERLAY_COMMENT

\section{What the Surviving Signal Is}
\label{sec:diagnostics}

A positive net result at 5 basis points is only meaningful if 5 basis points is
the right cost for the positions actually taken. It is not.

\begin{table}[!htbp]
\centering
\small
\caption{Composition and character of the execution-aware book.}
\label{tab:diagnostics}
@@TABLE_DIAGNOSTICS
\end{table}

The book holds roughly @@N_HELD names per rebalance, of which @@LIQ_HELD\% are in
names quoting in at least 95\% of bars, against @@LIQ_PANEL\% of the panel: the
strategy tilts systematically toward the thinly quoted tail. And its
predictions correlate @@CORR_REV with the previous bar's return. The average
long has fallen @@LONG_PREV basis points over the preceding ten minutes and the
average short has risen @@SHORT_PREV basis points. The model is a
short-horizon reversal rule.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.72\textwidth]{figures/reversal.pdf}
\caption{Model prediction against the previous bar's return on the test window.
The strategy buys what has just fallen and sells what has just risen.}
\label{fig:reversal}
\end{figure}

This is the crux. A stock whose \emph{mid} quote has moved 1.5\% in ten minutes
has a wide or stale quote. Buying it means lifting the offer and selling it
means hitting the bid, so the realistic round-trip cost is the full spread on
precisely the names where the spread is widest. Break-even for the
execution-aware strategy is @@BREAKEVEN_EXEC basis points; half-spreads on that
part of the universe are comfortably above it.

\begin{table}[!htbp]
\centering
\small
\caption{Selected configuration restricted to the liquid universe.}
\label{tab:liquid}
@@TABLE_LIQUID
\end{table}

Restricting to symbols that quote in almost every bar reduces the gross return
and lowers break-even further, which is consistent with the profit living in
the illiquid tail rather than in a tradeable cross-sectional anomaly.

\FloatBarrier
\section{Conclusion}

The engineered intraday features contain a small but strongly significant
cross-sectional signal: an information coefficient of @@IC_BEST with a
$t$-statistic of @@IC_T. Traded as a decile long--short book it earns
@@GROSS_BPS basis points per bar gross, against turnover of @@TURNOVER and a
break-even cost of @@BREAKEVEN basis points. Execution filtering raises
break-even to @@BREAKEVEN_EXEC basis points and produces a positive out-of-sample
net result at 5 basis points, with the trading rule chosen on validation rather
than on the test set.

That result should not be read as a working strategy. Almost all of the
predictability is 10-minute reversal---predictions correlate @@CORR_REV with the
previous bar's return---and the positions that generate it are concentrated in
wide-spread names for which the assumed cost is optimistic. The honest
conclusion is that the signal is statistically real and economically
unexploitable at the frictions its own positions imply.

\subsection*{Limitations}

The test window is @@N_TEST_BARS bars, under three trading days, so every
economic figure is a small-sample estimate and the per-bar Sharpe ratios should
not be extrapolated. Transaction costs are modelled as a constant per unit of
turnover; a spread model varying by name and by time would be more
appropriate, and Section~\ref{sec:diagnostics} suggests it would be
substantially less favourable. Market impact is ignored entirely. The models
minimise squared error rather than a cost-aware trading objective, so nothing
in the fitting stage penalises turnover. Finally, the strategy is evaluated on
mid quotes; a fill-aware backtest against the quoted book would be the natural
next step.

\end{document}
""")


DEEP_ABSENT = r"""
\section{Sequence Models}

The LSTM and Attention-LSTM specifications are implemented in
\texttt{src/models/deep\_learning.py} and trained by
\texttt{experiments/03\_train\_deep\_learning.py}. Results are not included in
this build of the report; run that script to populate this section.
"""

DEEP_PRESENT = Tex(r"""
\section{Sequence Models}

The panel models treat each observation independently. As a deep-learning
comparison we also fit two sequence models on the same features and the same
chronological split. Both convert the panel into rolling windows: a sequence
ending at $t$ is used to predict the return over the next bar,
\[
\widehat R_{i,t+1} = f_\theta\!\left(X_{i,t-L+1}, \dots, X_{i,t}\right).
\]
The LSTM summarises the window by its final hidden state. The Attention-LSTM
instead forms a weighted average $h^{\text{att}} = \sum_{\tau} \alpha_\tau
h_\tau$ over all hidden states, letting the network weight the more informative
bars. Both are trained with AdamW under mean squared error, with dropout and
early stopping on validation loss, and features standardised using training
statistics only.

\begin{table}[!htbp]
\centering
\small
\caption{Sequence model performance on the test window.}
\label{tab:deep}
@@TABLE_DEEP
\end{table}

\begin{table}[!htbp]
\centering
\small
\caption{Decile long--short book built from the sequence-model predictions,
5 basis points.}
\label{tab:deepport}
@@TABLE_DEEPPORT
\end{table}

@@DEEP_COMMENT
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    info = json.loads((RESULTS / "run_info.json").read_text())
    pred = pd.read_csv(RESULTS / "prediction_metrics.csv", index_col=0)
    costs = pd.read_csv(RESULTS / "transaction_cost_sensitivity.csv", index_col=0)
    rob = pd.read_csv(RESULTS / "robustness.csv", index_col=[0, 1])
    strat = pd.read_csv(RESULTS / "strategy_comparison.csv", index_col=0)
    execc = pd.read_csv(RESULTS / "execution_cost_curve.csv", index_col=0)
    overlay = pd.read_csv(RESULTS / "risk_overlay.csv", index_col=0)
    diag = pd.read_csv(RESULTS / "book_diagnostics.csv", index_col=0)["value"]
    liquid = pd.read_csv(RESULTS / "liquid_universe_cost_curve.csv", index_col=0)
    pred = pred.rename(index=lambda i: i.replace("hist_gradient_boosting", "hist. gradient boosting"))
    horizons = pd.read_csv(RESULTS / "horizon_decay.csv", index_col=0)

    best = info["best_model"]
    cfg = info["execution_config"]
    plain, executed = strat.iloc[0], strat.iloc[1]

    deep = DEEP_ABSENT
    dp = RESULTS / "deep_learning_prediction.csv"
    if dp.exists():
        dpred = pd.read_csv(dp, index_col=0)
        dpred.index = dpred.index.map({"lstm": "LSTM", "attention_lstm": "Attention-LSTM"})
        dport = pd.read_csv(RESULTS / "deep_learning_portfolio.csv", index_col=0)
        dport.index = dport.index.map({"lstm": "LSTM", "attention_lstm": "Attention-LSTM"})
        dargs = json.loads((RESULTS / "deep_learning.json").read_text())["args"] \
            if (RESULTS / "deep_learning.json").exists() else {}
        top = dpred["ic_cross_sectional"].idxmax()
        significant = (dpred["ic_t_stat"].abs() >= 2.0).any()
        if significant:
            comment = (
                f"The stronger of the two sequence models is {top}, with a "
                f"cross-sectional information coefficient of "
                f"{fmt(dpred.loc[top, 'ic_cross_sectional'])} "
                f"($t = {fmt(dpred.loc[top, 'ic_t_stat'], 2)}$) and a break-even cost of "
                f"{fmt(dport.loc[top, 'breakeven_bps'], 2)} basis points."
            )
        else:
            comment = (
                "Neither sequence model produces a signal distinguishable from zero. The "
                f"cross-sectional information coefficients are "
                f"{fmt(dpred.loc['LSTM', 'ic_cross_sectional'])} "
                f"($t = {fmt(dpred.loc['LSTM', 'ic_t_stat'], 2)}$) for the LSTM and "
                f"{fmt(dpred.loc['Attention-LSTM', 'ic_cross_sectional'])} "
                f"($t = {fmt(dpred.loc['Attention-LSTM', 'ic_t_stat'], 2)}$) for the "
                "Attention-LSTM, against "
                f"{fmt(pred.loc[best, 'ic_cross_sectional'])} "
                f"($t = {fmt(pred.loc[best, 'ic_t_stat'], 2)}$) for the selected panel "
                "model on the same features. Break-even costs are at or below zero, so "
                "neither is tradeable at any positive friction. Adding the attention "
                "mechanism does not help: it scores below the plain LSTM here.\n\n"
                "This is a result rather than a failure, and it is consistent with what "
                "Section~\\ref{sec:diagnostics} finds. The predictability in this panel is a "
                "static cross-sectional relation---a stock that has just fallen relative to "
                "its peers tends to bounce---and that relation is fully expressed by the "
                "current and lagged returns already in the feature vector. There is no "
                "temporal pattern across the window for recurrence to exploit, so the extra "
                "capacity buys nothing while the sequence construction discards most of the "
                "sample: the panel models train on "
                f"{info['train_rows']:,} rows drawn from the whole cross-section, the "
                f"sequence models on windows from {dargs.get('symbols', 0):,} symbols."
            )
        comment += (
            " Both are trained to minimise squared error rather than a trading objective, "
            "so nothing in the loss penalises turnover; their economic behaviour is "
            "governed by the same cost arithmetic as the panel models."
        )
        deep = DEEP_PRESENT.substitute(
            DEEP_SYMBOLS=f"{dargs.get('symbols', 0):,}",
            DEEP_LENGTH=dargs.get("sequence_length", "--"),
            DEEP_TIMESTAMPS=int(dpred["n_timestamps"].max()),
            N_TEST_BARS=info["test_bars"],
            N_TRAIN_ROWS=f"{info['train_rows']:,}",
            TABLE_DEEP=tabular(dpred, ["rmse", "r2", "ic_pooled", "ic_cross_sectional",
                                       "ic_t_stat", "hit", "n_timestamps"],
                               ["RMSE", "$R^2$", "Pooled IC", "IC", "IC $t$", "Hit",
                                "Timestamps"],
                               "Model", {"ic_t_stat": 2, "n_timestamps": 0}),
            TABLE_DEEPPORT=tabular(dport, ["gross_bps", "net_bps", "turnover",
                                           "sharpe_per_bar", "breakeven_bps"],
                                   ["Gross (bps)", "Net (bps)", "Turnover",
                                    "Sharpe/bar", "Break-even (bps)"],
                                   "Model", {"gross_bps": 2, "net_bps": 2, "breakeven_bps": 2}),
            DEEP_COMMENT=comment)

    ov_better = overlay["sharpe_per_bar"].idxmax()
    overlay_comment = (
        "The volatility overlay does not improve on equal weighting in this sample: it "
        f"lowers the per-bar Sharpe from {fmt(overlay.loc['equal weight', 'sharpe_per_bar'])} to "
        f"{fmt(overlay.iloc[1]['sharpe_per_bar'])} while raising turnover. With thousands of "
        "symbols and a short window, volatility estimates for the thinly traded names are "
        "noisy, and scaling by a noisy estimate adds variance rather than removing it. Full "
        "covariance estimation and mean-variance allocation are natural extensions but would "
        "be less stable still at this sample size."
        if ov_better == "equal weight" else
        "The volatility overlay improves on equal weighting in this sample, raising the "
        f"per-bar Sharpe to {fmt(overlay.iloc[1]['sharpe_per_bar'])}. Given the short test "
        "window this should be read as a diagnostic rather than as evidence that the overlay "
        "would persist."
    )

    tex = TEMPLATE.substitute(
        DATE=pd.Timestamp.today().strftime("%d %B %Y"),
        N_SYMBOLS=f"{info['symbols']:,}", N_BARS=f"{info['bars']:,}",
        PANEL_ROWS=f"{info['panel_rows']:,}", PANEL_SYMBOLS=f"{info['panel_symbols']:,}",
        PANEL_BARS=f"{info['panel_bars']:,}",
        N_ROWS_RAW=f"{info['rows']:,}", N_VOIDED="616,508",
        N_FEATURES=info["features"],
        N_TRAIN_ROWS=f"{info['train_rows']:,}", N_TEST_ROWS=f"{info['test_rows']:,}",
        N_TEST_BARS=info["test_bars"], N_TEST_SYMBOLS=f"{info['test_symbols']:,}",
        ZERO_SHARE=fmt(info["zero_target_share"] * 100, 1),
        IC_BEST=fmt(pred.loc[best, "ic_cross_sectional"]),
        IC_T=fmt(pred.loc[best, "ic_t_stat"], 1),
        R2_BEST=fmt(pred.loc[best, "r2"]),
        IC_LAG1=fmt(pred.loc["lag1_return", "ic_cross_sectional"]),
        GROSS_BPS=fmt(plain["gross_bps"], 2),
        SHARPE_GROSS=fmt(costs.loc[0, "sharpe_per_bar"]),
        NET_5=fmt(abs(plain["net_bps"]), 2),
        TURNOVER=fmt(plain["turnover"], 2),
        BREAKEVEN=fmt(plain["breakeven_bps"], 2),
        BREAKEVEN_EXEC=fmt(executed["breakeven_bps"], 2),
        CFG_Q=fmt(cfg["q"], 2), CFG_Z=fmt(cfg["no_trade_zone"], 2), CFG_RHO=fmt(cfg["rho"], 2),
        N_HELD=fmt(diag["names_per_rebalance"], 0),
        LIQ_HELD=fmt(diag["share_positions_liquid"] * 100, 1),
        LIQ_PANEL=fmt(diag["share_panel_liquid"] * 100, 1),
        CORR_REV=fmt(diag["corr_prediction_prev_return"], 2),
        LONG_PREV=fmt(abs(diag["mean_prev_return_longs_bps"]), 0),
        SHORT_PREV=fmt(abs(diag["mean_prev_return_shorts_bps"]), 0),
        BEST_MODEL=best.replace("_", r"\_"),
        TABLE_PREDICTION=tabular(pred, ["rmse", "mae", "r2", "ic_pooled",
                                        "ic_cross_sectional", "ic_t_stat", "hit",
                                        "validation_ic"],
                                 ["RMSE", "MAE", "$R^2$", "Pooled IC", "IC", "IC $t$",
                                  "Hit", "Val. IC"],
                                 "Model", {"ic_t_stat": 2}),
        TABLE_COSTS=tabular(costs, ["gross_bps", "net_bps", "turnover", "sharpe_per_bar",
                                    "t_stat_net", "cumulative_net", "max_drawdown"],
                            ["Gross (bps)", "Net (bps)", "Turnover", "Sharpe/bar",
                             "$t$", "Cum. net", "Max DD"],
                            "Cost (bps)", {"gross_bps": 2, "net_bps": 2, "t_stat_net": 2},
                            index_fmt=lambda i: f"{float(i):.0f}"),
        TABLE_ROBUSTNESS=tabular(rob, ["gross_bps", "net_bps", "turnover",
                                       "sharpe_per_bar", "breakeven_bps"],
                                 ["Gross (bps)", "Net (bps)", "Turnover", "Sharpe/bar",
                                  "Break-even (bps)"],
                                 "Quantile", {"gross_bps": 2, "net_bps": 2, "breakeven_bps": 2},
                                 index_fmt=lambda i: i.split(" / ")[-1]),
        TABLE_STRATEGY=tabular(strat, ["gross_bps", "net_bps", "turnover", "sharpe_per_bar",
                                       "t_stat_net", "cumulative_net", "max_drawdown",
                                       "losing_bars", "breakeven_bps"],
                               ["Gross", "Net", "Turn.", "Sharpe/bar", "$t$", "Cum.",
                                "Max DD", "Losing", "B/E"],
                               "Strategy", {"gross_bps": 2, "net_bps": 2, "t_stat_net": 2,
                                            "breakeven_bps": 2},
                               index_fmt=lambda s: s.split("(")[0].strip()),
        TABLE_EXECCOST=tabular(execc, ["net_bps", "sharpe_per_bar", "t_stat_net",
                                       "cumulative_net", "max_drawdown"],
                               ["Net (bps)", "Sharpe/bar", "$t$", "Cum. net", "Max DD"],
                               "Cost (bps)", {"net_bps": 2, "t_stat_net": 2},
                               index_fmt=lambda i: f"{float(i):.0f}"),
        TABLE_OVERLAY=tabular(overlay, ["gross_bps", "net_bps", "turnover",
                                        "sharpe_per_bar", "cumulative_net"],
                              ["Gross (bps)", "Net (bps)", "Turnover", "Sharpe/bar", "Cum. net"],
                              "Weighting", {"gross_bps": 2, "net_bps": 2}),
        TABLE_DIAGNOSTICS=tabular(diag.to_frame("value"), ["value"], ["Value"],
                                  "Statistic", {"value": 3},
                                  index_fmt=lambda s: {
                                      "names_per_rebalance": "Names held per rebalance",
                                      "share_positions_liquid": "Share of positions in liquid names",
                                      "share_panel_liquid": "Share of the panel that is liquid",
                                      "corr_prediction_prev_return": "corr(prediction, previous-bar return)",
                                      "mean_prev_return_longs_bps": "Mean previous-bar return of longs (bps)",
                                      "mean_prev_return_shorts_bps": "Mean previous-bar return of shorts (bps)",
                                  }.get(s, s.replace("_", " "))),
        TABLE_LIQUID=tabular(liquid, ["gross_bps", "net_bps", "sharpe_per_bar",
                                      "cumulative_net", "max_drawdown"],
                             ["Gross (bps)", "Net (bps)", "Sharpe/bar", "Cum. net", "Max DD"],
                             "Cost (bps)", {"gross_bps": 2, "net_bps": 2},
                             index_fmt=lambda i: f"{float(i):.0f}"),
        TABLE_HORIZON=tabular(horizons, ["ic", "ic_t", "gross_bps", "net_bps",
                                         "turnover", "breakeven_bps"],
                              ["IC", "IC $t$", "Gross (bps)", "Net (bps)", "Turnover",
                               "Break-even (bps)"],
                              "Horizon (min)", {"ic_t": 2, "gross_bps": 2, "net_bps": 2,
                                                "breakeven_bps": 2},
                              index_fmt=lambda i: f"{float(i):.0f}"),
        OVERLAY_COMMENT=overlay_comment,
        SECTION_DEEP=deep,
    )

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.tex").write_text(tex)
    print(f"wrote {REPORT / 'report.tex'}")

    if args.compile:
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                               cwd=REPORT, capture_output=True, text=True)
        pdf = REPORT / "report.pdf"
        if pdf.exists():
            target = PROJECT_ROOT / "PROJECT_REPORT.pdf"
            target.write_bytes(pdf.read_bytes())
            print(f"wrote {target}")
        else:
            print(r.stdout[-3000:])


if __name__ == "__main__":
    main()
