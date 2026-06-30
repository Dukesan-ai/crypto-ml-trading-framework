# crypto-ml-trading-framework

**End-to-end machine-learning system for crypto perpetual futures — signal research, leakage-resistant validation, and live execution on real capital.**

Built solo on the methodology of *Advances in Financial Machine Learning* (Marcos López de Prado), extended with market-microstructure alpha, regime modeling, and concept-drift–aware research.

> **What this is not.** Most public repos referencing *Advances in Financial Machine Learning* are book exercises or model demos that predict price with no out-of-sample discipline — exactly the failure mode the book warns about. This is the opposite: a **system that has run live on real capital**, where the entire design centers on *separating real edge from backtest overfitting*. Several pieces here are not in the book — they are problems I hit in production and had to solve myself: how conditionally-useful features get penalized by an averaged selection criterion, why the selection metric must differ from the validation metric, and how to keep feature computation identical between training and live inference. The judgment is the point, not the code.

> **Scope of this repository.** This documents the **architecture and research methodology** of a live system. Proprietary signal logic, feature indices, and tuned parameters are intentionally omitted. The goal is to show *how the system is designed and how edge is separated from overfitting* — not to disclose tradeable strategy.

---

## Overview

A complete, independently built pipeline that takes raw tick-level market data through feature engineering, model training, out-of-sample validation, and live 24/7 execution. The system is designed around a single hard problem in quantitative trading: **telling real edge apart from backtest overfitting**, and only deploying what survives honest out-of-sample scrutiny.

It currently runs **live, forward-tested on real capital**, tracking out-of-sample performance against pre-registered metrics on a rolling basis.

---

## Architecture: Two-Model Meta-Labeling

The system separates *what to trade* from *whether and how much* — the meta-labeling design from López de Prado.

```
            ┌──────────────────────┐
 market ───▶│   Primary model      │── direction (long / short) + conviction
 features   │   (RF, dollar bars)  │
            └──────────┬───────────┘
                       │ conviction (out-of-fold, leakage-free)
                       ▼
            ┌──────────────────────┐
 orthogonal │   Meta-model         │── P(primary is right)
 regime  ──▶│   (trade? size?)     │──▶ bet size  m = 2·Φ(z) − 1
 features   └──────────┬───────────┘
                       ▼
              risk-managed, sized position  (false positives filtered out)
```

- **Primary model** — predicts trade direction from market features, using triple-barrier labels on information-driven (dollar) bars.
- **Meta-model** — takes the primary model's conviction plus **orthogonal regime features**, then decides (a) whether the signal is worth trading (false-positive filter) and (b) position size, via a probability-to-size mapping.
- **Key design choice:** the meta-model never sees the primary model's own features — only information the primary model could *not* exploit. Reusing the primary's features yields no information advantage; the value comes from features the primary model leaves on the table. This is enforced explicitly, and confirmed empirically (the meta-model consistently selects regime signals the primary model discards as noise).
- **Long and short are handled by separate meta-models** (a two-model / asymmetric architecture), reflecting the distinct informational drivers of up- and down-moves.

---

## Core Components

### 1. Feature engineering — microstructure alpha
Diverse alpha signals engineered from tick-level market-microstructure data — order-flow, liquidity, and informed-flow proxies — alongside trend, volatility, and macro-regime features. Bars are sampled on traded volume (dollar bars), not clock time, to keep information content per bar roughly constant.

### 2. Feature selection — robustness over raw fit
Features are selected for a **robust, low-redundancy** set rather than maximal in-sample fit:
- **Clustered feature importance (CFI)** — cluster correlated features, then rank and select at the *cluster* level, so the substitution effect doesn't fragment importance across correlated twins.
- **Stability selection / cross-run consensus** (Meinshausen–Bühlmann) — run selection across many resamples and keep what is *consistently* chosen, which adds false-discovery control that single-run selection lacks.
- A recurring, hard-won lesson: on thin, highly-correlated financial data, the "best" single-run subset is unstable because the features are interchangeable. **Consensus across runs — not any one run's output — is what's deployable.**

### 3. Validation — separating edge from overfitting *(the core of the system)*
The part that matters most. In-sample performance is treated as ~zero evidence; the question is always *does it survive honest out-of-sample testing.*
- **Combinatorial Purged Cross-Validation (CPCV)** with purging and embargo — produces a *distribution* of out-of-sample performance across many train/test paths, not a single fragile backtest, and removes look-ahead leakage from overlapping labels.
- **Deflated / Probabilistic Sharpe (DSR / PSR)** — discounts measured performance by the number of configurations tried, controlling for selection under multiple testing.
- **An out-of-sample holdout used strictly as a judge** — feature selection happens only on the in-sample partition; the holdout is never used to select, only to evaluate.
- Selection metric and validation metric are deliberately *different* (a proper scoring rule for selection; risk-adjusted return as the validation gate) — chosen so the validation step cannot be gamed by the selection step.

### 4. Live deployment & concept drift
Deployed to trade **24/7 on real capital**, with a rolling forward record that tracks live out-of-sample performance against the metrics fixed at research time. Research is explicitly **concept-drift–aware**: the system distinguishes ordinary model instability (the substitution effect) from genuine regime drift, so it doesn't chase noise — but does flag when a once-reliable signal is decaying.

Surviving honest validation is the prerequisite, not the finish line. Running this live surfaced a second discipline that matters as much as the validation itself: **knowing when a correct method is the wrong move** — when a "better" component is quietly destroying value, when a rule should override the model and when overriding is the gambler's fallacy, and when the obvious architectural fix is a category error. That judgment, and the evidence behind each call, is the subject of notes 04 and 05.

---

## Design principles

- **Optimize for money, justified by method — not for statistics for its own sake.** Every choice is grounded in a specific, defensible reason (and, where possible, in the literature), then verified on out-of-sample results.
- **In-sample fit is not edge.** A subset that wins in-sample but loses out-of-sample is overfitting, full stop — and that signature is actively watched for.
- **Smaller and stable beats large and fitted** on thin, noisy data.
- **Provenance over vibes.** Methodology decisions are documented with their rationale and the evidence that backed them.

---

## Methodology notes

Longer write-ups on how I think about the hard problems in this system. They
contain no strategy, parameters, or signals — only reasoning and the evidence
behind specific decisions.

- **[01 — Feature selection: why a single run is not deployable](docs/01-feature-selection-consensus.md)** · the substitution effect, and cross-run consensus over single-run selection.
- **[02 — Separating real edge from overfitting](docs/02-validation-overfitting.md)** · how I design validation: CPCV, deflated/probabilistic Sharpe, and reading the numbers honestly.
- **[03 — Meta-labeling done right](docs/03-meta-labeling-information-advantage.md)** · orthogonal information and where a second model's edge actually comes from.
- **[04 — The eleventh reason ML strategies fail](docs/04-production-lessons.md)** · the gap between a correct strategy and a profitable one — delivery, train/serve consistency, and applying correct methods correctly.
- **[05 — When the right move is the wrong one](docs/05-judgment-over-machinery.md)** · two cases where the obvious optimization is the mistake: a "better" entry-price model that the data said to delete, and why regime *detection* is the wrong frame.
- **[06 — Arguing against myself](docs/06-arguing-against-myself.md)** · steelmanning the schools that disagree with this methodology — including one where my own data sided with the opposition for a while — and what that exercise is actually for.

---

## Tech stack

`Python` · scikit-learn · NumPy / pandas · parallelized feature construction · custom CPCV / DSR / PSR validation · live execution & data pipeline (exchange API, real-time).

---

## Contact

✉️ 117966377duke@gmail.com

---

*This repository is a methodology and architecture overview. It intentionally contains no tradeable parameters, signal definitions, or strategy code.*
