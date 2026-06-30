"""
Selection-Metric / Validation-Metric Separation
===============================================

Tooling for the central idea in `docs/02-validation-overfitting.md`: the metric
you *select* features on and the metric you *validate* on must be different, so
the optimiser cannot reach through the selection step and inflate the validation
result. When they differ, a characteristic *overfitting signature* becomes
visible — a candidate that wins the in-sample selection metric but loses out of
sample is overfitting; one that *loses* the selection metric but *wins* out of
sample has learned signal that generalises.

This module provides:

* ``return_weighted_negll`` — a proper scoring rule (weighted log-loss) used as
  the SELECTION criterion. It rewards calibrated probabilities and, crucially, is
  *not* a backtest return — so optimising it does not open the backtest-overfitting
  channel that selecting directly on Sharpe would.
* ``net_sharpe`` — a simple risk-adjusted return used as the (different)
  VALIDATION gate, scored out of sample.
* ``compare_candidates`` — fits each candidate feature set in-sample, scores the
  selection metric in-sample and the validation metric out of sample, and labels
  each candidate's *signature* relative to a baseline.

Generic; no strategy, parameters, or signals. The classifier ``make_model`` and
the feature sets are injected by the caller.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def return_weighted_negll(y_true: np.ndarray, proba: np.ndarray,
                          returns: np.ndarray, eps: float = 1e-6) -> float:
    """|return|-weighted negative log-loss. Lower = better.

    A proper scoring rule: it is minimised by well-calibrated probabilities, and
    weighting each sample by the magnitude of its associated return focuses the
    criterion on the observations that actually matter economically — without
    ever becoming a tradeable P&L itself (which is what keeps it safe to optimise).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    w = np.abs(np.asarray(returns, dtype=float))
    w = w / (w.mean() + eps)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return float(-(w * ll).mean())


def net_sharpe(strategy_returns: np.ndarray) -> float:
    """Non-annualised Sharpe of a realised return stream (the validation gate)."""
    r = np.asarray(strategy_returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def _evaluate(make_model: Callable, X: pd.DataFrame, y: pd.Series,
              returns: pd.Series, features: Sequence,
              is_idx: np.ndarray, oos_idx: np.ndarray) -> dict:
    """Fit on in-sample; selection metric in-sample, validation metric out-of-sample."""
    model = make_model()
    model.fit(X.iloc[is_idx][list(features)], y.iloc[is_idx])

    # selection criterion: in-sample, proper scoring rule
    p_is = model.predict_proba(X.iloc[is_idx][list(features)])[:, 1]
    sel = return_weighted_negll(y.iloc[is_idx].values, p_is, returns.iloc[is_idx].values)

    # validation criterion: OUT-OF-SAMPLE realised risk-adjusted return
    p_oos = model.predict_proba(X.iloc[oos_idx][list(features)])[:, 1]
    side = np.sign(p_oos - 0.5)                       # take the model's call
    strat = side * returns.iloc[oos_idx].values
    val = net_sharpe(strat)
    return {"selection_negll": sel, "validation_sharpe": val}


def compare_candidates(make_model: Callable, X: pd.DataFrame, y: pd.Series,
                       returns: pd.Series, candidates: dict[str, Sequence],
                       is_idx: np.ndarray, oos_idx: np.ndarray,
                       baseline: str) -> pd.DataFrame:
    """Score every candidate on both metrics and classify its signature.

    Signature of each candidate *relative to* ``baseline``:
      * ``overfitting``   : better selection metric, worse validation metric
                            (fit in-sample noise; failed out of sample).
      * ``generalizing``  : worse selection metric, better validation metric
                            (could not fit in-sample noise as well, yet wins OOS
                            => it captured signal, not noise).
      * ``consistent``    : both metrics move the same way.

    The signature only exists because the two metrics are different. Reading the
    selection metric alone would keep the overfitting candidate and discard the
    generalizing one — exactly backwards.
    """
    rows = {name: _evaluate(make_model, X, y, returns, feats, is_idx, oos_idx)
            for name, feats in candidates.items()}
    df = pd.DataFrame(rows).T
    base = df.loc[baseline]

    sig = []
    for name in df.index:
        if name == baseline:
            sig.append("— baseline —")
            continue
        better_sel = df.at[name, "selection_negll"] < base["selection_negll"]   # lower better
        better_val = df.at[name, "validation_sharpe"] > base["validation_sharpe"]
        if better_sel and not better_val:
            sig.append("overfitting")
        elif not better_sel and better_val:
            sig.append("generalizing")
        else:
            sig.append("consistent")
    df["signature_vs_baseline"] = sig
    return df


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    n_is, n_oos = 700, 500
    n = n_is + n_oos
    is_idx = np.arange(n_is)
    oos_idx = np.arange(n_is, n)

    # A persistent driver present in BOTH partitions -> drives forward returns.
    driver = rng.normal(size=n)
    fwd_ret = 0.012 * driver + rng.normal(scale=0.012, size=n)     # realised return
    y = pd.Series((fwd_ret > 0).astype(int))
    returns = pd.Series(fwd_ret)

    # SIGNAL feature: a noisy view of the persistent driver (works in/out of sample).
    feat_signal = driver + rng.normal(scale=0.5, size=n)

    # NOISE feature: aligned with the in-sample LABEL but random out of sample.
    # This is the classic overfitting trap — an in-sample relationship that does
    # not generalise.
    feat_noise = np.empty(n)
    feat_noise[is_idx] = (y.iloc[is_idx].values * 2 - 1) + rng.normal(scale=0.4, size=n_is)
    feat_noise[oos_idx] = rng.normal(size=n_oos)

    X = pd.DataFrame({"signal": feat_signal, "noise": feat_noise})

    candidates = {
        "noise_set":  ["noise"],     # fits in-sample, should fail OOS
        "signal_set": ["signal"],    # weaker in-sample, should win OOS
    }

    result = compare_candidates(
        make_model=lambda: LogisticRegression(max_iter=200),
        X=X, y=y, returns=returns, candidates=candidates,
        is_idx=is_idx, oos_idx=oos_idx, baseline="noise_set",
    )

    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    print("Selection metric is IN-SAMPLE (lower negll = better fit).")
    print("Validation metric is OUT-OF-SAMPLE (higher Sharpe = better).\n")
    print(result.to_string())
    print("\nThe noise_set wins the in-sample selection metric but loses OOS;")
    print("the signal_set is flagged 'generalizing' — worse in-sample, better OOS.")
    print("Reading the selection metric alone would have kept the wrong one.")
