"""
Cross-Run Consensus Feature Selection
=====================================

Tooling for the problem described in `docs/01-feature-selection-consensus.md`:
on thin, correlated financial data, any *single* feature-selection run is a
coin-flip among interchangeable features, so the deployable object is the
*cross-run consensus*, not any one run's output.

This module wraps an arbitrary selection procedure and adds the three things a
single run lacks:

1. ``selection_frequencies`` — run the selector across many resamples/seeds and
   record how often each feature is chosen (Stability Selection; Meinshausen &
   Bühlmann, 2010).
2. ``consensus_features``    — lock the features that survive a frequency
   threshold. This is the candidate set you then *validate* (see
   `validation/metric_separation.py`); the frequency only nominates.
3. ``seed_stability``        — the diagnostic that separates stochasticity from
   genuine drift: hold the sample fixed, vary only the seed. Low set-overlap on
   *identical data* proves the instability is the substitution effect, not the
   data changing.

The selector is injected, so this works with any procedure (RFE, importance
thresholding, L1, …). It is generic; it contains no strategy, parameters, or
signals. ``selector(X, y, random_state) -> sequence of selected column labels``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Hashable, Sequence

import numpy as np
import pandas as pd

Selector = Callable[[pd.DataFrame, pd.Series, int], Sequence[Hashable]]


def selection_frequencies(
    selector: Selector,
    X: pd.DataFrame,
    y: pd.Series,
    n_runs: int = 50,
    subsample: float = 0.75,
    random_state: int = 0,
) -> pd.Series:
    """Fraction of resampled runs in which each feature is selected.

    Each run draws a ``subsample`` fraction of rows without replacement and runs
    ``selector`` on it with a fresh seed. Returns a Series indexed by feature,
    sorted high→low. Note: these frequencies are *biased low for correlated
    features* (vote-splitting) — use them to nominate, not as importances.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    m = max(1, int(n * subsample))
    counts = pd.Series(0.0, index=X.columns, dtype=float)
    for _ in range(n_runs):
        rows = rng.choice(n, size=m, replace=False)
        chosen = selector(X.iloc[rows], y.iloc[rows], int(rng.integers(0, 2**31 - 1)))
        counts.loc[list(chosen)] += 1.0
    return (counts / n_runs).sort_values(ascending=False)


def consensus_features(frequencies: pd.Series, threshold: float = 0.6) -> list:
    """Features selected in at least ``threshold`` of runs — the locked candidate
    set to hand to out-of-sample validation."""
    return frequencies[frequencies >= threshold].index.tolist()


def seed_stability(selector: Selector, X: pd.DataFrame, y: pd.Series,
                   seeds: Sequence[int]) -> dict:
    """Drift-vs-stochasticity diagnostic on a *fixed* sample.

    Runs ``selector`` on the **same** rows under each seed and measures pairwise
    Jaccard overlap of the selected sets. A low mean Jaccard means the procedure
    returns different subsets on identical data — i.e. the instability is the
    substitution effect, not the data changing. (If you instead want to test for
    drift, vary the *data* window and hold the seed fixed.)
    """
    sets = [set(selector(X, y, int(s))) for s in seeds]
    jac = []
    for a, b in combinations(sets, 2):
        union = len(a | b)
        jac.append(len(a & b) / union if union else 1.0)
    return {"mean_jaccard": float(np.mean(jac)) if jac else 1.0, "sets": sets}


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 800

    # Three "signal families", each represented by 3 CORRELATED twins -> this is
    # what creates the substitution effect / vote-splitting.
    cols: dict[str, np.ndarray] = {}
    drivers = {f"sig{f}": rng.normal(size=n) for f in range(3)}
    for fam, s in drivers.items():
        for j in range(3):
            cols[f"{fam}_{j}"] = s + rng.normal(scale=0.30, size=n)   # twin
    # One LONE signal with no twins (equal strength, but no vote to split).
    lone = rng.normal(size=n)
    cols["lone_signal"] = lone + rng.normal(scale=0.30, size=n)
    # Pure-noise distractors.
    for k in range(15):
        cols[f"noise_{k}"] = rng.normal(size=n)

    X = pd.DataFrame(cols)
    logit = sum(drivers.values()) + lone               # label depends on signals only
    y = pd.Series((logit + rng.normal(scale=0.5, size=n) > 0).astype(int))

    def rf_topk(Xi, yi, seed, k=6):
        rf = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=-1)
        rf.fit(Xi, yi)
        imp = pd.Series(rf.feature_importances_, index=Xi.columns)
        return imp.nlargest(k).index.tolist()

    print("=== 1. Single-run instability on IDENTICAL data (8 seeds, top-6) ===")
    diag = seed_stability(rf_topk, X, y, seeds=range(8))
    for i, s in enumerate(diag["sets"][:4]):
        print(f"  seed {i}: {sorted(s)}")
    print(f"  mean pairwise Jaccard = {diag['mean_jaccard']:.2f}   "
          f"(< 1.0 on identical data => the swing is stochastic, not drift)\n")

    print("=== 2. Cross-run consensus (60 resampled runs) ===")
    freq = selection_frequencies(rf_topk, X, y, n_runs=60)
    print(freq.head(11).to_string())
    print(f"\n  consensus @0.6: {consensus_features(freq, 0.6)}")
    print("  note: 'lone_signal' scores higher than any individual twin of equal")
    print("        true strength — that gap IS vote-splitting (see doc 01).")
