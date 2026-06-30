"""
Purged & Combinatorial Purged Cross-Validation (CPCV)
=====================================================

Leakage-resistant cross-validation for financial machine learning, following
Marcos López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 7 & 12.

Why this exists
---------------
Plain k-fold CV leaks information on financial data for two reasons:

1. **Overlapping labels.** A label at time ``t`` is usually derived from a window
   of *future* prices (e.g. triple-barrier outcomes). If a training sample's
   label window overlaps a test sample's, information bleeds across the split.
   *Purging* removes the offending training samples.

2. **Serial correlation across the split boundary.** Even non-overlapping samples
   just after the test block are correlated with it. *Embargo* drops a small band
   of training samples immediately following each test block.

``PurgedKFold`` is a single-path purged splitter. ``CombinatorialPurgedCV``
generalises it: instead of one train/test partition per fold, it tests every
``C(n_groups, n_test_groups)`` combination of groups, yielding many backtest
*paths*. Performance then becomes a **distribution** across paths rather than a
single fragile number — which is what makes deflated/probabilistic Sharpe (see
``deflated_sharpe.py``) meaningful.

This is a generic, self-contained reference implementation (NumPy/pandas only).
It contains no strategy logic, parameters, or signals.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Purging                                                                      #
# --------------------------------------------------------------------------- #
def purge_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """Remove training observations whose label window overlaps any test label
    window (AFML snippet 7.1).

    Parameters
    ----------
    t1 : pd.Series
        Full label spans. Index = observation start time, value = label *end*
        time. (For triple-barrier labels this is the barrier-touch time.)
    test_times : pd.Series
        Same structure, restricted to the test observations.

    Returns
    -------
    pd.Series
        ``t1`` with every training observation that overlaps the test set removed.
    """
    train = t1.copy(deep=True)
    for start, end in test_times.items():
        # three ways a training span can overlap the test span [start, end]:
        starts_within = train[(start <= train.index) & (train.index <= end)].index
        ends_within = train[(start <= train) & (train <= end)].index
        envelops = train[(train.index <= start) & (end <= train)].index
        train = train.drop(starts_within.union(ends_within).union(envelops))
    return train


def embargo_index(t1: pd.Series, pct_embargo: float) -> pd.Series:
    """Map each observation to the first observation that is *clear* of its
    embargo band (AFML snippet 7.2).

    A non-zero embargo additionally drops ``pct_embargo`` of observations right
    after each test block to kill leakage from serial correlation.
    """
    n = t1.shape[0]
    step = int(n * pct_embargo)
    if step == 0:
        ans = pd.Series(t1.index, index=t1.index)
    else:
        ans = pd.Series(t1.index[step:], index=t1.index[: n - step])
        ans = pd.concat([ans, pd.Series(t1.index[-1], index=t1.index[n - step:])])
    return ans


# --------------------------------------------------------------------------- #
# Single-path purged k-fold                                                    #
# --------------------------------------------------------------------------- #
class PurgedKFold:
    """K-fold splitter with purging + embargo, sklearn-style ``split`` API.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        Label spans (index = start time, value = end time), aligned to ``X``.
    pct_embargo : float
        Fraction of samples to embargo after each test block. ``0.0`` = pure purge.

    Notes
    -----
    ``shuffle`` is deliberately unsupported: shuffling time series destroys the
    temporal structure that purging exists to protect.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 pct_embargo: float = 0.0) -> None:
        if t1 is None:
            raise ValueError("t1 (label end times) is required for purging.")
        self.n_splits = int(n_splits)
        self.t1 = t1
        self.pct_embargo = float(pct_embargo)

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if (X.index != self.t1.index).any():
            raise ValueError("X and t1 must share the same index.")
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        test_blocks = [(b[0], b[-1] + 1) for b in np.array_split(indices, self.n_splits)]

        for start, end in test_blocks:
            t0 = self.t1.index[start]                       # test block start time
            test_idx = indices[start:end]
            # last training row whose label could overlap this test block
            max_t1 = self.t1.index.searchsorted(self.t1.iloc[test_idx].max())
            # left side: everything whose label ends before the test starts
            train_idx = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index
            )
            # right side: everything after the test block + embargo band
            if max_t1 < X.shape[0]:
                train_idx = np.concatenate((train_idx, indices[max_t1 + embargo:]))
            yield train_idx, test_idx


# --------------------------------------------------------------------------- #
# Combinatorial Purged CV                                                      #
# --------------------------------------------------------------------------- #
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (AFML Ch. 12).

    Split the sample into ``n_groups`` contiguous groups, then for every
    combination of ``n_test_groups`` test groups, train on the rest (purged +
    embargoed) and test on the held-out groups. This produces

        n_paths = C(n_groups, n_test_groups) * n_test_groups / n_groups

    distinct backtest paths, so out-of-sample performance is a distribution.

    Parameters
    ----------
    n_groups : int
        Number of contiguous groups N.
    n_test_groups : int
        Groups held out per combination, k (1 <= k < N).
    t1 : pd.Series
        Label spans (index = start, value = end), aligned to ``X``.
    pct_embargo : float
        Embargo fraction.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 t1: pd.Series | None = None, pct_embargo: float = 0.0) -> None:
        if t1 is None:
            raise ValueError("t1 (label end times) is required for purging.")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("require 1 <= n_test_groups < n_groups.")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.t1 = t1
        self.pct_embargo = float(pct_embargo)

    @property
    def n_splits(self) -> int:
        from math import comb
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return self.n_splits * self.n_test_groups // self.n_groups

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if (X.index != self.t1.index).any():
            raise ValueError("X and t1 must share the same index.")
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        groups = np.array_split(indices, self.n_groups)

        for test_grp_ids in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in test_grp_ids])
            test_idx.sort()
            test_times = self.t1.iloc[test_idx]

            # start from all candidate training rows, then purge against test
            train_pool = self.t1.drop(test_times.index, errors="ignore")
            train_pool = purge_train_times(train_pool, test_times)
            train_idx = self.t1.index.searchsorted(train_pool.index)

            # embargo: drop a band right after each contiguous test block
            if embargo > 0:
                banned = set()
                for g in test_grp_ids:
                    last = groups[g][-1]
                    banned.update(range(last + 1, min(last + 1 + embargo, X.shape[0])))
                if banned:
                    train_idx = np.array([i for i in train_idx if i not in banned])
            yield train_idx, test_idx


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Toy example: 1,000 observations, each label spanning the next ~5 steps.
    n = 1_000
    idx = pd.RangeIndex(n)
    X = pd.DataFrame({"feature": np.random.randn(n)}, index=idx)
    # label end = start + a small random horizon (this is what creates overlap)
    horizons = np.random.randint(1, 8, size=n)
    t1 = pd.Series(np.minimum(idx.values + horizons, n - 1), index=idx)

    print("PurgedKFold (5 folds, 1% embargo)")
    pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    for i, (tr, te) in enumerate(pkf.split(X)):
        print(f"  fold {i}: train={tr.size:>4d}  test={te.size:>4d}")

    print("\nCombinatorialPurgedCV (N=6, k=2, 1% embargo)")
    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t1=t1, pct_embargo=0.01)
    print(f"  combinations = {cpcv.n_splits}, backtest paths = {cpcv.n_paths}")
    for i, (tr, te) in enumerate(cpcv.split(X)):
        if i < 3:
            print(f"  split {i}: train={tr.size:>4d}  test={te.size:>4d}")
    print("  ...")
