"""
Leakage-resistant cross-validation for financial ML.

Implements the two splitters this framework relies on:

  * ``PurgedKFold``           — K-fold with *purging* of train labels that
                                overlap the test interval, plus an *embargo*
                                on the samples immediately following each test
                                block (López de Prado, AFML Ch. 7).
  * ``CombinatorialPurgedCV`` — Combinatorial Purged Cross-Validation (CPCV):
                                test on every combination of ``k`` of ``N``
                                groups, purge + embargo each split, and expose
                                the number of back-test *paths* the scheme
                                produces (López de Prado, AFML Ch. 12).

Why this exists
---------------
Naïve K-fold leaks on time series in two ways: (1) a label built from a window
of future returns overlaps neighbouring samples, so a training label can share
information with a test sample; (2) serial correlation bleeds across the
train/test boundary. Purging removes the overlapping training samples; the
embargo drops a thin band after the test block. Without both, out-of-sample
scores are silently inflated — which corrupts the one metric you rely on to
tell edge from overfitting.

This module contains **no strategy, features, or parameters** — only the
validation machinery. Labels are described purely by ``t1``: a ``pd.Series``
whose index is each observation's start time and whose value is the time the
observation's label is *determined* (e.g. the triple-barrier touch time).

References
----------
Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley 2018,
Ch. 7 (cross-validation, purging, embargo) and Ch. 12 (CPCV).
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------- #
# Purging / embargo primitives
# --------------------------------------------------------------------------- #
def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """Purge training observations whose label window overlaps any test window.

    A training observation ``(start=i, end=t1[i])`` overlaps a test window
    ``(start=j0, end=j1)`` in three ways, all of which are removed:

    1. train starts inside the test window,
    2. train ends inside the test window,
    3. train envelops the test window.

    Parameters
    ----------
    t1
        Series of *all* observations: index = observation start time,
        value = label end time.
    test_times
        Series of test-set label spans: index = test start, value = test end.

    Returns
    -------
    pd.Series
        The subset of ``t1`` that is safe to train on (AFML snippet 7.1).
    """
    trn = t1.copy(deep=True)
    for start, end in test_times.items():
        starts_within = trn[(start <= trn.index) & (trn.index <= end)].index
        ends_within = trn[(start <= trn) & (trn <= end)].index
        envelops = trn[(trn.index <= start) & (end <= trn)].index
        trn = trn.drop(starts_within.union(ends_within).union(envelops))
    return trn


def get_embargo_times(times: pd.DatetimeIndex, pct_embargo: float) -> pd.Series:
    """Map each timestamp to the first timestamp allowed after its embargo.

    The embargo is expressed as a fraction of the sample; ``pct_embargo=0.01``
    embargoes 1% of observations after each test block (AFML snippet 7.2).
    """
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        return pd.Series(times, index=times)
    embargo = pd.Series(times[step:], index=times[:-step])
    embargo = pd.concat([embargo, pd.Series(times[-1], index=times[-step:])])
    return embargo


# --------------------------------------------------------------------------- #
# PurgedKFold
# --------------------------------------------------------------------------- #
class PurgedKFold(KFold):
    """K-fold cross-validation with purging and embargo for overlapping labels.

    Drop-in for :class:`sklearn.model_selection.KFold` but requires ``t1`` so it
    can purge. Folds are contiguous in time (never shuffled).

    Parameters
    ----------
    n_splits
        Number of folds.
    t1
        Label end times; index must align with ``X`` passed to :meth:`split`.
    pct_embargo
        Fraction of the sample embargoed after each test block.

    Notes
    -----
    Canonical implementation of AFML snippet 7.3. Yields *positional* indices,
    matching the scikit-learn splitter contract.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 pct_embargo: float = 0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series (index=start, value=label end).")
        if not (0.0 <= pct_embargo < 1.0):
            raise ValueError("pct_embargo must be in [0, 1).")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError("X and t1 must have the same length.")
        if not np.array_equal(np.asarray(X.index), np.asarray(self.t1.index)):
            raise ValueError("X and t1 must share the same index.")

        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        test_blocks = [(b[0], b[-1] + 1) for b in np.array_split(indices, self.n_splits)]

        for start, stop in test_blocks:
            t0 = self.t1.index[start]                      # test block start time
            test_idx = indices[start:stop]
            # `side="right"` so the right-side train block begins strictly AFTER
            # the test's last label ends — a train label that merely touches the
            # boundary is still a leak, so it is excluded.
            max_t1_idx = self.t1.index.searchsorted(
                self.t1.iloc[test_idx].max(), side="right")

            # left train: labels that end strictly before the test block starts
            train_idx = self.t1.index.searchsorted(
                self.t1[self.t1 < t0].index)
            # right train: everything after the test block's last label + embargo
            if max_t1_idx < X.shape[0]:
                right = indices[max_t1_idx + embargo:]
                train_idx = np.concatenate((train_idx, right))

            yield train_idx, test_idx


# --------------------------------------------------------------------------- #
# Combinatorial Purged Cross-Validation (CPCV)
# --------------------------------------------------------------------------- #
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (López de Prado, AFML Ch. 12).

    Split the sample into ``n_groups`` contiguous groups and use every
    combination of ``n_test_groups`` groups as the test set. Each split is
    purged and embargoed. Testing every combination yields a *distribution* of
    out-of-sample results over many back-test paths rather than a single
    fragile estimate.

    The number of distinct paths is::

        n_paths = C(n_groups, n_test_groups) * n_test_groups / n_groups

    Parameters
    ----------
    n_groups
        Number of contiguous groups the sample is partitioned into (``N``).
    n_test_groups
        Number of groups held out per split (``k``). ``k=2`` is the common
        choice; ``k=1`` reduces to purged K-fold.
    t1
        Label end times; index must align with ``X`` passed to :meth:`split`.
    pct_embargo
        Fraction of the sample embargoed after each test group.

    Examples
    --------
    >>> cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t1=t1)
    >>> cv.n_splits            # C(6, 2)
    15
    >>> cv.n_paths             # 15 * 2 / 6
    5
    >>> for train_idx, test_idx in cv.split(X):
    ...     ...                # fit on train_idx, score on test_idx
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 t1: pd.Series | None = None, pct_embargo: float = 0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series (index=start, value=label end).")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError("Require 1 <= n_test_groups < n_groups.")
        if not (0.0 <= pct_embargo < 1.0):
            raise ValueError("pct_embargo must be in [0, 1).")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.pct_embargo = pct_embargo

    @property
    def n_splits(self) -> int:
        """Number of train/test combinations = C(n_groups, n_test_groups)."""
        return len(list(combinations(range(self.n_groups), self.n_test_groups)))

    @property
    def n_paths(self) -> int:
        """Number of distinct back-test paths recoverable from the splits."""
        return self.n_splits * self.n_test_groups // self.n_groups

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError("X and t1 must have the same length.")
        if not np.array_equal(np.asarray(X.index), np.asarray(self.t1.index)):
            raise ValueError("X and t1 must share the same index.")

        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        group_bounds = [(b[0], b[-1] + 1) for b in np.array_split(indices, self.n_groups)]

        for test_group_ids in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate(
                [indices[group_bounds[g][0]:group_bounds[g][1]] for g in test_group_ids])
            test_idx = np.sort(test_idx)

            # test label span drives purging
            test_times = pd.Series(
                self.t1.iloc[test_idx].values, index=self.t1.index[test_idx])
            train_t1 = get_train_times(self.t1, test_times)
            train_idx = self.t1.index.searchsorted(train_t1.index)

            # embargo: drop a band after each contiguous test block
            if embargo > 0:
                banned = set()
                for g in test_group_ids:
                    stop = group_bounds[g][1]
                    banned.update(range(stop, min(stop + embargo, X.shape[0])))
                if banned:
                    train_idx = np.array([i for i in train_idx if i not in banned])

            yield train_idx, test_idx
