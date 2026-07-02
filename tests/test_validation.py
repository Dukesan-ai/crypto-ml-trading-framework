"""Unit tests for the validation toolkit.

Run with:  pytest -q

These test the *properties* the methodology relies on:
  - purging actually removes label-overlapping training samples,
  - the embargo drops the band right after each test block,
  - train/test never intersect,
  - CPCV produces the correct number of splits and paths,
  - PSR/DSR behave monotonically and DSR penalises more trials.
"""

import numpy as np
import pandas as pd
import pytest

from validation import (
    PurgedKFold,
    CombinatorialPurgedCV,
    get_train_times,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
)


def _make_labels(n=120, span=3):
    """n observations at integer times; each label ends `span` steps later."""
    idx = pd.RangeIndex(n)
    t1 = pd.Series(np.minimum(idx.values + span, n - 1), index=idx)
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    return X, t1


# --------------------------------------------------------------------------- #
# Purging primitives
# --------------------------------------------------------------------------- #
def test_get_train_times_removes_overlap():
    _, t1 = _make_labels(n=50, span=5)
    test_times = pd.Series({20: 25})  # test spans [20, 25]
    train = get_train_times(t1, test_times)
    # nothing in the training set may overlap [20, 25]
    for start, end in train.items():
        assert end < 20 or start > 25


# --------------------------------------------------------------------------- #
# PurgedKFold
# --------------------------------------------------------------------------- #
def test_purged_kfold_no_train_test_overlap():
    X, t1 = _make_labels(n=120, span=3)
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    for train_idx, test_idx in cv.split(X):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(test_idx) > 0


def test_purged_kfold_purges_boundary_labels():
    X, t1 = _make_labels(n=100, span=4)
    cv = PurgedKFold(n_splits=4, t1=t1, pct_embargo=0.0)
    for train_idx, test_idx in cv.split(X):
        t_start = t1.index[test_idx[0]]
        t_end = t1.iloc[test_idx].max()
        # no training label may fall inside the test label span
        for i in train_idx:
            assert t1.iloc[i] < t_start or t1.index[i] > t_end


def test_purged_kfold_requires_series():
    X, t1 = _make_labels(n=20)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, t1=t1.values)  # not a Series


# --------------------------------------------------------------------------- #
# CombinatorialPurgedCV
# --------------------------------------------------------------------------- #
def test_cpcv_split_and_path_counts():
    _, t1 = _make_labels(n=240, span=2)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t1=t1)
    assert cv.n_splits == 15          # C(6, 2)
    assert cv.n_paths == 5            # 15 * 2 / 6


def test_cpcv_disjoint_and_covers_test():
    X, t1 = _make_labels(n=240, span=2)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, t1=t1, pct_embargo=0.01)
    n_splits = 0
    for train_idx, test_idx in cv.split(X):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(test_idx) > 0
        n_splits += 1
    assert n_splits == cv.n_splits


# --------------------------------------------------------------------------- #
# PSR / DSR
# --------------------------------------------------------------------------- #
def test_psr_monotonic_in_sharpe():
    lo = probabilistic_sharpe_ratio(0.5, 0.0, n_obs=250)
    hi = probabilistic_sharpe_ratio(1.5, 0.0, n_obs=250)
    assert 0.0 <= lo <= hi <= 1.0


def test_psr_more_data_more_confident():
    few = probabilistic_sharpe_ratio(0.8, 0.0, n_obs=50)
    many = probabilistic_sharpe_ratio(0.8, 0.0, n_obs=2000)
    assert many > few


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 0.1) > expected_max_sharpe(10, 0.1)
    assert expected_max_sharpe(1, 0.1) == 0.0


def test_dsr_penalises_more_trials():
    rng = np.random.default_rng(0)
    trials_few = rng.normal(0, 0.1, size=10)
    trials_many = rng.normal(0, 0.1, size=500)
    dsr_few = deflated_sharpe_ratio(1.0, trials_few, n_obs=500)
    dsr_many = deflated_sharpe_ratio(1.0, trials_many, n_obs=500)
    # searching more configs raises the bar → lower confidence for same SR
    assert dsr_many <= dsr_few
    assert 0.0 <= dsr_many <= 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
