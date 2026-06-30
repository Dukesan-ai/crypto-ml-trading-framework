"""
Meta-Labeling
=============

The two-model architecture from López de Prado, *Advances in Financial Machine
Learning* (2018), Ch. 3, with the orthogonality discipline argued in
`docs/03-meta-labeling-information-advantage.md`.

A primary model decides *direction*; a meta-model decides *whether to act on that
call and how much to size it*. The meta-model is a binary classifier: given that
the primary says "trade," is this one of the cases where the primary tends to be
right? Its calibrated probability becomes the bet size, and it filters out the
primary's false positives.

Two things this module makes explicit:

* ``make_meta_labels`` — meta-labels are ``1`` when the primary's directional call
  was profitable, ``0`` otherwise. The meta-model learns to predict *the primary's
  correctness*, not direction.
* ``assert_orthogonal`` — the hard constraint from the methodology note: the
  meta-model must **not** see the primary model's own features. Sharing features
  yields no information advantage; the meta-model's value comes from information
  the primary could not use (regime, the primary's own conviction/reliability).
  This is enforced, because the failure mode is silent.

Bet sizing follows the probability-to-size transform (AFML Ch. 10):
``m = 2*Phi(z) - 1`` with ``z = (p - 1/2) / sqrt(p*(1-p))``.

Generic, self-contained reference implementation. No strategy, parameters, or
signals; the meta-model and meta-features are injected by the caller.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def make_meta_labels(primary_side: np.ndarray, realized_return: np.ndarray) -> np.ndarray:
    """``1`` if the primary's directional call made money, else ``0``.

    ``primary_side`` is +1 / -1; ``realized_return`` is the underlying's signed
    return over the bet horizon (e.g. the triple-barrier return). The product is
    the primary's P&L; its sign is the meta-label.
    """
    pnl = np.asarray(primary_side, dtype=float) * np.asarray(realized_return, dtype=float)
    return (pnl > 0).astype(int)


def assert_orthogonal(primary_features: Sequence, meta_features: Sequence) -> None:
    """Enforce that the meta-model sees none of the primary model's own features.

    Raises ``ValueError`` on any overlap. Keeping the two feature sets disjoint is
    what gives the meta-model an *information advantage* (see docs/03); violating
    it quietly turns the meta-model into a re-run of the primary.
    """
    overlap = set(primary_features) & set(meta_features)
    if overlap:
        raise ValueError(
            f"meta features overlap primary features {sorted(overlap)}: no "
            "information advantage — the meta-model must see only what the "
            "primary could not (see docs/03)."
        )


def bet_size(proba: np.ndarray, take_only: bool = True) -> np.ndarray:
    """Map meta-model probability to a position size via ``2*Phi(z) - 1``.

    ``proba`` = P(primary is right). With ``take_only=True`` (the meta-labeling
    convention) the size is clipped to ``[0, 1]`` — the meta-model decides how
    much of the primary's bet to take, never to reverse it. Direction stays the
    primary's.
    """
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1.0 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1.0 - p))
    size = 2.0 * norm.cdf(z) - 1.0
    return np.clip(size, 0.0, 1.0) if take_only else size


class MetaLabeler:
    """Train a meta-model on orthogonal meta-features and size the primary's bets.

    Parameters
    ----------
    make_model : Callable[[], estimator]
        Factory for a fresh sklearn-style classifier with ``predict_proba``.
    primary_features, meta_features : sequence
        Used only to assert orthogonality at construction time.
    """

    def __init__(self, make_model: Callable, primary_features: Sequence,
                 meta_features: Sequence) -> None:
        assert_orthogonal(primary_features, meta_features)
        self.make_model = make_model
        self.meta_features = list(meta_features)
        self.model_ = None

    def fit(self, meta_X: pd.DataFrame, primary_side: np.ndarray,
            realized_return: np.ndarray) -> "MetaLabeler":
        y_meta = make_meta_labels(primary_side, realized_return)
        self.model_ = self.make_model()
        self.model_.fit(meta_X[self.meta_features], y_meta)
        return self

    def predict_proba(self, meta_X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(meta_X[self.meta_features])[:, 1]

    def position(self, meta_X: pd.DataFrame, primary_side: np.ndarray) -> np.ndarray:
        """Final position = primary direction x meta-derived size."""
        size = bet_size(self.predict_proba(meta_X), take_only=True)
        return np.asarray(primary_side, dtype=float) * size


def _sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(5)
    n = 3_000
    split = 2_000
    tr, te = np.arange(split), np.arange(split, n)

    # A REGIME variable that governs when the primary is right — and which the
    # primary itself does NOT see (this is the orthogonal information).
    regime = rng.normal(size=n)
    p_correct = 1.0 / (1.0 + np.exp(-(0.15 + 1.3 * regime)))   # right more often in some regimes
    correct = rng.random(n) < p_correct

    primary_side = rng.choice([-1.0, 1.0], size=n)              # the primary's calls
    move = np.abs(rng.normal(0.010, 0.004, size=n))            # size of the move
    realized_return = primary_side * np.where(correct, move, -move)

    # Meta-feature: a noisy view of the regime (orthogonal to the primary).
    meta_X = pd.DataFrame({"regime_proxy": regime + rng.normal(scale=0.6, size=n)})

    meta = MetaLabeler(
        make_model=lambda: GradientBoostingClassifier(random_state=0),
        primary_features=["primary_feat_A", "primary_feat_B"],   # disjoint by construction
        meta_features=["regime_proxy"],
    )
    meta.fit(meta_X.iloc[tr], primary_side[tr], realized_return[tr])

    # Out-of-sample comparison: trade EVERY primary signal vs meta-filtered/sized.
    raw_ret = primary_side[te] * realized_return[te]                       # take all
    pos = meta.position(meta_X.iloc[te], primary_side[te])
    meta_ret = pos * realized_return[te]                                   # filtered + sized

    print("Out-of-sample, primary hit-rate vs meta-labeled:")
    print(f"  primary hit-rate            : {correct[te].mean():.1%}")
    print(f"  trades taken (size>0)       : {(pos != 0).mean():.1%} of signals")
    print(f"  avg size when taken         : {np.abs(pos[pos != 0]).mean():.2f}")
    print()
    print(f"  Sharpe — take every signal  : {_sharpe(raw_ret):+.3f}")
    print(f"  Sharpe — meta-labeled       : {_sharpe(meta_ret):+.3f}")
    print("\nThe meta-model sizes down / skips the regimes where the primary is")
    print("unreliable, lifting risk-adjusted return without touching direction.")
