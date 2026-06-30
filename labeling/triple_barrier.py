"""
Triple-Barrier Labeling
=======================

Path-dependent labeling for financial ML, following López de Prado,
*Advances in Financial Machine Learning* (2018), Ch. 3.

Idea
----
Fixed-horizon labels ("up or down after N bars") ignore *how* the path got there
— a position that hit a stop-loss intra-path is mislabeled as a win if it
recovers by bar N. Triple-barrier labeling fixes this by setting three barriers
per event and labeling by **whichever is touched first**:

* upper barrier    = profit-take    (``pt * volatility``)   -> label +1
* lower barrier    = stop-loss      (``sl * volatility``)   -> label -1
* vertical barrier = time limit     (``t1``)                -> label  0

Barrier widths scale with recent volatility, so labels are comparable across
calm and turbulent regimes. The label end time (the touch time) is exactly the
``t1`` series that purged cross-validation (see ``purged_cv.py``) needs to
prevent leakage from overlapping labels.

(The vertical-barrier outcome is labeled ``0`` here — a three-class scheme. The
original ``getBins`` variant instead labels it by the sign of the realized
return; both are common, and the choice depends on whether the timed-out case
should be a distinct class.)

Generic, self-contained reference implementation (NumPy/pandas only). No alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FAR_FUTURE = pd.Timestamp.max


def realized_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """EWM standard deviation of log returns — the barrier-scaling volatility."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=span).std()


def vertical_barriers(close: pd.Series, events: pd.Index, n_bars: int) -> pd.Series:
    """Time limit for each event: ``n_bars`` ahead, clipped to the series end."""
    positions = close.index.searchsorted(events) + n_bars
    inside = positions[positions < close.shape[0]]
    return pd.Series(close.index[inside], index=events[: inside.shape[0]])


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
) -> pd.DataFrame:
    """Find the first-touch timestamp of each barrier for every event.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by time.
    events : pd.DataFrame
        Index = event start times. Columns:
          * ``t1``   : vertical-barrier (time-limit) timestamp.
          * ``trgt`` : target width (volatility) for the horizontal barriers.
          * ``side`` : +1 long / -1 short (direction the barriers apply to).
    pt_sl : (float, float)
        Multipliers on ``trgt`` for the (profit-take, stop-loss) barriers.
        A ``0`` disables that horizontal barrier.

    Returns
    -------
    pd.DataFrame
        Columns ``t1``, ``sl``, ``pt`` — the timestamps at which each barrier was
        first touched (NaT if never). The earliest of the three is the label end.
    """
    out = events[["t1"]].copy()
    out["sl"] = pd.NaT
    out["pt"] = pd.NaT

    pt_mult, sl_mult = pt_sl
    for start, t1 in events["t1"].fillna(close.index[-1]).items():
        path = close.loc[start:t1]
        side = events.at[start, "side"]
        trgt = events.at[start, "trgt"]
        if not np.isfinite(trgt) or trgt <= 0:
            continue
        # signed log return along the path, in the position's direction
        path_ret = np.log(path / close.loc[start]) * side
        if pt_mult > 0:
            hit = path_ret[path_ret >= pt_mult * trgt]
            if not hit.empty:
                out.at[start, "pt"] = hit.index[0]
        if sl_mult > 0:
            hit = path_ret[path_ret <= -sl_mult * trgt]
            if not hit.empty:
                out.at[start, "sl"] = hit.index[0]
    return out


def get_labels(touches: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Label each event by the first barrier touched.

    label = +1  profit-take hit first
            -1  stop-loss hit first
             0  vertical barrier (time limit) reached first

    Returns columns ``t1_touch`` (label end time), ``ret`` (realized log return
    to that time) and ``label``.
    """
    pt_t = touches["pt"].fillna(_FAR_FUTURE)
    sl_t = touches["sl"].fillna(_FAR_FUTURE)
    t1_t = touches["t1"].fillna(_FAR_FUTURE)
    first = pd.concat([pt_t, sl_t, t1_t], axis=1).min(axis=1)

    label = pd.Series(0, index=touches.index, dtype=int)   # default: vertical
    label[sl_t == first] = -1
    label[pt_t == first] = 1                                # pt wins ties over sl

    out = pd.DataFrame(index=touches.index)
    out["t1_touch"] = first.where(first < _FAR_FUTURE)
    valid = out["t1_touch"].notna()
    out["ret"] = np.nan
    out.loc[valid, "ret"] = (
        np.log(close.loc[out.loc[valid, "t1_touch"].values].values
               / close.loc[out.index[valid]].values)
    )
    out["label"] = label
    return out


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 2_000
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

    vol = realized_volatility(close, span=50)

    # Take an event every 5 bars (in practice: from a primary model / sampler).
    event_idx = idx[50::5]
    t1 = vertical_barriers(close, event_idx, n_bars=12)        # 12-bar time limit
    events = pd.DataFrame(index=t1.index)
    events["t1"] = t1
    events["trgt"] = vol.loc[events.index]                     # vol-scaled width
    events["side"] = 1                                         # long-only demo

    touches = apply_triple_barrier(close, events, pt_sl=(3.0, 3.0))   # ±3 vol barriers
    labels = get_labels(touches, close).dropna(subset=["t1_touch"])

    counts = labels["label"].value_counts().sort_index()
    print(f"events labeled : {len(labels)}")
    print("label balance  :  (-1 stop-loss | 0 timed-out | +1 profit-take)")
    for k in (-1, 0, 1):
        print(f"   {k:+d}: {counts.get(k, 0)}")
    print("\nthe 't1_touch' column is exactly the label-end series that")
    print("PurgedKFold / CombinatorialPurgedCV consume to prevent leakage.")
