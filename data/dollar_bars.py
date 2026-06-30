"""
Dollar Bars (Information-Driven Bars)
=====================================

Sampling tick data into bars by **cumulative traded value** instead of clock
time, following López de Prado, *Advances in Financial Machine Learning* (2018),
Ch. 2.

Why not time bars
-----------------
Markets do not deliver information at a constant rate — activity clusters. Fixed
time bars (1-minute, 1-hour) therefore oversample quiet periods and undersample
busy ones, which gives them poor statistical properties: returns are
heteroskedastic and heavy-tailed, and serial correlation is high. Sampling a new
bar every time a fixed amount of *dollar value* (price x volume) has traded
synchronises the clock to market activity instead. The result is bars whose
returns are closer to IID and noticeably closer to normal — a better substrate
for ML.

This is a generic, self-contained reference implementation (NumPy / pandas only).
No strategy, parameters, or signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def suggest_threshold(ticks: pd.DataFrame, bars_per_day: float,
                      price_col: str = "price", volume_col: str = "volume") -> float:
    """A reasonable dollar-per-bar threshold to target ``bars_per_day`` on average.

    threshold = (total dollar volume) / (number of days) / (bars per day).
    """
    dollar = float((ticks[price_col] * ticks[volume_col]).sum())
    span_days = (ticks.index[-1] - ticks.index[0]) / pd.Timedelta(days=1)
    return dollar / max(span_days, 1e-9) / bars_per_day


def dollar_bars(ticks: pd.DataFrame, threshold: float,
                price_col: str = "price", volume_col: str = "volume") -> pd.DataFrame:
    """Aggregate ticks into dollar bars.

    Parameters
    ----------
    ticks : pd.DataFrame
        Tick records indexed by timestamp, with a price and a volume column.
    threshold : float
        Close a new bar once cumulative ``price * volume`` reaches this value.
    price_col, volume_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        One row per bar, indexed by the bar's close timestamp, with columns
        ``t_open, open, high, low, close, volume, dollar, n_ticks``. The final
        (possibly partial) bar is included.
    """
    px = ticks[price_col].to_numpy(dtype=float)
    vol = ticks[volume_col].to_numpy(dtype=float)
    ts = ticks.index
    dollar = px * vol

    bars = []
    cum = 0.0
    seg_start = 0
    n = px.shape[0]
    for i in range(n):
        cum += dollar[i]
        if cum >= threshold or i == n - 1:
            seg = slice(seg_start, i + 1)
            p = px[seg]
            bars.append({
                "t_open": ts[seg_start],
                "t_close": ts[i],
                "open": p[0],
                "high": p.max(),
                "low": p.min(),
                "close": p[-1],
                "volume": float(vol[seg].sum()),
                "dollar": float(dollar[seg].sum()),
                "n_ticks": i - seg_start + 1,
            })
            cum = 0.0
            seg_start = i + 1
    return pd.DataFrame(bars).set_index("t_close")


def _excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4 or x.std() == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3.0)


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(3)

    # Synthesise ~3 days of irregular ticks with CLUSTERED activity: three busy
    # bursts where both tick frequency and size jump (this is what time bars
    # handle badly and dollar bars adapt to).
    minutes = []
    t = pd.Timestamp("2024-01-01")
    for day in range(3):
        for m in range(24 * 60):
            now = t + pd.Timedelta(days=day, minutes=m)
            # base activity, with bursts around three times of day
            busy = any(abs(m - c) < 30 for c in (300, 720, 1140))
            n_ticks = rng.poisson(8 if busy else 2)
            minutes.extend([now] * n_ticks)
    stamps = pd.DatetimeIndex(minutes)
    n = len(stamps)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    volume = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    ticks = pd.DataFrame({"price": price, "volume": volume}, index=stamps)

    thr = suggest_threshold(ticks, bars_per_day=96)     # ~96 bars/day target
    bars = dollar_bars(ticks, threshold=thr)

    print(f"ticks: {n:,}   threshold: {thr:,.0f} $/bar   ->   {len(bars)} dollar bars")
    print("\nbar duration adapts to activity (shorter during bursts):")
    dur_min = bars.index.to_series().diff().dt.total_seconds().div(60).dropna()
    print(f"  median {dur_min.median():.1f} min, but ranges {dur_min.min():.1f}–{dur_min.max():.1f} min")
    print(f"  ticks/bar: median {bars['n_ticks'].median():.0f}, "
          f"range {bars['n_ticks'].min()}–{bars['n_ticks'].max()}")

    # The pay-off: dollar-bar returns are closer to normal than time-bar returns.
    dbar_ret = np.log(bars["close"]).diff().dropna().to_numpy()
    tbar = ticks["price"].resample("15min").last().dropna()
    tbar_ret = np.log(tbar).diff().dropna().to_numpy()
    print("\nexcess kurtosis of returns (closer to 0 = closer to normal):")
    print(f"  15-min time bars : {_excess_kurtosis(tbar_ret):+.2f}")
    print(f"  dollar bars      : {_excess_kurtosis(dbar_ret):+.2f}")
