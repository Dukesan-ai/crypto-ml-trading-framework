"""
Selection-bias-aware performance statistics.

Implements the Probabilistic and Deflated Sharpe Ratios (Bailey & López de
Prado, 2014). These answer the question a raw Sharpe cannot: *given how many
configurations I searched, and given short, skewed, fat-tailed returns, is this
Sharpe distinguishable from luck?*

  * ``probabilistic_sharpe_ratio`` (PSR) — probability the true SR exceeds a
    benchmark, correcting for sample length, skewness and kurtosis.
  * ``expected_max_sharpe``            — the SR you'd expect to see as the *max*
    of ``N`` independent trials with no real edge (the false-discovery bar).
  * ``deflated_sharpe_ratio`` (DSR)    — PSR evaluated against that inflated
    benchmark, i.e. deflated for the number of trials actually run.

The discipline that makes DSR honest is counting *every* configuration searched
— not just the winner. Under-count the trials and the deflated metric is itself
inflated. This module contains no strategy or parameters, only the statistics.

References
----------
D. Bailey & M. López de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting, and Non-Normality," *Journal of Portfolio
Management*, 2014. Also AFML Ch. 8 (feature importance) and Ch. 11–12.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sr``.

    All Sharpe ratios must be expressed in the **same frequency** as the return
    series used for ``n_obs``, ``skew`` and ``kurtosis`` (do not annualise one
    side only).

    Parameters
    ----------
    observed_sr
        Estimated Sharpe ratio of the strategy.
    benchmark_sr
        Threshold Sharpe to beat (0 for "better than nothing"; for DSR this is
        the expected-maximum benchmark from :func:`expected_max_sharpe`).
    n_obs
        Number of return observations.
    skew
        Skewness of the returns (0 for Normal).
    kurtosis
        Kurtosis of the returns (3 for Normal — *not* excess kurtosis).

    Returns
    -------
    float
        PSR in ``[0, 1]``. Values near 1 mean the observed SR is very unlikely
        to be a statistical artifact of the benchmark.
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(
        1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    )
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent, edgeless trials.

    Uses the analytic approximation from Bailey & López de Prado (2014)::

        E[max SR] ≈ σ_SR · [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    where ``γ`` is the Euler–Mascheroni constant and ``Z⁻¹`` is the inverse
    standard normal CDF. This is the benchmark a genuine strategy must clear to
    be distinguishable from the best of many noise trials.

    Parameters
    ----------
    n_trials
        Number of configurations searched (independent trials).
    sr_std
        Standard deviation of the Sharpe ratios across those trials.
    """
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_trials: np.ndarray,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected-maximum benchmark.

    Deflates the observed Sharpe for the number of configurations searched, so
    a high Sharpe found by trying many variants is not mistaken for edge.

    Parameters
    ----------
    observed_sr
        Sharpe of the selected (best) configuration.
    sr_trials
        Sharpe ratios of *all* configurations tried (including discarded ones).
        Its length is the trial count and its dispersion sets the deflation.
    n_obs
        Number of return observations for the selected configuration.
    skew, kurtosis
        Moments of the selected configuration's returns (kurtosis = 3 for Normal).

    Returns
    -------
    float
        DSR in ``[0, 1]``. A common decision rule accepts a strategy only if
        DSR > 0.95.
    """
    sr_trials = np.asarray(sr_trials, dtype=float)
    n_trials = sr_trials.size
    if n_trials < 2:
        raise ValueError("Need at least 2 trials to deflate; pass every config searched.")
    sr_std = float(np.std(sr_trials, ddof=1))
    benchmark = expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe_ratio(observed_sr, benchmark, n_obs, skew, kurtosis)
