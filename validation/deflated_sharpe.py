"""
Probabilistic & Deflated Sharpe Ratio (PSR / DSR)
=================================================

Overfitting-aware Sharpe-ratio statistics, following Bailey & López de Prado,
"The Deflated Sharpe Ratio" (2014) and *Advances in Financial Machine Learning*
(2018), Ch. 8 & 14.

The problem
-----------
A high in-sample Sharpe ratio means little on its own. Two effects inflate it:

* **Short, non-normal samples.** The Sharpe estimator has variance that grows
  with non-normality (skew/kurtosis) and shrinks with sample length ``T``.
  *PSR* asks: given ``T``, skew and kurtosis, what is the probability the *true*
  Sharpe exceeds a benchmark ``SR*``?

* **Multiple testing / selection bias.** If you try ``N`` configurations and keep
  the best, its Sharpe is upward-biased even with zero true edge. *DSR* deflates
  the benchmark ``SR*`` to the **expected maximum** Sharpe under ``N`` independent
  trials, then runs PSR against that.

DSR > 0.95 ≈ "this Sharpe is unlikely to be a multiple-testing fluke." It is a
*necessary* check, not a sufficient one — live forward performance is still the
final judge.

Generic, self-contained reference implementation (NumPy/SciPy only).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Non-annualised Sharpe ratio of a return series (same frequency as input)."""
    r = np.asarray(returns, dtype=float) - risk_free
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    r"""Probabilistic Sharpe Ratio: :math:`P(\text{true SR} > \text{benchmark})`.

    .. math::
        \widehat{PSR}(SR^*) = \Phi\!\left(
            \frac{(\widehat{SR}-SR^*)\sqrt{T-1}}
                 {\sqrt{1 - \hat\gamma_3\,\widehat{SR}
                          + \tfrac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}
        \right)

    Parameters
    ----------
    observed_sr, benchmark_sr : float
        Estimated Sharpe and the threshold ``SR*`` (same frequency as ``n_obs``).
    n_obs : int
        Number of return observations ``T``.
    skew : float
        Skewness :math:`\gamma_3` of the returns.
    kurtosis : float
        **Non-excess** kurtosis :math:`\gamma_4` (a normal distribution = 3).

    Returns
    -------
    float in [0, 1]
    """
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(
        max(1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2, 1e-12)
    )
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    r"""Expected maximum of ``n_trials`` i.i.d. Sharpe estimates with zero mean
    and variance ``sr_variance`` — the deflated benchmark ``SR*_0``.

    .. math::
        SR^*_0 = \sqrt{\widehat V}\,\Big[(1-\gamma)\,\Phi^{-1}\!\big(1-\tfrac1N\big)
                 + \gamma\,\Phi^{-1}\!\big(1-\tfrac1N e^{-1}\big)\Big]

    where :math:`\gamma` is the Euler–Mascheroni constant and :math:`\widehat V`
    is the variance of the Sharpe estimates across the ``N`` trials.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - g) * q1 + g * q2))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_estimates: np.ndarray,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: PSR against a benchmark deflated for multiple testing.

    Parameters
    ----------
    observed_sr : float
        Sharpe of the *selected* (best) configuration.
    sr_estimates : np.ndarray
        Sharpe ratios of **all** configurations tried. Their count gives ``N``
        and their spread gives the variance used to deflate the benchmark.
        Pass at least the trials you searched over — under-counting inflates DSR.
    n_obs : int
        Number of return observations for the selected configuration.
    skew, kurtosis : float
        Moments of the selected configuration's returns (kurtosis non-excess).

    Returns
    -------
    float in [0, 1]
        P(true SR > expected-max-under-noise). Higher = less likely a fluke.
    """
    sr_estimates = np.asarray(sr_estimates, dtype=float)
    n_trials = sr_estimates.size
    sr_star = expected_max_sharpe(sr_estimates.var(ddof=1), n_trials)
    return probabilistic_sharpe_ratio(observed_sr, sr_star, n_obs, skew, kurtosis)


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from scipy.stats import skew as _skew, kurtosis as _kurt

    rng = np.random.default_rng(0)

    # A genuinely-skilled strategy: small positive daily edge.
    real = rng.normal(0.0008, 0.01, size=1500)
    sr = sharpe_ratio(real)
    g3 = float(_skew(real))
    g4 = float(_kurt(real, fisher=False))   # non-excess
    print(f"Observed Sharpe (per-obs)   : {sr:+.4f}  (skew {g3:+.2f}, kurt {g4:.2f})")
    print(f"PSR vs 0                    : {probabilistic_sharpe_ratio(sr, 0.0, len(real), g3, g4):.4f}")

    # Now suppose we searched 50 configs and kept the best — deflate for that.
    trial_srs = rng.normal(0.0, 0.03, size=50)      # noise-only trial Sharpes
    trial_srs[trial_srs.argmax()] = sr              # the one we "selected"
    dsr = deflated_sharpe_ratio(sr, trial_srs, len(real), g3, g4)
    print(f"DSR (deflated for 50 trials): {dsr:.4f}")

    # A pure-noise 'winner' cherry-picked from 200 trials should NOT survive.
    noise = rng.normal(0.0, 0.01, size=1500)
    noise_sr = sharpe_ratio(noise)
    noise_trials = rng.normal(0.0, 0.03, size=200)
    noise_trials[noise_trials.argmax()] = noise_sr
    print(f"\nCherry-picked noise Sharpe  : {noise_sr:+.4f}")
    print(f"DSR (deflated for 200)      : {deflated_sharpe_ratio(noise_sr, noise_trials, len(noise)):.4f}  "
          f"(low = correctly flagged as luck)")
