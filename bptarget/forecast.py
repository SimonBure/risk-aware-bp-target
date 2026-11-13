"""The law of the next readings, the risk of hypotension and the score (Section II.B, Appendix C).

To first order in eta, the readings of a window at s_1 < ... < s_n are
Y_j = yhat_j + (A xi)_j + eps_j, A = S_w L, P_t = L L^T, xi ~ N(0, I), eps_j ~ N(0, sigma_t^2)
independent (eq. (readings), ``Predictive``). The risk is that of the lowest reading below c,
R_t = P(min_j Y_j < c) = 1 - E prod_j Phi((yhat_j + (A xi)_j - c) / sigma_t), eq. (risk): the noise
integrated out, xi sampled (``Draws``). The score e_t^2 / V_t has mean one under the law.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import log_ndtr

HYPO = 65.0                        # mmHg
ALPHAS = (0.01, 0.02, 0.05, 0.10)  # the risk levels
N_DRAWS = 10_000                   # draws of xi
TOL = 0.05                         # mmHg, the target's bisection width


@dataclass(frozen=True)
class Predictive:
    """The readings of a window under the linearised posterior: ``mean`` yhat (n,), ``a`` = S_w L (n, d),
    the noise variance ``sigma2``, at ``times`` (n,)."""
    times: np.ndarray
    mean: np.ndarray
    a: np.ndarray
    sigma2: float

    @property
    def var(self) -> np.ndarray:
        """The fit's share of each reading's variance, (S_w P S_w^T)_jj."""
        return np.sum(self.a ** 2, axis=1)


def predictive(times, mean, s, p, sigma2: float) -> Predictive:
    """The law of the readings at ``times``: the forecast ``mean``, its sensitivities ``s`` (n, d), the
    posterior covariance ``p`` (d, d) and the noise variance ``sigma2``."""
    return Predictive(np.asarray(times, float), np.asarray(mean, float), np.asarray(s, float) @ np.linalg.cholesky(p),
                      float(sigma2))


def score(pred: Predictive, y: np.ndarray) -> float:
    """e^2 / V of the readings ``y``: their mean squared error over the mean variance of a reading."""
    return float(np.mean((np.asarray(y, float) - pred.mean) ** 2) / (np.mean(pred.var) + pred.sigma2))


@dataclass(frozen=True)
class Draws:
    """Standard normal draws of xi, (K, d), the same for every target of a decision."""
    xi: np.ndarray
    eps: np.ndarray


def draws(d: int, n: int, k: int = N_DRAWS, seed=0) -> Draws:
    rng = np.random.default_rng(seed)
    return Draws(rng.standard_normal((k, d)), rng.standard_normal((k, n)))  # eps: drawn to keep the seeds' streams


def risk(pred: Predictive, c: float, dr: Draws) -> float:
    """P(min_j Y_j < c), eq. (risk)."""
    mu = pred.mean + dr.xi @ pred.a.T  # (K, n): the MAP of each draw of the parameters
    sig = np.sqrt(pred.sigma2)
    if sig == 0.0:
        return float(np.mean(mu.min(axis=1) < c))
    return float(1.0 - np.mean(np.exp(np.sum(log_ndtr((mu - c) / sig), axis=1))))


def pit(pred: Predictive, values: np.ndarray, dr: Draws) -> float:
    """Where the lowest recorded reading falls in the forecast's law of the minimum: P(min_j Y_j < its
    recorded value). Calibrated, it is uniform, and the minimum falls below the forecast's alpha-quantile
    when it is below alpha."""
    return risk(pred, float(np.min(values)), dr)


def smallest_target(risk_of: Callable[[float], float], alpha: float, lo: float, hi: float,
                    tol: float = TOL) -> float | None:
    """The smallest y in [lo, hi] with ``risk_of(y)`` <= alpha, by bisection to ``tol`` (the risk falls as
    the target rises); None when even ``hi`` fails, or when hi < lo."""
    if hi < lo or risk_of(hi) > alpha:
        return None
    if risk_of(lo) <= alpha:
        return float(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if risk_of(mid) <= alpha:
            hi = mid
        else:
            lo = mid
    return float(hi)
