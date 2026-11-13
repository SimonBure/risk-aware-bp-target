"""The discussion's sensor noise: how much the virtual sensor's correlated noise changes the risk (eq. (risk))
against the white noise the forecast assumes.

A constant predicted MAP, 50 readings 10 s apart (W_t), noise of the same variance, white or the
sensor's stationary AR(1) of coefficient rho = e^-1. The threshold is set so that the white-noise risk
is alpha: each reading stays above it with probability (1 - alpha)^(1/50). The AR(1) risk,
1 - P(all 50 readings above), is computed by integrating its Gaussian transition density over the
region above the threshold, reading after reading, on a grid. The paper: 5 % becomes 4.87 %, and every
level changes by less than 4 % relative.

Run ``python drivers/sensor_noise.py`` (seconds).
"""
import numpy as np
from scipy.stats import norm

from bptarget.forecast import ALPHAS
from bptarget.virtual import RHO

N_READINGS = 50
SPAN, N_GRID = 12.0, 6000  # the grid above the threshold, in SDs, and its points


def correlated_risk(alpha: float, n: int = N_READINGS, rho: float = RHO) -> float:
    z = norm.ppf(1 - (1 - alpha) ** (1 / n))  # the standardised threshold
    x = np.linspace(z, z + SPAN, N_GRID)
    w = np.full(N_GRID, x[1] - x[0])
    w[[0, -1]] /= 2                           # the trapezoid rule
    s = np.sqrt(1 - rho ** 2)
    kernel = norm.pdf((x[:, None] - rho * x[None, :]) / s) / s
    f = norm.pdf(x)                           # the density of the first reading, kept above the threshold
    for _ in range(n - 1):
        f = kernel @ (w * f)
    return float(1 - np.sum(w * f))


if __name__ == "__main__":
    print(f"{N_READINGS} readings, AR(1) rho = {RHO:.4f}")
    for a in ALPHAS:
        r = correlated_risk(a)
        print(f"white noise {100 * a:5.2f} %  ->  correlated {100 * r:5.2f} %  ({100 * (r - a) / a:+.1f} % relative)")
