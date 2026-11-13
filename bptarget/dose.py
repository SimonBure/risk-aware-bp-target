"""The control input of Section II.C: a bolus over 10 s on top of a constant infusion.

The infusion holds the target at equilibrium, u = kel h^-1(y), clipped to [0, u_max] (an unreachable
target is projected onto [F(0), F(u_max)]). The bolus is the smallest in [0, B_max] whose MAP reaches
the target within T_r, by bisection: the reduced model's PK is positive and linear, so the MAP grows
with the bolus at every time (Appendix D). The whole cap when even it misses T_r.
"""
from dataclasses import dataclass

import numpy as np

from bptarget.model import PK, Grid, reduced_map, reduced_map_inv, simulate_exact

RATE_MAX = 0.5          # nmol/s, u_max
BOLUS_MAX = 75.0        # nmol, B_max
BOLUS_DURATION = 10.0   # s
DEADLINE = 100.0        # s, T_r
HORIZON = 600.0         # s, the infusion is held this long
T_CMD = np.arange(0.0, HORIZON + 0.5, 1.0)  # s from the dose's start: where the MAP is checked
BOLUS_XTOL = 1e-4       # nmol, the bisection's width


@dataclass(frozen=True)
class Dose:
    bolus: float         # nmol, over BOLUS_DURATION
    rate: float          # nmol/s
    deadline_met: bool   # the MAP reaches the (projected) target within T_r


def map_under(theta: dict, x0: np.ndarray, rate: float, bolus: float, t: np.ndarray) -> np.ndarray:
    """The reduced model's MAP at ``t`` (s from the dose's start) from the state ``x0``, under the rate plus
    the bolus over its first 10 s; exact."""
    nodes = np.union1d(t, [BOLUS_DURATION])  # the bolus ends on a node
    steps = [[(b - a, rate + (bolus / BOLUS_DURATION if a < BOLUS_DURATION else 0.0))]
             for a, b in zip(nodes[:-1], nodes[1:])]
    x = simulate_exact([theta[k] for k in PK], Grid(nodes, steps, np.arange(len(nodes))), x0)
    return np.asarray(reduced_map(theta, x[1, np.isin(nodes, t)]))


def dose(theta: dict, x0: np.ndarray, target: float, deadline: float = DEADLINE) -> Dose:
    """The dose that reaches ``target`` from the state ``x0`` under the reduced model ``theta``."""
    raw = np.inf if target >= theta["Emax"] else float(reduced_map_inv(theta, target)) * theta["kel"]
    rate = float(np.clip(raw, 0.0, RATE_MAX))
    if not np.isfinite(rate):
        raise ValueError(f"target MAP {target} is at or above Emax and the infusion is unbounded")
    goal = float(reduced_map(theta, rate / theta["kel"]))  # the target, projected

    def reach(bolus: float) -> float:
        y = map_under(theta, x0, rate, bolus, T_CMD)
        hit = np.flatnonzero(y >= goal)
        return float(T_CMD[hit[0]]) if len(hit) else float(np.inf)

    if reach(0.0) <= deadline:
        return Dose(0.0, rate, True)
    if reach(BOLUS_MAX) > deadline:
        return Dose(float(BOLUS_MAX), rate, False)
    lo, hi = 0.0, BOLUS_MAX
    while hi - lo > BOLUS_XTOL:
        mid = 0.5 * (lo + hi)
        if reach(mid) <= deadline:
            hi = mid
        else:
            lo = mid
    return Dose(float(hi), rate, True)
