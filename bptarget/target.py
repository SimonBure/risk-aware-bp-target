"""The personalised target y* on the virtual cohort (Sections II.B and III.B).

At a decision time t, y* is the smallest MAP in [65 mmHg, F(u_max)], F the fit's equilibrium MAP,
whose dose (``bptarget.dose``) keeps the risk of a reading below 65 mmHg at most alpha over the window
W_t = (t + T_r, t + 10 min], eq. (target); no dose when even F(u_max) fails (``Forecasts``,
``targets``). ``decide`` scores it on a virtual patient against three other rules: the fit's dose at
70 mmHg (``FIXED``), the population's dose at 70 mmHg, the same for everyone (``POPULATION``), and the
oracle, the same rule on the true patient with the sensor's noise as its only uncertainty
(``ORACLE``, ``Truth``). Each dose is applied to the true patient (``applied``): its readings over W_t
(the collection's noise series continued) give the event, a reading below 65 mmHg; its true MAP,
noise-free on a 1 s grid, the truth. ``levels`` and ``at_levels``: the population's dose at every
target from 65 mmHg, the same for everyone, applied to each patient (the single target in hindsight).
"""
import numpy as np
from scipy.optimize import brentq

from bptarget.dose import BOLUS_MAX, DEADLINE, HORIZON, RATE_MAX
from bptarget.fit import N_BOX, PatientModel
from bptarget.forecast import ALPHAS, HYPO, N_DRAWS, draws, predictive, risk, smallest_target
from bptarget.model import Doses, equilibrium_map, reduced_map, simulate
from bptarget.virtual import (RHO, SEED, SIGMA, T_MAX, TARGET, collection, command, noise, record, sample_times,
                              shown, split, state, true_map, with_dose)

FIXED = "fixed"            # the fit's dose at 70 mmHg, whatever its risk
POPULATION = "population"  # the population's dose (eta = 0) at 70 mmHg
YSTAR = "ystar"            # y*
ORACLE = "oracle"          # y*'s rule on the true patient
LEVEL_STEP = 1.0           # mmHg between the population's targets of ``levels``
ORACLE_TOL = 0.01          # nmol, the oracle's bisection on the bolus


def window(times: np.ndarray, t: float, end: float | None = None) -> np.ndarray:
    """The readings of W_t = (t + T_r, t + 10 min], cut at ``end``."""
    hi = t + HORIZON if end is None else min(t + HORIZON, end)
    return (times > t + DEADLINE) & (times <= hi)


def f_max(theta: dict) -> float:
    """The reduced model's equilibrium MAP at u_max: the highest target it holds."""
    return float(reduced_map(theta, RATE_MAX / theta["kel"]))


def rate_for(params: dict, y: float) -> float:
    """The complete model's equilibrium rate holding the MAP y: 0 when F(0) >= y, NaN when F(u_max) < y."""
    def gap(u: float) -> float:
        return float(equilibrium_map(u, params)) - y

    if gap(0.0) >= 0.0:
        return 0.0
    if gap(RATE_MAX) < 0.0:
        return float("nan")
    return float(brentq(gap, 0.0, RATE_MAX, xtol=1e-12))


def readings_at(i: int, t: float, rep: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Virtual patient i's reading times in W_t and their noise, the collection's series continued."""
    times = sample_times(t + HORIZON)
    w = window(times, t)
    return times[w], noise(SEED, i, rep, t + HORIZON)[w]


def applied(params: dict, t: float, bolus: float, rate: float, shown_at: np.ndarray, eps: np.ndarray) -> dict:
    """The dose given at t after the collection, applied to the true patient: ``event``, a reading of W_t
    below 65 mmHg (the true MAP plus ``eps`` at ``shown_at``); the truth, noise-free on a 1 s grid from 0:
    over W_t, its minimum, whether it goes below 65 mmHg and its seconds below; its peak after t; its
    seconds below 65 during the collection (0, t] and the rise (t, t + T_r]."""
    g = np.arange(1.0, t + HORIZON + 0.5, 1.0)
    true = true_map(params, with_dose(collection(t), t, bolus, rate), g)
    on_grid = np.searchsorted(g, shown_at)
    assert np.allclose(g[on_grid], shown_at)
    readings = true[on_grid] + eps
    w, after = g > t + DEADLINE, g > t
    return dict(event=float(np.min(readings) < HYPO), true_min=float(true[w].min()),
                true_any=float(true[w].min() < HYPO), true_below=float(np.sum(true[w] < HYPO)),
                true_peak=float(true[after].max()), below_collection=float(np.sum(true[~after] < HYPO)),
                below_rise=float(np.sum(true[after & ~w] < HYPO)))


# --------------------------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------------------------- #
def true_state(params: dict, t: float) -> np.ndarray:
    """The true patient's state at t under the collection, from empty compartments at 0."""
    return simulate(params, collection(t), np.array([0.0, t]))[0][-1]


def true_map_from(params: dict, doses: Doses, t: float, x: np.ndarray, times: np.ndarray) -> np.ndarray:
    """The true MAP at ``times`` from the true state ``x`` at t, the clock started there (the doses before t
    enter after it, by Tlag); within LSODA's tolerance of the run from 0."""
    return simulate(params, Doses(doses.times - t, doses.amounts, doses.durations), np.asarray(times, float) - t, x)[1]


def true_command(params: dict, t: float, y: float, x: np.ndarray, tol: float = ORACLE_TOL) -> tuple[float, float, bool]:
    """Section II.C's dose on the true patient at t, from its state x: the equilibrium rate holding y (u_max
    when none does) and the smallest bolus whose true MAP reaches y within T_r, by bisection to ``tol``
    nmol, the whole cap when none does. Returns (bolus, rate, deadline met)."""
    rate = rate_for(params, y)
    if np.isnan(rate):
        rate = RATE_MAX
    rise = t + np.arange(0.0, DEADLINE + 0.5, 1.0)

    def reaches(bolus: float) -> bool:
        return bool((true_map_from(params, with_dose(collection(t), t, bolus, rate), t, x, rise) >= y).any())

    if reaches(0.0):
        return 0.0, rate, True
    if not reaches(BOLUS_MAX):
        return BOLUS_MAX, rate, False
    lo, hi = 0.0, BOLUS_MAX
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if reaches(mid) else (mid, hi)
    return hi, rate, True


class Truth:
    """A decision time of virtual patient i seen by the oracle: per target y, its dose on the true patient
    and the true MAP at W_t's readings (cached); the risk under the sensor's noise alone, its paths drawn
    once per patient and t."""

    def __init__(self, i: int, params: dict, t: float, k: int = N_DRAWS):
        self.params, self.t = params, t
        self.x = true_state(params, t)
        self.shown, _ = readings_at(i, t)
        self.top = float(equilibrium_map(RATE_MAX, params))
        z = np.random.default_rng([SEED, i, int(round(t)), 1]).standard_normal((k, len(self.shown)))
        z[:, 1:] *= np.sqrt(1.0 - RHO ** 2)
        for j in range(1, z.shape[1]):
            z[:, j] += RHO * z[:, j - 1]
        self.eps = SIGMA * z
        self._cache = {}

    def __call__(self, y: float):
        key = round(float(y), 9)
        if key not in self._cache:
            bolus, rate, met = true_command(self.params, self.t, y, self.x)
            u = with_dose(collection(self.t), self.t, bolus, rate)
            self._cache[key] = (bolus, rate, met), true_map_from(self.params, u, self.t, self.x, self.shown)
        return self._cache[key]

    def risk(self, y: float) -> float:
        """P(a reading of W_t below 65 mmHg) under the dose of y, the noise the only uncertainty."""
        return float(np.mean((self(y)[1] + self.eps).min(axis=1) < HYPO))


def oracle_targets(truth: Truth, alphas=ALPHAS) -> list[tuple[float, float | None]]:
    """``(alpha, y)``: the oracle's target at each alpha, None when even its F(u_max) fails."""
    found, hi = [], truth.top
    for alpha in sorted(alphas):
        y = smallest_target(truth.risk, alpha, HYPO, hi)
        found.append((alpha, y))
        hi = truth.top if y is None else y
    return found


# --------------------------------------------------------------------------------------------- #
# y*
# --------------------------------------------------------------------------------------------- #
class Forecasts:
    """A decision time of virtual patient i: per target, its dose and the law of W_t's readings under it,
    by the fit at t (cached)."""

    def __init__(self, i: int, fit, theta: dict, x: np.ndarray, n_box: float = N_BOX, sigma2_floor: float = 0.0):
        self.i, self.fit, self.theta, self.x, self.n_box = i, fit, theta, x, n_box
        self.sigma2 = max(fit[3], sigma2_floor)  # the noise level of the forecast (a floor: the discussion's)
        self.t = fit[0]
        self.times = sample_times(self.t + HORIZON)
        self.w = window(self.times, self.t)
        self._cache = {}

    def __call__(self, y: float):
        key = round(float(y), 9)
        if key not in self._cache:
            ctrl = command(self.theta, self.x, y)
            u = with_dose(collection(self.t), self.t, ctrl.bolus, ctrl.rate)
            mean, s = PatientModel(shown(self.i, u, self.times, np.zeros(len(self.times))), self.n_box)(self.fit[2])
            self._cache[key] = ctrl, predictive(self.times[self.w], mean[self.w], s[self.w], self.fit[4], self.sigma2)
        return self._cache[key]


def targets(at: Forecasts, top: float, dr, alphas=ALPHAS) -> list[tuple[float, float | None]]:
    """``(alpha, y*)`` at each alpha, None when even ``top`` fails; from the smallest alpha, whose target
    bounds the next ones."""
    found, hi = [], top
    for alpha in sorted(alphas):
        y = smallest_target(lambda v: risk(at(v)[1], HYPO, dr), alpha, HYPO, hi)
        found.append((alpha, y))
        hi = top if y is None else y
    return found


def setup(i: int, params: dict, fit, model, n_box: float = N_BOX, sigma2_floor: float = 0.0):
    """The forecasts of a decision, the fit's F(u_max) and the draws of xi (seeded by the cohort's seed, i
    and t)."""
    t, eta = fit[0], fit[2]
    theta, x0 = split(model.params(eta))
    at = Forecasts(i, fit, theta, state(theta, x0, t), n_box, sigma2_floor)
    dr = draws(len(eta), int(at.w.sum()), N_DRAWS, seed=[SEED, i, int(round(t))])
    return at, f_max(theta), dr


def decide(i: int, params: dict, fit, model, n_box: float = N_BOX, alphas=ALPHAS) -> list[dict]:
    """Virtual patient i at the decision time of ``fit``: a row per rule (y* and the oracle at each alpha,
    the fit's and the population's doses at 70 mmHg), with its target, dose, risk and outcome on the true
    patient (``applied``)."""
    t = fit[0]
    at, top, dr = setup(i, params, fit, model, n_box)
    shown_at, eps = readings_at(i, t)
    truth = Truth(i, params, t)
    base = dict(patient=i, t_min=t / 60, f_max=top, f_true=truth.top, rate65=rate_for(params, HYPO), sigma2_t=fit[3])
    rows = []
    for rule, alpha, y in [(YSTAR, a, y) for a, y in targets(at, top, dr, alphas)] + [(FIXED, np.nan, TARGET)]:
        row = dict(base, rule=rule, alpha=alpha, target=np.nan if y is None else y, feasible=y is not None)
        if y is not None:
            ctrl, pred = at(y)
            row.update(bolus=ctrl.bolus, rate=ctrl.rate, deadline_met=ctrl.deadline_met, risk=risk(pred, HYPO, dr),
                       **applied(params, t, ctrl.bolus, ctrl.rate, shown_at, eps))
        rows.append(row)
    pop, pop_x0 = split(model.params(np.zeros(len(model.names))))
    ctrl = command(pop, state(pop, pop_x0, t), TARGET)
    rows.append(dict(base, rule=POPULATION, alpha=np.nan, target=TARGET, feasible=True, bolus=ctrl.bolus,
                     rate=ctrl.rate, deadline_met=ctrl.deadline_met,
                     **applied(params, t, ctrl.bolus, ctrl.rate, shown_at, eps)))
    for alpha, y in oracle_targets(truth, alphas):
        row = dict(base, rule=ORACLE, alpha=alpha, target=np.nan if y is None else y, feasible=y is not None)
        if y is not None:
            (b, u, met), _ = truth(y)
            row.update(bolus=float(b), rate=float(u), deadline_met=bool(met), risk=truth.risk(y),
                       **applied(params, t, b, u, shown_at, eps))
        rows.append(row)
    return rows


def levels(decisions, n_box: float = N_BOX, step: float = LEVEL_STEP) -> dict[int, list[tuple]]:
    """The population's doses (eta = 0, from its state at t under the collection), the same for every
    patient, at the targets 65, 66, ... mmHg up to its F(u_max) and at it: ``(target, bolus, rate)`` per
    decision time (keyed by t in s)."""
    times = sample_times(T_MAX)
    model = PatientModel(shown(0, collection(T_MAX), times, np.zeros(len(times))), n_box)
    pop, x0 = split(model.params(np.zeros(len(model.names))))
    top = f_max(pop)
    goals = np.r_[np.arange(HYPO, top, step), top]
    found = {}
    for t in decisions:
        x = state(pop, x0, t)
        found[int(round(t))] = [(float(y), (d := command(pop, x, y)).bolus, d.rate) for y in goals]
    return found


def at_levels(i: int, params: dict, t: float, doses: list[tuple]) -> list[dict]:
    """Virtual patient i at t under each of ``levels``' doses, applied to the true patient."""
    shown_at, eps = readings_at(i, t)
    rate65 = rate_for(params, HYPO)
    return [dict(patient=i, t_min=t / 60, rate65=rate65, target=level, bolus=bolus, rate=rate,
                 **applied(params, t, bolus, rate, shown_at, eps)) for level, bolus, rate in doses]


def decide_all(i: int, params: dict, found: list, n_box: float = N_BOX,
               doses: dict[int, list[tuple]] | None = None) -> tuple[list[dict], list[dict]]:
    """``decide`` at each of virtual patient i's fits ``found``, and with ``doses`` (``levels``) the
    population's doses at each decision time: the rules' rows, the levels' rows."""
    model = PatientModel(record(i, params), n_box)
    rows = [row for fit in found for row in decide(i, params, fit, model, n_box)]
    common = [] if doses is None else [row for fit in found for row in at_levels(i, params, fit[0], doses[int(round(fit[0]))])]
    return rows, common
