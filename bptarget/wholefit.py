"""Whole-record fits (Section II.A, Fig. 1): the three models fitted by IPOPT to a patient's record.

The unknowns are the parameters, in the box pop exp(+-3 omega), and the state at the first sample,
its amounts capped (``X0_CAP``); [0, t0] only gives the population's state at t0 from the recorded
doses. The states at every sample (and Cc sample) are NLP variables (multiple shooting):

- ``reduced``: the reduced model, its linear PK chained exactly between samples (``transition``),
  fitted to the MAP (mean squared error);
- ``complete``: the complete model of Joachim et al., its 12 parameters free (Tlag, beta, gamma
  included), and ``windkessel``, its second-order variant; between samples, sub-steps of at most 1 s,
  one Radau IIA step of 2 stages each (its stages NLP variables), the input delayed by Tlag and
  averaged over each sub-step (``delayed_input``); fitted to the MAP and Cc, mean squared error of each.

Under the Hill or reduced output, the MAP without NE is at least 30 mmHg and the map increasing.
``multistart``: the population, then scrambled Sobol starts pop exp(k omega), k in [-3, 3]^d, in
stages of 64, 256 and 1024, until 3 converged starts end within 1 % of the best cost.
"""
import os
from typing import NamedTuple

import casadi as ca
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc

from bptarget.model import (HILL, OMEGA, PK, POP, WINDKESSEL_OMEGA, WINDKESSEL_POP, X0_CAP, Patient,
                            Readings, first_row, from_first_sample, grid, hill, reduced_map, reduced_omega,
                            reduced_pop, simulate_exact, simulate_from, transition)

if os.name == "nt":  # CasADi fails to load its IPOPT plugin from a non-ASCII path; loaded first, it is found
    import ctypes
    ctypes.WinDLL(os.path.join(os.path.dirname(ca.__file__), "libcasadi_nlpsol_ipopt.dll"))

MODELS = ("reduced", "complete", "windkessel")
N_BOX = 3.0
MAP_FLOOR = 30.0           # mmHg, the lowest MAP without NE
PD_LOG_MARGIN = 1e-6       # the increasing map's log slack
IPOPT = {"ipopt.max_iter": 10000, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4, "ipopt.acceptable_iter": 15,
         "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
DT, DELTA = 1.0, 0.1       # s: the longest sub-step; half-width of the cumulative dose's smoothed corners
RADAU_A = ((5 / 12, -1 / 12), (3 / 4, 1 / 4))  # Radau IIA of 2 stages at c = (1/3, 1)
RADAU_C = (1 / 3, 1.0)
EPS = 1e-3                 # nmol: Ad^beta smoothed to Ad (Ad^2 + EPS^2)^((beta - 1) / 2)
SOBOL_STAGES, SOBOL_SEED, N_SIGMA = (64, 256, 1024), 0, 3.0
MIN_HITS, HIT_RTOL = 3, 0.01


class Fit(NamedTuple):
    params: dict[str, float]
    x: np.ndarray        # (n_states, N + 1): the states at the nodes, from the first sample
    t: np.ndarray        # the nodes
    y: np.ndarray        # the fitted MAP at the readings
    cost: float
    converged: bool
    status: str


# --------------------------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------------------------- #
def population(model: str) -> dict[str, float]:
    if model == "reduced":
        return reduced_pop()
    if model == "complete":
        return dict(POP)
    return {**{k: v for k, v in POP.items() if k not in HILL}, **WINDKESSEL_POP}


def omega(model: str) -> dict[str, float]:
    if model == "reduced":
        return reduced_omega()
    if model == "complete":
        return dict(OMEGA)
    return {**{k: v for k, v in OMEGA.items() if k not in HILL}, **WINDKESSEL_OMEGA}


def n_states(model: str) -> int:
    return 5 if model == "windkessel" else 3


def output_state(model: str) -> int:
    """The state the MAP reads: Ac, or E under the Windkessel."""
    return 3 if model == "windkessel" else 1


def effect(model: str, p, z):
    """The MAP from the output state."""
    if model == "reduced":
        return reduced_map(p, z)
    return hill(p, z) if model == "complete" else p["E0"] + z


def rhs_names(model: str) -> tuple[str, ...]:
    pk = ("ka", "k12", "k21", "kel", "beta")
    return pk + ("Cendo", "Vc", "wn", "zeta", "nu") if model == "windkessel" else pk


def casadi_rhs(model: str, p, x, r):
    """The complete model's right-hand side, Ad^beta smoothed at an empty depot (IPOPT's Hessian is
    finite there); then the Windkessel's two rows."""
    ad, ac, ap = x[0, :], x[1, :], x[2, :]
    ra = p["ka"] * ad * (ad ** 2 + EPS ** 2) ** ((p["beta"] - 1.0) / 2.0)
    pk = ca.vertcat(r - ra, ra - (p["k12"] + p["kel"]) * ac + p["k21"] * ap, p["k12"] * ac - p["k21"] * ap)
    if model != "windkessel":
        return pk
    cc = p["Cendo"] + x[1, :] / p["Vc"]
    return ca.vertcat(pk, x[4, :], p["nu"] * cc - 2 * p["zeta"] * p["wn"] * x[4, :] - p["wn"] ** 2 * x[3, :])


def pd_log_slack(p, reduced: bool):
    """log(Emax b / a) (reduced) or log(Emax / E0): positive when the map is increasing."""
    names = ("Emax", "b", "a") if reduced else ("Emax", "E0")
    log = ca.log if any(isinstance(p[n], (ca.MX, ca.SX)) for n in names) else np.log
    return log(p["Emax"]) + log(p["b"]) - log(p["a"]) if reduced else log(p["Emax"]) - log(p["E0"])


def baseline_map(p, reduced: bool):
    """The MAP without NE: Emax - a / b, or E0 + (Emax - E0) H(Cendo)."""
    if reduced:
        return p["Emax"] - p["a"] / p["b"]
    c, g = p["Cendo"], p["gamma"]
    return p["E0"] + (p["Emax"] - p["E0"]) * c ** g / (p["EC50"] ** g + c ** g)


# --------------------------------------------------------------------------------------------- #
# The complete model's discretisation
# --------------------------------------------------------------------------------------------- #
def _ramp(u):
    """max(u, 0) with a quadratic corner on [-DELTA, DELTA]."""
    return (ca.fmin(ca.fmax(u, -DELTA), DELTA) + DELTA) ** 2 / (4 * DELTA) + ca.fmax(u - DELTA, 0)


def delayed_input(doses, a, b, tlag):
    """The mean rate of ``doses`` delayed by ``tlag`` over each sub-step [a_i, b_i], a (1, n) row: the
    cumulative dose (corners smoothed) differenced. ``tlag`` a number or a CasADi symbol."""
    s, d = ca.DM(doses.times), ca.DM(doses.durations)
    r = ca.DM(doses.amounts / doses.durations)
    a, b = ca.DM(np.asarray(a, float)).T, ca.DM(np.asarray(b, float)).T

    def cum(t):
        u = ca.repmat(t - tlag, s.numel(), 1) - ca.repmat(s, 1, t.numel())
        return ca.mtimes(r.T, _ramp(u) - _ramp(u - ca.repmat(d, 1, t.numel())))

    return (cum(b) - cum(a)) / (b - a)


def substeps(t: np.ndarray):
    """The sub-steps between the nodes ``t``, at most DT and equal per interval: each one's interval k,
    rank j there, start a and length dt; and the sub-steps that end on a node."""
    h = np.diff(t)
    n = np.maximum(1, np.ceil(h / DT - 1e-9)).astype(int)
    k = np.repeat(np.arange(len(h)), n)
    j = np.arange(len(k)) - np.repeat(np.cumsum(n) - n, n)
    dt = h[k] / n[k]
    return k, j, t[k] + j * dt, dt, np.cumsum(n) - 1


# --------------------------------------------------------------------------------------------- #
# One NLP
# --------------------------------------------------------------------------------------------- #
def solve(patient: Patient, model: str, free=None, theta0: dict | None = None, x_start=None,
          n_box: float = N_BOX) -> Fit:
    """The fit of ``model`` to the patient, ``free`` its free parameters (all by default; the others at
    their population values), from the parameters ``theta0`` (the population's where absent) and the
    node states ``x_start`` (the population's simulation by default)."""
    pop, om = population(model), omega(model)
    free = list(pop) if free is None else list(free)
    with_cc = model != "reduced"
    full = grid(patient, with_cc)
    g = from_first_sample(full)
    ns = n_states(model)

    opti = ca.Opti()
    p = {n: opti.variable() if n in free else pop[n] for n in pop}
    x = opti.variable(ns, len(g.t))
    for n in pop:
        if isinstance(p[n], ca.MX):
            opti.subject_to(p[n] >= pop[n] * np.exp(-n_box * om[n]))
            opti.subject_to(p[n] <= pop[n] * np.exp(+n_box * om[n]))
    if model != "windkessel":
        reduced = model == "reduced"
        slack, base = pd_log_slack(p, reduced), baseline_map(p, reduced)
        if isinstance(slack, ca.MX):
            opti.subject_to(slack >= PD_LOG_MARGIN)
        if isinstance(base, ca.MX):
            opti.subject_to(base >= MAP_FLOOR)
    opti.subject_to(ca.vec(x[:3, :]) >= 0)
    opti.subject_to(x[:3, 0] <= X0_CAP)
    z = _exact_dynamics(opti, p, x, g) if model == "reduced" else _radau_dynamics(opti, p, x, g, model, patient.doses)

    bp = patient.map.values
    y = effect(model, p, x[output_state(model), g.obs_idx.tolist()].T)
    cost = ca.sumsqr(y - bp) / y.shape[0]
    if with_cc:
        cc = p["Cendo"] + x[1, g.cc_idx.tolist()].T / p["Vc"]
        cost = cost + ca.sumsqr(cc - patient.cc.values) / cc.shape[0]
    opti.minimize(cost)
    opti.solver("ipopt", IPOPT)

    if x_start is None:
        x_start = population_start(patient, model, pop, full)
    start = dict(pop, **(theta0 or {}))
    for n, v in p.items():
        if isinstance(v, ca.MX):
            opti.set_initial(v, start[n])
    opti.set_initial(x, x_start)
    if z is not None:  # the stages on the start's own simulation
        t0, c1, c2 = z[1]
        times, back = np.unique(np.r_[t0, c1, c2], return_inverse=True)
        xs = simulate_from(start, patient.doses, x_start[:, 0], times)[:, back[1:]]
        opti.set_initial(z[0], np.vstack([xs[:, :len(c1)], xs[:, len(c1):]]))

    try:
        sol, converged, status = opti.solve(), True, "Solve_Succeeded"
    except RuntimeError as e:
        msg = str(e)
        status = msg.split("return_status is")[1].strip().strip("'") if "return_status is" in msg else "Solver_Error"
        sol, converged = opti.debug, False
    values = {n: float(sol.value(v)) if isinstance(v, ca.MX) else float(v) for n, v in p.items()}
    if model != "windkessel":  # an acceptable point may break the physiology
        reduced = model == "reduced"
        with np.errstate(over="ignore", invalid="ignore"):
            vals = {n: np.float64(v) for n, v in values.items()}
            slack, base = float(pd_log_slack(vals, reduced)), float(baseline_map(vals, reduced))
        if not np.isfinite(slack) or slack <= 0 or not np.isfinite(base) or base < MAP_FLOOR * (1 - 1e-6):
            converged, status = False, "Nonphysical_PD_Solution"
    xf = np.array(sol.value(x)).reshape(ns, -1)
    y_fit = np.asarray(effect(model, values, xf[output_state(model), g.obs_idx]))
    return Fit(values, xf, g.t, y_fit, float(sol.value(cost)), converged, status)


def _exact_dynamics(opti, p, x, g):
    """x_{k+1} = the exact transition of x_k over the pieces of interval k; the intervals of one piece
    sharing a length, all at once."""
    th = ca.vertcat(*(p[k] for k in PK))
    memo = {}

    def tr(h):
        if h not in memo:
            memo[h] = transition(th, h)
        return memo[h]

    single = {}
    for k, segs in enumerate(g.steps):
        if len(segs) == 1:
            single.setdefault(float(segs[0][0]), []).append(k)
        else:
            xk = x[:, k]
            for h, r in segs:
                phi, gam = tr(float(h))
                xk = phi @ xk + gam * float(r)
            opti.subject_to(x[:, k + 1] == xk)
    for h, ks in single.items():
        phi, gam = tr(h)
        rates = ca.DM([float(g.steps[k][0][1]) for k in ks]).T
        opti.subject_to(x[:, [k + 1 for k in ks]] == phi @ x[:, ks] + gam @ rates)
    return None


def _radau_dynamics(opti, p, x, g, model, doses):
    """One Radau IIA step per sub-step, from the node or the previous sub-step's end; each interval ends
    on its node. Returns the stage variables and their times."""
    k, j, a, dt, ends = substeps(g.t)
    m, ns = len(k), n_states(model)
    tl = ca.SX.sym("tlag")
    u = ca.Function("u", [tl], [delayed_input(doses, a, a + dt, tl)])(p["Tlag"])
    names = rhs_names(model)
    xs_, r_, th_ = ca.SX.sym("x", ns), ca.SX.sym("r"), ca.SX.sym("th", len(names))
    f = ca.Function("f", [xs_, r_, th_], [casadi_rhs(model, dict(zip(names, ca.vertsplit(th_))), xs_, r_)]).map(m)
    thv = ca.vertcat(*(p[n] for n in names))
    z = opti.variable(2 * ns, m)
    z1, z2 = z[:ns, :], z[ns:, :]
    start = np.where(j == 0, k, len(g.t) + np.arange(m) - 1)
    xs = ca.horzcat(x, z2)[:, start.tolist()]
    f1, f2 = f(z1, u, thv), f(z2, u, thv)
    dts = ca.repmat(ca.DM(dt).T, ns, 1)
    (a11, a12), (a21, a22) = RADAU_A
    opti.subject_to(z1 == xs + dts * (a11 * f1 + a12 * f2))
    opti.subject_to(z2 == xs + dts * (a21 * f1 + a22 * f2))
    opti.subject_to(x[:, 1:] == z2[:, ends.tolist()])
    return z, (g.t[0], a + RADAU_C[0] * dt, a + RADAU_C[1] * dt)


def population_start(patient: Patient, model: str, theta: dict, full=None) -> np.ndarray:
    """The node states of ``theta``'s simulation from empty compartments at t = 0, from the first sample
    on: exact for the reduced model, LSODA for the complete ones."""
    full = grid(patient, model != "reduced") if full is None else full
    if model == "reduced":
        pk = [theta[k] for k in PK]
        nominal = simulate_exact(pk, full, np.zeros(3))[:, first_row(full)]
        return simulate_exact(pk, from_first_sample(full), nominal)
    return simulate_from(theta, patient.doses, np.zeros(n_states(model)), full.t)[:, first_row(full):]


# --------------------------------------------------------------------------------------------- #
# Starts
# --------------------------------------------------------------------------------------------- #
def starts(n_sobol: int, d: int) -> np.ndarray:
    """(n_sobol + 1, d) k-values: 0, then scrambled Sobol points in [-3, 3]^d (a smaller set a prefix of a
    larger one)."""
    u = qmc.Sobol(d=d, scramble=True, seed=SOBOL_SEED).random(n_sobol)
    return np.vstack([np.zeros(d), N_SIGMA * (2 * u - 1)])


def fit_start(patient: Patient, model: str, k: np.ndarray, free=None, n_box: float = N_BOX) -> Fit:
    """The fit from pop exp(k omega) on ``free``, the states guessed by that start's own simulation."""
    pop, om = population(model), omega(model)
    free = list(pop) if free is None else list(free)
    theta0 = {n: float(pop[n] * np.exp(ki * om[n])) for n, ki in zip(free, k)}
    return solve(patient, model, free, theta0, population_start(patient, model, dict(pop, **theta0)), n_box)


def sse(patient: Patient, fit: Fit) -> float:
    return float(np.sum((patient.map.values - fit.y) ** 2))


def hits(score: np.ndarray, converged: np.ndarray) -> int:
    """Converged starts within HIT_RTOL of the best converged score."""
    return int(np.sum(converged & (score <= score[converged].min() * (1 + HIT_RTOL))))


def multistart(patient: Patient, model: str, n_jobs: int = 1, max_starts: int | None = None, free=None,
               by: str = "cost", n_box: float = N_BOX) -> tuple[Fit, dict]:
    """Starts in stages until MIN_HITS converged starts reach the best score (``by``: the NLP's cost, or
    the MAP's sum of squares), or ``max_starts`` (1: the population only). The fit of the best converged
    score (the best score when none converged) and the record: ``n_starts``, ``hits``, ``start``."""
    d = len(population(model) if free is None else free)
    scale = min(n_box, N_SIGMA) / N_SIGMA
    runs = []
    for n_sobol in SOBOL_STAGES:
        if max_starts is not None:
            n_sobol = min(n_sobol, max_starts - 1)
        ks = scale * starts(n_sobol, d)
        runs += Parallel(n_jobs=n_jobs)(delayed(fit_start)(patient, model, k, free, n_box) for k in ks[len(runs):])
        score = np.array([r.cost if by == "cost" else sse(patient, r) for r in runs])
        converged = np.array([r.converged for r in runs])
        n_hits = hits(score, converged) if converged.any() else 0
        if n_hits >= MIN_HITS or len(runs) == max_starts:
            break
    pool = converged if converged.any() else np.ones_like(converged)
    i = int(np.flatnonzero(pool)[np.argmin(score[pool])])
    return runs[i], dict(n_starts=len(runs), hits=n_hits, start="population" if i == 0 else f"sobol-{i}")


# --------------------------------------------------------------------------------------------- #
# What a fit shows
# --------------------------------------------------------------------------------------------- #
def rmse(fit: np.ndarray, data: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(fit) - np.asarray(data)) ** 2)))


def metrics(patient: Patient, model: str, fit: Fit) -> dict:
    """The fit's MAP RMSE at the readings and, for the complete models, its Cc RMSE at the Cc samples
    (both nodes of the NLP)."""
    row = dict(id=patient.id, converged=fit.converged, cost=fit.cost, rmse_map=rmse(fit.y, patient.map.values))
    if model != "reduced":
        ac = fit.x[1, np.searchsorted(fit.t, patient.cc.times)]
        row["rmse_cc"] = rmse(fit.params["Cendo"] + ac / fit.params["Vc"], patient.cc.values)
    return row


def outputs(patient: Patient, model: str, params: dict, x0: np.ndarray, times: np.ndarray):
    """MAP and Cc (None for the reduced model) at ``times`` from the state x0 at ``times[0]``, a fit's first
    node, under the patient's doses: the exact transition, or LSODA for the complete models."""
    times = np.asarray(times, float)
    if model == "reduced":
        shown = Patient(patient.id, patient.doses, Readings(times, np.zeros_like(times)))
        x = simulate_exact([params[k] for k in PK], from_first_sample(grid(shown)), x0)
        return np.asarray(reduced_map(params, x[1])), None
    x = simulate_from(params, patient.doses, x0, times)
    return np.asarray(effect(model, params, x[output_state(model)])), params["Cendo"] + x[1] / params["Vc"]
