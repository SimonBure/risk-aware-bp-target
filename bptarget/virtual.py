"""The virtual cohort of Section III.B and Appendix E: its patients, sensor and collection.

A virtual patient is the complete model with its twelve parameters drawn independently,
theta_j = pop_j exp(omega_j z_j), z_j ~ N(0, 1) (``propose``), from a fixed seed until n are
accepted (``draw``); a draw is rejected for the first rule it fails (``reject_reason``). The MAP is
read every 10 s with a stationary Ornstein-Uhlenbeck noise of SD 2 mmHg and correlation
exp(-|dt| / 10 s) (``noise``). Every patient receives the same collection (``collection``): by default
0.36 nmol/s over the first 2.5 min, then 0.18 nmol/s; ``use`` swaps it for another one, written as an
initial rate and events, each a new rate or a bolus over 10 s on top of the running rate.
"""
import numpy as np
from scipy.signal import lfilter

from bptarget.dose import BOLUS_DURATION, BOLUS_MAX, HORIZON, RATE_MAX, Dose, dose
from bptarget.fit import NAMES, PatientModel, fits
from bptarget.model import (OMEGA, POP, PK, Doses, Patient, Readings, equilibrium_map, from_first_sample, grid,
                            simulate, simulate_exact)

SEED = 2026101501
SAMPLING = 10.0                # s, between readings
SIGMA, TAU = 2.0, 10.0         # mmHg, s: the sensor's noise
RHO = float(np.exp(-SAMPLING / TAU))
T_MAX = 1500.0                 # s, the collection's record
BASELINE = (30.0, 65.0)        # mmHg: F(0), hypotensive without NE
CEILING = 200.0                # mmHg: the highest MAP at u_max and under the heaviest input
REACH = 70.0                   # mmHg: F(u_max) at least this
TARGET = 70.0                  # mmHg: the fixed target
IC = ["Ad0", "Ac0", "Ap0"]

CURRENT = {"r0": 0.36, "events": [[150.0, "rate", 0.18]]}  # the paper's collection
_spec = CURRENT


# --------------------------------------------------------------------------------------------- #
# The collection
# --------------------------------------------------------------------------------------------- #
def use(spec: dict | None = None) -> dict:
    """Make ``collection`` follow ``spec``, ``{"r0": rate, "events": [[t, "rate" | "bolus", value], ...]}``
    (nmol/s and nmol, t in s from 0); the paper's by default. Module state: a worker process sets it
    itself. Returns the spec in use."""
    global _spec
    _spec = CURRENT if spec is None else spec
    return _spec


def rate_at(spec: dict, t: float) -> float:
    """The collection's rate at t: the running rate plus any bolus over its 10 s."""
    r = spec["r0"]
    for te, kind, v in sorted(spec["events"]):
        if kind == "rate" and te <= t:
            r = v
    return r + sum(v / BOLUS_DURATION for te, kind, v in spec["events"]
                   if kind == "bolus" and te <= t < te + BOLUS_DURATION)


def collection(T: float, spec: dict | None = None) -> Doses:
    """The collection up to T (s), as constant pieces, zero rates left out."""
    spec = _spec if spec is None else spec
    edges = {0.0, T}
    for te, kind, _ in spec["events"]:
        edges |= {te, te + BOLUS_DURATION} if kind == "bolus" else {te}
    edges = sorted(e for e in edges if 0.0 <= e <= T)
    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = rate_at(spec, 0.5 * (a + b))
        if pieces and np.isclose(pieces[-1][2], r) and np.isclose(pieces[-1][0] + pieces[-1][1], a):
            pieces[-1][1] += b - a
        else:
            pieces.append([a, b - a, r])
    pieces = [p for p in pieces if p[2] > 0]
    return Doses(np.array([p[0] for p in pieces]), np.array([p[2] * p[1] for p in pieces]),
                 np.array([p[1] for p in pieces]))


def _pieces(pieces: list[tuple[float, float, float]]) -> Doses:
    """``Doses`` from ``(start, duration, rate)`` pieces in time order."""
    t, h, r = (np.array(c, float) for c in zip(*pieces))
    return Doses(t, r * h, h)


def with_dose(before: Doses, T: float, bolus: float, rate: float, horizon: float = HORIZON) -> Doses:
    """``before``, then from T the bolus over 10 s on top of the rate, held up to T + horizon."""
    then = _pieces([(T, BOLUS_DURATION, rate + bolus / BOLUS_DURATION),
                    (T + BOLUS_DURATION, horizon - BOLUS_DURATION, rate)])
    return Doses(np.concatenate([before.times, then.times]), np.concatenate([before.amounts, then.amounts]),
                 np.concatenate([before.durations, then.durations]))


def sample_times(T: float) -> np.ndarray:
    """The reading times up to T: every 10 s, 0 and T included."""
    return np.arange(0.0, T + SAMPLING / 2.0, SAMPLING)


def noise(seed: int, patient: int, rep: int, T: float) -> np.ndarray:
    """The sensor's noise at ``sample_times(T)``: an AR(1) of coefficient rho = e^-1 from its stationary
    law, one series per (patient, repetition); a shorter record reads the start of the same series."""
    z = np.random.default_rng([seed, patient, rep]).standard_normal(int(round(T / SAMPLING)) + 1)
    drive = SIGMA * np.sqrt(1.0 - RHO ** 2) * z
    drive[0] = SIGMA * z[0]
    return lfilter([1.0], [1.0, -RHO], drive)


# --------------------------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------------------------- #
def propose(rng: np.random.Generator) -> dict[str, float]:
    """One parameter vector, the twelve log-normal factors independent."""
    z = rng.standard_normal(len(POP))
    return {n: float(POP[n] * np.exp(OMEGA[n] * zn)) for n, zn in zip(POP, z)}


def true_map(params: dict, doses: Doses, t) -> np.ndarray:
    """The true MAP under ``doses`` at ``t``, from empty compartments at t = 0."""
    return simulate(params, doses, t)[1]


def envelope(params: dict, inputs: list[tuple[Doses, float]]) -> float:
    """The highest MAP under each ``(input, end)`` from 0 to its end, every second."""
    highest = -np.inf
    for u, end in inputs:
        highest = max(highest, float(np.max(true_map(params, u, np.arange(0.0, end + 0.5, 1.0)))))
    return highest


def ceiling_inputs(decisions) -> list[tuple[Doses, float]]:
    """At each decision time t, u_max from 0 to t, then the largest dose (B_max on top of u_max for
    10 min): by the positivity and monotonicity of the PK, its MAP bounds that of any collection at
    most u_max followed by any dose at t."""
    return [(with_dose(Doses(np.array([0.0]), np.array([RATE_MAX * t]), np.array([t])), t, BOLUS_MAX, RATE_MAX),
             t + HORIZON) for t in decisions]


def reject_reason(params: dict, *, inputs, baseline=BASELINE, ceiling=CEILING, reach=REACH,
                  e0_min: float | None = None, emax_min: float | None = None) -> str | None:
    """The first rule ``params`` fails, None when accepted: a parameter not positive; Emax <= E0; F(0)
    outside ``baseline``; E0 or Emax under its minimum, when given; F(u_max) above ``ceiling``; F(u_max)
    under ``reach``; the MAP above ``ceiling`` under one of ``inputs`` (the envelope)."""
    values = np.array([params[n] for n in POP], float)
    if not np.all(np.isfinite(values) & (values > 0.0)):
        return "nonpositive"
    if params["Emax"] <= params["E0"]:
        return "decreasing"
    if not baseline[0] <= equilibrium_map(0.0, params) <= baseline[1]:
        return "baseline"
    if e0_min is not None and params["E0"] < e0_min:
        return "e0"
    if emax_min is not None and params["Emax"] < emax_min:
        return "emax"
    highest = equilibrium_map(RATE_MAX, params)
    if highest > ceiling:
        return "ceiling"
    if reach is not None and highest < reach:
        return "unreachable"
    if envelope(params, inputs) > ceiling:
        return "envelope"
    return None


def draw(n: int, decisions, seed: int = SEED, **rules) -> tuple[list[dict], list[tuple[int, str | None]]]:
    """n accepted patients and the log of every proposal, ``(index, reason)``, reason None when accepted.
    The paper's rules by default (Appendix E); the envelope for the decision times ``decisions`` (s)."""
    rng = np.random.default_rng(seed)
    rules.setdefault("inputs", ceiling_inputs(decisions))
    patients, log = [], []
    while len(patients) < n:
        params = propose(rng)
        reason = reject_reason(params, **rules)
        log.append((len(log), reason))
        if reason is None:
            patients.append(params)
    return patients, log


# --------------------------------------------------------------------------------------------- #
# A patient under the protocol
# --------------------------------------------------------------------------------------------- #
def shown(i: int, doses: Doses, times: np.ndarray, values: np.ndarray) -> Patient:
    """A virtual record: ``doses`` and readings ``values`` at ``times``."""
    return Patient(i, doses, Readings(times, values))


def record(i: int, params: dict, T: float = T_MAX, rep: int = 0) -> Patient:
    """Virtual patient i's collection record: a reading every 10 s from 0 to T, the true MAP plus the
    sensor's noise (repetition ``rep``)."""
    times, u = sample_times(T), collection(T)
    return shown(i, u, times, true_map(params, u, times) + noise(SEED, i, rep, T))


def state(theta: dict, x0: np.ndarray, t: float) -> np.ndarray:
    """The reduced model's state at t under the collection, from x0 at 0."""
    times = sample_times(t)
    g = from_first_sample(grid(shown(0, collection(t), times, np.zeros(len(times)))))
    return simulate_exact([theta[k] for k in PK], g, x0)[:, -1]


def split(values: dict[str, float]) -> tuple[dict[str, float], np.ndarray]:
    """``PatientModel.params``' values as the reduced parameters and the amounts at the first reading."""
    return {k: values[k] for k in NAMES}, np.array([values[k] for k in IC])


def command(theta: dict, x: np.ndarray, target: float) -> Dose:
    """The dose of Section II.C for ``target`` under the reduced model ``theta`` from the state x."""
    return dose(theta, x, target)


def collect(i: int, params: dict, n_box: float, until: float | None = None, rep: int = 0) -> list:
    """Virtual patient i's online fits on its collection record, up to ``until`` (s)."""
    patient = record(i, params, rep=rep)
    return list(fits(patient, PatientModel(patient, n_box), until))
