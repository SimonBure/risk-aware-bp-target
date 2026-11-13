"""The NE PK/PD models of Section II.A and their simulation.

State x = (Ad, Ac, Ap), the exogenous NE in a depot, a central and a peripheral compartment (nmol),
driven by the infusion rate u (nmol/s) delayed by Tlag::

    dAd/dt = u(t - Tlag) - ka Ad^beta
    dAc/dt = ka Ad^beta - (k12 + kel) Ac + k21 Ap
    dAp/dt = k12 Ac - k21 Ap

The complete model of Joachim et al. (2024) reads the MAP from the total concentration
C = Cendo + Ac / Vc by a Hill map of slope gamma (``hill``): twelve parameters, population values
and inter-individual SDs in ``POP`` and ``OMEGA`` (their Table 2). Its second-order variant replaces
the Hill map by a Windkessel driven by C (``WINDKESSEL``), used only in Fig. 1. The reduced model
sets Tlag = 0 and beta = 1, and reads the MAP as Emax - a / (b + Ac) (``reduced_map``): seven
parameters, its linear PK integrated exactly (``transition``, ``simulate_exact``).
"""
from dataclasses import dataclass
from typing import NamedTuple

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

# Joachim et al. (2024), Table 2: population values and log-normal SDs, in this order.
POP = {"Cendo": 0.81, "Tlag": 25.31, "ka": 0.02, "Vc": 0.49, "k12": 0.06, "k21": 0.04, "kel": 0.05,
       "beta": 1.21, "E0": 57.09, "Emax": 113.52, "EC50": 15.7, "gamma": 1.46}
OMEGA = {"Cendo": 0.51, "Tlag": 0.33, "ka": 0.65, "Vc": 0.36, "k12": 0.28, "k21": 0.58, "kel": 0.31,
         "beta": 0.28, "E0": 0.22, "Emax": 0.51, "EC50": 0.59, "gamma": 0.7}
# Its second pharmacodynamic model: E'' + 2 zeta wn E' + wn^2 E = nu C, MAP = E0 + E.
WINDKESSEL_POP = {"wn": 1.01, "zeta": 19.44, "nu": 2.12}
WINDKESSEL_OMEGA = {"wn": 0.2, "zeta": 0.66, "nu": 0.4}
HILL = ("Emax", "EC50", "gamma")  # the parameters the Windkessel replaces

PK = ("ka", "k12", "k21", "kel")
REDUCED = ("ka", "k12", "k21", "kel", "Emax", "a", "b")
RTOL, ATOL = 1e-10, 1e-12  # LSODA's tolerances


@dataclass
class Readings:
    """Samples of a signal: ``times`` (s), ``values`` (MAP in mmHg, or Cc in nmol/L)."""
    times: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.values = np.asarray(self.values, dtype=float)


@dataclass
class Doses:
    """A piecewise-constant NE input: dose i delivers ``amounts[i]`` (nmol) over
    [``times[i]``, ``times[i]`` + ``durations[i]``) at a constant rate; zero elsewhere."""
    times: np.ndarray
    amounts: np.ndarray
    durations: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.amounts = np.asarray(self.amounts, dtype=float)
        self.durations = np.asarray(self.durations, dtype=float)

    def __call__(self, t: float) -> float:
        """The rate at t (nmol/s)."""
        for i, start in enumerate(self.times):
            if start <= t < start + self.durations[i]:
                return self.amounts[i] / self.durations[i]
        return 0.0


@dataclass
class Patient:
    """A record: its NE ``doses``, its ``map`` readings and, for the clinical cohort, its plasma
    concentrations ``cc``."""
    id: int
    doses: Doses
    map: Readings
    cc: Readings | None = None


# --------------------------------------------------------------------------------------------- #
# The complete model and its second-order variant
# --------------------------------------------------------------------------------------------- #
def hill(p, ac):
    """MAP (mmHg) from the central amount ``ac`` (nmol), eq. (emax)."""
    z = ((p["Cendo"] + ac / p["Vc"]) / p["EC50"]) ** p["gamma"]
    return p["E0"] + (p["Emax"] - p["E0"]) * z / (1.0 + z)


def equilibrium_map(u, p):
    """The complete model's MAP at equilibrium under the rate u: Ac = u / kel whatever beta."""
    return hill(p, np.asarray(u) / p["kel"])


def windkessel_params(p: dict) -> bool:
    return "wn" in p


def simulate(p: dict, doses: Doses, t_eval, x0=None) -> tuple[np.ndarray, np.ndarray]:
    """The complete model (its second-order variant when ``p`` holds ``wn``) from ``x0`` at t = 0
    (default: empty), at ``t_eval`` (sorted, from 0 on): the states ``(len(t_eval), n)`` and the MAP.
    LSODA, piece by piece between the edges of the delayed input."""
    second = windkessel_params(p)

    def rhs(_t, x, u):
        ra = p["ka"] * max(x[0], 0.0) ** p["beta"]  # Ad floored at 0: the solver may step a hair under it
        pk = np.array([u - ra, ra - (p["k12"] + p["kel"]) * x[1] + p["k21"] * x[2], p["k12"] * x[1] - p["k21"] * x[2]])
        if not second:
            return pk
        c = p["Cendo"] + x[1] / p["Vc"]
        return np.r_[pk, x[4], p["nu"] * c - 2 * p["zeta"] * p["wn"] * x[4] - p["wn"] ** 2 * x[3]]

    t_eval = np.asarray(t_eval, dtype=float)
    lag = p["Tlag"]
    edges = np.concatenate([doses.times, doses.times + doses.durations]) + lag
    cuts = np.unique(np.concatenate([[0.0], edges[(edges > 0.0) & (edges < t_eval[-1])], [t_eval[-1]]]))
    x = np.zeros(5 if second else 3) if x0 is None else np.asarray(x0, dtype=float)
    out = np.zeros((len(x), len(t_eval)))
    out[:, t_eval == 0.0] = x[:, None]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a > 1e-9 * max(1.0, b):  # shorter: two edges apart by rounding only
            inside = (t_eval > a) & (t_eval < b)
            u = doses(0.5 * (a + b) - lag)
            sol = solve_ivp(lambda t, x: rhs(t, x, u), (a, b), x, t_eval=np.append(t_eval[inside], b),
                            method="LSODA", rtol=RTOL, atol=ATOL)
            out[:, inside] = sol.y[:, :-1]
            x = sol.y[:, -1]
        out[:, t_eval == b] = x[:, None]
    y = p["E0"] + out[3] if second else hill(p, out[1])
    return out.T, np.asarray(y)


def simulate_from(p: dict, doses: Doses, x0, times) -> np.ndarray:
    """States ``(n, len(times))`` from ``x0`` at ``times[0]``: ``simulate`` with the clock started
    there (the doses before it enter after it, by Tlag)."""
    t0 = float(times[0])
    x, _ = simulate(p, Doses(doses.times - t0, doses.amounts, doses.durations), np.asarray(times, float) - t0, x0)
    return x.T


# --------------------------------------------------------------------------------------------- #
# The reduced model
# --------------------------------------------------------------------------------------------- #
def reduced_pop() -> dict[str, float]:
    """The population's reduced parameters: a = (Emax - E0) EC50 Vc, b = (Cendo + EC50) Vc."""
    a = (POP["Emax"] - POP["E0"]) * POP["EC50"] * POP["Vc"]
    b = (POP["Cendo"] + POP["EC50"]) * POP["Vc"]
    return {"ka": POP["ka"], "k12": POP["k12"], "k21": POP["k21"], "kel": POP["kel"], "Emax": POP["Emax"],
            "a": a, "b": b}


def reduced_omega() -> dict[str, float]:
    """Log-SDs of the reduced parameters: a and b products of independent log-normals."""
    w = OMEGA
    omega = {k: w[k] for k in ("ka", "k12", "k21", "kel", "Emax")}
    omega["a"] = float(np.sqrt(w["Emax"] ** 2 + w["EC50"] ** 2 + w["Vc"] ** 2))
    omega["b"] = float(np.sqrt(w["EC50"] ** 2 + w["Vc"] ** 2))
    return omega


def reduced_map(p, ac):
    """MAP (mmHg) from the central amount (nmol), eq. (pd): h(Ac) = Emax - a / (b + Ac)."""
    return p["Emax"] - p["a"] / (p["b"] + ac)


def reduced_map_inv(p, y):
    """The central amount whose MAP is y: h^-1(y)."""
    return p["a"] / (p["Emax"] - y) - p["b"]


def box(pop: dict, omega: dict, n_sigma: float) -> dict[str, tuple[float, float]]:
    """Bounds pop exp(+-n_sigma omega) on each parameter."""
    return {k: (pop[k] * np.exp(-n_sigma * omega[k]), pop[k] * np.exp(+n_sigma * omega[k])) for k in pop}


def _linear_map_inv(y):
    """The central amount whose MAP is y under the complete model with gamma = 1."""
    z = (y - POP["E0"]) / (POP["Emax"] - y)
    return POP["Vc"] * (POP["EC50"] * z - POP["Cendo"])


# The amounts at the first reading are capped: Ad0, Ac0 <= 4 h^-1(0.9 Emax) at the population (gamma = 1),
# about 121 nmol, and Ap0 at its equilibrium with that Ac0. IC_SD scales them in the normalised coordinates.
_UB = 4.0 * float(_linear_map_inv(np.array(0.9 * POP["Emax"])))
X0_CAP = np.array([_UB, _UB, POP["k12"] / POP["k21"] * _UB])
IC_SD = _UB / 6.0


# --------------------------------------------------------------------------------------------- #
# The exact transition of the linear PK
# --------------------------------------------------------------------------------------------- #
# dx/dt = A x + e1 u is linear: under a constant rate r over h, x(t + h) = Phi(h) x(t) + Gamma(h) r,
# Phi = e^{Ah}, Gamma = A^-1 (Phi - I) e1, in closed form. Ad is decoupled; the (Ac, Ap) block has real,
# distinct eigenvalues; the coupling uses divided differences (e^{ah} - e^{bh}) / (a - b), regular at a = b.

def _sinhc(z):
    """sinh(z) / z, by its series near 0."""
    small = ca.fabs(z) < 1e-3
    zs = ca.if_else(small, 1.0, z)
    return ca.if_else(small, 1.0 + z ** 2 / 6.0 + z ** 4 / 120.0, ca.sinh(zs) / zs)


def _dd(a, b, h):
    """(e^{ah} - e^{bh}) / (a - b): regular at a = b, finite far apart (the slower exponential factored out)."""
    z = (a - b) * h / 2.0
    near = ca.fabs(z) < 1.0
    zn = ca.if_else(near, z, 0.0)
    d = ca.if_else(near, 1.0, ca.fabs(a - b))
    far = ca.exp(ca.fmax(a, b) * h) * -ca.expm1(-d * h) / d
    return ca.if_else(near, h * ca.exp((a + b) * h / 2.0) * _sinhc(zn), far)


def _build_transition() -> ca.Function:
    th, h = ca.SX.sym("th", 4), ca.SX.sym("h")
    ka, k12, k21, kel = th[0], th[1], th[2], th[3]
    c = ca.vertcat(ca.horzcat(-(k12 + kel), k21), ca.horzcat(k12, -k21))
    tr = -(k12 + kel + k21)
    s = ca.sqrt((k12 + kel - k21) ** 2 + 4.0 * k12 * k21)
    l1, l2 = (tr + s) / 2.0, (tr - s) / 2.0
    eye = ca.SX.eye(2)
    exp_c = ca.exp(l2 * h) * eye + _dd(l1, l2, h) * (c - l2 * eye)
    w = ca.vertcat(ka, 0.0)
    v = (_dd(l1, -ka, h) * ((c - l2 * eye) @ w) - _dd(l2, -ka, h) * ((c - l1 * eye) @ w)) / (l1 - l2)
    phi = ca.vertcat(ca.horzcat(ca.exp(-ka * h), ca.SX.zeros(1, 2)), ca.horzcat(v, exp_c))
    a = ca.vertcat(ca.horzcat(-ka, 0.0, 0.0), ca.horzcat(ka, -(k12 + kel), k21), ca.horzcat(0.0, k12, -k21))
    gamma = ca.solve(a, (phi - ca.SX.eye(3))[:, 0])
    return ca.Function("transition", [th, h], [phi, gamma], ["th", "h"], ["phi", "gamma"])


transition = _build_transition()
"""``transition(th, h) -> (Phi(h), Gamma(h))`` for ``th = (ka, k12, k21, kel)``, numeric or symbolic."""


class Grid(NamedTuple):
    """The nodes of a record: t = 0 and every reading (and, for a fit to Cc, every Cc sample);
    per interval between nodes, its pieces of constant rate ``(h, rate)``."""
    t: np.ndarray
    steps: list[list[tuple[float, float]]]
    obs_idx: np.ndarray            # the nodes of the MAP readings
    cc_idx: np.ndarray | None = None  # the nodes of the Cc samples, when fitted


def grid(patient: Patient, with_cc: bool = False) -> Grid:
    """Nodes {0} and the readings (and the Cc samples), each interval cut at the dose edges."""
    t_obs, doses = patient.map.times, patient.doses
    t_cc = patient.cc.times if with_cc else np.empty(0)
    t = np.unique(np.concatenate([[0.0], t_obs, t_cc]))
    edges = np.unique(np.concatenate([doses.times, doses.times + doses.durations]))

    def pieces(a: float, b: float) -> list[tuple[float, float]]:
        cuts = np.concatenate([[a], edges[(edges > a) & (edges < b)], [b]])
        return [(hi - lo, doses(0.5 * (lo + hi))) for lo, hi in zip(cuts[:-1], cuts[1:])]

    steps = [pieces(a, b) for a, b in zip(t[:-1], t[1:])]
    cc_idx = np.searchsorted(t, t_cc) if with_cc else None
    return Grid(t, steps, np.searchsorted(t, t_obs), cc_idx)


def first_row(g: Grid) -> int:
    """The node of the first sample, MAP or (when fitted) Cc."""
    k0 = int(g.obs_idx[0])
    return k0 if g.cc_idx is None else min(k0, int(g.cc_idx[0]))


def from_first_sample(g: Grid) -> Grid:
    """The grid from the first sample on, where the state is estimated: [0, t0] only gives the
    population's state at t0, the recorded doses from empty compartments."""
    k0 = first_row(g)
    cc_idx = None if g.cc_idx is None else g.cc_idx - k0
    return Grid(g.t[k0:], g.steps[k0:], g.obs_idx - k0, cc_idx)


def simulate_exact(th, g: Grid, x0) -> np.ndarray:
    """The linear PK's states ``(3, N + 1)`` at the nodes of ``g`` from ``x0`` at its first node."""
    th = np.asarray(th, float)
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    xs = [np.asarray(x0, float)]
    for segs in g.steps:
        x = xs[-1]
        for h, r in segs:
            if h not in cache:
                phi, gam = transition(th, h)
                cache[h] = (np.array(phi), np.array(gam).ravel())
            phi, gam = cache[h]
            x = phi @ x + gam * r
        xs.append(x)
    return np.array(xs).T


def ic_guess(patient: Patient) -> np.ndarray:
    """The population's state at the first reading: the recorded doses from empty compartments."""
    g = grid(patient)
    return simulate_exact([POP[k] for k in PK], g, np.zeros(3))[:, g.obs_idx[0]]
