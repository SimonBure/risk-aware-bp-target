"""The online fit of the reduced model (Section II.B, eq. (fit)) and its linearised posterior.

The unknowns are eta in normalised coordinates (Appendix A): the seven log-parameters over their
population SDs, then the three amounts at the first reading, x0 = x0_bar + IC_SD eta, around the
population's state there. The prior is N(0, I), the box +-3 on the parameters and [0, cap] on the
amounts, and the MAP without NE h(0) at least ``MAP_FLOOR``. At a decision time t the readings up to
t are fitted; sigma_t^2 by fixed point (Appendix B); P_t = (S_t^T S_t / sigma_t^2 + I)^-1.
"""
import casadi as ca
import numpy as np
from scipy.optimize import Bounds, least_squares, minimize

from bptarget.model import (IC_SD, REDUCED, X0_CAP, from_first_sample, grid, ic_guess, reduced_map,
                            reduced_omega, reduced_pop, transition)

H = 300.0                 # s, the forecast window of the online fits
DT = 150.0                # s, between decision times
T_FIRST = 300.0           # s, the first decision time after the first reading
SIGMA2_0 = 3.987          # mmHg^2, start and floor of the fixed point (Appendix B)
FP_TOL, FP_MAX = 0.05, 5  # fixed point: relative change, iterations
MIN_DOF = 10.0            # keep SIGMA2_0 when n_t - p_t < MIN_DOF
N_BOX = 3.0               # the box: pop exp(+-3 omega)
MAP_FLOOR = 30.0          # mmHg, the lowest MAP without NE
FLOOR_TOL = 1e-6          # mmHg: a fit this far under the floor counts as on it (SLSQP's precision)
NAMES = list(REDUCED)
IC_NAMES = ["Ad0", "Ac0", "Ap0"]


def bounds(x_bar: np.ndarray, n_params: int, n_box: float = N_BOX) -> tuple[np.ndarray, np.ndarray]:
    """The box in eta: +-n_box on the parameters; the amounts x_bar + IC_SD eta in [0, X0_CAP]."""
    lo = np.concatenate([np.full(n_params, -n_box), -x_bar / IC_SD])
    hi = np.concatenate([np.full(n_params, n_box), (X0_CAP - x_bar) / IC_SD])
    return lo, hi


def central_amounts(th, x, g):
    """Ac at every node of ``g``, chained by the exact transition from the state ``x`` at its first node
    (symbolic)."""
    memo = {}
    ac = [x[1]]
    for segs in g.steps:
        for h, r in segs:
            if h not in memo:
                memo[h] = transition(th, h)
            phi, gam = memo[h]
            x = phi @ x + gam * r
        ac.append(x[1])
    return ca.vertcat(*ac)


class PatientModel:
    """The reduced model on one record: ``model(eta) -> (y, S)``, the MAP at every reading under the
    record's doses and its sensitivity to eta (cached per eta); ``baseline``, ``params``, the box
    ``lo``, ``hi``."""

    def __init__(self, patient, n_box: float = N_BOX):
        x_bar = ic_guess(patient)
        self.names = NAMES + IC_NAMES
        self.theta_bar = dict(reduced_pop(), **dict(zip(IC_NAMES, x_bar.tolist())))
        omega = dict(reduced_omega(), **{n: IC_SD for n in IC_NAMES})
        self.omega = np.array([omega[n] for n in self.names])
        self._sens = self._build(from_first_sample(grid(patient)), omega)
        self.lo, self.hi = bounds(x_bar, len(NAMES), n_box)
        self._key, self._val = None, None

    def _build(self, g, omega) -> ca.Function:
        """y and S = dy/d(raw) diag(omega) at the readings, raw = omega eta (log-deviations, nmol)."""
        raw = ca.MX.sym("eta", len(self.names))
        comps = [0.0, 0.0, 0.0]
        p = self.theta_bar.copy()
        for i, name in enumerate(self.names):
            if name in IC_NAMES:
                comps[IC_NAMES.index(name)] = self.theta_bar[name] + raw[i]
            else:
                p[name] = self.theta_bar[name] * ca.MX.exp(raw[i])
        th = ca.vertcat(p["ka"], p["k12"], p["k21"], p["kel"])
        ac = central_amounts(th, ca.vertcat(*comps), g)
        y = reduced_map(p, ac[g.obs_idx.tolist()])
        s = ca.jacobian(y, raw) @ ca.diag(ca.vertcat(*[omega[n] for n in self.names]))
        return ca.Function("sens", [raw], [y, s])

    def __call__(self, eta):
        key = eta.tobytes()
        if key != self._key:
            y, s = self._sens(self.omega * eta)
            self._key, self._val = key, (np.array(y).ravel(), np.array(s))
        return self._val

    def baseline(self, eta) -> tuple[float, np.ndarray]:
        """The MAP without NE, h(0) = Emax - a / b, at eta, and its gradient in eta."""
        i = [self.names.index(n) for n in ("Emax", "a", "b")]
        emax, a, b = (self.theta_bar[n] * np.exp(self.omega[k] * eta[k]) for n, k in zip(("Emax", "a", "b"), i))
        grad = np.zeros(len(eta))
        grad[i] = self.omega[i] * np.array([emax, -a / b, a / b])
        return emax - a / b, grad

    def params(self, eta) -> dict[str, float]:
        """The physical values at eta, by name: parameters, then the amounts at the first reading (nmol)."""
        bar, om = self.theta_bar, self.omega
        return {n: float(bar[n] * np.exp(om[k] * eta[k]) if n in NAMES else bar[n] + om[k] * eta[k])
                for k, n in enumerate(self.names)}


def fit_prefix(model, y_obs, m, sigma2, starts):
    """argmin ||y_obs[:m] - y(eta)[:m]||^2 / sigma2 + ||eta||^2 in the box, from each start; the lowest
    kept. When its MAP without NE is under ``MAP_FLOOR``, the fits are redone with the floor imposed
    (SLSQP on the cost over its value at the start: SLSQP's tests are absolute) and the lowest on it
    or above, within ``FLOOR_TOL``, is kept."""
    sig = np.sqrt(sigma2)

    def res(eta):
        y, _ = model(eta)
        return np.concatenate([(y_obs[:m] - y[:m]) / sig, eta])

    def jac(eta):
        _, s = model(eta)
        return np.vstack([-s[:m] / sig, np.eye(len(eta))])

    fits = [least_squares(res, np.clip(x0, model.lo, model.hi), jac=jac, bounds=(model.lo, model.hi), method="trf")
            for x0 in starts]
    best = min(fits, key=lambda r: r.cost)
    if model.baseline(best.x)[0] >= MAP_FLOOR:
        return best

    def cost(eta):
        r = res(eta)
        return 0.5 * r @ r, jac(eta).T @ r

    floor = {"type": "ineq", "fun": lambda e: model.baseline(e)[0] - MAP_FLOOR,
             "jac": lambda e: model.baseline(e)[1]}
    for x0 in starts:
        x0 = np.clip(x0, model.lo, model.hi)
        j0 = cost(x0)[0]
        r = minimize(lambda e: tuple(v / j0 for v in cost(e)), x0, jac=True, method="SLSQP",
                     bounds=Bounds(model.lo, model.hi), constraints=[floor], options={"maxiter": 500})
        r.cost = r.fun * j0
        fits.append(r)
    return min((r for r in fits if model.baseline(r.x)[0] >= MAP_FLOOR - FLOOR_TOL), key=lambda r: r.cost)


def posterior(model, eta, m, sigma2):
    """P = (S_m^T S_m / sigma2 + I)^-1 and p_t = tr(I - P) at eta."""
    _, s = model(eta)
    f = s[:m].T @ s[:m] / sigma2
    lam = np.linalg.eigvalsh(f)
    p = np.linalg.inv(f + np.eye(len(eta)))
    return p, float(np.sum(lam / (1 + lam)))


def fit_fixed_point(model, y_obs, m, starts):
    """The fit of the first m readings with sigma^2 by fixed point: eta, sigma2, P, p_t and an info dict
    (``iters``, ``floor``: SIGMA2_0 kept, ``converged``: the solver succeeded)."""
    sigma2, floor, it = SIGMA2_0, False, 0
    for it in range(1, FP_MAX + 1):
        r = fit_prefix(model, y_obs, m, sigma2, starts)
        p, p_eff = posterior(model, r.x, m, sigma2)
        y, _ = model(r.x)
        sse = float(np.sum((y_obs[:m] - y[:m]) ** 2))
        if m - p_eff < MIN_DOF:
            if sigma2 != SIGMA2_0:  # back to the floor, refitted there
                sigma2, floor = SIGMA2_0, True
                r = fit_prefix(model, y_obs, m, sigma2, starts)
                p, p_eff = posterior(model, r.x, m, sigma2)
            floor = True
            break
        new = sse / (m - p_eff)
        if abs(new - sigma2) / sigma2 < FP_TOL:
            break
        sigma2 = new
    return r.x, sigma2, p, p_eff, dict(iters=it, floor=floor, sse=sse, converged=bool(r.success))


def decision_times(t0, T, first=T_FIRST, step=DT, horizon=H) -> np.ndarray:
    """t0 + first, then one every step, while t + horizon <= T (s)."""
    return np.arange(t0 + first, T - horizon + 1e-9, step)


def fits(patient, model, until: float | None = None):
    """The online fits of one record: at each decision time t (up to ``until``), the fit of the readings
    up to t from eta = 0 and from the previous estimate, ``(t, m, eta, sigma2, P, p_t, info)``."""
    t_obs, y_obs = patient.map.times, patient.map.values
    zero, warm = np.zeros(len(model.names)), None
    for t in decision_times(t_obs[0], t_obs[-1]):
        if until is not None and t > until + 1e-6:
            return
        m = int(np.searchsorted(t_obs, t, side="right"))
        eta, sigma2, p, p_eff, info = fit_fixed_point(model, y_obs, m, [zero] + ([warm] if warm is not None else []))
        warm = eta
        yield t, m, eta, sigma2, p, p_eff, info
