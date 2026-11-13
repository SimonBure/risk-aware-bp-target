"""Which of the reduced model's seven parameters to estimate (Section II.A): the criteria of Wu et al.
(2011), McLean et al. (2012) and Chu et al. (2009), as McLean et al. (2012) set them out.

The state at the first sample is always estimated, never a candidate. Each patient is fitted on its
own (``wholefit``), the parameters left out at their population values.

- ``reference``: the all-free fit, multistart (``wholefit.multistart`` on the MAP's sum of squares).
- Wu (O + r_CC): one ranking for the cohort (``yao_ranking`` of the stacked sensitivities at the
  population, each patient's initial state projected out), then the nested subsets along it, each fit
  from the population and from the previous one (``fit_subset``); the corrected critical ratio pooled
  over the patients (``r_cc_pooled``), and by vote and worst case.
- McLean (SrCC): at each step, the candidate whose subset has the lowest pooled r_CC joins.
- Chu: no data; the forward selection on the Monte Carlo mean bias over population draws of the
  parameters and initial state under jittered recorded protocols, stopping when a step buys less than
  sigma^2, the median time-averaged residual variance of the reference fits.
"""
from typing import NamedTuple, Sequence

import numpy as np
import scipy.linalg as sl
from joblib import Parallel, delayed

from bptarget.fit import IC_NAMES, PatientModel
from bptarget.model import IC_SD, OMEGA, POP, REDUCED, Doses, Patient, Readings, ic_guess, reduced_omega, reduced_pop
from bptarget.wholefit import multistart, solve, sse

NAMES = list(REDUCED)
FULL_NAMES = ["Cendo", "ka", "Vc", "k12", "k21", "kel", "E0", "Emax", "EC50"]  # the linear model's, drawn by Chu
N_JITTER, N_THETA, JITTER, CHU_SEED = 10, 32, 0.2, 0  # Chu: protocols per record, truths per protocol


class SubsetFit(NamedTuple):
    patient: int
    free: tuple[str, ...]
    sse: float
    n_obs: int
    converged: bool
    params: dict[str, float]
    x: np.ndarray        # the node states, a warm start for the next subset
    t: np.ndarray
    residual: np.ndarray
    start: str           # population, warm, floor (the previous fit kept) or the multistart's


# --------------------------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------------------------- #
def reference(patient: Patient, n_jobs: int = 1) -> SubsetFit:
    """The all-free fit from many starts."""
    fit, record = multistart(patient, "reduced", n_jobs, by="sse")
    return SubsetFit(patient.id, tuple(NAMES), sse(patient, fit), len(patient.map.times), True, fit.params, fit.x,
                     patient.map.times, patient.map.values - fit.y, record["start"])


def fit_subset(patient: Patient, free: Sequence[str], warm: SubsetFit | None = None) -> SubsetFit:
    """The fit of ``free`` (in NAMES order) from the population and, when given, from ``warm``'s solution;
    the lower sum of squares kept, or ``warm`` itself when both end above it (a feasible point of this
    model), so nested fits never get worse."""
    free = tuple(n for n in NAMES if n in free)
    starts = {"population": (None, None)} if warm is None else {"population": (None, None), "warm": (warm.x, warm.params)}
    best = None
    for label, (x_start, theta0) in starts.items():
        fit = solve(patient, "reduced", free, theta0, x_start)
        s = sse(patient, fit)
        if best is None or s < best[1]:
            best = (label, s, fit)
    label, s, fit = best
    if warm is not None and warm.sse < s:
        return warm._replace(free=free, start="floor")
    return SubsetFit(patient.id, free, s, len(patient.map.times), fit.converged, fit.params, fit.x,
                     patient.map.times, patient.map.values - fit.y, label)


def fit_cohort(patients: list[Patient], free: Sequence[str], warm: dict[int, SubsetFit] | None = None,
               n_jobs: int = 8) -> list[SubsetFit]:
    warm = warm or {}
    return Parallel(n_jobs=n_jobs)(delayed(fit_subset)(p, tuple(free), warm.get(p.id)) for p in patients)


# --------------------------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------------------------- #
def patient_s(patient: Patient, eta: np.ndarray | None = None) -> np.ndarray:
    """The scaled sensitivities (mmHg per prior SD) at the readings, columns NAMES then the initial state,
    at eta (the population by default)."""
    return PatientModel(patient)(np.zeros(len(NAMES) + 3) if eta is None else np.asarray(eta, float))[1]


def residual_after_ic(s: np.ndarray, n_ic: int = 3) -> np.ndarray:
    """The parameters' columns with the initial state's projected out."""
    q, _ = np.linalg.qr(s[:, -n_ic:])
    body = s[:, :-n_ic]
    return body - q @ (q.T @ body)


def stacked_residuals(patients: list[Patient]) -> np.ndarray:
    return np.vstack([residual_after_ic(patient_s(p)) for p in patients])


def yao_ranking(s: np.ndarray, names: Sequence[str]) -> list[tuple[str, float]]:
    """Yao et al.'s orthogonalisation ranking, most to least estimable: QR with column pivoting
    (Householder), each score the RMS of the part of the column the previous ones do not imitate."""
    _, r, pivots = sl.qr(s.astype(float), pivoting=True, mode="economic")
    scores = np.zeros(s.shape[1])
    diag = np.abs(np.diag(r)) / np.sqrt(s.shape[0])
    scores[:len(diag)] = diag
    scores[np.linalg.norm(s[:, pivots], axis=0) == 0.0] = 0.0
    return [(names[j], float(g)) for j, g in zip(pivots, scores)]


# --------------------------------------------------------------------------------------------- #
# Wu and McLean
# --------------------------------------------------------------------------------------------- #
def _r_cc(J: np.ndarray, n_obs: int, n_runs: int) -> np.ndarray:
    """r_CC,k = (p - k) / N (max(r_C,k - 1, 2 r_C,k / (p - k + 2)) - 1), r_C,k = (J_k - J_p) / (p - k),
    each fixed parameter counted once per patient; 0 for the full model."""
    p, out = len(J), np.zeros(len(J))
    for k in range(1, p):
        fixed = n_runs * (p - k)
        rc = (J[k - 1] - J[-1]) / fixed
        out[k - 1] = fixed / n_obs * (max(rc - 1.0, 2.0 * rc / (fixed + 2)) - 1.0)
    return out


def r_cc(J: Sequence[float], n_obs: int) -> np.ndarray:
    return _r_cc(np.asarray(J, float), n_obs, 1)


def r_cc_pooled(J_by_patient, n_obs_by_patient) -> np.ndarray:
    J = np.sum([np.asarray(j, float) for j in J_by_patient], axis=0)
    return _r_cc(J, int(np.sum(n_obs_by_patient)), len(J_by_patient))


def wu_select(rcc: np.ndarray) -> int:
    return int(np.argmin(rcc)) + 1


def wu_vote(ks: Sequence[int]) -> int:
    values, counts = np.unique(np.asarray(ks), return_counts=True)
    return int(values[np.argmax(counts)])


def wu_worst_case(rcc_by_patient: np.ndarray) -> int:
    return int(np.argmin(np.max(np.asarray(rcc_by_patient), axis=0))) + 1


def standardised(fits: dict, full: tuple) -> tuple[dict, dict]:
    """Per patient sigma^2 = SSE_p / (N - p - 3), and J = SSE / sigma^2 per subset."""
    sigma2 = {f.patient: f.sse / (f.n_obs - len(full) - 3) for f in fits[full]}
    return sigma2, {s: {f.patient: f.sse / sigma2[f.patient] for f in fs} for s, fs in fits.items()}


def wu(patients: list[Patient], fits: dict, n_jobs: int = 8) -> dict:
    """Wu's selection; ``fits`` (subset -> fits, the reference under the full subset) is filled in place."""
    ranking = yao_ranking(stacked_residuals(patients), NAMES)
    order = [n for n, _ in ranking]
    nested = [tuple(n for n in NAMES if n in order[:k]) for k in range(1, len(NAMES) + 1)]
    warm = {}
    for subset in nested[:-1]:
        if subset not in fits:
            fits[subset] = fit_cohort(patients, subset, warm, n_jobs)
        warm = {f.patient: f for f in fits[subset]}
    full = nested[-1]
    sigma2, J = standardised(fits, full)
    ids = sorted(f.patient for f in fits[full])
    n_obs = {f.patient: f.n_obs for f in fits[full]}
    per = {i: r_cc([J[s][i] for s in nested], n_obs[i]) for i in ids}
    pooled = r_cc_pooled([[J[s][i] for s in nested] for i in ids], [n_obs[i] for i in ids])
    return dict(ranking=ranking, nested=nested, pooled=pooled, k_pooled=wu_select(pooled),
                k_vote=wu_vote([wu_select(per[i]) for i in ids]),
                k_worst_case=wu_worst_case(np.array([per[i] for i in ids])), per_patient=per)


def mclean(patients: list[Patient], fits: dict, n_jobs: int = 8) -> dict:
    """McLean's simultaneous ranking and selection (SrCC); ``fits`` filled in place."""
    full = tuple(NAMES)
    sigma2, J = standardised(fits, full)
    n_obs = {f.patient: f.n_obs for f in fits[full]}
    ids = sorted(n_obs)
    chosen, path, warm = [], [], {}
    while len(chosen) < len(NAMES):
        k, scores = len(chosen) + 1, {}
        for cand in [n for n in NAMES if n not in chosen]:
            subset = tuple(n for n in NAMES if n in chosen + [cand])
            if subset not in fits:
                fits[subset] = fit_cohort(patients, subset, warm, n_jobs)
            J_k = {f.patient: f.sse / sigma2[f.patient] for f in fits[subset]}
            vectors = [[J_k[i] if j == k - 1 else J[full][i] for j in range(len(NAMES))] for i in ids]
            scores[cand] = float(r_cc_pooled(vectors, [n_obs[i] for i in ids])[k - 1])
        best = min(scores, key=scores.get)
        chosen.append(best)
        path.append((k, best, scores[best], scores))
        warm = {f.patient: f for f in fits[tuple(n for n in NAMES if n in chosen)]}
    k_best = int(np.argmin([r for _, _, r, _ in path])) + 1
    return dict(ranking=chosen, k=k_best, path=path)


# --------------------------------------------------------------------------------------------- #
# Chu
# --------------------------------------------------------------------------------------------- #
def residual_variance(t: np.ndarray, r: np.ndarray) -> float:
    """(1/T) int r^2 dt by trapezoid: the residual variance per unit time, whatever the sampling."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    return float(np.trapezoid(r ** 2, t) / (t[-1] - t[0]))


def sample_population(n: int, rng) -> dict[str, np.ndarray]:
    """n parameter sets of the linear model (independent log-normals, Emax <= E0 redrawn), reduced."""
    full = {k: np.empty(0) for k in FULL_NAMES}
    while len(full["ka"]) < n:
        batch = {k: POP[k] * np.exp(OMEGA[k] * rng.standard_normal(n)) for k in FULL_NAMES}
        keep = batch["Emax"] > batch["E0"]
        full = {k: np.concatenate([full[k], batch[k][keep]]) for k in FULL_NAMES}
    f = {k: v[:n] for k, v in full.items()}
    return {"ka": f["ka"], "k12": f["k12"], "k21": f["k21"], "kel": f["kel"], "Emax": f["Emax"],
            "a": (f["Emax"] - f["E0"]) * f["EC50"] * f["Vc"], "b": (f["Cendo"] + f["EC50"]) * f["Vc"]}


def sample_protocol(sources: list[Patient], rng, jitter: float = JITTER) -> Patient:
    """A recorded protocol drawn from ``sources``, each dose's rate and duration times exp(jitter N(0, 1)),
    durations clipped so doses never overlap; the source's reading times."""
    src = sources[int(rng.integers(len(sources)))]
    d = src.doses
    rates = d.amounts / d.durations * np.exp(jitter * rng.standard_normal(len(d.times)))
    durations = d.durations * np.exp(jitter * rng.standard_normal(len(d.times)))
    durations[:-1] = np.minimum(durations[:-1], np.diff(d.times))
    return Patient(src.id, Doses(d.times.copy(), rates * durations, durations),
                   Readings(src.map.times.copy(), src.map.values.copy()))


def chu_bias(s: np.ndarray, free: Sequence[int], eta: np.ndarray) -> float:
    """||(I - P_free) S_fixed eta_fixed||^2: the output error left by fitting ``free`` when the others sit
    eta (prior SDs) away from their population values."""
    fixed = [j for j in range(s.shape[1]) if j not in free]
    v = s[:, fixed] @ np.asarray(eta, float)[fixed]
    if free:
        q, _ = np.linalg.qr(s[:, list(free)])
        v = v - q @ (q.T @ v)
    return float(v @ v)


def chu_select(s_draws, eta_draws, sigma2: float, names: Sequence[str], forced: Sequence[str]):
    """Forward ranking on the mean bias; a parameter counts while its step lowers it by at least sigma^2.
    Returns the ranking, the mean bias after each step (first: the forced only) and the count kept."""
    names = list(names)
    free = [names.index(n) for n in forced]
    remaining = [j for j in range(len(names)) if j not in free]

    def mean_bias(cols):
        return float(np.mean([chu_bias(s, cols, e) for s, e in zip(s_draws, eta_draws)]))

    bias, ranking, k = [mean_bias(free)], [], None
    while remaining:
        scores = [mean_bias(free + [j]) for j in remaining]
        i = int(np.argmin(scores))
        if k is None and bias[-1] - scores[i] < sigma2:
            k = len(ranking)
        free.append(remaining.pop(i))
        ranking.append(names[free[-1]])
        bias.append(scores[i])
    return ranking, bias, len(ranking) if k is None else k


def chu(patients: list[Patient], sigma2: float, seed: int = CHU_SEED):
    """Chu's selection over N_JITTER jittered protocols per record and N_THETA truths per protocol."""
    rng = np.random.default_rng(seed)
    pop, om = reduced_pop(), reduced_omega()
    s_draws, eta_draws = [], []
    for src in patients:
        for _ in range(N_JITTER):
            proto = sample_protocol([src], rng)
            model = PatientModel(proto)
            theta = sample_population(N_THETA, rng)
            eta_par = np.column_stack([(np.log(theta[k]) - np.log(pop[k])) / om[k] for k in NAMES])
            nominal = ic_guess(proto)
            ic = np.clip(nominal + IC_SD * rng.standard_normal((N_THETA, 3)), 0.0, None)
            for e in np.hstack([eta_par, (ic - nominal) / IC_SD]):
                s_draws.append(np.array(model(e)[1]))
                eta_draws.append(e)
    return chu_select(s_draws, eta_draws, sigma2, NAMES + IC_NAMES, forced=IC_NAMES)
