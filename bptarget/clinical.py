"""The clinical cohort of Section III.A: its records, their forecast windows and the score.

The records are not public (``data/README.md``). ``load`` reads them in the layout of Joachim et al.'s
files: ``data.csv`` (``id``, ``time(s)``, ``obs``, ``obsid``: 0 for Cc, 1 for the MAP, ``obs`` '.' when
none) and ``injections.csv`` (``patient_id``, ``injection_time_s``, ``amount_nmol``, ``duration_s``).

``windows``: the online fits of a record at t = 5 min after its first reading, then every 2.5 min,
and over W_t = (t + T_r, t + 10 min], cut at the record's end, under the recorded NE, the score
e_t^2 / V_t and where the lowest reading falls in the forecast's law of the minimum (``pit``). A record
sampled every second is thinned to one reading every 10 s first (``subsample``), as the virtual
patients are read.

The discussion's two checks on correlated errors: ``windows(..., whiten=True)`` computes P_t with the
sensitivities whitened by an Ornstein-Uhlenbeck model of the fit's residuals (``residual_tau``,
``ou_whiten``), the rest unchanged; ``acf`` is the residuals' autocorrelation at time lags.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bptarget.fit import N_BOX, PatientModel, fits
from bptarget.forecast import ALPHAS, N_DRAWS, draws, pit, predictive, score
from bptarget.model import Doses, Patient, Readings
from bptarget.stats import median_interval
from bptarget.target import window
from bptarget.virtual import SAMPLING, SEED

BINS = {"all": (-np.inf, np.inf), "5-10": (5.0, 10.0), "10-15": (10.0, 15.0), ">=15": (15.0, np.inf)}  # t, min
BOLUS_SPAN = 60.0  # s: a dose over at most this long is a bolus, a longer one a piece of infusion
MAX_TAU = 120.0    # s: the OU's correlation time is capped, slow drift left to the noise
LAGS = (1, 2, 3, 5, 10, 20, 30, 60, 120, 300)  # s: the autocorrelation's lags, up to the forecast window


def load(data_dir: Path) -> list[Patient]:
    """The records of ``data_dir``, by increasing id."""
    data_dir = Path(data_dir)
    if not (data_dir / "data.csv").exists():
        raise FileNotFoundError(f"no data.csv in {data_dir}: the clinical records are available on request "
                                "(data/README.md)")
    obs = pd.read_csv(data_dir / "data.csv")
    inj = pd.read_csv(data_dir / "injections.csv")
    obs = obs[obs["obs"].astype(str) != "."].astype({"obs": float, "obsid": int})
    patients = []
    for pid in sorted(obs["id"].unique()):
        mine = obs[obs["id"] == pid]
        bp, cc = mine[mine["obsid"] == 1], mine[mine["obsid"] == 0]
        d = inj[inj["patient_id"] == pid].sort_values("injection_time_s")
        doses = Doses(d["injection_time_s"].to_numpy(float), d["amount_nmol"].to_numpy(float),
                      d["duration_s"].to_numpy(float))
        patients.append(Patient(int(pid), doses, Readings(bp["time(s)"], bp["obs"]),
                                Readings(cc["time(s)"], cc["obs"]) if len(cc) else None))
    return patients


def sampling(patient: Patient) -> int:
    """The record's MAP sampling period, s: the median gap between readings."""
    return round(float(np.median(np.diff(patient.map.times))))


def subsample(patient: Patient, period: float = SAMPLING) -> Patient:
    """The record with one MAP reading every ``period`` s from the first: each reading kept is the first at
    least ``period`` s (half a second's tolerance) after the last kept, so a gap restarts the series."""
    t, y = patient.map.times, patient.map.values
    keep = [0]
    for j in range(1, len(t)):
        if t[j] - t[keep[-1]] >= period - 0.5:
            keep.append(j)
    return Patient(patient.id, patient.doses, Readings(t[keep], y[keep]), patient.cc)


def residual_tau(t: np.ndarray, r: np.ndarray, max_tau: float = MAX_TAU) -> float:
    """The correlation time of an OU process through the residuals r at times t, s: -dt / ln(rho), rho
    their lag-one correlation clipped to [0, 0.999] and dt the median gap, capped at ``max_tau``; 0 when
    they are uncorrelated."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    rho = 0.0
    if len(r) >= 4 and np.std(r[:-1]) > 1e-12 and np.std(r[1:]) > 1e-12:
        rho = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.0, 0.999))
    return 0.0 if rho <= 1e-8 else min(max_tau, float(-np.median(np.diff(t)) / np.log(rho)))


def ou_whiten(t: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    """The rows of x (one per time t) as the innovations of an OU process of correlation time tau, so
    that x^T x under OU noise reads as under white noise: x_0, then (x_j - phi_j x_{j-1}) / sqrt(1 - phi_j^2),
    phi_j = exp(-(t_j - t_{j-1}) / tau). x unchanged when tau = 0."""
    x = np.array(x, float)
    if tau <= 0 or len(x) < 2:
        return x
    phi = np.exp(-np.diff(np.asarray(t, float)) / tau).reshape(-1, *[1] * (x.ndim - 1))
    x[1:] = (x[1:] - phi * x[:-1]) / np.sqrt(1 - phi ** 2)
    return x


def whitened_posterior(model: PatientModel, eta: np.ndarray, t: np.ndarray, y: np.ndarray, m: int,
                       sigma2: float) -> tuple[np.ndarray, float]:
    """P_t with the first m sensitivities whitened by the OU of the fit's residuals there, and its tau."""
    mean, s = model(eta)
    tau = residual_tau(t[:m], y[:m] - mean[:m])
    sw = ou_whiten(t[:m], s[:m], tau)
    return np.linalg.inv(sw.T @ sw / sigma2 + np.eye(len(eta))), tau


def windows(patient: Patient, n_box: float = N_BOX, k: int = N_DRAWS, whiten: bool = False) -> list[dict]:
    """A row per forecast window of the record (none when a sampling gap covers it): its decision time
    ``t_min`` after the first reading, its readings ``n_w``, sigma_t^2, whether the fit converged, the
    ``pit`` of its lowest reading, the score ``e2_v`` = e^2 / V, e^2, V and the fit's part of V,
    tr(S_w P S_w^T) / n_w. ``sampling``: the record's own period, before any thinning. ``whiten``: P_t
    from ``whitened_posterior``, its ``tau`` in the row."""
    own = sampling(patient)
    if own < SAMPLING:
        patient = subsample(patient)
    model = PatientModel(patient, n_box)
    t_obs, y_obs = patient.map.times, patient.map.values
    rows = []
    for t, m, eta, sigma2, p, _, info in fits(patient, model):
        w = window(t_obs, t, t_obs[-1])
        if not w.any():
            continue
        tau = np.nan
        if whiten:
            p, tau = whitened_posterior(model, eta, t_obs, y_obs, m, sigma2)
        mean, s = model(eta)
        pred = predictive(t_obs[w], mean[w], s[w], p, sigma2)
        dr = draws(len(eta), int(w.sum()), k, seed=[SEED, patient.id, int(round(t))])
        y = y_obs[w]
        rows.append(dict(patient=patient.id, sampling=own, t_min=(t - t_obs[0]) / 60, n_w=int(w.sum()),
                         sigma2_t=sigma2, converged=info["converged"], pit=pit(pred, y, dr), e2_v=score(pred, y),
                         e2=float(np.mean((y - pred.mean) ** 2)), v_fit=float(np.mean(pred.var)),
                         v=float(np.mean(pred.var) + pred.sigma2), tau=tau))
    return rows


def acf(t: np.ndarray, r: np.ndarray, lags=LAGS, tol: float = 0.5) -> np.ndarray:
    """The autocorrelation of r at each time lag h, s: the mean of e_i e_j over the pairs with
    |t_j - t_i - h| <= tol over the mean of e^2, e = r - mean(r). Pairs are matched in time, so a gap
    does not shift the lags; nan where no pair exists."""
    t = np.asarray(t, float)
    e = np.asarray(r, float) - np.mean(r)
    var = np.mean(e ** 2)
    out = []
    for h in lags:
        j = np.searchsorted(t, t + h - tol)
        i = np.flatnonzero(j < len(t))
        j = j[i]
        pair = np.abs(t[j] - t[i] - h) <= tol
        out.append(np.mean(e[i[pair]] * e[j[pair]]) / var if pair.any() else np.nan)
    return np.array(out)


def _bins(rows: pd.DataFrame):
    for label, (lo, hi) in BINS.items():
        yield label, rows[(rows.t_min >= lo) & (rows.t_min < hi)]


def score_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Table I: per bin of t, the windows and their patients, the median score with its 95 % interval
    (patients resampled), its mean and its 90th and 95th percentiles."""
    out = {}
    for label, g in _bins(rows):
        lo, hi = median_interval(g.e2_v, g.patient)
        out[label] = dict(windows=len(g), patients=g.patient.nunique(), median=g.e2_v.median(), lo=lo, hi=hi,
                          mean=g.e2_v.mean(), q90=g.e2_v.quantile(0.90), q95=g.e2_v.quantile(0.95))
    return pd.DataFrame.from_dict(out, orient="index")


def sides(rows: pd.DataFrame) -> pd.DataFrame:
    """The score's two sides per bin of t, mmHg^2: the median of e^2, of V, of its fit part and of
    sigma_t^2; the median RMSE and sqrt(V), mmHg."""
    out = {}
    for label, g in _bins(rows):
        out[label] = dict(e2=g.e2.median(), v=g.v.median(), v_fit=g.v_fit.median(), sigma2_t=g.sigma2_t.median(),
                          rmse=np.sqrt(g.e2).median(), sqrt_v=np.sqrt(g.v).median())
    return pd.DataFrame.from_dict(out, orient="index")


def calibration(rows: pd.DataFrame, alphas=ALPHAS) -> pd.DataFrame:
    """Per level alpha, the windows whose lowest reading falls below the forecast's alpha-quantile of the
    minimum: their count and share (alpha when calibrated)."""
    below = {a: rows.pit < a for a in alphas}
    return pd.DataFrame({"windows": [int(b.sum()) for b in below.values()],
                         "share": [float(b.mean()) for b in below.values()]}, index=list(alphas))


def describe(patients: list[Patient]) -> dict:
    """The cohort as Section III.A.1 describes it: the records by sampling period, their lengths, the
    boluses (doses over at most ``BOLUS_SPAN``), the infusion pieces and the Cc samples."""
    periods = pd.Series([sampling(p) for p in patients])
    length = pd.Series([(p.map.times[-1] - p.map.times[0]) / 60 for p in patients])
    short = [p.doses.durations <= BOLUS_SPAN for p in patients]
    bolus = np.concatenate([p.doses.amounts[s] for p, s in zip(patients, short)])
    span = np.concatenate([p.doses.durations[s] for p, s in zip(patients, short)])
    rate = np.concatenate([p.doses.amounts[~s] / p.doses.durations[~s] for p, s in zip(patients, short)])
    n_cc = [0 if p.cc is None else len(p.cc.times) for p in patients]
    return {"patients": len(patients), "by sampling period, s": periods.value_counts().sort_index().to_dict(),
            "record, min: median [min, max]": (length.median(), length.min(), length.max()),
            "boluses": len(bolus), "patients by boluses": pd.Series([int(s.sum()) for s in short]).value_counts()
            .sort_index().to_dict(), "bolus, nmol: median": float(np.median(bolus)),
            "bolus duration, s: [min, max]": (float(span.min()), float(span.max())),
            "infusion pieces": len(rate), "infusion rate, nmol/s: median, max": (float(np.median(rate)), float(rate.max())),
            "Cc samples per patient: [min, max]": (min(n_cc), max(n_cc))}
