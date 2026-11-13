"""Fig. 1 and Section II.A's RMSEs: the three models' whole-record fits (``bptarget.wholefit``) on the
clinical cohort, and the figure of one patient. Needs the clinical records in ``data/``.

By default (``STARTS``) the reduced model is fitted from the population, the complete models from as many
starts as the multistart needs; ``--starts N`` caps every model's starts (1: the population only). Run
``python drivers/clinical_fits.py [--models reduced complete windkessel] [--starts N] [--patient 4]
[--n-jobs 8]`` (the reduced model in a minute; the complete models from many starts, hours). Writes
``outputs/clinical/fits-{model}-{starts}.pkl`` (the fits), ``fits.csv`` (one row per patient and
model), ``fits.txt`` and ``fits-{patient}.{pdf,png}``.
"""
import common  # noqa: F401, first: one thread per worker

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_config

from bptarget.wholefit import MODELS, metrics, multistart, outputs, rmse

NAMES = {"complete": "Complete", "windkessel": "Complete, second order", "reduced": "Reduced"}
COLOR = {"complete": "#0072B2", "windkessel": "#E69F00", "reduced": "#009E73"}  # Okabe-Ito
DASH = {"complete": (0, (4, 1.5)), "windkessel": (0, (1.2, 1.2)), "reduced": "-"}
N_PLOT = 1000
STARTS = {"reduced": 1, "complete": None, "windkessel": None}  # at most this many starts (None: no cap)


def fit(patient, model: str, starts: int | None, n_jobs: int):
    return multistart(patient, model, n_jobs, max_starts=starts)


def cohort(patients, model: str, starts: int | None, n_jobs: int) -> list:
    """Each record's fit and its multistart record: one record per worker from the population, else the
    starts in parallel."""
    def compute():
        if starts == 1:
            return common.parallel(fit, [(p, model, 1, 1) for p in patients], n_jobs)
        with parallel_config(backend="multiprocessing"):
            return [fit(p, model, starts, n_jobs) for p in patients]
    tag = {1: "population", None: "multistart"}.get(starts, f"{starts}-starts")
    return common.cached(common.OUT / "clinical" / f"fits-{model}-{tag}.pkl", compute)


def fitted(patient, model: str, f, times: np.ndarray):
    """The fit's MAP and Cc at ``times`` (on or after its first node)."""
    t0 = f.t[0]
    first = bool(times[0] > t0 + 1e-9)
    y, cc = outputs(patient, model, f.params, f.x[:, 0], np.r_[t0, times] if first else times)
    return y[first:], None if cc is None else cc[first:]


def figure(patient, fits: dict, path) -> None:
    common.style()
    bp, cc = patient.map, patient.cc
    end = max(bp.times[-1], cc.times[-1])
    fig, (ax_y, ax_u) = plt.subplots(2, 1, sharex=True, figsize=(common.COLUMN, 2.5), height_ratios=(3, 1))
    ax_y.plot(bp.times / 60, bp.values, ".", ms=2, color=common.GRAY, label="data")
    grid = np.linspace(bp.times[0], end, N_PLOT)  # no MAP before its first reading
    for model, f in fits.items():
        e = rmse(fitted(patient, model, f, bp.times)[0], bp.values)
        ax_y.plot(grid / 60, fitted(patient, model, f, grid)[0], color=COLOR[model], ls=DASH[model],
                  label=f"{NAMES[model]} ({e:.1f})")
    t = np.arange(0.0, end + 1.0)
    ax_u.step(t / 60, [patient.doses(s) for s in t], where="post", color=common.GRAY)
    ax_y.set_ylabel("MAP (mmHg)")
    ax_u.set_ylabel("NE (nmol/s)")
    ax_u.set_xlabel("time (min)")
    ax_y.legend(fontsize=6, loc="best")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--starts", type=int, default=None, help="at most this many starts (1: the population)")
    parser.add_argument("--patient", type=int, default=4, help="the patient drawn")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    out = common.OUT / "clinical"
    patients = common.records()
    rows, shown = [], {}
    for model in args.models:
        got = cohort(patients, model, STARTS[model] if args.starts is None else args.starts, args.n_jobs)
        rows += [dict(model=model, **metrics(p, model, f), **record) for p, (f, record) in zip(patients, got)]
        shown[model] = next(f for p, (f, _) in zip(patients, got) if p.id == args.patient)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fits.csv", index=False)
    cols = [c for c in ("rmse_map", "rmse_cc") if c in df]
    def sd(s):  # the population SD, as the paper
        return s.std(ddof=0)
    summary = df.groupby("model")[cols].agg(["mean", sd, "median", "min", "max"])
    summary["converged"] = df.groupby("model").converged.sum().astype(int).astype(str) + "/" + \
        df.groupby("model").size().astype(str)
    own = df[df.id == args.patient].set_index("model")[cols]
    with pd.option_context("display.width", 200):
        text = "\n".join(["RMSE over the cohort (MAP mmHg, Cc nmol/L):", summary.round(2).to_string(), "",
                          f"Patient {args.patient}:", own.round(2).to_string()])
    (out / "fits.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    figure(next(p for p in patients if p.id == args.patient), shown, out / f"fits-{args.patient}")


if __name__ == "__main__":
    main()
