"""The clinical residuals' correlation (Appendix E, the virtual sensor's noise; the discussion): the
autocorrelation of the reduced model's whole-record fit residuals at time lags from 1 s to 5 min, per
sampling period, against the single OU process set by the first lag. Needs the clinical records in
``data/``.

Run ``python drivers/clinical_correlation.py [--n-jobs 8]`` (the reference fits, shared with
``clinical_selection.py``, take about an hour the first time). Writes ``outputs/clinical/acf.csv`` and
``acf.txt``.
"""
import common  # noqa: F401, first: one thread per worker

import argparse

import numpy as np
import pandas as pd

from bptarget.clinical import LAGS, acf, sampling

MAX_GAP = 0.10  # one OU is enough if |median - OU| is at most this at every lag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    out = common.OUT / "clinical"
    patients = common.records()
    refs = common.references(patients, args.n_jobs)
    rows = []
    for p, f in zip(patients, refs):
        dt = sampling(p)
        lags = [h for h in LAGS if h >= dt]
        rows += [dict(patient=p.id, sampling=dt, lag=h, rho=r, rms=float(np.sqrt(np.mean(f.residual ** 2))))
                 for h, r in zip(lags, acf(f.t, f.residual, lags))]
    df = pd.DataFrame(rows)
    df.to_csv(out / "acf.csv", index=False)
    lines = [f"Median RMS residual of the fits: {df.drop_duplicates('patient').rms.median():.2f} mmHg", ""]
    for dt, g in df.groupby("sampling"):
        q = g.groupby("lag").rho.quantile([0.25, 0.5, 0.75]).unstack()
        q.columns = ["q25", "median", "q75"]
        q["single_ou"] = max(q.loc[dt, "median"], 0.0) ** (q.index.to_numpy() / dt)
        q["gap"] = q["median"] - q["single_ou"]
        lines += [f"== records sampled every {dt} s: {g.patient.nunique()} patients", q.round(3).to_string(),
                  f"one OU is enough (|median - single OU| <= {MAX_GAP} at every lag): {bool((q.gap.abs() <= MAX_GAP).all())}",
                  ""]
    text = "\n".join(lines)
    (out / "acf.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
