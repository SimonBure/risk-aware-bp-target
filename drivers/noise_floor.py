"""The discussion's noise level: y* on the virtual cohort with a 20 min collection, sigma_t as estimated
(A) and never below 2 mmHg, the virtual sensor's SD, in the forecast's noise (B).

The paper's draw for a decision at 20 min, one fit at 20 min per patient, y* at alpha = 1, 2, 5, 10 %
under A and B from the same fit and draws, each dose applied to the true patient. The paper: 6.8 % at
5 % under A, 5.9 % under B, which holds 10 %.

Run ``python drivers/noise_floor.py [--n-patients 4096] [--n-jobs 8]`` (about 30 CPU hours). Writes ``outputs/noise_floor/`` (``rows/``, one file per 256 patients, and
``summary.txt``).
"""
import common  # noqa: F401, first: one thread per worker

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bptarget.fit import N_BOX, PatientModel
from bptarget.forecast import ALPHAS
from bptarget.stats import interval, verdict
from bptarget.target import applied, readings_at, setup, targets
from bptarget.virtual import SIGMA, collect, draw, record

DECISION = 1200.0  # s
ARMS = {"A": 0.0, "B": SIGMA ** 2}  # the floor of the forecast's noise variance, mmHg^2
CHUNK = 256


def patient(i: int, params: dict) -> list[dict]:
    fit = next(f for f in collect(i, params, N_BOX, until=DECISION) if abs(f[0] - DECISION) < 1e-6)
    model = PatientModel(record(i, params), N_BOX)
    shown_at, eps = readings_at(i, DECISION)
    rows = []
    for arm, floor in ARMS.items():
        at, top, dr = setup(i, params, fit, model, N_BOX, sigma2_floor=floor)
        for alpha, y in targets(at, top, dr, ALPHAS):
            row = dict(patient=i, arm=arm, alpha=alpha, sigma2_t=fit[3], feasible=y is not None,
                       target=np.nan if y is None else y)
            if y is not None:
                ctrl, _ = at(y)
                out = applied(params, DECISION, ctrl.bolus, ctrl.rate, shown_at, eps)
                row.update(bolus=ctrl.bolus, rate=ctrl.rate, event=out["event"], true_any=out["true_any"])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-patients", type=int, default=4096)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=common.OUT / "noise_floor")
    args = parser.parse_args()
    (args.out / "rows").mkdir(parents=True, exist_ok=True)
    patients, _ = draw(args.n_patients, [DECISION])
    for a in range(0, len(patients), CHUNK):
        b = min(a + CHUNK, len(patients))
        path = args.out / "rows" / f"{a:04d}-{b:04d}.csv"
        if not path.exists():
            got = common.parallel(patient, [(i, patients[i]) for i in range(a, b)], args.n_jobs)
            pd.DataFrame([r for rows in got for r in rows]).to_csv(path, index=False)
            print(f"patients {a}:{b} done", flush=True)
    rows = pd.concat([pd.read_csv(p) for p in sorted((args.out / "rows").glob("*.csv"))], ignore_index=True)
    rows = rows[rows.patient < len(patients)]
    n = rows.patient.nunique()
    lines = [f"y* at 20 min on {n:,} patients; A: sigma_t as estimated, B: at least {SIGMA:g} mmHg in the forecast",
             f"{'arm':<4}{'alpha':>6}{'selected %':>12}{'reading < 65 % [95 %]':>26}{'verdict':>9}{'NE':>8}"]
    for (arm, alpha), g in rows.groupby(["arm", "alpha"]):
        fed = g[g.feasible.astype(bool)]
        r = interval(fed.event, fed.patient)
        lines.append(f"{arm:<4}{100 * alpha:>5g}%{100 * len(fed) / n:>12.1f}{100 * r[0]:>11.2f} "
                     f"[{100 * r[1]:.2f}, {100 * r[2]:.2f}]{verdict(r, alpha):>9}"
                     f"{(fed.bolus + 600 * fed.rate).median():>8.1f}")
    text = "\n".join(lines)
    (args.out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
