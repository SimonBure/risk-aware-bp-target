"""Section III.A: the clinical cohort, Table I, the minimum's calibration, and the score's two sides
(discussion). Needs the clinical records in ``data/`` (``data/README.md``).

Run ``python drivers/clinical_windows.py [--n-jobs 8]`` (minutes). Writes ``outputs/clinical/``:
``windows.csv`` (one row per forecast window) and ``summary.txt`` (what it prints).
"""
import common  # noqa: F401, first: one thread per worker

import argparse

import pandas as pd

from bptarget.clinical import calibration, describe, score_table, sides, windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    out = common.OUT / "clinical"
    out.mkdir(parents=True, exist_ok=True)
    patients = common.records()
    rows = pd.DataFrame([r for rows in common.parallel(windows, [(p,) for p in patients], args.n_jobs) for r in rows])
    rows.to_csv(out / "windows.csv", index=False)
    cal = calibration(rows)
    with pd.option_context("display.width", 200):
        text = "\n".join([
            "The cohort:", *[f"  {k}: {v}" for k, v in describe(patients).items()], "",
            f"{len(rows)} windows from {rows.patient.nunique()} patients, {int(rows.converged.sum())} fits converged", "",
            "Table I, e^2/V by decision time t (min after the first reading):", score_table(rows).round(2).to_string(), "",
            "The minimum's calibration: windows whose lowest reading falls below the forecast's alpha-quantile:",
            cal.round(3).to_string(), "",
            "The score's two sides, medians (mmHg^2; RMSE and sqrt(V) in mmHg):", sides(rows).round(2).to_string()])
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
