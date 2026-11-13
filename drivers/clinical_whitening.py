"""The discussion's whitening: the clinical windows of Table I with P_t from white residuals (the paper's)
and from sensitivities whitened by an OU model of the fit's residuals (``bptarget.clinical.whitened_posterior``),
the fit and the noise level unchanged. Needs the clinical records in ``data/``.

Run ``python drivers/clinical_whitening.py [--n-jobs 8]`` (minutes). Writes ``outputs/clinical/``:
``whitened.csv`` and ``whitening.txt``.
"""
import common  # noqa: F401, first: one thread per worker

import argparse

import pandas as pd

from bptarget.clinical import calibration, score_table, windows
from bptarget.fit import N_BOX
from bptarget.forecast import N_DRAWS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    out = common.OUT / "clinical"
    out.mkdir(parents=True, exist_ok=True)
    patients = common.records()
    both = {}
    for name, whiten in (("white", False), ("whitened", True)):
        got = common.parallel(windows, [(p, N_BOX, N_DRAWS, whiten) for p in patients], args.n_jobs)
        both[name] = pd.DataFrame([r for rows in got for r in rows])
    both["whitened"].to_csv(out / "whitened.csv", index=False)
    w = both["whitened"]
    with pd.option_context("display.width", 200):
        text = "\n".join(
            [f"OU correlation time of the fits' residuals, s: median {w.tau.median():.1f} [IQR {w.tau.quantile(0.25):.1f}, "
             f"{w.tau.quantile(0.75):.1f}], {int((w.tau == 0).sum())} windows uncorrelated", ""]
            + [line for name, rows in both.items() for line in (
                f"== P_t from {name} residuals",
                "e^2/V by decision time:", score_table(rows).round(2).to_string(),
                "below the alpha-quantile of the minimum:", calibration(rows).round(3).to_string(), "")])
    (out / "whitening.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
