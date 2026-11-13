"""Section II.A's model selection: which of the reduced model's seven parameters to estimate, by the criteria
of Wu et al. (2011), McLean et al. (2012) and Chu et al. (2009) (``bptarget.selection``). Needs the
clinical records in ``data/``.

Run ``python drivers/clinical_selection.py [--n-jobs 8]`` (a few hours: the reference fits from many
starts, then a fit of every nested subset per record). Writes ``outputs/clinical/selection.txt``, the
subset fits in ``selection.pkl``.
"""
import common  # noqa: F401, first: one thread per worker

import argparse

import numpy as np
from joblib import parallel_config

from bptarget.selection import NAMES, chu, mclean, residual_variance, wu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    out = common.OUT / "clinical"
    patients = common.records()
    refs = common.references(patients, args.n_jobs)
    sigma2 = float(np.median([residual_variance(f.t, f.residual) for f in refs]))

    def compute():
        fits = {tuple(NAMES): refs}
        with parallel_config(backend="multiprocessing"):
            w = wu(patients, fits, args.n_jobs)  # first: McLean reuses its fits
            m = mclean(patients, fits, args.n_jobs)
        return w, m, fits
    w, m, fits = common.cached(out / "selection.pkl", compute)
    ranking, bias, k = chu(patients, sigma2)
    n = len(NAMES)
    lines = [f"Reference fits: {len(refs)} records, residual variance per unit time median {sigma2:.3f} mmHg^2", "",
             "Wu et al. (2011): ranking " + ", ".join(f"{p} ({v:.3g})" for p, v in w["ranking"]),
             "  pooled r_CC per k: " + " ".join(f"{v:.3f}" for v in w["pooled"]),
             f"  parameters kept: pooled {w['k_pooled']}, by vote {w['k_vote']}, worst case {w['k_worst_case']} of {n}",
             "McLean et al. (2012): ranking " + ", ".join(m["ranking"]),
             "  pooled r_CC per step: " + " ".join(f"{r:.3f}" for _, _, r, _ in m["path"]),
             f"  parameters kept: {m['k']} of {n}",
             "Chu et al. (2009): ranking " + ", ".join(ranking),
             "  mean bias after each step: " + " ".join(f"{b:.3g}" for b in bias),
             f"  parameters kept: {k} of {n}, fixed: {', '.join(p for p in NAMES if p not in ranking[:k])}"]
    text = "\n".join(lines)
    (out / "selection.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
