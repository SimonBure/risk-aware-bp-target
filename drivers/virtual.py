"""Section III.B: the virtual cohort, Table II and the abstract's numbers.

The paper's 4,096 virtual patients (Appendix E), the collection of Section III.B, one fit at 10 min,
then per patient: y* and the oracle at alpha = 1, 2, 5, 10 %, the fit's and the population's doses at
70 mmHg, each applied to the true patient (``bptarget.target.decide_all``); and the population's dose
at every target from 65 mmHg, the same for everyone (``levels``), from which the single targets in
hindsight are read: the lowest whose rate over the cohort is at most alpha.

Run ``python drivers/virtual.py [--n-patients 4096] [--n-jobs 8]`` (about 20 CPU hours for the 4,096).
Writes to ``outputs/virtual/``: ``draw.csv`` (every proposal, its reason for rejection, empty when
kept), ``rows/`` and ``levels/`` (one file per 256 patients, so an interrupted run resumes),
``summary.txt`` (what it prints).
"""
import common  # noqa: F401, first: one thread per worker

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from bptarget.dose import HORIZON
from bptarget.fit import N_BOX
from bptarget.forecast import ALPHAS, HYPO
from bptarget.stats import interval, verdict
from bptarget.target import FIXED, ORACLE, POPULATION, YSTAR, decide_all, levels
from bptarget.virtual import collect, draw

DECISION = 600.0  # s
CHUNK = 256


def patient(i: int, params: dict, doses: dict) -> tuple[list[dict], list[dict]]:
    found = [f for f in collect(i, params, N_BOX, until=DECISION) if abs(f[0] - DECISION) < 1e-6]
    return decide_all(i, params, found, doses=doses)


def run(patients: list[dict], out: Path, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    doses = levels([DECISION])
    for d in ("rows", "levels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for a in range(0, len(patients), CHUNK):
        b = min(a + CHUNK, len(patients))
        name = f"{a:04d}-{b:04d}.csv"
        if (out / "levels" / name).exists():
            continue
        got = common.parallel(patient, [(i, patients[i], doses) for i in range(a, b)], n_jobs)
        pd.DataFrame([r for rows, _ in got for r in rows]).to_csv(out / "rows" / name, index=False)
        pd.DataFrame([r for _, rows in got for r in rows]).to_csv(out / "levels" / name, index=False)
        print(f"patients {a}:{b} done", flush=True)
    read = lambda d: pd.concat([pd.read_csv(p) for p in sorted((out / d).glob("*.csv"))], ignore_index=True)
    rows, lv = read("rows"), read("levels")
    return rows[rows.patient < len(patients)], lv[lv.patient < len(patients)]


def nmol(g: pd.DataFrame) -> pd.Series:
    """The NE over the dose's 10 min: the bolus and the rate over 600 s."""
    return g.bolus + HORIZON * g.rate


def line(name: str, alpha, g: pd.DataFrame, n: int) -> dict:
    """Table II's columns for the patients ``g`` of a rule given a dose, out of n."""
    lo = interval(g.event, g.patient)
    return dict(rule=name, alpha="--" if pd.isna(alpha) else f"{100 * alpha:g} %", selected=100 * len(g) / n,
                n=len(g), reading=100 * lo[0], lo=100 * lo[1], hi=100 * lo[2],
                verdict="" if pd.isna(alpha) else verdict(lo, alpha), true=100 * g.true_any.mean(),
                ne=nmol(g).median(), margin=(g.target - HYPO).median())


def table(rows: pd.DataFrame, lv: pd.DataFrame) -> pd.DataFrame:
    n = rows.patient.nunique()
    fed = rows[rows.feasible.astype(bool)]
    out = []
    for rule, name in ((YSTAR, "y*"), (ORACLE, "oracle")):
        for a in ALPHAS:
            out.append(line(name, a, fed[(fed.rule == rule) & np.isclose(fed.alpha, a)], n))
    rate = lv.groupby("target").event.mean()
    for a in ALPHAS:
        held = rate.index[rate <= a]
        if len(held):
            out.append(line(f"single, {held.min():.1f} mmHg", a, lv[np.isclose(lv.target, held.min())], n))
        else:
            out.append(dict(rule=f"single: none up to {rate.index.max():.1f} mmHg", alpha=f"{100 * a:g} %"))
    fixed = fed[fed.rule == FIXED]
    out.append(line("fit at 70", np.nan, fixed, n))
    chosen = fed[(fed.rule == YSTAR) & np.isclose(fed.alpha, 0.05)].patient
    out.append(line("fit at 70, y*'s at 5 %", np.nan, fixed[fixed.patient.isin(chosen)], n))
    out.append(line("population at 70", np.nan, fed[fed.rule == POPULATION], n))
    return pd.DataFrame(out)


def smallest_n(power: float, p0: float = 0.02, p1: float = 0.01, size: float = 0.05) -> int:
    """The fewest doses for which the one-sided exact binomial test of a rate p0 against p1 < p0, at
    ``size``, has at least ``power`` (the exact test's power is not monotone in n: a few larger n fall
    just short)."""
    for n in range(10, 100_000):
        k = np.arange(n + 1)
        reject = k[binom.cdf(k, n, p0) <= size]
        if len(reject) and binom.cdf(reject[-1], n, p1) >= power:
            return n
    raise ValueError


def summary(rows: pd.DataFrame, lv: pd.DataFrame, log: pd.DataFrame) -> str:
    t = table(rows, lv)
    fixed = rows[rows.rule == FIXED]
    below = fixed.below_collection
    kept = log.reason.isna()
    lines = [f"Draw: {int(kept.sum()):,} of {len(log):,} proposals kept; rejected for "
             + ", ".join(f"{k} {v:,}" for k, v in log.reason.value_counts().items()),
             f"Size: the exact one-sided test of 2 % against 1 % at 5 % needs {smallest_n(0.80):,} doses for 80 % "
             f"power, {smallest_n(0.95):,} for 95 %",
             "",
             "Table II (selected: % of the patients given a dose; reading: % with a reading of W_t below 65 mmHg "
             "[95 %, patients resampled]; true: % whose true MAP goes below 65 mmHg; NE: median over the 10 min, "
             "nmol; margin: median y - 65, mmHg)",
             t.round(2).to_string(index=False),
             "",
             f"During the collection, time below 65 mmHg: median {below.median():.0f} s [IQR {below.quantile(0.25):.0f}, "
             f"{below.quantile(0.75):.0f}], {100 * (below > 0).mean():.1f} % of the patients some time"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-patients", type=int, default=4096)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=common.OUT / "virtual")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    patients, log = draw(args.n_patients, [DECISION])
    log = pd.DataFrame(log, columns=["proposal", "reason"])
    log.to_csv(args.out / "draw.csv", index=False)
    rows, lv = run(patients, args.out, args.n_jobs)
    text = summary(rows, lv, log)
    (args.out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
