# A risk-aware individualized blood pressure target to avoid hypotension in surgery

Code of the paper by Simon Buré, Jona Joachim and Florent Di Meglio (submitted to ECC 2027).

At a decision time, a reduced PK/PD model of norepinephrine (NE) is fitted to the mean arterial
pressure (MAP) readings, and its linearised posterior gives the law of the next readings. The
personalised target y* is the lowest MAP whose dose (a bolus and a constant infusion) keeps the
predicted probability of a reading below 65 mmHg under a level alpha; when no target does, the
decision returns to the clinician.

## Install

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

or with pip, in a virtual environment:

```bash
pip install -e .
```

The dependencies are pinned to the versions the paper's numbers were computed with.

## Layout

- `bptarget/`: the method and its evaluation.
  - `model.py`: the NE PK/PD models (complete, second order, reduced) and their simulation.
  - `wholefit.py`: fits of whole records by IPOPT (Section II.A, Fig. 1).
  - `selection.py`: which parameters of the reduced model to estimate (Section II.A).
  - `fit.py`: the online fit and its linearised posterior (Section II.B).
  - `forecast.py`: the law of the next readings, the risk of hypotension and the score.
  - `dose.py`: the bolus and infusion that reach a target (Section II.C).
  - `clinical.py`: the clinical records and their forecast windows (Section III.A).
  - `virtual.py`: the virtual cohort, its sensor and its collection (Section III.B, Appendix E).
  - `target.py`: y*, the oracle and the fixed doses (Section III.B).
  - `collection.py`: the search over collections (discussion).
  - `stats.py`: the intervals and verdicts of the tables.
- `drivers/`: one script per result of the paper, run as `python drivers/<name>.py`. Each prints its
  numbers and writes them, with its intermediate results, under `outputs/`. The long ones save their
  progress and resume when run again. `--n-jobs` sets the number of worker processes.
- `data/`: where the clinical records go (below).

## The paper's results

| Result | Driver | Time |
|---|---|---|
| Abstract and Table II: y*, the oracle, the single targets, the fixed doses; the draw (Appendix E); the time below 65 mmHg during the collection | `virtual.py` | about 20 CPU hours |
| Fig. 2: one decision on patients 5 and 109 | `decision.py` | seconds |
| Discussion, the noise level: sigma_t floored at 2 mmHg, 20 min collection | `noise_floor.py` | about 30 CPU hours |
| Discussion, the sensor's correlated noise: 5 % becomes 4.87 % | `sensor_noise.py` | seconds |
| Discussion, collection informativeness: the search over collections | `collection.py` | about 60 CPU hours |
| Section III.A: the cohort, Table I, the minimum's calibration; discussion, the score's two sides | `clinical_windows.py` | minutes |
| Fig. 1 and the RMSEs of the three models | `clinical_fits.py` | hours (the complete models from many starts) |
| Section II.A: model selection (Wu, McLean, Chu) | `clinical_selection.py` | hours |
| Discussion: whitening the sensitivities by the residuals' correlation | `clinical_whitening.py` | minutes |
| Appendix E and discussion: the residuals' correlation | `clinical_correlation.py` | an hour (shares its fits with `clinical_selection.py`) |

The virtual cohort is drawn from fixed seeds, so the drivers without clinical data reproduce the
paper's numbers exactly. A run on fewer patients, for instance `python drivers/virtual.py
--n-patients 64`, gives the first patients of the same cohort.

## Clinical data

The clinical records of Joachim et al. (2024) are identifying health data and cannot be shared
openly; they are available from the corresponding author upon reasonable request, subject to
approval by AP-HP. The drivers `clinical_*.py` need them in `data/`, or in the folder named by the
environment variable `BPTARGET_DATA` (`data/README.md`); without them they stop with that message.

## Licence

MIT, see `LICENSE`.
