"""The search over collections (discussion of Section III.B): which NE doses every virtual patient
receives during the 10 min before the first decision.

A collection is an initial rate and up to four events on a 10 s grid in [0, 10 min), each a new rate
or a bolus over 10 s on top of the running rate (``virtual.use``'s format). It is scored on y* at
alpha = 5 % at 10 min (``evaluate``): the patients given a dose minus (1 - alpha) / alpha = 19 per dose
followed by a reading of W_t below 65 mmHg, so that a rate of events of exactly alpha scores zero.

``search``, in stages: (1) the paper's collection, Joachim et al.'s (35 nmol over 10 s, then
0.18 nmol/s) and 148 random ones on 64 training patients; (2) 40 moves from the best, each kept when
it scores higher; (3) the 10 best of (1) and (2) on the 128 training patients; then the winner, the
paper's and Joachim et al.'s once on 128 confirmation patients. The training cohort follows the
paper's rules from another seed, with another noise series; the confirmation cohort is the paper's
first 128 patients.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bptarget.fit import N_BOX, PatientModel
from bptarget.forecast import ALPHAS
from bptarget.target import applied, readings_at, setup, targets
from bptarget.virtual import CURRENT, SEED, collect, draw, record, use

DECISION = 600.0                  # s
GRID, T_END = 10.0, DECISION      # s: the events' grid, and the end of the collection
R_MAX, B_MIN, B_MAX = 0.5, 5.0, 75.0
ALPHA = 0.05
COST = (1 - ALPHA) / ALPHA        # a dose followed by a reading below 65 mmHg costs 19 safe ones
JOACHIM = {"r0": 0.18, "events": [[0.0, "bolus", 35.0]]}
N_RANDOM, N_MOVES, N_TOP = 148, 40, 10
SEEDS = {"random": 77, "moves": 78}


# --------------------------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------------------------- #
def random_event(rng) -> list:
    """A bolus (one in three) anywhere on the grid, or a new rate from 10 s."""
    if rng.random() < 1 / 3:
        return [float(GRID * rng.integers(0, T_END / GRID)), "bolus", float(np.round(rng.uniform(B_MIN, B_MAX)))]
    return [float(GRID * rng.integers(1, T_END / GRID)), "rate", float(np.round(rng.uniform(0, R_MAX), 2))]


def random_spec(rng) -> dict:
    return dict(r0=float(np.round(rng.uniform(0, R_MAX), 2)),
                events=sorted(random_event(rng) for _ in range(rng.integers(0, 5))))


def perturb(spec: dict, rng) -> dict:
    """One element changed: the initial rate +-0.1, an event's time +-1 to 3 min, a rate +-0.1, a bolus
    +-15 nmol, an event added or removed."""
    s = json.loads(json.dumps(spec))
    moves = ["rate0"] + (["time", "value", "remove"] if s["events"] else []) + (["add"] if len(s["events"]) < 4 else [])
    move = moves[rng.integers(len(moves))]
    sign = rng.choice([-1.0, 1.0])
    if move == "rate0":
        s["r0"] = float(np.clip(np.round(s["r0"] + 0.1 * sign, 2), 0, R_MAX))
    elif move == "add":
        s["events"].append(random_event(rng))
    elif move == "remove":
        s["events"].pop(rng.integers(len(s["events"])))
    else:
        e = s["events"][rng.integers(len(s["events"]))]
        if move == "time":
            lo = 0.0 if e[1] == "bolus" else GRID
            e[0] = float(np.clip(GRID * np.round((e[0] + sign * 60 * rng.uniform(1, 3)) / GRID), lo, T_END - GRID))
        elif e[1] == "rate":
            e[2] = float(np.clip(np.round(e[2] + 0.1 * sign, 2), 0, R_MAX))
        else:
            e[2] = float(np.clip(e[2] + 15 * sign, B_MIN, B_MAX))
    s["events"] = sorted(s["events"])
    return s


# --------------------------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------------------------- #
def patient_row(spec: dict, i: int, params: dict, rep: int) -> dict:
    """Virtual patient i (noise repetition ``rep``) under the collection: y* at ALPHA at 10 min (every
    level of ALPHAS bisected, as in Table II), and its dose's outcome."""
    use(spec)
    try:
        fit = next(f for f in collect(i, params, N_BOX, until=DECISION, rep=rep) if abs(f[0] - DECISION) < 1e-6)
        at, top, dr = setup(i, params, fit, PatientModel(record(i, params, rep=rep)))
        y = dict(targets(at, top, dr, ALPHAS))[ALPHA]
        row = dict(patient=i, dosed=y is not None, target=np.nan if y is None else y, event=np.nan, error="")
        if y is not None:
            ctrl, _ = at(y)
            out = applied(params, DECISION, ctrl.bolus, ctrl.rate, *readings_at(i, DECISION, rep))
            row.update(event=out["event"], below_collection=out["below_collection"])
        return row
    except Exception as e:  # counted as no dose, reported
        return dict(patient=i, dosed=False, target=np.nan, event=np.nan, error=repr(e))


def evaluate(spec: dict, patients: list[dict], rep: int, n_jobs: int) -> tuple[dict, pd.DataFrame]:
    """The collection's score on ``patients``, and its rows."""
    rows = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(patient_row)(spec, i, p, rep) for i, p in enumerate(patients)))
    dosed = int(rows.dosed.sum())
    events = int(rows.event[rows.dosed].sum())
    return dict(n=len(rows), dosed=dosed, events=events, safe=dosed - events, score=dosed - COST * events,
                rate=events / dosed if dosed else np.nan, errors=int((rows.error != "").sum()),
                below_collection=float(rows.below_collection.median()) if "below_collection" in rows else np.nan), rows


def ranked(df: pd.DataFrame) -> pd.DataFrame:
    """Highest score, then more safe doses, then earlier."""
    return df.assign(order=range(len(df))).sort_values(["score", "safe", "order"], ascending=[False, False, True])


# --------------------------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------------------------- #
def _rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _log(path: Path, row: dict) -> None:
    pd.concat([_rows(path), pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    print(f"{path.stem} #{len(_rows(path)) - 1}: score {row['score']:g}, dosed {row['dosed']}, events {row['events']}"
          f" | {row['spec']}", flush=True)


def search(out: Path, n_jobs: int) -> None:
    """The stages of the module's doc, each row appended to {out}/stage{1,2,3}.csv as it is scored, so an
    interrupted search resumes; the confirmations in {out}/confirm-{winner,current,joachim}/."""
    out.mkdir(parents=True, exist_ok=True)
    train, _ = draw(128, [DECISION], seed=SEED + 1)
    confirm, _ = draw(128, [DECISION], seed=SEED)
    rng = np.random.default_rng(SEEDS["random"])
    specs = [CURRENT, JOACHIM] + [random_spec(rng) for _ in range(N_RANDOM)]

    s1 = out / "stage1.csv"
    for k, spec in enumerate(specs):
        if k >= len(_rows(s1)):
            summary, _ = evaluate(spec, train[:64], 1, n_jobs)
            _log(s1, dict(id=f"s1-{k}", spec=json.dumps(spec), **summary))

    s2 = out / "stage2.csv"
    top = ranked(_rows(s1)).iloc[0]
    current, best = json.loads(top.spec), (top.score, top.safe)
    rng = np.random.default_rng(SEEDS["moves"])
    for k in range(N_MOVES):  # replays the moves already scored, then goes on
        cand = perturb(current, rng)
        done = _rows(s2)
        if k < len(done):
            row = done.iloc[k]
            assert json.loads(row.spec) == cand, "stage 2's replay diverged"
        else:
            summary, _ = evaluate(cand, train[:64], 1, n_jobs)
            row = dict(id=f"s2-{k}", spec=json.dumps(cand), **summary)
            _log(s2, row)
            row = pd.Series(row)
        if (row.score, row.safe) > best:
            current, best = cand, (row.score, row.safe)

    s3 = out / "stage3.csv"
    pool = pd.concat([_rows(s1), _rows(s2)], ignore_index=True).drop_duplicates("spec")
    for k, (_, r) in enumerate(ranked(pool).head(N_TOP).iterrows()):
        if k >= len(_rows(s3)):
            summary, _ = evaluate(json.loads(r.spec), train, 1, n_jobs)
            _log(s3, dict(id=r.id, spec=r.spec, score64=r.score, **summary))
    winner = ranked(_rows(s3)).iloc[0]

    for name, spec in [("winner", json.loads(winner.spec)), ("current", CURRENT), ("joachim", JOACHIM)]:
        d = out / f"confirm-{name}"
        if (d / "summary.json").exists():
            continue
        d.mkdir(exist_ok=True)
        summary, rows = evaluate(spec, confirm, 0, n_jobs)
        rows.to_csv(d / "patients.csv", index=False)
        (d / "summary.json").write_text(json.dumps(dict(spec=spec, **summary), indent=1))
        print(f"confirm {name}: {summary}", flush=True)
    print(f"{len(pool)} distinct collections scored on the training cohort", flush=True)
