"""What the drivers share: one thread per worker, the output folder, the worker pool and the figures' style.

Import it first in a driver (before numpy), so that each worker computes on one thread.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # figures to files only

from pathlib import Path

from joblib import Parallel, delayed, parallel_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = Path(os.environ.get("BPTARGET_DATA", ROOT / "data"))  # the clinical records (data/README.md)

# Figures: one IEEE column is 3.45 in wide, the page 7.16 in.
COLUMN, PAGE = 3.45, 7.16
BLUE, RED, GRAY, INK, MUTED, PALE = "#2a78d6", "#e34948", "#8a8985", "#0b0b0b", "#52514e", "#f2f1ed"


def parallel(fn, args: list[tuple], n_jobs: int) -> list:
    """fn(*a) for each a of ``args``, on ``n_jobs`` processes (``multiprocessing``: joblib's default pool
    fails on some Windows set-ups), in order."""
    with parallel_config(backend="multiprocessing", n_jobs=n_jobs):
        return Parallel()(delayed(fn)(*a) for a in args)


def records() -> list:
    """The clinical records, or exit with where to get them."""
    import sys
    from bptarget.clinical import load
    try:
        return load(DATA)
    except FileNotFoundError as e:
        sys.exit(str(e))


def cached(path: Path, compute):
    """compute()'s value, pickled to ``path`` the first time and read back after."""
    import pickle
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return value


def references(patients: list, n_jobs: int) -> list:
    """The reduced model's all-free whole-record fit of each record, from many starts
    (``bptarget.selection.reference``), cached in ``outputs/clinical/reference.pkl``."""
    from bptarget.selection import reference

    def compute():
        with parallel_config(backend="multiprocessing"):
            return [reference(p, n_jobs) for p in patients]
    return cached(OUT / "clinical" / "reference.pkl", compute)


def style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                         "legend.frameon": False, "lines.linewidth": 1.2, "svg.fonttype": "path"})
