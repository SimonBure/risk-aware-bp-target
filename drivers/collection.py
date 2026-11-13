"""The discussion's collection informativeness: the search over collections (``bptarget.collection``).

Run ``python drivers/collection.py [--n-jobs 8]`` (about 200 collections on 64 to 128 patients, each a
fit at 10 min and y*; it resumes from its files). Writes ``outputs/collection/``: ``stage1.csv`` to
``stage3.csv`` (one row per collection scored), ``confirm-{winner,current,joachim}/``.
"""
import common  # noqa: F401, first: one thread per worker

import argparse
from pathlib import Path

from joblib import parallel_config

from bptarget.collection import search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=common.OUT / "collection")
    args = parser.parse_args()
    with parallel_config(backend="multiprocessing"):
        search(args.out, args.n_jobs)
