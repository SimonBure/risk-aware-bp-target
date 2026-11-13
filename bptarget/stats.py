"""95 % intervals with the patients resampled, and the verdict on a rate against its level."""
import numpy as np
import pandas as pd

N_BOOT, BOOT_SEED = 2000, 0  # resamples of the patients, and their seed


def interval(values: pd.Series, patient: pd.Series, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """The mean of ``values`` and its 95 % interval, the patients resampled (each resample's mean, a ratio
    of sums)."""
    frame = pd.DataFrame(dict(v=values.to_numpy(float), p=patient.to_numpy())).dropna()
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    sums = frame.groupby("p").v.agg(["sum", "count"]).to_numpy()
    pick = np.random.default_rng(seed).integers(0, len(sums), (N_BOOT, len(sums)))
    boot = sums[pick, 0].sum(axis=1) / sums[pick, 1].sum(axis=1)
    return float(frame.v.mean()), *map(float, np.quantile(boot, [0.025, 0.975]))


def median_interval(values: pd.Series, patient: pd.Series, seed: int = BOOT_SEED) -> tuple[float, float]:
    """The 95 % interval of the median of ``values``, the patients resampled."""
    groups = [g.to_numpy(float) for _, g in values.groupby(patient.to_numpy())]
    rng = np.random.default_rng(seed)
    boot = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
            for _ in range(N_BOOT)]
    return tuple(map(float, np.quantile(boot, [0.025, 0.975])))


def verdict(rate: tuple[float, float, float], alpha: float) -> str:
    """``holds`` when the rate is at most alpha, ``fails`` when its interval is above alpha, else ``unclear``."""
    if rate[0] <= alpha:
        return "holds"
    return "fails" if rate[1] > alpha else "unclear"
