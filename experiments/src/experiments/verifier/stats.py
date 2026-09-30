"""Hearing-level bootstrap and multiple-comparison helpers shared by the verifier experiments."""

from typing import Any

import numpy as np

from experiments.common.stats import holm
from experiments.udv.calibrate_threshold import interval, resample_rows, unit_groups

Record = dict[str, Any]

MISSING_P_VALUE = 1.0
MIN_HEARINGS_FOR_P_VALUE = 2


def hearing_draws(
    hearing_ids: np.ndarray, samples: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Row indices of `samples` bootstrap replicates that resample whole hearings."""
    groups = unit_groups(hearing_ids.tolist(), "hearing")
    return [resample_rows(rng, groups) for _ in range(samples)]


def finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def finite_interval(values: np.ndarray, level: float) -> Record:
    """Percentile interval over the replicates where the statistic was defined."""
    return interval(finite_values(values).tolist(), level)


def bootstrap_p_value(deltas: np.ndarray) -> float | None:
    """Two-sided bootstrap p-value of a paired difference against zero."""
    valid = finite_values(deltas)
    if len(valid) == 0:
        return None
    below = int(np.sum(valid <= 0))
    above = int(np.sum(valid >= 0))
    return float(min(1.0, 2 * (min(below, above) + 1) / (len(valid) + 1)))


def apply_holm(entries: list[Record]) -> None:
    """Add Holm-adjusted p-values in place, counting an undefined p-value as one."""
    p_values = [
        MISSING_P_VALUE if entry.get("p_value") is None else float(entry["p_value"])
        for entry in entries
    ]
    for entry, adjusted in zip(entries, holm(p_values), strict=True):
        entry["p_holm"] = adjusted
        entry["p_holm_counts_missing_as_one"] = entry.get("p_value") is None
