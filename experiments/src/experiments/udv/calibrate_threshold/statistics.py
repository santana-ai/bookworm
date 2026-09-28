"""Summaries and resampling shared by the calibration rules."""

from typing import Any

import numpy as np

from experiments.common.reporting import rounded

Record = dict[str, Any]


def describe(values: np.ndarray) -> Record:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": rounded(values.min()),
        "q25": rounded(np.quantile(values, 0.25)),
        "median": rounded(np.median(values)),
        "q75": rounded(np.quantile(values, 0.75)),
        "max": rounded(values.max()),
        "mean": rounded(values.mean()),
    }


def interval(values: list[float], confidence_level: float) -> Record:
    """Percentile interval, median and spread of bootstrap replicates."""
    if not values:
        return {"replicates": 0}
    array = np.array(values, dtype=np.float64)
    tail = (1 - confidence_level) / 2
    return {
        "replicates": len(values),
        "low": rounded(np.quantile(array, tail)),
        "high": rounded(np.quantile(array, 1 - tail)),
        "median": rounded(np.median(array)),
        "std": rounded(array.std(ddof=1)) if len(values) > 1 else None,
    }


def unit_groups(hearing_ids: list[int], unit: str) -> list[np.ndarray]:
    """Row indices grouped by bootstrap unit: one group per query or per hearing."""
    if unit == "query":
        return [np.array([row], dtype=np.int64) for row in range(len(hearing_ids))]
    groups: dict[int, list[int]] = {}
    for row, hearing_id in enumerate(hearing_ids):
        groups.setdefault(hearing_id, []).append(row)
    return [np.array(rows, dtype=np.int64) for _, rows in sorted(groups.items())]


def resample_rows(rng: np.random.Generator, groups: list[np.ndarray]) -> np.ndarray:
    picks = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[pick] for pick in picks])


def rule_rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])
