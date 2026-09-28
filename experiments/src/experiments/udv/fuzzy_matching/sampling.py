"""Seeded draws, the hearing bootstrap of a ratio and score histograms."""

import zlib
from collections import Counter
from typing import Any

import numpy as np

from experiments.common.reporting import rounded
from experiments.udv.fuzzy_matching.config import MAX_SCORE, FuzzyConfig

Record = dict[str, Any]
HearingCounts = dict[int, float]


def draw_rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def increment(counts: HearingCounts, hearing_id: int) -> None:
    counts[hearing_id] = counts.get(hearing_id, 0.0) + 1


def bootstrap_ratio(
    numerators: HearingCounts,
    denominators: HearingCounts,
    hearing_ids: list[int],
    config: FuzzyConfig,
    stream: str,
) -> Record:
    """Pooled ratio of per-hearing counts with a percentile interval over hearing resamples."""
    total = sum(denominators.values())
    if not hearing_ids or total == 0:
        return {"value": None, "low": None, "high": None}
    top = np.array([numerators.get(hearing_id, 0.0) for hearing_id in hearing_ids])
    bottom = np.array([denominators.get(hearing_id, 0.0) for hearing_id in hearing_ids])
    rng = draw_rng(config.seed, zlib.crc32(stream.encode()))
    picks = rng.integers(0, len(hearing_ids), size=(config.bootstrap_samples, len(hearing_ids)))
    sampled_top, sampled_bottom = top[picks].sum(axis=1), bottom[picks].sum(axis=1)
    valid = sampled_bottom > 0
    ratios = sampled_top[valid] / sampled_bottom[valid]
    tail = (1 - config.confidence_level) / 2
    return {
        "value": rounded(top.sum() / total),
        "low": rounded(float(np.quantile(ratios, tail))) if ratios.size else None,
        "high": rounded(float(np.quantile(ratios, 1 - tail))) if ratios.size else None,
        "replicates_with_denominator": int(valid.sum()),
    }


def score_histogram(values: list[float], width: int) -> dict[str, int]:
    bins = Counter(
        str(MAX_SCORE) if value >= MAX_SCORE else str(int(value // width) * width)
        for value in values
    )
    return dict(sorted(bins.items(), key=lambda item: float(item[0])))


def sorted_counts(values: list[Any]) -> dict[Any, int]:
    return dict(sorted(Counter(values).items()))
