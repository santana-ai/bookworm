"""Scores, identification ranks and hearing-bootstrap intervals of the profile validation."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from bookworm.data.io import JsonObject
from bookworm.features.encoders import FloatMatrix
from bookworm.profiles.config import ProfileValidationConfig
from bookworm.profiles.schemas import ProfilePair

ROUND_DECIMALS = 4
BOOTSTRAP_INTERVAL = "percentile interval over replicates that resample whole hearings"
SCORE_QUANTILES = (0.25, 0.5, 0.75)


def round_value(value: float) -> float:
    return round(float(value), ROUND_DECIMALS)


def rounded(value: float | None) -> float | None:
    return None if value is None else round_value(value)


def unit_rows(matrix: FloatMatrix) -> NDArray[np.float64]:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


@dataclass(frozen=True)
class ProfileScores:
    scores: NDArray[np.float64]
    best_sentence: NDArray[np.int64]


def score_profiles(
    propositions: FloatMatrix, sentences: FloatMatrix, sentence_counts: Sequence[int]
) -> ProfileScores:
    similarities = unit_rows(propositions) @ unit_rows(sentences).T
    bounds = np.cumsum([0, *sentence_counts])
    scores = np.empty((similarities.shape[0], len(sentence_counts)), dtype=np.float64)
    best = np.empty_like(scores, dtype=np.int64)
    for column, (start, end) in enumerate(pairwise(bounds)):
        block = similarities[:, start:end]
        best[:, column] = block.argmax(axis=1)
        scores[:, column] = block.max(axis=1)
    return ProfileScores(scores, best)


def identification_rank(scores: NDArray[np.float64], true_index: int) -> int:
    others = np.delete(scores, true_index)
    return 1 + int(np.count_nonzero(others >= scores[true_index]))


def chance_mrr(n_candidates: int) -> float:
    return sum(1 / rank for rank in range(1, n_candidates + 1)) / n_candidates


def percentile_interval(values: NDArray[np.float64], confidence_level: float) -> list[float]:
    alpha = 1 - confidence_level
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return [round_value(low), round_value(high)]


def bootstrap_by_hearing(
    pairs: Sequence[ProfilePair],
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> JsonObject:
    hearings = sorted({pair.hearing_id for pair in pairs})
    position = {hearing_id: index for index, hearing_id in enumerate(hearings)}
    counts = np.zeros(len(hearings))
    sums = np.zeros((3, len(hearings)))
    for pair in pairs:
        index = position[pair.hearing_id]
        counts[index] += 1
        sums[:, index] += (pair.score, pair.rank == 1, 1 / pair.rank)
    weights = np.stack(
        [
            np.bincount(rng.integers(0, len(hearings), len(hearings)), minlength=len(hearings))
            for _ in range(samples)
        ]
    ).astype(np.float64)
    replicates = (weights @ sums.T) / (weights @ counts)[:, None]
    return {
        "unit": "hearing",
        "method": BOOTSTRAP_INTERVAL,
        "samples": samples,
        "confidence_level": confidence_level,
        "mean_score": percentile_interval(replicates[:, 0], confidence_level),
        "acc_at_1": percentile_interval(replicates[:, 1], confidence_level),
        "mrr": percentile_interval(replicates[:, 2], confidence_level),
    }


def group_metrics(
    pairs: Sequence[ProfilePair],
    n_candidates: int,
    config: ProfileValidationConfig,
    rng: np.random.Generator,
) -> JsonObject:
    chance = {"acc_at_1": rounded(1 / n_candidates), "mrr": rounded(chance_mrr(n_candidates))}
    summary: JsonObject = {
        "pairs": len(pairs),
        "hearings": len({pair.hearing_id for pair in pairs}),
        "actors": len({pair.actor for pair in pairs}),
        "splits": sorted({pair.split for pair in pairs}),
        "n_candidates": n_candidates,
        "chance": chance,
    }
    if not pairs:
        return {**summary, "score": None, "identification": None, "bootstrap": None}
    scores = np.array([pair.score for pair in pairs])
    ranks = np.array([pair.rank for pair in pairs])
    q25, median, q75 = np.quantile(scores, SCORE_QUANTILES)
    return {
        **summary,
        "score": {
            "mean": rounded(scores.mean()),
            "median": rounded(median),
            "q25": rounded(q25),
            "q75": rounded(q75),
            "min": rounded(scores.min()),
            "max": rounded(scores.max()),
        },
        "identification": {
            "acc_at_1": rounded(float(np.mean(ranks == 1))),
            "mrr": rounded(float(np.mean(1 / ranks))),
        },
        "bootstrap": bootstrap_by_hearing(
            pairs, config.bootstrap_samples, config.confidence_level, rng
        ),
    }
