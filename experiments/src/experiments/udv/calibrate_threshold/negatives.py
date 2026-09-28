"""Negative sentences for the pair rules: a pool of other hearings or the speaker's own."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

Record = dict[str, Any]

QUERY_SCORE_DECIMALS = 6


@dataclass(frozen=True)
class SentencePool:
    hearing_ids: np.ndarray
    sentence_indices: np.ndarray
    scores: np.ndarray


class NegativeDraw(NamedTuple):
    scores: np.ndarray
    hearing_ids: np.ndarray
    sentence_indices: np.ndarray


NegativeSampler = Callable[[np.random.Generator, np.ndarray], NegativeDraw]


def build_pool(
    query_embeddings: np.ndarray, sentence_embeddings: dict[int, np.ndarray]
) -> SentencePool:
    hearing_ids = sorted(h for h, embeddings in sentence_embeddings.items() if len(embeddings))
    return SentencePool(
        hearing_ids=np.concatenate(
            [np.full(len(sentence_embeddings[h]), h, dtype=np.int64) for h in hearing_ids]
        ),
        sentence_indices=np.concatenate(
            [np.arange(len(sentence_embeddings[h]), dtype=np.int64) for h in hearing_ids]
        ),
        scores=np.hstack(
            [cosine_similarity(query_embeddings, sentence_embeddings[h]) for h in hearing_ids]
        ),
    )


def draw_pool_negatives(
    rng: np.random.Generator,
    rows: np.ndarray,
    negatives_per_query: int,
    query_hearing_ids: np.ndarray,
    pool: SentencePool,
) -> NegativeDraw:
    repeated = np.repeat(rows, negatives_per_query)
    hearings = query_hearing_ids[repeated]
    picks = rng.integers(0, len(pool.hearing_ids), size=len(repeated))
    clash = pool.hearing_ids[picks] == hearings
    while clash.any():
        picks[clash] = rng.integers(0, len(pool.hearing_ids), size=int(clash.sum()))
        clash = pool.hearing_ids[picks] == hearings
    return NegativeDraw(
        pool.scores[repeated, picks], pool.hearing_ids[picks], pool.sentence_indices[picks]
    )


def draw_hard_negatives(
    rng: np.random.Generator, rows: np.ndarray, negatives_per_query: int, queries: list[Record]
) -> NegativeDraw:
    repeated = np.repeat(rows, negatives_per_query)
    sizes = np.array([len(queries[row]["non_target_indices"]) for row in repeated], dtype=np.int64)
    picks = rng.integers(0, sizes) if len(repeated) else np.empty(0, dtype=np.int64)
    candidates = [
        int(queries[row]["non_target_indices"][pick])
        for row, pick in zip(repeated, picks, strict=True)
    ]
    return NegativeDraw(
        np.array(
            [
                queries[row]["candidate_scores"][candidate]
                for row, candidate in zip(repeated, candidates, strict=True)
            ],
            dtype=np.float64,
        ),
        np.array([queries[row]["hearing_id"] for row in repeated], dtype=np.int64),
        np.array(
            [
                queries[row]["sentence_offset"] + candidate
                for row, candidate in zip(repeated, candidates, strict=True)
            ],
            dtype=np.int64,
        ),
    )


def negatives_by_query(
    queries: list[Record], draw: NegativeDraw, negatives_per_query: int
) -> dict[str, list[Record]]:
    return {
        query["id"]: [
            {
                "hearing_id": int(draw.hearing_ids[index]),
                "sentence_index": int(draw.sentence_indices[index]),
                "score": round(float(draw.scores[index]), QUERY_SCORE_DECIMALS),
            }
            for index in range(position * negatives_per_query, (position + 1) * negatives_per_query)
        ]
        for position, query in enumerate(queries)
    }
