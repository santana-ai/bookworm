"""Calibration queries: opinions with a trusted quote, their target sentence and their scores."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.common.transcript import (
    find_opinion_turn_quote_match,
    is_trusted_quote,
    sentences_agree,
)
from experiments.common.udv_run import (
    UdvConfig,
    encode_with_cache,
    resolve_hearing_people,
    sentence_slices_by_person,
)

Record = dict[str, Any]

EMBEDDING_KINDS = ("sentences", "opinions", "masked")

Encode = Callable[[list[str], str], np.ndarray]


@dataclass
class EncodingLedger:
    """Which hearings were encoded, per embedding kind, and whether the cache answered."""

    allowed_hearing_ids: frozenset[int]
    encoded_hearing_ids: set[int] = field(default_factory=set)
    by_kind: dict[str, Record] = field(default_factory=dict)


def encode_guarded(
    encoder: SentenceTransformer,
    texts: list[str],
    udv_config: UdvConfig,
    device: str,
    kind: str,
    hearing_id: int,
    ledger: EncodingLedger,
) -> np.ndarray:
    if hearing_id not in ledger.allowed_hearing_ids:
        raise SystemExit(f"refusing hearing {hearing_id}: it is outside the calibration splits")
    label = f"{kind}_{hearing_id}"
    before = set(udv_config.cache_dir.glob(f"{label}_*.npy"))
    embeddings = encode_with_cache(encoder, texts, udv_config, device, label)
    written = set(udv_config.cache_dir.glob(f"{label}_*.npy")) - before
    counts = ledger.by_kind.setdefault(
        kind, {"calls": 0, "texts": 0, "cache_hits": 0, "encoded": 0, "empty": 0}
    )
    counts["calls"] += 1
    counts["texts"] += len(texts)
    counts["empty" if not texts else "encoded" if written else "cache_hits"] += 1
    ledger.encoded_hearing_ids.add(hearing_id)
    return embeddings


def trusted_quote_queries(hearing: Record, people: list[Record]) -> tuple[list[Record], Record]:
    slices = sentence_slices_by_person(people)
    opinions = [
        (person, opinion_index, opinion_text)
        for person in people
        for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"])
    ]
    counts = {
        "opinions": len(opinions),
        "resolved_person": 0,
        "trusted_quote": 0,
        "target_not_in_candidates": 0,
    }
    queries = []
    for position, (person, opinion_index, opinion_text) in enumerate(opinions):
        if not person["matched_turns"]:
            continue
        counts["resolved_person"] += 1
        quote_match = find_opinion_turn_quote_match(opinion_text, person["matched_turns"])
        if quote_match is None or not is_trusted_quote(quote_match):
            continue
        counts["trusted_quote"] += 1
        target = quote_match["sentence"]
        target_indices = [
            index
            for index, sentence in enumerate(person["sentences"])
            if sentences_agree(sentence, target)
        ]
        if not target_indices:
            counts["target_not_in_candidates"] += 1
            continue
        queries.append(
            {
                "id": f"udv-{hearing['id']}-{person['index']}-{opinion_index}",
                "hearing_id": hearing["id"],
                "person_index": person["index"],
                "opinion": opinion_text,
                "opinion_position": position,
                "target_text": target,
                "target_indices": target_indices,
                "candidates": person["sentences"],
                "sentence_offset": slices[person["index"]].start,
            }
        )
    return queries, counts


def masked_row_problems(row: Record, base_queries: dict[str, Record]) -> list[str]:
    query = base_queries.get(row["id"])
    if query is None:
        return ["not_a_trusted_quote_query"]
    checks = {
        "opinion_mismatch": row["opinion"] == query["opinion"],
        "person_mismatch": row["person"]["index"] == query["person_index"],
        "candidates_mismatch": row["candidates"] == query["candidates"],
        "target_text_mismatch": row["target"]["text"] == query["target_text"],
        "target_indices_mismatch": row["target_indices"] == query["target_indices"],
    }
    return [problem for problem, passed in checks.items() if not passed]


def score_query(
    query: Record, kind: str, query_embedding: np.ndarray, candidate_embeddings: np.ndarray
) -> Record:
    scores = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    targets = query["target_indices"]
    top1 = int(scores.argmax())
    return {
        **query,
        "query": kind,
        "candidate_scores": scores,
        "non_target_indices": np.setdiff1d(np.arange(len(scores)), targets),
        "positive_score": float(scores[targets].max()),
        "top1_index": top1,
        "top1_score": float(scores[top1]),
        "top1_correct": top1 in targets,
    }


def masked_problems(rows: list[Record], by_id: dict[str, Record]) -> dict[str, list[str]]:
    problems = {row["id"]: masked_row_problems(row, by_id) for row in rows}
    return {row_id: found for row_id, found in problems.items() if found}


def score_unmasked(
    people: list[Record],
    queries: list[Record],
    candidate_embeddings: np.ndarray,
    slices: dict[int, slice],
    encode: Encode,
) -> tuple[list[Record], np.ndarray]:
    """Score each opinion against its speaker's candidates; also return the query embeddings."""
    if not queries:
        return [], np.empty((0, candidate_embeddings.shape[1]), dtype=np.float32)
    opinion_texts = [text for person in people for text in person["participant"]["opinioes"]]
    opinion_embeddings = encode(opinion_texts, "opinions")
    query_embeddings = opinion_embeddings[[q["opinion_position"] for q in queries]]
    unmasked = [
        score_query(
            query,
            "opinion",
            opinion_embeddings[query["opinion_position"]],
            candidate_embeddings[slices[query["person_index"]]],
        )
        for query in queries
    ]
    return unmasked, query_embeddings


def score_masked(
    rows: list[Record],
    by_id: dict[str, Record],
    candidate_embeddings: np.ndarray,
    slices: dict[int, slice],
    encode: Encode,
) -> list[Record]:
    """Score the masked opinion of each row whose query is in ``by_id``."""
    masked_embeddings = encode([row["masked_opinion"] for row in rows], "masked")
    return [
        score_query(
            {**by_id[row["id"]], "masked_opinion": row["masked_opinion"]},
            "masked_opinion",
            masked_embeddings[position],
            candidate_embeddings[slices[row["person"]["index"]]],
        )
        for position, row in enumerate(rows)
        if row["id"] in by_id
    ]


def guarded_encoder(
    encoder: SentenceTransformer,
    udv_config: UdvConfig,
    device: str,
    hearing_id: int,
    ledger: EncodingLedger,
) -> Encode:
    def encode(texts: list[str], kind: str) -> np.ndarray:
        return encode_guarded(encoder, texts, udv_config, device, kind, hearing_id, ledger)

    return encode


def collect_hearing(
    hearing: Record,
    rows: list[Record],
    encoder: SentenceTransformer,
    udv_config: UdvConfig,
    device: str,
    ledger: EncodingLedger,
) -> Record:
    """Score the unmasked and masked queries of one hearing against the speaker's sentences."""
    encode = guarded_encoder(encoder, udv_config, device, hearing["id"], ledger)
    people = resolve_hearing_people(hearing)
    slices = sentence_slices_by_person(people)
    sentences = [sentence for person in people for sentence in person["sentences"]]
    sentence_embeddings = encode(sentences, "sentences")
    base_queries, funnel = trusted_quote_queries(hearing, people)
    unmasked, query_embeddings = score_unmasked(
        people, base_queries, sentence_embeddings, slices, encode
    )
    by_id = {query["id"]: query for query in base_queries}
    problems = masked_problems(rows, by_id)
    masked: list[Record] = []
    if rows and not problems:
        masked = score_masked(rows, by_id, sentence_embeddings, slices, encode)
    return {
        "unmasked": unmasked,
        "query_embeddings": query_embeddings,
        "masked": masked,
        "sentence_embeddings": sentence_embeddings,
        "funnel": funnel,
        "problems": problems,
    }
