"""Cosine similarity baseline between premise items and the opinion."""

import dataclasses
import time
from dataclasses import dataclass
from typing import Any

import huggingface_hub
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.common.hub_offline import pinned_weights_file
from experiments.common.transcript import split_sentences
from experiments.common.udv_run import (
    UdvConfig,
    encode_with_cache,
    load_encoder,
)
from experiments.verifier.nli.benchmark import PremiseUnit, distinct_items, unit_header
from experiments.verifier.nli.config import ScorerSpec, VerifierConfig
from experiments.verifier.nli.cross_encoder import minimum_scores, release_device, token_counts

Record = dict[str, Any]


@dataclass(frozen=True)
class TextVectors:
    vectors: dict[str, np.ndarray]
    sentence_lists: dict[str, tuple[list[str], bool]]


def cosine_encoder(config: VerifierConfig, spec: ScorerSpec, device: str) -> tuple[Any, UdvConfig]:
    udv = dataclasses.replace(config.udv, cache_dir=config.cache_dir)
    encoder = load_encoder(udv, device)
    if not huggingface_hub.is_offline_mode():
        raise SystemExit(f"{spec.key}: the encoder must be loaded in offline mode")
    if encoder.max_seq_length != spec.max_length:
        raise SystemExit(
            f"{spec.key}: max_seq_length {encoder.max_seq_length} != {spec.max_length}"
        )
    return encoder, udv


def encode_texts(
    encoder: SentenceTransformer, texts: list[str], udv: UdvConfig, device: str, label: str
) -> dict[str, np.ndarray]:
    distinct = list(dict.fromkeys(texts))
    embeddings = encode_with_cache(encoder, distinct, udv, device, label)
    return {text: embeddings[index] for index, text in enumerate(distinct)}


def item_sentences(item: str) -> tuple[list[str], bool]:
    sentences = split_sentences(item)
    return (sentences, False) if sentences else ([item], True)


def cosines(vector: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    return cosine_similarity(vector.reshape(1, -1), np.vstack(others)).flatten()


def cosine_item_record(
    item: str,
    hypothesis_vector: np.ndarray,
    vectors: TextVectors,
    tokens: dict[str, int],
    max_length: int,
) -> Record:
    sentences, fallback = vectors.sentence_lists[item]
    sentence_scores = cosines(hypothesis_vector, [vectors.vectors[s] for s in sentences])
    return {
        "cosine": float(cosines(hypothesis_vector, [vectors.vectors[item]])[0]),
        "tokens": tokens[item],
        "truncated": tokens[item] > max_length,
        "sentences": len(sentences),
        "sentences_truncated": sum(1 for s in sentences if tokens[s] > max_length),
        "sentence_max": float(sentence_scores.max()),
        "sentence_fallback": fallback,
    }


def cosine_unit_row(
    unit: PremiseUnit, vectors: TextVectors, tokens: dict[str, int], spec: ScorerSpec
) -> Record:
    hypothesis_vector = vectors.vectors[unit.hypothesis]
    records = {
        item: cosine_item_record(item, hypothesis_vector, vectors, tokens, spec.max_length)
        for item in distinct_items(unit.items)
    }
    if records:
        scores = {
            "max.cosine": max(record["cosine"] for record in records.values()),
            "sentence_max.cosine": max(record["sentence_max"] for record in records.values()),
        }
    else:
        scores = minimum_scores(list(spec.scores))
    return {
        **unit_header(unit),
        "items": [records[item] if item else None for item in unit.items],
        "concatenated": None,
        "scores": scores,
    }


def encode_units(
    units: list[PremiseUnit],
    encoder: SentenceTransformer,
    udv: UdvConfig,
    device: str,
    label: str,
) -> TextVectors:
    items = list(dict.fromkeys(item for unit in units for item in distinct_items(unit.items)))
    sentence_lists = {item: item_sentences(item) for item in items}
    sentence_texts = [s for listed, _ in sentence_lists.values() for s in listed]
    texts = [*(unit.hypothesis for unit in units), *items, *sentence_texts]
    return TextVectors(
        vectors=encode_texts(encoder, texts, udv, device, label),
        sentence_lists=sentence_lists,
    )


def score_units_cosine(
    units: list[PremiseUnit], spec: ScorerSpec, config: VerifierConfig, device: str, prefix: str
) -> tuple[list[Record], Record]:
    started = time.perf_counter()
    encoder, udv = cosine_encoder(config, spec, device)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = encode_units(units, encoder, udv, device, f"{prefix}_all")
    seconds = time.perf_counter() - started
    tokens = token_counts(encoder.tokenizer, list(vectors.vectors), True)
    rows = [cosine_unit_row(unit, vectors, tokens, spec) for unit in units]
    details = {
        "model": {
            "name": spec.name,
            "revision": spec.revision,
            "device": device,
            "max_seq_length": encoder.max_seq_length,
            "weights_file": pinned_weights_file(spec.name, spec.revision),
            "embedding_dimension": encoder.get_embedding_dimension(),
            "batch_size": udv.batch_size,
            "cache_dir": str(udv.cache_dir),
            "cache_label": f"{prefix}_all: distinct opinions, chunks and chunk sentences",
            "load_seconds": round(load_seconds, 1),
        },
        "label_probes": None,
        "timing": {
            "distinct_texts": len(vectors.vectors),
            "seconds": round(seconds, 2),
            "texts_per_second": round(len(vectors.vectors) / seconds, 1) if seconds else None,
        },
    }
    del encoder
    release_device(device)
    return rows, details
