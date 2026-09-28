"""The UDV run settings the experiment scripts read, and the sentence encoder with its cache.

The UDV runs themselves are built by ``bookworm build-udvs``. The experiment scripts that reuse a
run's encoder and embeddings (calibration, retrieval, verifier and measurement scripts) read the
same TOML through ``load_config`` and share the embedding cache with the library: the cache file
names come from ``bookworm.cache_file_name``, so both sides read and write the same files.
"""

import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bookworm import SUPPORT_TYPES, TIERS, cache_file_name, load_gated_jsonl
from bookworm import cache_key as library_cache_key
from bookworm.features.encoders import sentence_encoder_identity
from bookworm.features.sentence_transformer import load_sentence_transformer, select_device
from sentence_transformers import SentenceTransformer

from experiments.common.transcript import (
    resolve_person_speech,
    split_into_turns,
    split_turn_sentences,
)

__all__ = [
    "SUPPORT_TYPES",
    "TIERS",
    "UdvConfig",
    "cache_key",
    "encode_with_cache",
    "load_config",
    "load_encoder",
    "load_lds_records",
    "resolve_hearing_people",
    "seed_everything",
    "select_device",
    "sentence_slices_by_person",
]

Record = dict[str, Any]


@dataclass(frozen=True)
class UdvConfig:
    lds_path: Path
    expected_sha256: str
    model_name: str
    model_revision: str
    batch_size: int
    device: str
    embedding_threshold: float
    seed: int
    output_dir: Path
    cache_dir: Path
    source: Record = field(default_factory=dict)


def load_config(config_path: Path) -> UdvConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return UdvConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        expected_sha256=raw["dataset"]["sha256"],
        model_name=raw["encoder"]["name"],
        model_revision=raw["encoder"]["revision"],
        batch_size=raw["encoder"]["batch_size"],
        device=raw["encoder"]["device"],
        embedding_threshold=raw["evidence"]["embedding_threshold"],
        seed=raw["run"]["seed"],
        output_dir=Path(raw["run"]["output_dir"]),
        cache_dir=Path(raw["run"]["cache_dir"]),
        source=raw,
    )


def load_lds_records(config: UdvConfig) -> list[Record]:
    return load_gated_jsonl(config.lds_path, config.expected_sha256)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_encoder(config: UdvConfig, device: str) -> SentenceTransformer:
    return load_sentence_transformer(config.model_name, config.model_revision, device)


def encoder_identity(config: UdvConfig, device: str) -> str:
    return sentence_encoder_identity(config.model_name, config.model_revision, device)


def cache_key(texts: list[str], config: UdvConfig, device: str) -> str:
    return library_cache_key(encoder_identity(config, device), texts)


def encode_with_cache(
    encoder: SentenceTransformer,
    texts: list[str],
    config: UdvConfig,
    device: str,
    label: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, encoder.get_embedding_dimension() or 0), dtype=np.float32)
    cache_path = config.cache_dir / cache_file_name(label, encoder_identity(config, device), texts)
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = encoder.encode(
        texts, batch_size=config.batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def resolve_hearing_people(hearing: Record) -> list[Record]:
    turns = split_into_turns(hearing["transcricao"])
    people = []
    for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
        matched_turns, speech = resolve_person_speech(participant, turns)
        units = split_turn_sentences(matched_turns)
        people.append(
            {
                "index": person_index,
                "participant": participant,
                "matched_turns": matched_turns,
                "speech": speech,
                "sentences": [unit["text"] for unit in units],
                "sentence_turns": [unit["turn_index"] for unit in units],
            }
        )
    return people


def sentence_slices_by_person(people: list[Record]) -> dict[int, slice]:
    slices: dict[int, slice] = {}
    offset = 0
    for person in people:
        slices[person["index"]] = slice(offset, offset + len(person["sentences"]))
        offset += len(person["sentences"])
    return slices
