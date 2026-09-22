import argparse
import hashlib
import json
import platform
import random
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.dataset_io import load_gated_jsonl, write_json, write_jsonl
from utils.udv_pipeline import (
    enclosing_sentence,
    find_opinion_quote_match,
    is_trusted_quote,
    locate_sentence_span,
    resolve_person_speech,
    sentences_agree,
    split_into_turns,
    split_sentences,
)

Record = dict[str, Any]

TIERS = (
    "quote_found",
    "semantic_match_high",
    "semantic_match_weak",
    "no_evidence",
    "person_not_resolved",
)
SUPPORT_TYPES = ("direct_quote", "semantic_with_short_quote", "semantic_similarity")
EMPTY_SPAN: Record = {"start_char": None, "end_char": None, "speaker_turn": None}


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


def select_hearings(
    records: list[Record], limit: int | None, ids: list[int] | None
) -> list[Record]:
    if ids is not None:
        wanted = set(ids)
        return [record for record in records if record["id"] in wanted]
    if limit is not None:
        return records[:limit]
    return records


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_encoder(config: UdvConfig, device: str) -> SentenceTransformer:
    return SentenceTransformer(config.model_name, revision=config.model_revision, device=device)


def cache_key(texts: list[str], config: UdvConfig, device: str) -> str:
    digest = hashlib.sha256(f"{config.model_name}@{config.model_revision}@{device}".encode())
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def encode_with_cache(
    encoder: SentenceTransformer,
    texts: list[str],
    config: UdvConfig,
    device: str,
    label: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, encoder.get_embedding_dimension() or 0), dtype=np.float32)
    cache_path = config.cache_dir / f"{label}_{cache_key(texts, config, device)[:16]}.npy"
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
        people.append(
            {
                "index": person_index,
                "participant": participant,
                "matched_turns": matched_turns,
                "speech": speech,
                "sentences": split_sentences(speech) if matched_turns else [],
            }
        )
    return people


def build_quote_evidence(quote_match: Record, person: Record, transcript: str) -> Record:
    sentence = enclosing_sentence(quote_match["prefix"], person["speech"])
    span = locate_sentence_span(sentence, transcript, person["matched_turns"])
    return {
        "text": sentence,
        "support_type": "direct_quote",
        "score": None,
        "quote_prefix": quote_match["prefix"],
        **(span or EMPTY_SPAN),
    }


def short_quote_supports(quote_match: Record | None, sentence: str, person: Record) -> bool:
    if quote_match is None:
        return False
    return sentences_agree(enclosing_sentence(quote_match["prefix"], person["speech"]), sentence)


def build_semantic_evidence(
    opinion_embedding: np.ndarray,
    person: Record,
    sentence_embeddings: np.ndarray,
    transcript: str,
    quote_match: Record | None,
) -> Record:
    similarities = cosine_similarity(
        opinion_embedding.reshape(1, -1), sentence_embeddings
    ).flatten()
    best_index = int(similarities.argmax())
    sentence = person["sentences"][best_index]
    supported = short_quote_supports(quote_match, sentence, person)
    span = locate_sentence_span(sentence, transcript, person["matched_turns"])
    return {
        "text": sentence,
        "support_type": "semantic_with_short_quote" if supported else "semantic_similarity",
        "score": float(similarities[best_index]),
        "quote_prefix": quote_match["prefix"] if supported else None,
        **(span or EMPTY_SPAN),
    }


def classify_tier(evidence: Record | None, person: Record, threshold: float) -> str:
    if not person["matched_turns"]:
        return "person_not_resolved"
    if evidence is None:
        return "no_evidence"
    if evidence["support_type"] == "direct_quote":
        return "quote_found"
    return "semantic_match_high" if evidence["score"] >= threshold else "semantic_match_weak"


def provenance_for(tier: str) -> str | None:
    if tier == "quote_found":
        return "weak"
    if tier in ("semantic_match_high", "semantic_match_weak"):
        return "model"
    return None


def build_udv_record(
    hearing: Record,
    person: Record,
    opinion_index: int,
    opinion_text: str,
    evidence: Record | None,
    tier: str,
    config: UdvConfig,
) -> Record:
    return {
        "id": f"udv-{hearing['id']}-{person['index']}-{opinion_index}",
        "hearing_id": hearing["id"],
        "actor": {"name": person["participant"]["nome"], "role": person["participant"]["cargo"]},
        "proposition": opinion_text,
        "evidence": evidence,
        "tier": tier,
        "provenance": provenance_for(tier),
        "method": {
            "encoder": config.model_name,
            "revision": config.model_revision,
            "embedding_threshold": config.embedding_threshold,
        },
    }


def sentence_slices_by_person(people: list[Record]) -> dict[int, slice]:
    slices: dict[int, slice] = {}
    offset = 0
    for person in people:
        slices[person["index"]] = slice(offset, offset + len(person["sentences"]))
        offset += len(person["sentences"])
    return slices


def build_hearing_udvs(
    hearing: Record, encoder: SentenceTransformer, config: UdvConfig, device: str
) -> tuple[list[Record], list[Record]]:
    transcript = hearing["transcricao"]
    people = resolve_hearing_people(hearing)
    all_sentences = [sentence for person in people for sentence in person["sentences"]]
    sentence_embeddings = encode_with_cache(
        encoder, all_sentences, config, device, f"sentences_{hearing['id']}"
    )
    opinions = [
        (person, opinion_index, opinion_text)
        for person in people
        for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"])
    ]
    opinion_embeddings = encode_with_cache(
        encoder, [text for _, _, text in opinions], config, device, f"opinions_{hearing['id']}"
    )
    slices = sentence_slices_by_person(people)

    records = []
    for position, (person, opinion_index, opinion_text) in enumerate(opinions):
        evidence = None
        if person["matched_turns"]:
            quote_match = find_opinion_quote_match(opinion_text, person["speech"])
            if is_trusted_quote(quote_match):
                evidence = build_quote_evidence(quote_match, person, transcript)
            elif person["sentences"]:
                evidence = build_semantic_evidence(
                    opinion_embeddings[position],
                    person,
                    sentence_embeddings[slices[person["index"]]],
                    transcript,
                    quote_match,
                )
        tier = classify_tier(evidence, person, config.embedding_threshold)
        records.append(
            build_udv_record(hearing, person, opinion_index, opinion_text, evidence, tier, config)
        )
    return records, people


def summarize_run(
    run_name: str,
    records: list[Record],
    people: list[Record],
    hearings: list[Record],
    config: UdvConfig,
    encoder: SentenceTransformer,
    device: str,
    hearing_seconds: list[float],
) -> Record:
    evidences = [record["evidence"] for record in records if record["evidence"] is not None]
    return {
        "run_name": run_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hearings": {"count": len(hearings), "ids": [hearing["id"] for hearing in hearings]},
        "people": {
            "total": len(people),
            "resolved": sum(1 for person in people if person["matched_turns"]),
        },
        "opinions": {
            "total": len(records),
            "by_tier": {tier: sum(1 for r in records if r["tier"] == tier) for tier in TIERS},
        },
        "evidence_offsets": {
            "total": len(evidences),
            "located": sum(1 for e in evidences if e["start_char"] is not None),
        },
        "evidence_support_types": {
            support_type: sum(1 for e in evidences if e["support_type"] == support_type)
            for support_type in SUPPORT_TYPES
        },
        "encoder_runtime": {
            "device": device,
            "max_seq_length": encoder.max_seq_length,
            "embedding_dimension": encoder.get_embedding_dimension(),
        },
        "timing": {
            "elapsed_seconds": round(sum(hearing_seconds), 1),
            "mean_seconds_per_hearing": round(float(np.mean(hearing_seconds)), 2),
            "max_seconds_per_hearing": round(max(hearing_seconds), 2),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "platform": platform.platform(),
        },
        "config": config.source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build UDV v0 records (opinion -> transcript evidence) from the LDS file."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    parser.add_argument("--run-name", default="udv_v0", help="basename of the output files")
    parser.add_argument("--limit", type=int, default=None, help="only the first N LDS records")
    parser.add_argument("--ids", type=int, nargs="+", default=None, help="only these hearing ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.seed)
    device = select_device(config.device)
    hearings = select_hearings(load_lds_records(config), args.limit, args.ids)
    print(
        f"{len(hearings)} hearings | {config.model_name}@{config.model_revision[:7]} on {device}",
        flush=True,
    )
    encoder = load_encoder(config, device)

    all_records: list[Record] = []
    all_people: list[Record] = []
    hearing_seconds: list[float] = []
    for number, hearing in enumerate(hearings, start=1):
        started = time.perf_counter()
        records, people = build_hearing_udvs(hearing, encoder, config, device)
        hearing_seconds.append(time.perf_counter() - started)
        all_records.extend(records)
        all_people.extend(people)
        print(
            f"[{number}/{len(hearings)}] hearing {hearing['id']}: "
            f"{len(records)} opinions in {hearing_seconds[-1]:.1f}s",
            flush=True,
        )

    summary = summarize_run(
        args.run_name, all_records, all_people, hearings, config, encoder, device, hearing_seconds
    )
    write_jsonl(all_records, config.output_dir / f"{args.run_name}.jsonl")
    write_json(summary, config.output_dir / f"{args.run_name}_coverage.json")
    print(json.dumps({**summary["opinions"], **summary["timing"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
