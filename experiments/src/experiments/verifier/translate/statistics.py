"""Output flags, length statistics and time estimates of a translation run."""

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.reporting import file_record
from experiments.common.transcript import (
    is_sentence,
    split_sentences,
)
from experiments.verifier.translate.config import Segmenter, TranslationConfig, boundary_parts
from experiments.verifier.translate.selection import OpinionUnit, TextSelection
from experiments.verifier.translate.store import TranslationStore, has_content

Record = dict[str, Any]


WORD_PATTERN = re.compile(r"\w+")
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
FLAG_NAMES = (
    "empty",
    "identical",
    "degenerate",
    "degenerate_not_in_source",
    "input_truncated",
    "hit_max_new_tokens",
    "long_output",
    "doubled_length",
)
MAX_LISTED_KEYS = 200


def quantiles(values: Iterable[float], digits: int = 4) -> Record:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), digits),
        **{f"q{int(q * 100):02d}": round(float(np.quantile(array, q)), digits) for q in QUANTILES},
    }


def repeated_ngram(text: str, n: int, min_count: int) -> bool:
    words = WORD_PATTERN.findall(text.lower())
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(count >= min_count for count in grams.values())


def output_flags(source: str, record: Record, config: TranslationConfig) -> Record:
    translation = record["translation"]
    n, count = config.degenerate_ngram, config.degenerate_min_count
    degenerate = repeated_ngram(translation, n, count)
    decoding = config.decoding
    budget = decoding.max_new_tokens_ratio * record["input_tokens"] + decoding.max_new_tokens_margin
    return {
        "empty": not translation,
        "identical": translation == source,
        "degenerate": degenerate,
        "degenerate_not_in_source": degenerate and not repeated_ngram(source, n, count),
        "input_truncated": bool(record["input_truncated"]),
        "hit_max_new_tokens": bool(record["hit_max_new_tokens"]),
        "long_output": record["output_tokens"] > budget,
        "doubled_length": record["output_tokens"] > 2 * record["input_tokens"],
    }


def segmentation_check(chunks: list[str]) -> Record:
    dropped_words = total_words = chunks_with_drop = fallback = 0
    coverage = []
    for chunk in chunks:
        sentences = split_sentences(chunk)
        if not sentences:
            fallback += 1
            sentences = [chunk]
        words = len(chunk.split())
        kept = min(sum(len(sentence.split()) for sentence in sentences), words)
        dropped_words += words - kept
        total_words += words
        chunks_with_drop += kept < words
        coverage.append(kept / words)
    return {
        "method": (
            "for each distinct non-empty chunk, the words of the sentences split_sentences returns "
            "(the whole chunk when it returns none) against the words of the chunk; this is what "
            "joining only split_sentences sentences would drop, while the translation units keep "
            "every word"
        ),
        "distinct_chunks": len(chunks),
        "chunks_without_sentence": fallback,
        "chunks_losing_words": chunks_with_drop,
        "words": total_words,
        "words_dropped": dropped_words,
        "words_dropped_fraction": round(dropped_words / total_words, 6) if total_words else None,
        "chunk_word_coverage": quantiles(coverage),
        "distinct_split_sentences": len(
            {s for chunk in chunks for s in (split_sentences(chunk) or [chunk])}
        ),
    }


def selection_counts(
    units: list[OpinionUnit], selection: TextSelection, segmenter: Segmenter
) -> Record:
    segments_per_chunk = [len(segmenter.segments(chunk)) for chunk in selection.chunks]
    parts_per_chunk = [boundary_parts(chunk) for chunk in selection.chunks]
    distinct_parts = list(dict.fromkeys(part for parts in parts_per_chunk for part in parts))
    sentences = [segment for segment in selection.segments if is_sentence(segment)]
    chunk_slots = [chunk for unit in units for chunk in unit.chunks]
    return {
        "opinions": len(units),
        "opinions_by_split": dict(sorted(Counter(unit.split for unit in units).items())),
        "hearings": len({unit.hearing_id for unit in units}),
        "chunk_slots": len(chunk_slots),
        "empty_chunk_slots": sum(1 for chunk in chunk_slots if not chunk),
        "distinct_opinions_with_probes": len(selection.opinions),
        "distinct_chunks_with_probes": len(selection.chunks),
        "distinct_segments": len(selection.segments),
        "distinct_segments_accepted_by_is_sentence": len(sentences),
        "distinct_segments_shorter_than_a_sentence": len(selection.segments) - len(sentences),
        "distinct_verbatim_segments": len(selection.verbatim_segments),
        "distinct_boundary_parts": len(distinct_parts),
        "distinct_boundary_parts_ending_in_join_abbreviation": sum(
            1 for part in distinct_parts if part.split()[-1] in segmenter.join_abbreviations
        ),
        "distinct_boundary_parts_rejected_by_is_sentence": sum(
            1 for part in distinct_parts if not is_sentence(part)
        ),
        "join_abbreviation_last_token_counts": dict(
            sorted(
                Counter(
                    part.split()[-1]
                    for parts in parts_per_chunk
                    for part in parts[:-1]
                    if part.split()[-1] in segmenter.join_abbreviations
                ).items()
            )
        ),
        "chunks_with_joined_parts": sum(
            1
            for parts, count in zip(parts_per_chunk, segments_per_chunk, strict=True)
            if len(parts) != count
        ),
        "segments_per_chunk": quantiles(segments_per_chunk),
        "distinct_model_texts": len(selection.model_texts()),
    }


def token_plan(tokenizer: Any, texts: list[str], max_input: int, special: int) -> Record:
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]] if texts else []
    return {
        "texts": len(texts),
        "model_input_tokens": int(sum(lengths)),
        "model_input_tokens_distribution": quantiles(lengths, 1),
        "content_tokens_distribution": quantiles([length - special for length in lengths], 1),
        "over_max_input_tokens": sum(1 for length in lengths if length > max_input),
    }


def record_statistics(
    config: TranslationConfig, texts: list[str], store: TranslationStore
) -> Record:
    records = [(text, store.record(text)) for text in texts]
    present = [(text, record) for text, record in records if record is not None]
    flags: dict[str, list[str]] = {name: [] for name in FLAG_NAMES}
    for text, record in present:
        for name, value in output_flags(text, record, config).items():
            if value:
                flags[name].append(record["key"])
    return {
        "texts": len(texts),
        "translated": len(present),
        "missing": len(texts) - len(present),
        "flags": {name: len(keys) for name, keys in flags.items()},
        "flagged_keys": {name: keys[:MAX_LISTED_KEYS] for name, keys in flags.items() if keys},
        "input_tokens": quantiles((r["input_tokens"] for _, r in present), 1),
        "output_tokens": quantiles((r["output_tokens"] for _, r in present), 1),
        "token_ratio": quantiles(
            r["output_tokens"] / r["input_tokens"] for _, r in present if r["input_tokens"]
        ),
        "character_ratio": quantiles(len(r["translation"]) / len(t) for t, r in present),
        "devices": dict(sorted(Counter(r["device"] for _, r in present).items())),
    }


def chunk_statistics(chunks: list[str], store: TranslationStore) -> Record:
    ratios = []
    seconds = []
    incomplete = 0
    for chunk in chunks:
        segments = [s for s in dict.fromkeys(store.segmenter.segments(chunk)) if has_content(s)]
        records = [store.record(segment) for segment in segments]
        if any(record is None for record in records):
            incomplete += 1
            continue
        ratios.append(len(store.translate_chunk(chunk)) / len(chunk))
        seconds.append(sum(record["seconds"] for record in records if record is not None))
    return {
        "distinct_chunks": len(chunks),
        "chunks_with_missing_segments": incomplete,
        "character_ratio": quantiles(ratios),
        "seconds_per_chunk": quantiles(seconds),
        "seconds_per_chunk_rule": (
            "sum over the distinct parts of a chunk of the per-text seconds (batch seconds / "
            "batch texts) of their cache records; a part shared by chunks counts in each"
        ),
    }


def timing_by_device(records: list[Record]) -> Record:
    timing: Record = {}
    for device in sorted({record["device"] for record in records}):
        members = [record for record in records if record["device"] == device]
        seconds = sum(record["seconds"] for record in members)
        tokens = sum(record["model_input_tokens"] for record in members)
        timing[device] = {
            "records": len(members),
            "seconds": round(seconds, 2),
            "model_input_tokens": tokens,
            "seconds_per_text": round(seconds / len(members), 4),
            "seconds_per_model_input_token": round(seconds / tokens, 6) if tokens else None,
        }
    return timing


def cache_timing(selection: TextSelection, store: TranslationStore) -> Record:
    def present(texts: list[str]) -> list[Record]:
        return [record for text in texts if (record := store.record(text)) is not None]

    return {
        "rule": (
            "per-text seconds are batch seconds / batch texts, summed over the cache records of "
            "the selected texts; a text translated by an earlier run counts with its own timing"
        ),
        "all_texts": timing_by_device(present(selection.model_texts())),
        "opinions": timing_by_device(present(selection.opinions)),
        "segments": timing_by_device(present(selection.segments)),
    }


def estimate_from_plan(plan_path: Path, timing: Record) -> Record:
    with open(plan_path) as f:
        plan = json.load(f)
    tokens = plan["tokens"]["all"]["model_input_tokens"]
    texts = plan["counts"]["distinct_model_texts"]
    estimates = {}
    for device, measured in timing["all_texts"].items():
        per_token = measured["seconds_per_model_input_token"]
        estimates[device] = {
            "from_tokens_hours": round(per_token * tokens / 3600, 2) if per_token else None,
            "from_texts_hours": round(measured["seconds_per_text"] * texts / 3600, 2),
            "measured_records": measured["records"],
        }
    return {
        "plan": file_record(plan_path),
        "plan_splits": plan["splits"],
        "plan_distinct_model_texts": texts,
        "plan_model_input_tokens": tokens,
        "by_device": estimates,
        "devices_without_measurement": [d for d in ("cpu", "mps") if d not in estimates],
        "note": (
            "an extrapolation from the measured records, not a measurement; the time of a device "
            "without records is unknown until measured"
        ),
    }
