"""Benchmark opinions and NLI chunks selected for translation."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from bookworm import load_jsonl, sha256_of_file

from experiments.common.splits import SPLIT_NAMES, load_split_lookup
from experiments.common.transcript import (
    normalize_whitespace,
)
from experiments.verifier.benchmark_inputs import (
    check_benchmark_file,
    check_row_split,
    load_nli_chunks,
)
from experiments.verifier.translate.config import Segmenter, TranslationConfig
from experiments.verifier.translate.store import has_content

Record = dict[str, Any]


@dataclass(frozen=True)
class OpinionUnit:
    unit_id: str
    hearing_id: int
    split: str
    opinion: str
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class TextSelection:
    opinions: list[str]
    chunks: list[str]
    segments: list[str]
    verbatim_segments: list[str]

    def model_texts(self) -> list[str]:
        return list(dict.fromkeys([*self.opinions, *self.segments]))


def resolve_splits(
    config: TranslationConfig, requested: list[str] | None, final_test: bool
) -> tuple[str, ...]:
    allowed = (*config.default_splits, *(config.final_test_splits if final_test else ()))
    chosen = tuple(dict.fromkeys(requested)) if requested else allowed
    unknown = [split for split in chosen if split not in SPLIT_NAMES]
    if unknown:
        raise SystemExit(f"unknown splits {unknown}; known: {list(SPLIT_NAMES)}")
    refused = [split for split in chosen if split in config.final_test_splits and not final_test]
    if refused:
        raise SystemExit(f"splits {refused} are refused unless --final-test is passed")
    outside = [split for split in chosen if split not in allowed]
    if outside:
        raise SystemExit(f"splits {outside} are outside {list(allowed)}")
    return chosen


def benchmark_rows(
    config: TranslationConfig, split_of: dict[int, str], splits: tuple[str, ...], limit: int | None
) -> list[Record]:
    rows = []
    taken: Counter[str] = Counter()
    for row in load_jsonl(config.benchmark_path):
        check_row_split(row, split_of)
        if row["split"] not in splits:
            continue
        if limit is not None and taken[row["split"]] >= limit:
            continue
        taken[row["split"]] += 1
        rows.append(row)
    return rows


def load_units(
    config: TranslationConfig, splits: tuple[str, ...], limit: int | None
) -> tuple[list[OpinionUnit], Record]:
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    benchmark = check_benchmark_file(config)
    rows = benchmark_rows(config, split_of, splits, limit)
    chunks = load_nli_chunks(config, rows)
    units = [
        OpinionUnit(
            unit_id=row["id"],
            hearing_id=row["hearing_id"],
            split=row["split"],
            opinion=normalize_whitespace(row["opinion"]),
            chunks=tuple(normalize_whitespace(text) for text in chunks[row["id"]]),
        )
        for row in rows
    ]
    context = {
        "sources": {
            "nli": {"path": str(config.nli_path), "sha256": config.nli_sha256},
            "benchmark": benchmark,
            "splits": split_source,
            "nli_verifier_config": {
                "path": str(config.nli_config_path),
                "sha256": sha256_of_file(config.nli_config_path),
            },
        },
        "subset": (
            None
            if limit is None
            else {"limit_per_split": limit, "rule": "first rows of each split in file order"}
        ),
    }
    return units, context


def select_texts(
    opinions: Iterable[str], chunks: Iterable[str], segmenter: Segmenter
) -> TextSelection:
    distinct_chunks = list(dict.fromkeys(chunk for chunk in chunks if chunk))
    segments = list(
        dict.fromkeys(s for chunk in distinct_chunks for s in segmenter.segments(chunk))
    )
    return TextSelection(
        opinions=list(dict.fromkeys(o for o in opinions if has_content(o))),
        chunks=distinct_chunks,
        segments=[segment for segment in segments if has_content(segment)],
        verbatim_segments=[segment for segment in segments if not has_content(segment)],
    )


def unit_selection(units: list[OpinionUnit], config: TranslationConfig) -> TextSelection:
    return select_texts(
        [*(unit.opinion for unit in units), *map(normalize_whitespace, config.probe_hypotheses)],
        [
            *(chunk for unit in units for chunk in unit.chunks),
            *map(normalize_whitespace, config.probe_premises),
        ],
        config.segmenter,
    )
