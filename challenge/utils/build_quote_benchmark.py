import argparse
import json
import platform
import re
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils import udv_pipeline
from utils.dataset_io import load_gated_jsonl, sha256_of_file, write_json, write_jsonl
from utils.udv_pipeline import (
    MIN_SENTENCE_WORDS,
    QUOTE_PATTERNS,
    SENTENCE_BOUNDARY_PATTERN,
    TRUSTED_PREFIX_WORDS,
    extract_quotes,
    find_opinion_turn_quote_match,
    is_trusted_quote,
    locate_turn_sentence_span,
    normalize_whitespace,
    resolve_person_speech,
    sentences_agree,
    split_into_turns,
    split_turn_sentences,
    strip_accents,
)

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
DROP_REASONS = ("masked_too_short", "target_not_in_candidates")
WORD_CHARACTER_PATTERN = re.compile(r"\w")
OVERLAP_TOKEN_PATTERN = re.compile(r"\w+")
QUOTE_CHARACTERS = frozenset("\"'“”„‟‘’‚‛«»‹›")
PIPELINE: Record = {
    "person_resolution": "resolve_person_speech over split_into_turns",
    "sentence_segmentation": "per matched turn, concatenated in turn order",
    "sentence_boundary_pattern": SENTENCE_BOUNDARY_PATTERN.pattern,
    "min_sentence_words": MIN_SENTENCE_WORDS,
    "quote_patterns": [pattern.pattern for pattern in QUOTE_PATTERNS],
    "quote_search": "inside each matched turn",
    "quote_selection": "most prefix words over all quotes, earliest quote on ties",
    "quote_occurrence": "max token Jaccard with the opinion for trusted prefixes, first on ties",
    "trusted_prefix_words": TRUSTED_PREFIX_WORDS,
    "target_agreement": "sentences_agree: equal strings, or one contained in the other",
}


@dataclass(frozen=True)
class QuoteBenchmarkConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    quote_pairs: list[tuple[str, str]]
    word_bounded_quote_pairs: list[tuple[str, str]]
    min_masked_words: int
    min_masked_words_split: str
    audit_max_words: int
    benchmark_version: str
    output_dir: Path
    source: Record = field(default_factory=dict)


def parse_quote_pairs(masking: Record, key: str) -> list[tuple[str, str]]:
    if any(len(pair) != 2 or any(len(mark) != 1 for mark in pair) for pair in masking[key]):
        raise SystemExit(f"masking.{key} must list [open, close] pairs of single characters")
    return [(pair[0], pair[1]) for pair in masking[key]]


def load_config(config_path: Path) -> QuoteBenchmarkConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    masking = raw["masking"]
    pairs = parse_quote_pairs(masking, "quote_pairs")
    bounded_pairs = parse_quote_pairs(masking, "word_bounded_quote_pairs")
    unknown = {mark for pair in pairs + bounded_pairs for mark in pair} - QUOTE_CHARACTERS
    if unknown:
        raise SystemExit(f"masking pairs use marks outside QUOTE_CHARACTERS: {sorted(unknown)}")
    if masking["min_masked_words_split"] not in SPLIT_NAMES[:2]:
        raise SystemExit("masking.min_masked_words_split must be train or validation, never test")
    if masking["min_masked_words"] < 0 or masking["audit_max_words"] < 0:
        raise SystemExit("masking word limits must be non-negative")
    return QuoteBenchmarkConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        quote_pairs=pairs,
        word_bounded_quote_pairs=bounded_pairs,
        min_masked_words=masking["min_masked_words"],
        min_masked_words_split=masking["min_masked_words_split"],
        audit_max_words=masking["audit_max_words"],
        benchmark_version=raw["run"]["benchmark_version"],
        output_dir=Path(raw["run"]["output_dir"]),
        source=raw,
    )


def load_split_lookup(manifest_path: Path, lds_sha256: str) -> tuple[dict[int, str], Record]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != lds_sha256:
        raise SystemExit(f"{manifest_path} was built from another LDS file")
    lookup = {hearing_id: name for name in SPLIT_NAMES for hearing_id in manifest[name]}
    source = {
        "path": str(manifest_path),
        "sha256": sha256_of_file(manifest_path),
        "split_version": manifest["split_version"],
    }
    return lookup, source


def quote_span_pattern(
    pairs: list[tuple[str, str]], word_bounded_pairs: list[tuple[str, str]]
) -> re.Pattern[str]:
    plain = [
        f"{re.escape(opening)}[^{re.escape(closing)}]*{re.escape(closing)}"
        for opening, closing in pairs
    ]
    bounded = [
        rf"(?<!\w){re.escape(opening)}[^{re.escape(closing)}]*{re.escape(closing)}(?!\w)"
        for opening, closing in word_bounded_pairs
    ]
    return re.compile("|".join(plain + bounded))


def mask_quotes(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    masked, removed = pattern.subn(" ", text)
    return normalize_whitespace(masked), removed


def count_words(text: str) -> int:
    return sum(1 for token in text.split() if WORD_CHARACTER_PATTERN.search(token))


def has_quote_characters(text: str) -> bool:
    return any(character in QUOTE_CHARACTERS for character in text)


def has_extracted_quote_left(row: Record) -> bool:
    return any(quote in row["masked_opinion"] for quote in extract_quotes(row["opinion"]))


def word_bounded_pairs_effect(rows: list[Record], config: QuoteBenchmarkConfig) -> Record:
    pattern = quote_span_pattern(config.quote_pairs, [])
    ids = [
        row["id"] for row in rows if has_quote_characters(mask_quotes(row["opinion"], pattern)[0])
    ]
    return {"rows": len(ids), "ids": ids}


def overlap_tokens(text: str) -> list[str]:
    return OVERLAP_TOKEN_PATTERN.findall(strip_accents(text).lower())


def longest_common_run(text: str, other_text: str) -> int:
    tokens, other_tokens = overlap_tokens(text), overlap_tokens(other_text)
    best = 0
    previous = [0] * (len(other_tokens) + 1)
    for token in tokens:
        current = [0] * (len(other_tokens) + 1)
        for position, other_token in enumerate(other_tokens, start=1):
            if token == other_token:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


def resolve_people(hearing: Record) -> list[Record]:
    turns = split_into_turns(hearing["transcricao"])
    people = []
    for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
        matched_turns = resolve_person_speech(participant, turns)[0]
        units = split_turn_sentences(matched_turns)
        people.append(
            {
                "index": person_index,
                "participant": participant,
                "matched_turns": matched_turns,
                "sentences": [unit["text"] for unit in units],
                "sentence_turns": [unit["turn_index"] for unit in units],
            }
        )
    return people


def build_target(quote_match: Record, person: Record, transcript: str) -> Record:
    sentence = quote_match["sentence"]
    span = locate_turn_sentence_span(
        sentence, transcript, person["matched_turns"], quote_match["turn_index"]
    )
    return {
        "text": sentence,
        "turn_index": quote_match["turn_index"],
        "start_char": span["start_char"] if span else None,
        "end_char": span["end_char"] if span else None,
    }


def build_row(
    hearing: Record,
    split: str,
    person: Record,
    opinion_index: int,
    opinion_text: str,
    quote_match: Record,
    pattern: re.Pattern[str],
) -> Record:
    masked, removed = mask_quotes(opinion_text, pattern)
    target = build_target(quote_match, person, hearing["transcricao"])
    candidates = person["sentences"]
    participant = person["participant"]
    return {
        "id": f"udv-{hearing['id']}-{person['index']}-{opinion_index}",
        "hearing_id": hearing["id"],
        "split": split,
        "person": {
            "index": person["index"],
            "name": participant["nome"],
            "role": participant["cargo"],
        },
        "opinion_index": opinion_index,
        "opinion": opinion_text,
        "masked_opinion": masked,
        "masked_word_count": count_words(masked),
        "removed_quote_spans": removed,
        "quote": {
            "prefix": quote_match["prefix"],
            "prefix_words": quote_match["words"],
            "quote_index": quote_match["quote_index"],
            "occurrence_count": quote_match["occurrence_count"],
            "occurrence_index": quote_match["occurrence_index"],
        },
        "target": target,
        "verbatim_overlap": {
            "original_longest_run": longest_common_run(opinion_text, target["text"]),
            "masked_longest_run": longest_common_run(masked, target["text"]),
        },
        "candidates": candidates,
        "candidate_turns": person["sentence_turns"],
        "n_candidates": len(candidates),
        "target_indices": [
            index
            for index, sentence in enumerate(candidates)
            if sentences_agree(sentence, target["text"])
        ],
    }


def drop_reasons(row: Record, config: QuoteBenchmarkConfig) -> list[str]:
    reasons = []
    if row["masked_word_count"] < config.min_masked_words:
        reasons.append("masked_too_short")
    if not row["target_indices"]:
        reasons.append("target_not_in_candidates")
    return reasons


def funnel_entry(split: str, person: Record, opinion_text: str) -> tuple[Record, Record | None]:
    quote_match = (
        find_opinion_turn_quote_match(opinion_text, person["matched_turns"])
        if person["matched_turns"]
        else None
    )
    entry = {
        "split": split,
        "resolved": bool(person["matched_turns"]),
        "with_extracted_quote": bool(extract_quotes(opinion_text)),
        "with_quote_match": quote_match is not None,
        "with_trusted_quote": is_trusted_quote(quote_match),
    }
    return entry, quote_match


def build_benchmark(
    lds: list[Record], split_of: dict[int, str], config: QuoteBenchmarkConfig
) -> tuple[list[Record], list[Record]]:
    pattern = quote_span_pattern(config.quote_pairs, config.word_bounded_quote_pairs)
    considered: list[Record] = []
    funnel: list[Record] = []
    for hearing in lds:
        if hearing["id"] not in split_of:
            raise SystemExit(f"hearing {hearing['id']} is not in the split manifest")
        split = split_of[hearing["id"]]
        for person in resolve_people(hearing):
            for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"]):
                entry, quote_match = funnel_entry(split, person, opinion_text)
                funnel.append(entry)
                if quote_match is None or not entry["with_trusted_quote"]:
                    continue
                row = build_row(
                    hearing, split, person, opinion_index, opinion_text, quote_match, pattern
                )
                row["drop_reasons"] = drop_reasons(row, config)
                considered.append(row)
    return considered, funnel


def histogram(values: list[int]) -> dict[int, int]:
    return dict(sorted(Counter(values).items()))


def distribution(values: list[int]) -> Record:
    if not values:
        return {"count": 0}
    array = np.array(values, dtype=float)
    return {
        "count": len(values),
        "min": int(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 2),
    }


def random_acc_at_1(rows: list[Record]) -> float | None:
    if not rows:
        return None
    return round(float(np.mean([len(r["target_indices"]) / r["n_candidates"] for r in rows])), 4)


def target_checks(rows: list[Record], transcripts: dict[int, str]) -> Record:
    located = [row for row in rows if row["target"]["start_char"] is not None]
    quote_characters_left = [
        row["id"] for row in rows if has_quote_characters(row["masked_opinion"])
    ]
    return {
        "offsets_located": len(located),
        "offset_text_matches": sum(
            1
            for row in located
            if normalize_whitespace(
                transcripts[row["hearing_id"]][
                    row["target"]["start_char"] : row["target"]["end_char"]
                ]
            )
            == row["target"]["text"]
        ),
        "target_is_exact_candidate": sum(
            1
            for row in rows
            if any(row["candidates"][i] == row["target"]["text"] for i in row["target_indices"])
        ),
        "target_index_in_target_turn": sum(
            1
            for row in rows
            if any(
                row["candidate_turns"][i] == row["target"]["turn_index"]
                for i in row["target_indices"]
            )
        ),
        "target_index_in_other_turn": sum(
            1
            for row in rows
            if any(
                row["candidate_turns"][i] != row["target"]["turn_index"]
                for i in row["target_indices"]
            )
        ),
        "masked_with_quote_characters_left": len(quote_characters_left),
        "masked_with_quote_characters_left_ids": quote_characters_left,
        "masked_with_extracted_quote_left": sum(1 for row in rows if has_extracted_quote_left(row)),
    }


def summarize_overlap(rows: list[Record]) -> Record:
    masked = [row["verbatim_overlap"]["masked_longest_run"] for row in rows]
    return {
        "original_longest_run": distribution(
            [row["verbatim_overlap"]["original_longest_run"] for row in rows]
        ),
        "masked_longest_run": histogram(masked),
        "masked_run_at_least_trusted_prefix": sum(
            1 for run in masked if run >= TRUSTED_PREFIX_WORDS
        ),
    }


def summarize_funnel(entries: list[Record]) -> Record:
    return {
        "total": len(entries),
        "with_resolved_person": sum(1 for entry in entries if entry["resolved"]),
        "with_extracted_quote": sum(
            1 for entry in entries if entry["resolved"] and entry["with_extracted_quote"]
        ),
        "with_quote_match": sum(1 for entry in entries if entry["with_quote_match"]),
        "with_trusted_quote": sum(1 for entry in entries if entry["with_trusted_quote"]),
    }


def summarize_group(
    considered: list[Record],
    funnel: list[Record],
    hearing_ids: set[int],
    transcripts: dict[int, str],
    config: QuoteBenchmarkConfig,
) -> Record:
    kept = [row for row in considered if not row["drop_reasons"]]
    dropped = [row for row in considered if row["drop_reasons"]]
    return {
        "hearings": len(hearing_ids),
        "opinions": summarize_funnel(funnel),
        "considered": len(considered),
        "kept": len(kept),
        "dropped": {
            "total": len(dropped),
            "by_reason": {
                reason: sum(1 for row in dropped if reason in row["drop_reasons"])
                for reason in DROP_REASONS
            },
        },
        "masked_word_count_considered": {
            **distribution([row["masked_word_count"] for row in considered]),
            "histogram": histogram([row["masked_word_count"] for row in considered]),
        },
        "masked_word_count_kept": distribution([row["masked_word_count"] for row in kept]),
        "removed_quote_spans_kept": histogram([row["removed_quote_spans"] for row in kept]),
        "prefix_words_kept": histogram([row["quote"]["prefix_words"] for row in kept]),
        "chosen_quote_index_kept": histogram([row["quote"]["quote_index"] for row in kept]),
        "candidates_kept": {
            **distribution([row["n_candidates"] for row in kept]),
            "total": sum(row["n_candidates"] for row in kept),
        },
        "targets_per_row_kept": histogram([len(row["target_indices"]) for row in kept]),
        "random_acc_at_1": random_acc_at_1(kept),
        "target_checks_kept": target_checks(kept, transcripts),
        "quote_characters_left_without_word_bounded_pairs_kept": word_bounded_pairs_effect(
            kept, config
        ),
        "verbatim_overlap_kept": summarize_overlap(kept),
    }


def min_words_audit(considered: list[Record], config: QuoteBenchmarkConfig) -> Record:
    rows = [row for row in considered if row["split"] == config.min_masked_words_split]
    counts = [row["masked_word_count"] for row in rows]
    below = [count for count in counts if count < config.min_masked_words]
    above = [count for count in counts if count >= config.min_masked_words]
    return {
        "split": config.min_masked_words_split,
        "min_masked_words": config.min_masked_words,
        "rule": config.source["masking"]["min_masked_words_rule"],
        "considered": len(rows),
        "word_count_histogram": histogram(counts),
        "rows_below_min": len(below),
        "largest_dropped_word_count": max(below) if below else None,
        "smallest_kept_word_count": min(above) if above else None,
        "texts_up_to_audit_max_words": [
            {
                "id": row["id"],
                "masked_word_count": row["masked_word_count"],
                "masked_opinion": row["masked_opinion"],
            }
            for row in sorted(
                rows, key=lambda r: (r["masked_word_count"], r["hearing_id"], r["id"])
            )
            if row["masked_word_count"] <= config.audit_max_words
        ],
    }


def build_report(
    considered: list[Record],
    funnel: list[Record],
    split_source: Record,
    split_of: dict[int, str],
    transcripts: dict[int, str],
    artifact_path: Path,
    rows_written: int,
    elapsed_seconds: float,
    config: QuoteBenchmarkConfig,
) -> Record:
    groups = {name: [name] for name in SPLIT_NAMES} | {"all": list(SPLIT_NAMES)}
    return {
        "benchmark_version": config.benchmark_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "splits": split_source,
        },
        "label_semantics": {
            "target": (
                "silver label: the sentence of the person's own turn that encloses the matched "
                "quote prefix of 6+ words, found by string matching and not validated by a human; "
                "the masked opinion keeps only the text outside quote marks, which may describe "
                "a passage other than the one holding the quote"
            ),
            "task": (
                "rank the candidates (every sentence of the person, per-turn segmentation) for "
                "the masked opinion; a hit is any index in target_indices"
            ),
            "split_use": "rows carry split; the test split must not be used to choose anything",
        },
        "field_definitions": {
            "masked_opinion": (
                "opinion with every span between a configured pair of quote marks removed "
                "together with the marks, then whitespace normalized; for word_bounded pairs "
                "the opening mark must not follow a word character and the closing mark must "
                "not precede one, so an apostrophe inside a word is not a quote mark"
            ),
            "masked_with_quote_characters_left": (
                "kept rows whose masked_opinion still has any character of QUOTE_CHARACTERS, "
                "configured or not"
            ),
            "masked_with_extracted_quote_left": (
                "kept rows whose masked_opinion still contains a quote extracted by the "
                "pipeline's quote patterns"
            ),
            "quote_characters_left_without_word_bounded_pairs_kept": (
                "kept rows that would still have a quote character if the mask used only "
                "masking.quote_pairs, which measures what the word_bounded pairs remove"
            ),
            "masked_word_count": "whitespace tokens of masked_opinion with a word character",
            "removed_quote_spans": "number of quoted spans removed by the mask",
            "target.start_char": (
                "first match of the target text inside the target turn of the transcript"
            ),
            "target_indices": "candidate indices whose sentence agrees with target.text",
            "verbatim_overlap": (
                "longest run of consecutive shared tokens (lowercase, accents stripped, \\w+) "
                "between the target and the original or the masked opinion"
            ),
            "random_acc_at_1": "mean over rows of len(target_indices) / n_candidates",
            "splits.*.opinions": (
                "nested counts over every LDS opinion: resolved person, then at least one "
                "extracted quote, then a matched prefix of any length, then a trusted prefix"
            ),
            "splits.*.considered": "opinions with a trusted quote, before the drop rules",
        },
        "pipeline": PIPELINE,
        "quote_characters": "".join(sorted(QUOTE_CHARACTERS)),
        "min_masked_words_audit": min_words_audit(considered, config),
        "splits": {
            name: summarize_group(
                [row for row in considered if row["split"] in members],
                [entry for entry in funnel if entry["split"] in members],
                {hearing_id for hearing_id, split in split_of.items() if split in members},
                transcripts,
                config,
            )
            for name, members in groups.items()
        },
        "dropped_rows": [
            {
                "id": row["id"],
                "split": row["split"],
                "reasons": row["drop_reasons"],
                "masked_word_count": row["masked_word_count"],
            }
            for row in considered
            if row["drop_reasons"]
        ],
        "artifact": {
            "path": str(artifact_path),
            "rows": rows_written,
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_of_file(artifact_path),
        },
        "code": {
            "utils/udv_pipeline.py": sha256_of_file(Path(udv_pipeline.__file__)),
            "utils/build_quote_benchmark.py": sha256_of_file(Path(__file__)),
        },
        "timing": {"elapsed_seconds": round(elapsed_seconds, 1)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "config": config.source,
    }


def print_summary(report: Record) -> None:
    for name, summary in report["splits"].items():
        print(
            f"{name:10s} {summary['hearings']:3d} hearings "
            f"considered={summary['considered']:3d} kept={summary['kept']:3d} "
            f"dropped={summary['dropped']['by_reason']} "
            f"median_candidates={summary['candidates_kept'].get('median')} "
            f"random_acc@1={summary['random_acc_at_1']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the masked-quote benchmark: opinions with their quotes removed, "
        "ranked against the sentences of the person who was quoted."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/quote_benchmark.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    considered, funnel = build_benchmark(lds, split_of, config)
    rows = [
        {key: value for key, value in row.items() if key != "drop_reasons"}
        for row in considered
        if not row["drop_reasons"]
    ]
    artifact_path = config.output_dir / f"{config.benchmark_version}.jsonl"
    write_jsonl(rows, artifact_path)
    transcripts = {hearing["id"]: hearing["transcricao"] for hearing in lds}
    report = build_report(
        considered,
        funnel,
        split_source,
        split_of,
        transcripts,
        artifact_path,
        len(rows),
        time.perf_counter() - started,
        config,
    )
    write_json(report, config.output_dir / f"{config.benchmark_version}_report.json")
    print_summary(report)


if __name__ == "__main__":
    main()
