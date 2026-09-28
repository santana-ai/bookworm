import argparse
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, sha256_of_file, write_json, write_jsonl

from experiments.common import transcript
from experiments.common.provenance import source_hashes
from experiments.common.transcript import (
    DOUBLE_QUOTE_PATTERN,
    DOUBLE_QUOTE_PATTERNS,
    QUOTE_PATTERNS,
    SINGLE_QUOTE_PATTERN,
    find_opinion_turn_quote_match,
    find_turn_quote_match,
    is_trusted_quote,
    locate_turn_sentence_span,
    normalize_whitespace,
    quoted_text,
    resolve_person_speech,
    split_into_turns,
    token_jaccard,
)
from experiments.common.udv_run import load_config, load_lds_records

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
CURLY_SINGLE_QUOTE_PATTERN = re.compile(r"‘([^’]{10,})’")
APOSTROPHE_INSIDE_WORD_PATTERN = re.compile(r"\w['’]\w")
SPAN_KINDS = {
    "double": DOUBLE_QUOTE_PATTERN,
    "single": SINGLE_QUOTE_PATTERN,
    "curly_single_not_in_pipeline": CURLY_SINGLE_QUOTE_PATTERN,
}


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


def span_prefix_words(opinion_text: str, turns: list[Record]) -> dict[str, list[Record]]:
    spans: dict[str, list[Record]] = {kind: [] for kind in SPAN_KINDS}
    double_ranges = [match.span() for match in DOUBLE_QUOTE_PATTERN.finditer(opinion_text)]
    for kind, pattern in SPAN_KINDS.items():
        for match in pattern.finditer(opinion_text):
            quote_match = find_turn_quote_match(normalize_whitespace(quoted_text(match)), turns)
            spans[kind].append(
                {
                    "prefix_words": quote_match["words"] if quote_match else None,
                    "inside_double": kind != "double"
                    and any(a < match.start() and match.end() <= b for a, b in double_ranges),
                }
            )
    return spans


def change_kinds(old: Record | None, new: Record | None) -> list[str]:
    old_trusted, new_trusted = is_trusted_quote(old), is_trusted_quote(new)
    if old_trusted != new_trusted:
        return ["enters_quote_found" if new_trusted else "leaves_quote_found"]
    if old is None and new is None:
        return []
    if old is None:
        return ["short_match_appears"]
    if new is None:
        return ["short_match_disappears"]
    prefix = "trusted" if old_trusted else "short"
    if old["prefix"] != new["prefix"]:
        return [f"{prefix}_prefix_changed"]
    if (old["turn_index"], old["start"]) != (new["turn_index"], new["start"]):
        return [f"{prefix}_occurrence_changed"]
    return []


def describe_match(match: Record | None, opinion_text: str) -> Record | None:
    if match is None:
        return None
    return {
        "prefix": match["prefix"],
        "words": match["words"],
        "trusted": is_trusted_quote(match),
        "quote_index": match["quote_index"],
        "turn_index": match["turn_index"],
        "start": match["start"],
        "sentence": match["sentence"],
        "sentence_opinion_jaccard": round(token_jaccard(match["sentence"], opinion_text), 4),
    }


def empty_split_counts() -> Record:
    return {
        "opinions": 0,
        "resolved_opinions": 0,
        "with_double_span": 0,
        "with_single_span": 0,
        "spans": {kind: 0 for kind in SPAN_KINDS},
        "single_spans_inside_double": 0,
        "with_apostrophe_inside_word": 0,
        "span_prefix_words": {kind: Counter() for kind in SPAN_KINDS},
        "trusted_double_only": 0,
        "trusted_all_patterns": 0,
        "trusted_all_patterns_located_in_source_turn": 0,
        "short_double_only": 0,
        "short_all_patterns": 0,
        "change_kinds": Counter(),
    }


def finalize_split_counts(counts: Record) -> Record:
    words = {
        kind: {
            str(key): value
            for key, value in sorted(
                counter.items(), key=lambda item: (item[0] is None, item[0] or 0)
            )
        }
        for kind, counter in counts["span_prefix_words"].items()
    }
    trusted_share = {
        kind: round(
            sum(v for k, v in counter.items() if k is not None and k >= 6) / sum(counter.values()),
            4,
        )
        if counter
        else None
        for kind, counter in counts["span_prefix_words"].items()
    }
    return {
        **counts,
        "span_prefix_words": words,
        "span_trusted_share": trusted_share,
        "change_kinds": dict(counts["change_kinds"]),
    }


def measure(
    hearings: list[Record], split_of: dict[int, str], run_records: dict[str, Record]
) -> tuple[Record, list[Record]]:
    by_split = {name: empty_split_counts() for name in SPLIT_NAMES}
    run_tiers: Counter[str] = Counter()
    changes: list[Record] = []
    for hearing in hearings:
        split = split_of[hearing["id"]]
        counts = by_split[split]
        turns = split_into_turns(hearing["transcricao"])
        for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
            matched_turns = resolve_person_speech(participant, turns)[0]
            for opinion_index, opinion_text in enumerate(participant["opinioes"]):
                counts["opinions"] += 1
                if not matched_turns:
                    continue
                counts["resolved_opinions"] += 1
                spans = span_prefix_words(opinion_text, matched_turns)
                counts["with_double_span"] += bool(spans["double"])
                counts["with_single_span"] += bool(spans["single"])
                for kind, items in spans.items():
                    counts["spans"][kind] += len(items)
                    counts["span_prefix_words"][kind].update(i["prefix_words"] for i in items)
                counts["single_spans_inside_double"] += sum(
                    1 for item in spans["single"] if item["inside_double"]
                )
                counts["with_apostrophe_inside_word"] += bool(
                    APOSTROPHE_INSIDE_WORD_PATTERN.search(opinion_text)
                )
                old = find_opinion_turn_quote_match(
                    opinion_text, matched_turns, DOUBLE_QUOTE_PATTERNS
                )
                new = find_opinion_turn_quote_match(opinion_text, matched_turns, QUOTE_PATTERNS)
                counts["trusted_double_only"] += is_trusted_quote(old)
                counts["trusted_all_patterns"] += is_trusted_quote(new)
                if new is not None and is_trusted_quote(new):
                    counts["trusted_all_patterns_located_in_source_turn"] += (
                        locate_turn_sentence_span(
                            new["sentence"],
                            hearing["transcricao"],
                            matched_turns,
                            new["turn_index"],
                        )
                        is not None
                    )
                counts["short_double_only"] += old is not None and not is_trusted_quote(old)
                counts["short_all_patterns"] += new is not None and not is_trusted_quote(new)
                kinds = change_kinds(old, new)
                counts["change_kinds"].update(kinds)
                if not kinds:
                    continue
                udv_id = f"udv-{hearing['id']}-{person_index}-{opinion_index}"
                record = run_records.get(udv_id)
                run_tier = record["tier"] if record is not None else None
                if "enters_quote_found" in kinds:
                    run_tiers[f"{split}:{run_tier}"] += 1
                changes.append(
                    {
                        "id": udv_id,
                        "split": split,
                        "change": kinds,
                        "run_tier": run_tier,
                        "proposition": opinion_text,
                        "double_only": describe_match(old, opinion_text),
                        "all_patterns": describe_match(new, opinion_text),
                    }
                )
    summary = {
        "by_split": {name: finalize_split_counts(c) for name, c in by_split.items()},
        "entering_quote_found_by_run_tier": dict(sorted(run_tiers.items())),
    }
    return summary, changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure what the single-quote pattern adds to quote matching, per split, "
        "against the double-quote patterns alone, without any encoder."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits/temporal_v1.json"))
    parser.add_argument("--run-name", default="udv_v0", help="existing run to read tiers from")
    parser.add_argument("--output-name", default="quote_patterns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    hearings = load_lds_records(config)
    split_of, split_source = load_split_lookup(args.manifest, config.expected_sha256)
    missing = [hearing["id"] for hearing in hearings if hearing["id"] not in split_of]
    if missing:
        raise SystemExit(f"hearings missing from the split manifest: {missing}")
    run_records = {r["id"]: r for r in load_jsonl(config.output_dir / f"{args.run_name}.jsonl")}
    measured, changes = measure(hearings, split_of, run_records)
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "hearings": len(hearings),
        "splits": split_source,
        "patterns": {
            "double_only": [pattern.pattern for pattern in DOUBLE_QUOTE_PATTERNS],
            "all_patterns": [pattern.pattern for pattern in QUOTE_PATTERNS],
        },
        "field_definitions": {
            "span_prefix_words": (
                "per quoted span, the longest quote prefix (10, 6, 4 or 3 words) found in a "
                "turn of the speaker; None when no prefix is found"
            ),
            "single_spans_inside_double": (
                "single-quoted spans that lie inside a double-quoted span of the same opinion"
            ),
            "curly_single_not_in_pipeline": (
                "spans between ‘ and ’ with 10+ characters, measured only; they are not quote "
                "patterns of the pipeline"
            ),
            "with_apostrophe_inside_word": (
                "resolved opinions with ' or ’ between two word characters, which the word "
                "boundaries of the single-quote pattern keep out of a quote"
            ),
            "change_kinds": (
                "difference between the quote match chosen with the double-quote patterns "
                "alone and with every pattern, per opinion of a resolved person"
            ),
        },
        **measured,
        "code": {
            **source_hashes(transcript, *transcript.SOURCES),
            **source_hashes(Path(__file__)),
        },
        "config": config.source,
    }
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    write_json(summary, config.output_dir / f"{args.output_name}_summary.json")
    write_jsonl(changes, config.output_dir / f"{args.output_name}_changes.jsonl")
    print(f"{len(changes)} opinions change | {summary['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
