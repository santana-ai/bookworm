import argparse
import importlib.util
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from utils import udv_pipeline
from utils.build_udvs import load_config, load_lds_records
from utils.dataset_io import load_jsonl, sha256_of_file, write_json, write_jsonl
from utils.udv_pipeline import (
    DOUBLE_QUOTE_PATTERNS,
    SENTENCE_BOUNDARY_PATTERN,
    find_opinion_turn_quote_match,
    is_sentence,
    is_trusted_quote,
    locate_sentence_span,
    locate_turn_sentence_span,
    normalize_whitespace,
    resolve_person_speech,
    sentence_part_spans,
    split_into_turns,
    split_turn_sentences,
    token_jaccard,
    turn_text,
)

Record = dict[str, Any]

CLOSING_PARENTHESIS_PATTERN = re.compile(r"\(([^()]*[.!?])\)\s+(\S)")
ELISION = "..."
CONTENT_LABEL_LENGTH = 40
LONG_CONTENT_LABEL = "<longer than 40 characters>"
CONTEXT_CHARS = 60
EXAMPLE_CHARS = 120
BOUNDARY_VARIANTS = {
    "legacy": r"(?<=[.!?])\s+",
    "after_any_closing_parenthesis": r"(?<=[.!?])\s+|(?<=[.!?]\))\s+",
    "after_closing_parenthesis_before_uppercase": r"(?<=[.!?])\s+|(?<=[.!?]\))\s+(?=[A-ZÀ-Ý])",
    "after_closing_parenthesis_except_elision": SENTENCE_BOUNDARY_PATTERN.pattern,
}


def load_legacy_pipeline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("legacy_udv_pipeline", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load legacy pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolved_people(hearings: list[Record]) -> list[Record]:
    people = []
    for hearing in hearings:
        turns = split_into_turns(hearing["transcricao"])
        for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
            matched_turns, speech = resolve_person_speech(participant, turns)
            if matched_turns:
                people.append(
                    {
                        "hearing": hearing,
                        "index": person_index,
                        "participant": participant,
                        "matched_turns": matched_turns,
                        "speech": speech,
                    }
                )
    return people


def joined_segments(person: Record) -> tuple[str, list[Record]]:
    segments = []
    position = 0
    for turn in person["matched_turns"]:
        text = turn_text(turn)
        if not text:
            continue
        segments.append({"turn_index": turn["turn_index"], "start": position, "text": text})
        position += len(text) + 1
    joined = " ".join(segment["text"] for segment in segments)
    if joined != person["speech"]:
        raise SystemExit(f"joined speech differs from resolve_person_speech for {person['index']}")
    return joined, segments


def locate_in_segments(start: int, end: int, segments: list[Record]) -> Record:
    for segment in segments:
        segment_end = segment["start"] + len(segment["text"])
        if segment["start"] <= start < segment_end:
            return {
                "turn_index": segment["turn_index"],
                "start": start - segment["start"],
                "crosses_turns": end > segment_end,
            }
    raise SystemExit(f"position {start} outside every segment")


def split_with(pattern: re.Pattern[str], text: str) -> list[str]:
    parts = (normalize_whitespace(part) for part in pattern.split(text))
    return [part for part in parts if is_sentence(part)]


def per_turn_sentences(pattern: re.Pattern[str], person: Record) -> list[str]:
    return [
        sentence
        for turn in person["matched_turns"]
        for sentence in split_with(pattern, turn_text(turn))
    ]


def is_legacy_sentence(legacy: ModuleType, part: str) -> bool:
    return len(part.split()) >= 4 and not legacy.STAGE_DIRECTION_PATTERN.match(part)


def legacy_crossing_sentences(legacy: ModuleType, person: Record) -> int:
    joined, segments = joined_segments(person)
    joins = [segment["start"] - 1 for segment in segments[1:]]
    crossing = 0
    start = 0
    boundaries = list(legacy.SENTENCE_BOUNDARY_PATTERN.finditer(joined)) + [None]
    for boundary in boundaries:
        end = boundary.start() if boundary is not None else len(joined)
        part = joined[start:end]
        if is_legacy_sentence(legacy, part) and any(start < join < end for join in joins):
            crossing += 1
        if boundary is not None:
            start = boundary.end()
    return crossing


def measure_segmentation(legacy: ModuleType, people: list[Record]) -> Record:
    legacy_pattern = legacy.SENTENCE_BOUNDARY_PATTERN
    counts: Counter[str] = Counter()
    changed_hearings: set[int] = set()
    for person in people:
        old = legacy.split_sentences(person["speech"])
        per_turn_old_boundary = per_turn_sentences(legacy_pattern, person)
        new = [unit["text"] for unit in split_turn_sentences(person["matched_turns"])]
        crossing = legacy_crossing_sentences(legacy, person)
        if old != new:
            changed_hearings.add(person["hearing"]["id"])
        counts["people_resolved"] += 1
        counts["people_with_multiple_turns"] += len(person["matched_turns"]) > 1
        counts["legacy_sentences"] += len(old)
        counts["per_turn_legacy_boundary_sentences"] += len(per_turn_old_boundary)
        counts["new_sentences"] += len(new)
        counts["legacy_crossing_sentences"] += crossing
        counts["people_with_crossing_sentences"] += crossing > 0
        by_split = old != per_turn_old_boundary
        by_boundary = per_turn_old_boundary != new
        counts["people_changed_by_per_turn_split"] += by_split
        counts["people_changed_by_boundary_rule"] += by_boundary
        counts["people_changed_by_both_steps"] += by_split and by_boundary
        counts["people_changed_only_by_per_turn_split"] += by_split and not by_boundary
        counts["people_changed_only_by_boundary_rule"] += by_boundary and not by_split
        counts["people_changed_by_both_steps_but_unchanged_total"] += (
            by_split and by_boundary and old == new
        )
        counts["people_changed_total"] += old != new
        old_counter, new_counter = Counter(old), Counter(new)
        counts["legacy_sentences_gone"] += sum((old_counter - new_counter).values())
        counts["new_sentences_added"] += sum((new_counter - old_counter).values())
    return {**counts, "hearings_with_changed_sentence_lists": len(changed_hearings)}


def unique_resolved_turns(people: list[Record]) -> list[tuple[int, Record]]:
    seen: set[tuple[int, int]] = set()
    turns = []
    for person in people:
        for turn in person["matched_turns"]:
            key = (person["hearing"]["id"], turn["turn_index"])
            if key not in seen:
                seen.add(key)
                turns.append((person["hearing"]["id"], turn))
    return turns


def context_example(hearing_id: int, turn: Record, text: str, hit: re.Match[str]) -> Record:
    return {
        "hearing_id": hearing_id,
        "turn_index": turn["turn_index"],
        "context": text[max(0, hit.start() - CONTEXT_CHARS) : hit.end() + CONTEXT_CHARS],
    }


def character_class(character: str) -> str:
    if character.isupper():
        return "uppercase"
    if character.islower():
        return "lowercase"
    return "other"


def measure_closing_parenthesis(people: list[Record]) -> Record:
    content: Counter[str] = Counter()
    previous: Counter[str] = Counter()
    following: Counter[str] = Counter()
    elision_previous: Counter[str] = Counter()
    elision_following: Counter[str] = Counter()
    elision_examples: list[Record] = []
    unusual_examples: list[Record] = []
    turns = unique_resolved_turns(people)
    for hearing_id, turn in turns:
        text = turn_text(turn)
        for hit in CLOSING_PARENTHESIS_PATTERN.finditer(text):
            inner, next_character = hit.group(1), hit.group(2)
            before = text[: hit.start()].rstrip()[-1:] or "<turn start>"
            content[inner if len(inner) <= CONTENT_LABEL_LENGTH else LONG_CONTENT_LABEL] += 1
            if inner == ELISION:
                elision_previous[before] += 1
                elision_following[character_class(next_character)] += 1
                elision_examples.append(context_example(hearing_id, turn, text, hit))
                continue
            previous[before] += 1
            following[character_class(next_character)] += 1
            if before not in ".!?" or not next_character.isupper():
                unusual_examples.append(context_example(hearing_id, turn, text, hit))
    baseline_pattern = re.compile(BOUNDARY_VARIANTS["legacy"])
    baselines = [per_turn_sentences(baseline_pattern, person) for person in people]
    variants = {}
    for name, source in BOUNDARY_VARIANTS.items():
        pattern = re.compile(source)
        changed = 0
        sentences = 0
        for person, baseline in zip(people, baselines, strict=True):
            new = per_turn_sentences(pattern, person)
            sentences += len(new)
            changed += new != baseline
        variants[name] = {"pattern": source, "sentences": sentences, "people_changed": changed}
    return {
        "unique_resolved_turns": len(turns),
        "occurrences": sum(content.values()),
        "occurrences_by_content": dict(content.most_common()),
        "elision_preceded_by": dict(elision_previous.most_common()),
        "elision_followed_by": dict(elision_following),
        "non_elision_preceded_by": dict(previous.most_common()),
        "non_elision_followed_by": dict(following),
        "elision_examples": elision_examples,
        "non_elision_examples_not_between_punctuation_and_uppercase": unusual_examples,
        "per_turn_variants_per_person": variants,
    }


def legacy_quote_result(legacy: ModuleType, person: Record, opinion_text: str) -> Record | None:
    match = legacy.find_opinion_quote_match(opinion_text, person["speech"])
    if match is None:
        return None
    joined, segments = joined_segments(person)
    hit = legacy.quote_prefix_pattern(match["prefix"]).search(joined)
    location = locate_in_segments(hit.start(), hit.end(), segments)
    return {
        "prefix": match["prefix"],
        "words": match["words"],
        "trusted": legacy.is_trusted_quote(match),
        "text": legacy.enclosing_sentence(match["prefix"], person["speech"]),
        **location,
    }


def sentence_found_earlier(match: Record, turns: list[Record]) -> bool:
    text = next(turn_text(turn) for turn in turns if turn["turn_index"] == match["turn_index"])
    sentence_start = min(
        start
        for start, end in sentence_part_spans(text)
        if start < match["end"] and match["start"] < end
    )
    return text.find(match["sentence"]) != sentence_start


def new_quote_result(person: Record, opinion_text: str) -> Record | None:
    match = find_opinion_turn_quote_match(
        opinion_text, person["matched_turns"], DOUBLE_QUOTE_PATTERNS
    )
    if match is None:
        return None
    span = locate_turn_sentence_span(
        match["sentence"],
        person["hearing"]["transcricao"],
        person["matched_turns"],
        match["turn_index"],
    )
    return {
        "located_in_source_turn": span is not None,
        "sentence_found_earlier_in_turn": sentence_found_earlier(match, person["matched_turns"]),
        "prefix": match["prefix"],
        "words": match["words"],
        "trusted": is_trusted_quote(match),
        "text": match["sentence"],
        "turn_index": match["turn_index"],
        "start": match["start"],
        "quote_index": match["quote_index"],
        "occurrence_count": match["occurrence_count"],
        "occurrence_index": match["occurrence_index"],
    }


def quote_change_kinds(old: Record | None, new: Record | None) -> list[str]:
    old_trusted = old is not None and old["trusted"]
    new_trusted = new is not None and new["trusted"]
    if not old_trusted and not new_trusted:
        return []
    if not old_trusted:
        return ["enters_quote_found"]
    if not new_trusted:
        return ["leaves_quote_found"]
    kinds = []
    if old["prefix"] != new["prefix"]:
        kinds.append("prefix_changed")
    elif (old["turn_index"], old["start"]) != (new["turn_index"], new["start"]):
        kinds.append("occurrence_changed")
    if old["text"] != new["text"]:
        kinds.append("text_changed")
    return kinds


def short_change(old: Record | None, new: Record | None) -> str | None:
    old_short = old is not None and not old["trusted"]
    new_short = new is not None and not new["trusted"]
    if not old_short and not new_short:
        return None
    if old_short != new_short:
        return "short_match_appears_or_disappears"
    if old["prefix"] != new["prefix"]:
        return "short_prefix_changed"
    if old["text"] != new["text"]:
        return "short_sentence_changed"
    return None


def with_overlap(result: Record | None, opinion_text: str) -> Record | None:
    if result is None:
        return None
    return {**result, "text_opinion_jaccard": round(token_jaccard(result["text"], opinion_text), 4)}


def measure_quotes(
    legacy: ModuleType, people: list[Record], run_records: dict[str, Record]
) -> tuple[Record, list[Record]]:
    counts: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    short_changes: Counter[str] = Counter()
    changes: list[Record] = []
    for person in people:
        for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"]):
            udv_id = f"udv-{person['hearing']['id']}-{person['index']}-{opinion_index}"
            old = legacy_quote_result(legacy, person, opinion_text)
            new = new_quote_result(person, opinion_text)
            counts["opinions"] += 1
            counts["legacy_trusted"] += old is not None and old["trusted"]
            counts["new_trusted"] += new is not None and new["trusted"]
            counts["legacy_short"] += old is not None and not old["trusted"]
            counts["new_short"] += new is not None and not new["trusted"]
            counts["legacy_match_crossing_turns"] += old is not None and old["crosses_turns"]
            if new is not None and new["trusted"]:
                counts["new_trusted_multiple_occurrences"] += new["occurrence_count"] > 1
                counts["new_trusted_non_first_occurrence"] += new["occurrence_index"] > 0
                counts["new_trusted_non_first_quote"] += new["quote_index"] > 0
                counts["new_trusted_located_in_source_turn"] += new["located_in_source_turn"]
                counts["new_trusted_sentence_found_earlier_in_turn"] += new[
                    "sentence_found_earlier_in_turn"
                ]
            record = run_records.get(udv_id)
            evidence = record["evidence"] if record is not None else None
            if evidence is not None and evidence["support_type"] == "direct_quote":
                counts["run_direct_quote"] += 1
                counts["run_direct_quote_reproduced_by_legacy"] += (
                    old is not None
                    and old["trusted"]
                    and old["prefix"] == evidence["quote_prefix"]
                    and old["text"] == evidence["text"]
                )
            short_kind = short_change(old, new)
            if short_kind is not None:
                short_changes[short_kind] += 1
            change = quote_change_kinds(old, new)
            kinds.update(change)
            if short_kind is not None and short_kind not in change:
                change = [*change, short_kind]
            if change:
                changes.append(
                    {
                        "id": udv_id,
                        "run_tier": record["tier"] if record is not None else None,
                        "change": change,
                        "proposition": opinion_text,
                        "legacy": with_overlap(old, opinion_text),
                        "new": with_overlap(new, opinion_text),
                    }
                )
    summary = {
        "new_quote_patterns": [pattern.pattern for pattern in DOUBLE_QUOTE_PATTERNS],
        **counts,
        "trusted_change_kinds": dict(kinds),
        "short_changes": dict(short_changes),
    }
    return summary, changes


def raw_turn_sentences(turn: Record) -> list[Record]:
    units = []
    for start, end in sentence_part_spans(turn["speech"]):
        text = normalize_whitespace(turn["speech"][start:end])
        if is_sentence(text):
            units.append(
                {
                    "text": text,
                    "start_char": turn["start_char"] + start,
                    "end_char": turn["start_char"] + end,
                }
            )
    return units


def earlier_position_kind(span: Record, unit: Record, units: list[Record]) -> str:
    at_start = next((other for other in units if other["start_char"] == span["start_char"]), None)
    if at_start is not None:
        if at_start["text"] == unit["text"]:
            return "at_identical_sentence_start"
        if at_start["text"].startswith(unit["text"]):
            return "at_longer_sentence_start"
        return "at_other_sentence_start"
    if any(other["start_char"] < span["start_char"] < other["end_char"] for other in units):
        return "inside_candidate_sentence"
    return "outside_candidate_sentences"


def offset_example(person: Record, turn: Record, unit: Record, span: Record, kind: str) -> Record:
    transcript = person["hearing"]["transcricao"]
    return {
        "kind": kind,
        "hearing_id": person["hearing"]["id"],
        "turn_index": turn["turn_index"],
        "sentence": unit["text"],
        "found_at": normalize_whitespace(
            transcript[span["start_char"] : span["start_char"] + EXAMPLE_CHARS]
        ),
    }


def measure_offset_search(people: list[Record]) -> Record:
    counts: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    examples: list[Record] = []
    for person in people:
        transcript = person["hearing"]["transcricao"]
        for turn in person["matched_turns"]:
            units = raw_turn_sentences(turn)
            for unit in units:
                span = locate_sentence_span(unit["text"], transcript, [turn])
                counts["sentences"] += 1
                if span is None:
                    counts["not_found_in_source_turn"] += 1
                elif span["start_char"] != unit["start_char"]:
                    kind = earlier_position_kind(span, unit, units)
                    counts["earlier_position"] += 1
                    counts["earlier_position_at_sentence_start"] += kind.startswith("at_")
                    kinds[kind] += 1
                    if kind != "at_identical_sentence_start":
                        examples.append(offset_example(person, turn, unit, span, kind))
    return {
        "sentences_searched_per_person": counts.pop("sentences", 0),
        **counts,
        "earlier_position_by_kind": dict(kinds),
        "earlier_position_examples_not_identical": examples,
    }


def measure_run(people: list[Record], run_records: list[Record]) -> Record:
    by_key = {(p["hearing"]["id"], p["index"]): p for p in people}
    sentences_by_key = {
        key: {unit["text"] for unit in split_turn_sentences(person["matched_turns"])}
        for key, person in by_key.items()
    }
    counts: Counter[str] = Counter()
    outside: list[str] = []
    stale: list[str] = []
    for record in run_records:
        evidence = record["evidence"]
        if evidence is None:
            continue
        _, hearing_id, person_index, _ = record["id"].split("-")
        key = (int(hearing_id), int(person_index))
        person = by_key[key]
        counts["evidences"] += 1
        counts["null_offsets"] += evidence["start_char"] is None
        if not any(evidence["text"] in turn_text(t) for t in person["matched_turns"]):
            outside.append(record["id"])
        if evidence["support_type"] != "direct_quote":
            counts["semantic"] += 1
            if evidence["text"] not in sentences_by_key[key]:
                stale.append(record["id"])
    return {
        **counts,
        "evidence_not_in_single_actor_turn": outside,
        "semantic_text_not_a_new_candidate_sentence": len(stale),
        "semantic_text_not_a_new_candidate_sentence_examples": stale[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-turn sentence segmentation and quote matching against a legacy "
        "udv_pipeline, without any encoder."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    parser.add_argument(
        "--legacy-pipeline", type=Path, default=Path("../backup/udv_pipeline_2026-09-21.py")
    )
    parser.add_argument("--run-name", default="udv_v0", help="existing run to cross-check")
    parser.add_argument("--output-name", default="turn_segmentation")
    parser.add_argument("--skip-offset-search", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    legacy = load_legacy_pipeline(args.legacy_pipeline)
    hearings = load_lds_records(config)
    people = resolved_people(hearings)
    run_records = load_jsonl(config.output_dir / f"{args.run_name}.jsonl")
    quotes, changes = measure_quotes(legacy, people, {r["id"]: r for r in run_records})
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "legacy_pipeline": str(args.legacy_pipeline),
        "run_name": args.run_name,
        "hearings": len(hearings),
        "sentence_boundary_pattern": {
            "legacy": legacy.SENTENCE_BOUNDARY_PATTERN.pattern,
            "new": SENTENCE_BOUNDARY_PATTERN.pattern,
        },
        "segmentation": measure_segmentation(legacy, people),
        "closing_parenthesis": measure_closing_parenthesis(people),
        "quotes": quotes,
        "existing_run": measure_run(people, run_records),
        "offset_search": None if args.skip_offset_search else measure_offset_search(people),
        "config": config.source,
    }
    summary["code"] = {
        str(args.legacy_pipeline): sha256_of_file(args.legacy_pipeline),
        "utils/udv_pipeline.py": sha256_of_file(Path(udv_pipeline.__file__)),
        "utils/measure_turn_segmentation.py": sha256_of_file(Path(__file__)),
    }
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    write_json(summary, config.output_dir / f"{args.output_name}_summary.json")
    write_jsonl(changes, config.output_dir / f"{args.output_name}_quote_changes.jsonl")
    print(f"{len(changes)} quote changes | {summary['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
