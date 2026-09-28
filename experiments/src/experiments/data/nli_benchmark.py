import argparse
import bisect
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

import sklearn
from bookworm import load_gated_jsonl, sha256_of_file, write_json, write_jsonl
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from experiments.common import transcript
from experiments.common.provenance import source_hashes
from experiments.common.transcript import (
    normalize_name,
    resolve_person_speech,
    resolve_turn_name,
    split_into_turns,
)

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
JUDGE_KEY_PATTERN = re.compile(r"^prompt_(\d+)_(.+)$")
WHITESPACE_MODES = (("exact_whitespace", r"\s+"), ("whitespace_inserted", r"\s*"))
POLARITIES = ("inferable", "not_inferable")
CLASS_ORDER = (True, False)
CLASS_NAMES = {True: "inferable", False: "not_inferable"}
PAPER_CELLS = {
    "valid_as_valid": (True, True),
    "valid_as_hallucination": (True, False),
    "hallucination_as_valid": (False, True),
    "hallucination_as_hallucination": (False, False),
}
NOT_LOCATED_REASONS = ("empty_chunk", "hearing_not_in_lds", "not_found", "segment_not_found")


@dataclass(frozen=True)
class NliBenchmarkConfig:
    lds_path: Path
    lds_sha256: str
    nli_path: Path
    nli_sha256: str
    manifest_path: Path
    manual_field: str
    judge_field: str
    true_means: str
    paper_manual_counts: dict[str, int]
    paper_cell_order: list[str]
    paper_confusion: dict[str, list[int]]
    scopes: dict[str, list[str]]
    include_chunk_text: bool
    top_unreachable_speakers: int
    benchmark_version: str
    seed: int
    output_dir: Path
    source: Record = field(default_factory=dict)


@dataclass(frozen=True)
class HearingTurns:
    transcript: str
    turns: list[Record]
    starts: list[int]


def load_config(config_path: Path) -> NliBenchmarkConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    labels = raw["labels"]
    confusion = dict(labels["paper_confusion"])
    cell_order = confusion.pop("cell_order")
    if labels["true_means"] not in POLARITIES:
        raise SystemExit(f"labels.true_means must be one of {POLARITIES}")
    if sorted(cell_order) != sorted(PAPER_CELLS):
        raise SystemExit(f"labels.paper_confusion.cell_order must list {sorted(PAPER_CELLS)}")
    scopes = raw["evaluation"]["scopes"]
    for split_names in scopes.values():
        if not set(split_names) <= set(SPLIT_NAMES):
            raise SystemExit(f"evaluation scope {split_names} names an unknown split")
    return NliBenchmarkConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        nli_path=Path(raw["dataset"]["nli_path"]),
        nli_sha256=raw["dataset"]["nli_sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        manual_field=labels["manual_field"],
        judge_field=labels["judge_field"],
        true_means=labels["true_means"],
        paper_manual_counts=dict(labels["paper_manual_counts"]),
        paper_cell_order=cell_order,
        paper_confusion=confusion,
        scopes=scopes,
        include_chunk_text=raw["output"]["include_chunk_text"],
        top_unreachable_speakers=raw["output"]["top_unreachable_speakers"],
        benchmark_version=raw["run"]["benchmark_version"],
        seed=raw["run"]["seed"],
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


def iter_opinions(nli: list[Record]) -> list[tuple[Record, int, Record, int, Record]]:
    return [
        (hearing, person_index, person, opinion_index, opinion)
        for hearing in nli
        for person_index, person in enumerate(hearing["metadados_extraidos"]["envolvidos"])
        for opinion_index, opinion in enumerate(person["opinioes"])
    ]


def check_judge_keys(nli: list[Record], config: NliBenchmarkConfig) -> list[str]:
    expected = list(config.paper_confusion)
    for hearing, person_index, _, opinion_index, opinion in iter_opinions(nli):
        found = set(opinion["verificacao_alucinacao"]) - {config.manual_field}
        if found != set(expected):
            raise SystemExit(
                f"hearing {hearing['id']} person {person_index} opinion {opinion_index}: "
                f"judge keys {sorted(found)} differ from the configured {sorted(expected)}"
            )
    return expected


def to_inferable(value: Any, true_means: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise SystemExit(f"{where}: expected a bool, found {value!r}")
    return not value if true_means == "not_inferable" else value


def parse_judge_key(key: str) -> Record:
    match = JUDGE_KEY_PATTERN.match(key)
    if match is None:
        return {"prompt": None, "model": key}
    return {"prompt": int(match.group(1)), "model": match.group(2)}


def chunk_pattern(text: str, separator: str) -> re.Pattern[str]:
    return re.compile(separator.join(re.escape(token) for token in text.split()))


def find_occurrences(text: str, transcript: str) -> tuple[list[re.Match[str]], str | None]:
    for mode, separator in WHITESPACE_MODES:
        matches = list(chunk_pattern(text, separator).finditer(transcript))
        if matches:
            return matches, mode
    return [], None


def is_contiguous(text: str, transcript: str) -> bool:
    return any(
        chunk_pattern(text, separator).search(transcript) is not None
        for _, separator in WHITESPACE_MODES
    )


def split_into_segments(chunk: str, transcript: str) -> list[str]:
    segments: list[str] = []
    for line in (line for line in chunk.split("\n") if line.strip()):
        if segments and is_contiguous(f"{segments[-1]}\n{line}", transcript):
            segments[-1] = f"{segments[-1]}\n{line}"
        else:
            segments.append(line)
    return segments


def index_turns(transcript: str) -> HearingTurns:
    turns = split_into_turns(transcript)
    return HearingTurns(transcript, turns, [turn["start_char"] for turn in turns])


def overlapping_turns(start: int, end: int, hearing: HearingTurns) -> list[Record]:
    position = max(bisect.bisect_right(hearing.starts, start) - 1, 0)
    found: list[Record] = []
    while position < len(hearing.turns) and hearing.turns[position]["start_char"] < end:
        if hearing.turns[position]["end_char"] > start:
            found.append(hearing.turns[position])
        position += 1
    return found


def covered_turn_indices(match: re.Match[str], hearing: HearingTurns) -> list[int]:
    return [turn["turn_index"] for turn in overlapping_turns(match.start(), match.end(), hearing)]


def build_segment(
    chosen: re.Match[str],
    matches: list[re.Match[str]],
    mode: str | None,
    out_of_order: bool,
    hearing: HearingTurns,
) -> Record:
    covered = overlapping_turns(chosen.start(), chosen.end(), hearing)
    return {
        "start_char": chosen.start(),
        "end_char": chosen.end(),
        "turn_index": covered[0]["turn_index"] if covered else None,
        "speaker_name": resolve_turn_name(covered[0]) if covered else None,
        "turn_indices": [turn["turn_index"] for turn in covered],
        "match_mode": mode,
        "match_count": len(matches),
        "occurrence_turn_indices": [covered_turn_indices(match, hearing) for match in matches],
        "out_of_order": out_of_order,
    }


def not_located(reason: str) -> Record:
    return {"located": False, "structure": None, "reason": reason, "segments": []}


def locate_chunk(chunk: str, hearing: HearingTurns | None) -> Record:
    if hearing is None:
        return not_located("hearing_not_in_lds")
    if not chunk.split():
        return not_located("empty_chunk")
    matches, mode = find_occurrences(chunk, hearing.transcript)
    if matches:
        segment = build_segment(matches[0], matches, mode, False, hearing)
        return {"located": True, "structure": "contiguous", "reason": None, "segments": [segment]}
    segments = split_into_segments(chunk, hearing.transcript)
    if len(segments) <= 1:
        return not_located("not_found")
    located: list[Record] = []
    cursor = 0
    for segment_text in segments:
        matches, mode = find_occurrences(segment_text, hearing.transcript)
        if not matches:
            return not_located("segment_not_found")
        following = [match for match in matches if match.start() >= cursor]
        chosen = (following or matches)[0]
        located.append(build_segment(chosen, matches, mode, not following, hearing))
        cursor = chosen.end()
    return {"located": True, "structure": "multi_segment", "reason": None, "segments": located}


def inside_person(turn_indices: list[int], person_turns: set[int]) -> bool:
    return bool(turn_indices) and set(turn_indices) <= person_turns


def segment_reachable(segment: Record, person_turns: set[int]) -> bool:
    return any(
        inside_person(turn_indices, person_turns)
        for turn_indices in segment["occurrence_turn_indices"]
    )


def is_reachable(location: Record, person_turns: set[int]) -> bool:
    return location["located"] and all(
        segment_reachable(segment, person_turns) for segment in location["segments"]
    )


def is_reachable_at_chosen(location: Record, person_turns: set[int]) -> bool:
    return location["located"] and all(
        inside_person(segment["turn_indices"], person_turns) for segment in location["segments"]
    )


def build_chunk_record(
    position: int, text: str, location: Record, person_turns: set[int], include_text: bool
) -> Record:
    record: Record = {"position": position}
    if include_text:
        record["text"] = text
    record.update(location)
    record["reachable"] = is_reachable(location, person_turns)
    record["reachable_at_chosen_occurrence"] = is_reachable_at_chosen(location, person_turns)
    return record


def resolve_people(hearing: Record, turns: HearingTurns | None) -> list[Record]:
    people = []
    for person_index, participant in enumerate(hearing["metadados_extraidos"]["envolvidos"]):
        matched_turns = resolve_person_speech(participant, turns.turns)[0] if turns else []
        people.append(
            {
                "hearing_id": hearing["id"],
                "index": person_index,
                "name": participant["nome"],
                "opinions": len(participant["opinioes"]),
                "turn_indices": [turn["turn_index"] for turn in matched_turns],
            }
        )
    return people


def build_row(
    hearing_id: int,
    split: str,
    person: Record,
    participant: Record,
    opinion_index: int,
    opinion: Record,
    locations: dict[str, Record],
    judge_keys: list[str],
    config: NliBenchmarkConfig,
) -> Record:
    row_id = f"nli-{hearing_id}-{person['index']}-{opinion_index}"
    verification = opinion["verificacao_alucinacao"]
    person_turns = set(person["turn_indices"])
    chunks = [
        build_chunk_record(position, text, locations[text], person_turns, config.include_chunk_text)
        for position, text in enumerate(opinion["chunks_proximos"])
    ]
    return {
        "id": row_id,
        "hearing_id": hearing_id,
        "split": split,
        "person": {
            "index": person["index"],
            "name": participant["nome"],
            "role": participant["cargo"],
        },
        "opinion": opinion["opiniao"],
        "label_inferable": to_inferable(
            verification[config.manual_field], config.true_means, f"{row_id} manual"
        ),
        "judge_inferable": {
            key: to_inferable(verification[key][config.judge_field], config.true_means, row_id)
            for key in judge_keys
        },
        "person_resolution": {
            "resolved": bool(person_turns),
            "turn_indices": person["turn_indices"],
        },
        "chunks": chunks,
        "chunk_summary": {
            "count": len(chunks),
            "distinct": len(set(opinion["chunks_proximos"])),
            "located": sum(1 for chunk in chunks if chunk["located"]),
            "reachable": sum(1 for chunk in chunks if chunk["reachable"]),
        },
    }


def build_benchmark(
    nli: list[Record],
    lds: dict[int, Record],
    split_of: dict[int, str],
    judge_keys: list[str],
    config: NliBenchmarkConfig,
) -> tuple[list[Record], list[Record], list[Record]]:
    rows: list[Record] = []
    all_people: list[Record] = []
    distinct_locations: list[Record] = []
    for hearing in nli:
        if hearing["id"] not in split_of:
            raise SystemExit(f"hearing {hearing['id']} is not in the split manifest")
        lds_record = lds.get(hearing["id"])
        turns = index_turns(lds_record["transcricao"]) if lds_record else None
        people = resolve_people(hearing, turns)
        split = split_of[hearing["id"]]
        all_people.extend({**person, "split": split} for person in people)
        locations: dict[str, Record] = {}
        for person, participant in zip(
            people, hearing["metadados_extraidos"]["envolvidos"], strict=True
        ):
            for opinion_index, opinion in enumerate(participant["opinioes"]):
                for text in opinion["chunks_proximos"]:
                    if text not in locations:
                        locations[text] = locate_chunk(text, turns)
                        distinct_locations.append(locations[text])
                rows.append(
                    build_row(
                        hearing["id"],
                        split,
                        person,
                        participant,
                        opinion_index,
                        opinion,
                        locations,
                        judge_keys,
                        config,
                    )
                )
    return rows, all_people, distinct_locations


def paper_check(rows: list[Record], judge_keys: list[str], config: NliBenchmarkConfig) -> Record:
    manual = Counter(CLASS_NAMES[row["label_inferable"]] for row in rows)
    manual_computed = {name: manual[name] for name in POLARITIES}
    judges: Record = {}
    for key in judge_keys:
        cells = Counter((row["label_inferable"], row["judge_inferable"][key]) for row in rows)
        computed = [cells[PAPER_CELLS[name]] for name in config.paper_cell_order]
        expected = config.paper_confusion[key]
        judges[key] = {"expected": expected, "computed": computed, "match": computed == expected}
    manual_match = manual_computed == config.paper_manual_counts
    return {
        "scope": "all opinions, every split; a check of the label encoding, not a selection",
        "cell_order": config.paper_cell_order,
        "manual": {
            "expected": config.paper_manual_counts,
            "computed": manual_computed,
            "match": manual_match,
        },
        "judges": judges,
        "all_match": manual_match and all(judge["match"] for judge in judges.values()),
    }


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def judge_metrics(y_true: list[bool], y_pred: list[bool]) -> Record:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_ORDER), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_ORDER))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "per_class": {
            CLASS_NAMES[label]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(CLASS_ORDER)
        },
        "confusion": {
            "inferable_as_inferable": int(matrix[0, 0]),
            "inferable_as_not_inferable": int(matrix[0, 1]),
            "not_inferable_as_inferable": int(matrix[1, 0]),
            "not_inferable_as_not_inferable": int(matrix[1, 1]),
        },
        "predicted_not_inferable": int(matrix[0, 1] + matrix[1, 1]),
    }


def evaluate_judges(
    rows: list[Record], judge_keys: list[str], config: NliBenchmarkConfig
) -> Record:
    result: Record = {}
    for scope_name, split_names in config.scopes.items():
        scoped = [row for row in rows if row["split"] in split_names]
        y_true = [row["label_inferable"] for row in scoped]
        inferable = sum(y_true)
        result[scope_name] = {
            "splits": split_names,
            "opinions": len(scoped),
            "label_counts": {"inferable": inferable, "not_inferable": len(scoped) - inferable},
            "always_inferable_accuracy": rate(inferable, len(scoped)),
            "judges": {
                key: {
                    **parse_judge_key(key),
                    **judge_metrics(y_true, [row["judge_inferable"][key] for row in scoped]),
                }
                for key in judge_keys
            },
        }
    return result


def has_mixed_speakers(chunk: Record) -> bool:
    return len({normalize_name(s["speaker_name"] or "") for s in chunk["segments"]}) > 1


def summarize_chunks(chunks: list[Record]) -> Record:
    located = [chunk for chunk in chunks if chunk["located"]]
    segments = [segment for chunk in located for segment in chunk["segments"]]
    return {
        "total": len(chunks),
        "located": len(located),
        "location_rate": rate(len(located), len(chunks)),
        "by_structure": dict(sorted(Counter(chunk["structure"] for chunk in located).items())),
        "not_located_by_reason": {
            reason: sum(1 for chunk in chunks if chunk["reason"] == reason)
            for reason in NOT_LOCATED_REASONS
        },
        "segments": len(segments),
        "segments_by_match_mode": dict(
            sorted(Counter(segment["match_mode"] for segment in segments).items())
        ),
        "segments_per_multi_segment_chunk": dict(
            sorted(
                Counter(
                    len(chunk["segments"])
                    for chunk in located
                    if chunk["structure"] == "multi_segment"
                ).items()
            )
        ),
        "chunks_with_ambiguous_segment": sum(
            1 for chunk in located if any(s["match_count"] > 1 for s in chunk["segments"])
        ),
        "chunks_with_out_of_order_segment": sum(
            1 for chunk in located if any(s["out_of_order"] for s in chunk["segments"])
        ),
        "chunks_with_segment_outside_turns": sum(
            1 for chunk in located if any(not s["turn_indices"] for s in chunk["segments"])
        ),
        "chunks_with_segment_crossing_turns": sum(
            1 for chunk in located if any(len(s["turn_indices"]) > 1 for s in chunk["segments"])
        ),
        "chunks_with_mixed_speakers": sum(1 for chunk in located if has_mixed_speakers(chunk)),
        "chunks_with_mixed_speakers_and_unique_segments": sum(
            1
            for chunk in located
            if has_mixed_speakers(chunk) and all(s["match_count"] == 1 for s in chunk["segments"])
        ),
    }


def count_opinion_reachability(rows: list[Record]) -> Record:
    summaries = [row["chunk_summary"] for row in rows]
    return {
        "opinions": len(rows),
        "all_chunks_reachable": sum(1 for s in summaries if s["reachable"] == s["count"]),
        "any_chunk_reachable": sum(1 for s in summaries if s["reachable"] > 0),
        "no_chunk_reachable": sum(1 for s in summaries if s["reachable"] == 0),
    }


def summarize_reachability(rows: list[Record]) -> Record:
    chunk_pairs = [(row, chunk) for row in rows for chunk in row["chunks"] if chunk["located"]]
    resolved_pairs = [
        (row, chunk) for row, chunk in chunk_pairs if row["person_resolution"]["resolved"]
    ]
    with_chunks = [row for row in rows if row["chunks"]]
    reachable = sum(1 for _, chunk in chunk_pairs if chunk["reachable"])
    at_chosen = sum(1 for _, chunk in chunk_pairs if chunk["reachable_at_chosen_occurrence"])
    return {
        "located_chunks": len(chunk_pairs),
        "reachable_chunks": reachable,
        "reachable_rate": rate(reachable, len(chunk_pairs)),
        "reachable_at_chosen_occurrence_chunks": at_chosen,
        "reachable_at_chosen_occurrence_rate": rate(at_chosen, len(chunk_pairs)),
        "located_chunks_of_resolved_people": len(resolved_pairs),
        "reachable_rate_among_resolved_people": rate(
            sum(1 for _, chunk in resolved_pairs if chunk["reachable"]), len(resolved_pairs)
        ),
        "opinions_with_chunks": count_opinion_reachability(with_chunks),
        "opinions_with_chunks_by_label": {
            CLASS_NAMES[label]: count_opinion_reachability(
                [row for row in with_chunks if row["label_inferable"] is label]
            )
            for label in CLASS_ORDER
        },
    }


def summarize_split(rows: list[Record], people: list[Record], hearing_ids: set[int]) -> Record:
    speaking = [person for person in people if person["opinions"] > 0]
    inferable = sum(1 for row in rows if row["label_inferable"])
    resolved_speaking = sum(1 for person in speaking if person["turn_indices"])
    return {
        "hearings": len(hearing_ids),
        "people": {
            "total": len(people),
            "with_opinions": len(speaking),
            "resolved": sum(1 for person in people if person["turn_indices"]),
            "resolved_with_opinions": resolved_speaking,
            "resolution_rate_with_opinions": rate(resolved_speaking, len(speaking)),
        },
        "opinions": {
            "total": len(rows),
            "inferable": inferable,
            "not_inferable": len(rows) - inferable,
            "not_inferable_share": rate(len(rows) - inferable, len(rows)),
            "with_resolved_person": sum(1 for row in rows if row["person_resolution"]["resolved"]),
            "without_chunks": sum(1 for row in rows if not row["chunks"]),
            "with_repeated_chunks": sum(
                1
                for row in rows
                if row["chunk_summary"]["distinct"] < row["chunk_summary"]["count"]
            ),
        },
        "chunks": summarize_chunks([chunk for row in rows for chunk in row["chunks"]]),
        "reachability": summarize_reachability(rows),
    }


def top_pairs(pairs: Counter[tuple[str, str]], limit: int) -> list[Record]:
    return [
        {"person": person, "chunk_speaker": speaker, "chunks": count}
        for (person, speaker), count in pairs.most_common(limit)
    ]


def unreachable_diagnostics(rows: list[Record], limit: int) -> Record:
    elsewhere: Counter[tuple[str, str]] = Counter()
    unresolved: Counter[tuple[str, str]] = Counter()
    for row in rows:
        person_turns = set(row["person_resolution"]["turn_indices"])
        for chunk in row["chunks"]:
            if not chunk["located"] or chunk["reachable"]:
                continue
            target = elsewhere if row["person_resolution"]["resolved"] else unresolved
            for segment in chunk["segments"]:
                if segment_reachable(segment, person_turns):
                    continue
                speaker = segment["speaker_name"] or "<outside turns>"
                target[(row["person"]["name"], speaker)] += 1
    return {
        "method": (
            "segments with no occurrence inside the person's resolved turns, counted per "
            "(NLI person, speaker of the chosen occurrence)"
        ),
        "resolved_person_segment_elsewhere": {
            "segments": sum(elsewhere.values()),
            "top": top_pairs(elsewhere, limit),
        },
        "unresolved_person_segments": {
            "segments": sum(unresolved.values()),
            "top": top_pairs(unresolved, limit),
        },
    }


def build_report(
    rows: list[Record],
    people: list[Record],
    distinct_locations: list[Record],
    judge_keys: list[str],
    check: Record,
    split_source: Record,
    split_of: dict[int, str],
    artifact_path: Path,
    elapsed_seconds: float,
    config: NliBenchmarkConfig,
) -> Record:
    groups = {name: [name] for name in SPLIT_NAMES} | {"all": list(SPLIT_NAMES)}
    return {
        "benchmark_version": config.benchmark_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "nli": {"path": str(config.nli_path), "sha256": config.nli_sha256},
            "splits": split_source,
        },
        "label_semantics": {
            "label_inferable": (
                "True when the annotator judged the opinion inferable from the four retrieved "
                "chunks; it says nothing about text outside those chunks"
            ),
            "source_fields": {"manual": config.manual_field, "judges": config.judge_field},
            "source_true_means": config.true_means,
            "polarity_source": config.source["labels"]["polarity_source"],
            "paper_check": check,
        },
        "field_definitions": {
            "chunks[].segments": (
                "a chunk found contiguously has one segment; otherwise it is split at line breaks "
                "and consecutive lines are merged while they stay contiguous in the transcript"
            ),
            "segments[].match_mode": (
                "exact_whitespace joins chunk tokens with one or more whitespace characters; "
                "whitespace_inserted also allows none, for places where the transcript has no "
                "space after punctuation"
            ),
            "segments[].start_char": (
                "first occurrence for the first segment; for later segments, the first "
                "occurrence after the previous segment; match_count > 1 marks an ambiguous choice"
            ),
            "chunks[].reachable": (
                "every segment text occurs at least once entirely inside turns resolved for the "
                "NLI person, so a search restricted to that person's speech can return it"
            ),
            "chunks[].reachable_at_chosen_occurrence": (
                "same test applied only to the occurrence recorded in start_char/end_char"
            ),
            "chunks[].text": (
                "written only when output.include_chunk_text is true; otherwise the chunk text is "
                "chunks_proximos[position] of the same opinion in the gated NLI file, where the "
                "row id nli-{hearing}-{person}-{opinion} indexes metadados_extraidos.envolvidos "
                "and opinioes"
            ),
        },
        "splits": {
            name: summarize_split(
                [row for row in rows if row["split"] in members],
                [person for person in people if person["split"] in members],
                {hearing_id for hearing_id, split in split_of.items() if split in members},
            )
            for name, members in groups.items()
        },
        "distinct_chunks_per_hearing": summarize_chunks(distinct_locations),
        "unreachable_diagnostics": unreachable_diagnostics(rows, config.top_unreachable_speakers),
        "judge_evaluation": evaluate_judges(rows, judge_keys, config),
        "artifact": {
            "path": str(artifact_path),
            "rows": len(rows),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_of_file(artifact_path),
            "include_chunk_text": config.include_chunk_text,
        },
        "code": {
            **source_hashes(transcript, *transcript.SOURCES),
            **source_hashes(Path(__file__)),
        },
        "timing": {"elapsed_seconds": round(elapsed_seconds, 1)},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "config": config.source,
    }


def print_summary(report: Record) -> None:
    for name, summary in report["splits"].items():
        opinions, chunks = summary["opinions"], summary["chunks"]
        reach = summary["reachability"]
        print(
            f"{name:10s} {summary['hearings']:3d} hearings {opinions['total']:4d} opinions "
            f"not_inferable={opinions['not_inferable']:3d} "
            f"located={chunks['located']}/{chunks['total']} "
            f"people={summary['people']['resolution_rate_with_opinions']} "
            f"reachable={reach['reachable_rate']}"
        )
    for scope, evaluation in report["judge_evaluation"].items():
        print(f"[{scope}] {evaluation['opinions']} opinions")
        for key, metrics in evaluation["judges"].items():
            print(
                f"  {key:34s} acc={metrics['accuracy']:.4f} kappa={metrics['cohen_kappa']:.4f} "
                f"f1_not_inferable={metrics['per_class']['not_inferable']['f1']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the NLI benchmark: one row per NLI opinion, chunks located in the LDS."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/nli_benchmark.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    lds = {record["id"]: record for record in load_gated_jsonl(config.lds_path, config.lds_sha256)}
    nli = load_gated_jsonl(config.nli_path, config.nli_sha256)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    judge_keys = check_judge_keys(nli, config)
    rows, people, distinct_locations = build_benchmark(nli, lds, split_of, judge_keys, config)
    check = paper_check(rows, judge_keys, config)
    if not check["all_match"]:
        print(json.dumps(check, indent=2))
        raise SystemExit("label polarity does not reproduce the paper; benchmark not written")
    artifact_path = config.output_dir / f"{config.benchmark_version}.jsonl"
    write_jsonl(rows, artifact_path)
    report = build_report(
        rows,
        people,
        distinct_locations,
        judge_keys,
        check,
        split_source,
        split_of,
        artifact_path,
        time.perf_counter() - started,
        config,
    )
    write_json(report, config.output_dir / f"{config.benchmark_version}_report.json")
    print_summary(report)


if __name__ == "__main__":
    main()
