import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from experiments.common.transcript import (
    locate_turn_sentence_span,
    normalize_whitespace,
    resolve_person_speech,
    split_into_turns,
    split_sentences,
    turn_text,
)
from experiments.common.udv_run import resolve_hearing_people

Record = dict[str, Any]

UNIT_KINDS = ("sentence", "window2", "window3", "turn")
BENCHES = ("masked_quotes", "nli")
SPLIT_NAMES = ("train", "validation", "test")
HEARING_CONTEXT = "all"


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    turn_index: int
    position_in_turn: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Unit:
    unit_id: str
    turn_index: int
    sentence_positions: tuple[int, ...]
    text: str
    start_char: int
    end_char: int


@dataclass
class SpeakerContext:
    key: str
    hearing_id: int
    turn_indices: tuple[int, ...]
    sentences: list[SentenceSpan]
    units: dict[str, list[Unit]]


@dataclass(frozen=True)
class Query:
    query_id: str
    bench: str
    split: str
    hearing_id: int
    context_key: str
    text: str
    relevant: dict[str, tuple[int, ...]]


@dataclass
class HearingData:
    hearing_id: int
    split: str
    transcript: str
    turns: list[Record]
    turn_sentences: dict[int, list[SentenceSpan]]
    people: list[Record]
    contexts: dict[str, SpeakerContext] = field(default_factory=dict)
    span_checks: Counter[str] = field(default_factory=Counter)


def sentence_pattern(text: str) -> re.Pattern[str]:
    return re.compile(r"\s+".join(re.escape(token) for token in text.split()))


def locate_turn_sentences(
    turn: Record, transcript: str, checks: Counter[str]
) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    cursor = turn["start_char"]
    for position, text in enumerate(split_sentences(turn_text(turn))):
        match = sentence_pattern(text).search(transcript, cursor, turn["end_char"])
        if match is None:
            raise SystemExit(
                f"sentence not found after its predecessor in turn {turn['turn_index']}"
            )
        if normalize_whitespace(transcript[match.start() : match.end()]) != text:
            raise SystemExit(f"sentence span text differs in turn {turn['turn_index']}")
        first = locate_turn_sentence_span(text, transcript, [turn], turn["turn_index"])
        checks["sentences"] += 1
        if first is None or first["start_char"] != match.start():
            checks["first_match_differs_from_sequential"] += 1
        spans.append(SentenceSpan(text, turn["turn_index"], position, match.start(), match.end()))
        cursor = match.end()
    return spans


def load_hearing(record: Record, split: str) -> HearingData:
    transcript = record["transcricao"]
    turns = split_into_turns(transcript)
    checks: Counter[str] = Counter()
    turn_sentences = {
        turn["turn_index"]: locate_turn_sentences(turn, transcript, checks) for turn in turns
    }
    return HearingData(
        hearing_id=record["id"],
        split=split,
        transcript=transcript,
        turns=turns,
        turn_sentences=turn_sentences,
        people=resolve_hearing_people(record),
        span_checks=checks,
    )


def sentence_range_id(turn_index: int, first: int, last: int) -> str:
    if first == last:
        return f"t{turn_index}.s{first}"
    return f"t{turn_index}.s{first}-{last}"


def window_units(turn_spans: list[SentenceSpan], offset: int, size: int) -> list[Unit]:
    count = len(turn_spans)
    width = min(size, count)
    units = []
    for first in range(count - width + 1):
        members = turn_spans[first : first + width]
        units.append(
            Unit(
                unit_id=sentence_range_id(members[0].turn_index, first, first + width - 1),
                turn_index=members[0].turn_index,
                sentence_positions=tuple(range(offset + first, offset + first + width)),
                text=" ".join(member.text for member in members),
                start_char=members[0].start_char,
                end_char=members[-1].end_char,
            )
        )
    return units


def turn_unit(turn: Record, turn_spans: list[SentenceSpan], offset: int) -> Unit:
    return Unit(
        unit_id=f"t{turn['turn_index']}",
        turn_index=turn["turn_index"],
        sentence_positions=tuple(range(offset, offset + len(turn_spans))),
        text=turn_text(turn),
        start_char=turn["start_char"],
        end_char=turn["end_char"],
    )


def build_units(
    hearing: HearingData, turn_indices: tuple[int, ...], window_sizes: dict[str, int]
) -> tuple[list[SentenceSpan], dict[str, list[Unit]]]:
    turns_by_index = {turn["turn_index"]: turn for turn in hearing.turns}
    sentences: list[SentenceSpan] = []
    units: dict[str, list[Unit]] = {kind: [] for kind in UNIT_KINDS}
    for turn_index in turn_indices:
        turn_spans = hearing.turn_sentences[turn_index]
        if not turn_spans:
            continue
        offset = len(sentences)
        sentences.extend(turn_spans)
        units["sentence"].extend(window_units(turn_spans, offset, 1))
        for kind, size in window_sizes.items():
            units[kind].extend(window_units(turn_spans, offset, size))
        units["turn"].append(turn_unit(turns_by_index[turn_index], turn_spans, offset))
    return sentences, units


def context_key(hearing_id: int, turn_indices: tuple[int, ...] | None) -> str:
    if turn_indices is None:
        return f"{hearing_id}:{HEARING_CONTEXT}"
    return f"{hearing_id}:" + ",".join(str(index) for index in turn_indices)


def get_context(
    hearing: HearingData, turn_indices: tuple[int, ...] | None, window_sizes: dict[str, int]
) -> SpeakerContext:
    key = context_key(hearing.hearing_id, turn_indices)
    if key not in hearing.contexts:
        indices = (
            tuple(turn["turn_index"] for turn in hearing.turns)
            if turn_indices is None
            else turn_indices
        )
        sentences, units = build_units(hearing, indices, window_sizes)
        hearing.contexts[key] = SpeakerContext(key, hearing.hearing_id, indices, sentences, units)
    return hearing.contexts[key]


def overlaps(unit: Unit, spans: list[tuple[int, int]]) -> bool:
    return any(unit.start_char < end and start < unit.end_char for start, end in spans)


def relevant_by_sentences(context: SpeakerContext, targets: set[int]) -> dict[str, tuple[int, ...]]:
    return {
        kind: tuple(
            position
            for position, unit in enumerate(units)
            if targets.intersection(unit.sentence_positions)
        )
        for kind, units in context.units.items()
    }


def relevant_by_spans(
    context: SpeakerContext, spans: list[tuple[int, int]]
) -> dict[str, tuple[int, ...]]:
    return {
        kind: tuple(position for position, unit in enumerate(units) if overlaps(unit, spans))
        for kind, units in context.units.items()
    }


def masked_quote_query(
    row: Record, hearing: HearingData, window_sizes: dict[str, int], checks: Counter[str]
) -> Query | None:
    person = hearing.people[row["person"]["index"]]
    turn_indices = tuple(turn["turn_index"] for turn in person["matched_turns"])
    context = get_context(hearing, turn_indices, window_sizes)
    checks["rows"] += 1
    if [span.text for span in context.sentences] != row["candidates"]:
        checks["candidates_mismatch"] += 1
        return None
    if [span.turn_index for span in context.sentences] != row["candidate_turns"]:
        checks["candidate_turns_mismatch"] += 1
        return None
    if person["sentences"] != row["candidates"]:
        checks["pipeline_sentences_mismatch"] += 1
        return None
    targets = set(row["target_indices"])
    if not targets:
        checks["no_target"] += 1
        return None
    target = row["target"]
    if target["start_char"] is not None:
        target_span = [(target["start_char"], target["end_char"])]
        hits = [context.units["sentence"][index] for index in sorted(targets)]
        if not any(overlaps(unit, target_span) for unit in hits):
            checks["target_span_outside_target_sentences"] += 1
    else:
        checks["target_span_missing"] += 1
    checks["kept"] += 1
    return Query(
        query_id=row["id"],
        bench="masked_quotes",
        split=row["split"],
        hearing_id=row["hearing_id"],
        context_key=context.key,
        text=row["masked_opinion"],
        relevant=relevant_by_sentences(context, targets),
    )


def located_spans(row: Record) -> list[tuple[int, int]]:
    return [
        (segment["start_char"], segment["end_char"])
        for chunk in row["chunks"]
        if chunk["located"]
        for segment in chunk["segments"]
    ]


def nli_query(
    row: Record, hearing: HearingData, window_sizes: dict[str, int], checks: Counter[str]
) -> Query | None:
    checks["rows"] += 1
    if not row["label_inferable"]:
        checks["dropped_not_inferable"] += 1
        return None
    resolution = row["person_resolution"]
    if not resolution["resolved"]:
        checks["dropped_person_not_resolved"] += 1
        return None
    spans = located_spans(row)
    if not spans:
        checks["dropped_no_located_chunk"] += 1
        return None
    turn_indices = tuple(resolution["turn_indices"])
    rebuilt = resolve_person_speech({"nome": row["person"]["name"]}, hearing.turns)[0]
    if tuple(turn["turn_index"] for turn in rebuilt) != turn_indices:
        checks["turn_resolution_mismatch"] += 1
        return None
    for start, end in spans:
        if not normalize_whitespace(hearing.transcript[start:end]):
            checks["empty_chunk_span"] += 1
    located = [chunk for chunk in row["chunks"] if chunk["located"]]
    checks["located_chunks"] += len(located)
    checks["located_chunks_chosen_occurrence_outside_person"] += sum(
        1 for chunk in located if not chunk["reachable_at_chosen_occurrence"]
    )
    context = get_context(hearing, turn_indices, window_sizes)
    if not context.sentences:
        checks["dropped_person_without_sentences"] += 1
        return None
    relevant = relevant_by_spans(context, spans)
    if not relevant["sentence"]:
        checks["dropped_no_relevant_sentence_unit"] += 1
        if relevant["turn"]:
            checks["dropped_no_relevant_sentence_unit_but_relevant_turn"] += 1
        return None
    lds_turn_sets = {
        tuple(turn["turn_index"] for turn in person["matched_turns"]) for person in hearing.people
    }
    checks["kept_turn_set_equals_lds_person"] += int(turn_indices in lds_turn_sets)
    checks["kept"] += 1
    return Query(
        query_id=row["id"],
        bench="nli",
        split=row["split"],
        hearing_id=row["hearing_id"],
        context_key=context.key,
        text=row["opinion"],
        relevant=relevant,
    )


def build_queries(
    bench: str,
    rows: list[Record],
    hearing: HearingData,
    window_sizes: dict[str, int],
    checks: Counter[str],
) -> list[Query]:
    builder = masked_quote_query if bench == "masked_quotes" else nli_query
    queries = [builder(row, hearing, window_sizes, checks) for row in rows]
    return [query for query in queries if query is not None]
