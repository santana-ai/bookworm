from pathlib import Path

import pytest
from conftest import udv_artifact_path

from bookworm import HearingRecord, UdvRecord, load_udv_jsonl, resolve_hearing_people
from bookworm.transcript.offsets import sentence_pattern
from bookworm.transcript.sentences import sentence_part_spans, turn_text
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.build import PersonSpeech
from bookworm.udv.quotes import find_opinion_turn_quote_match, find_prefix_occurrences

pytestmark = pytest.mark.dataset

PersonKey = tuple[int, int]

REPEATED_PREFIXES = {
    "udv-23-0-1": ("semantic_with_short_quote", 2, 2, 0),
    "udv-28-0-0": ("semantic_with_short_quote", 2, 1, 0),
    "udv-44-3-0": ("direct_quote", 2, 1, 0),
    "udv-65-1-0": ("direct_quote", 2, 2, 1),
    "udv-68-1-2": ("semantic_with_short_quote", 2, 2, 0),
    "udv-71-4-0": ("semantic_with_short_quote", 2, 1, 0),
    "udv-101-0-0": ("semantic_with_short_quote", 2, 1, 0),
    "udv-103-3-0": ("semantic_with_short_quote", 3, 1, 0),
    "udv-106-2-0": ("semantic_with_short_quote", 2, 1, 0),
    "udv-150-2-0": ("semantic_with_short_quote", 3, 1, 0),
    "udv-168-0-4": ("semantic_with_short_quote", 2, 1, 0),
    "udv-192-0-1": ("direct_quote", 2, 1, 0),
    "udv-193-4-0": ("semantic_with_short_quote", 4, 1, 0),
    "udv-199-0-2": ("semantic_with_short_quote", 2, 2, 0),
    "udv-206-0-1": ("semantic_with_short_quote", 2, 1, 0),
}


@pytest.fixture(scope="module")
def people(lds_hearings: list[HearingRecord]) -> dict[PersonKey, PersonSpeech]:
    return {
        (hearing.id, person.index): person
        for hearing in lds_hearings
        for person in resolve_hearing_people(hearing)
    }


@pytest.fixture(scope="module")
def records(udv_artifacts_dir: Path) -> list[UdvRecord]:
    return load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, "udv_v1.jsonl"))


def person_of(record: UdvRecord, people: dict[PersonKey, PersonSpeech]) -> PersonSpeech:
    return people[(record.hearing_id, int(record.id.split("-")[2]))]


def test_recorded_quote_prefixes_that_occur_more_than_once(
    records: list[UdvRecord], people: dict[PersonKey, PersonSpeech]
) -> None:
    repeated: dict[str, tuple[str, int, int, int]] = {}
    for record in records:
        evidence = record.evidence
        if evidence is None or evidence.quote_prefix is None:
            continue
        person = person_of(record, people)
        occurrences = find_prefix_occurrences(evidence.quote_prefix, person.matched_turns)
        if len(occurrences) > 1:
            match = find_opinion_turn_quote_match(record.proposition, person.matched_turns)
            assert match is not None
            repeated[record.id] = (
                evidence.support_type,
                len(occurrences),
                len({occurrence.turn_index for occurrence in occurrences}),
                match.occurrence_index,
            )
    assert repeated == REPEATED_PREFIXES


def test_every_evidence_is_the_only_occurrence_between_sentence_boundaries(
    records: list[UdvRecord],
    people: dict[PersonKey, PersonSpeech],
    lds_hearings: list[HearingRecord],
) -> None:
    transcripts = {hearing.id: hearing.transcricao for hearing in lds_hearings}
    checked = 0
    for record in records:
        evidence = record.evidence
        if evidence is None:
            continue
        assert evidence.start_char is not None
        transcript = transcripts[record.hearing_id]
        turn = next(
            turn
            for turn in person_of(record, people).matched_turns
            if turn.turn_index == evidence.speaker_turn
        )
        hits = sentence_pattern(evidence.text).findall(transcript, turn.start_char, turn.end_char)
        assert len(hits) == 1, record.id
        text = turn_text(turn)
        before = normalize_whitespace(transcript[turn.start_char : evidence.start_char])
        start = len(before) + (1 if before else 0)
        end = start + len(evidence.text)
        parts = sentence_part_spans(text)
        assert start in {part_start for part_start, _ in parts}, record.id
        assert end in {part_end for _, part_end in parts}, record.id
        assert text[start:end] == evidence.text, record.id
        checked += 1
    assert checked == 2105
