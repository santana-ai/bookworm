import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import FIXTURES_DIR, import_challenge_module

from bookworm import (
    HearingRecord,
    QuoteMatch,
    Span,
    Turn,
    enclosing_sentence,
    extract_quotes,
    find_opinion_quote_match,
    is_trusted_quote,
    load_hearings,
    load_jsonl,
    resolve_person_speech,
    split_into_turns,
    split_sentences,
)
from bookworm.transcript.offsets import locate_turn_sentence_span
from bookworm.transcript.sentences import (
    MIN_SENTENCE_WORDS,
    SENTENCE_BOUNDARY_PATTERN,
    STAGE_DIRECTION_PATTERN,
    split_turn_sentences,
)
from bookworm.transcript.speakers import PARTY_INFO_MARKERS
from bookworm.transcript.turns import TURN_HEADER_PATTERN
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    DOUBLE_QUOTE_PATTERNS,
    QUOTE_PATTERNS,
    WORD_TOKEN_PATTERN,
    TurnQuoteMatch,
    find_opinion_turn_quote_match,
)

pytestmark = pytest.mark.dataset

STAGES = (
    "turns",
    "matched_turns",
    "speech",
    "sentences",
    "sentence_span",
    "concatenated_sentences",
    "extracted_quotes",
    "extracted_double_quotes",
    "turn_quote_match",
    "trusted_quote",
    "quote_sentence_span",
    "concatenated_quote_match",
    "enclosing_sentence",
)
EXAMPLES_PER_STAGE = 5
ABSENT = "<absent>"
HEARINGS = 206
PEOPLE = 1065
OPINIONS = 2203
SENTENCES = 115599


def load_reference(challenge_dir: Path) -> ModuleType:
    return import_challenge_module(challenge_dir, "utils.udv_pipeline")


@dataclass
class DriftReport:
    comparisons: Counter[str] = field(default_factory=Counter)
    mismatches: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def compare(self, stage: str, location: str, ours: object, theirs: object) -> None:
        self.comparisons[stage] += 1
        if ours != theirs:
            self.mismatches[stage] += 1
            stage_examples = self.examples.setdefault(stage, [])
            if len(stage_examples) < EXAMPLES_PER_STAGE:
                stage_examples.append(f"{location}: bookworm={ours!r} reference={theirs!r}")


def turn_as_dict(turn: Turn) -> dict[str, Any]:
    return dataclasses.asdict(turn)


def quote_match_as_dict(match: QuoteMatch | None) -> dict[str, Any] | None:
    return None if match is None else {"prefix": match.prefix, "words": match.words}


def turn_quote_match_as_dict(match: TurnQuoteMatch | None) -> dict[str, Any] | None:
    return None if match is None else dataclasses.asdict(match)


def span_as_dict(span: Span | None) -> dict[str, Any] | None:
    return None if span is None else dataclasses.asdict(span)


@dataclass(frozen=True)
class PersonSide:
    transcript: str
    matched: list[Any]
    speech: str


def compare_hearing(
    report: DriftReport,
    reference: ModuleType,
    hearing: HearingRecord,
    raw_hearing: dict[str, Any],
) -> None:
    transcript = hearing.transcricao
    reference_transcript = raw_hearing["transcricao"]
    turns = split_into_turns(transcript)
    reference_turns = reference.split_into_turns(reference_transcript)
    report.compare(
        "turns", f"hearing {hearing.id}", [turn_as_dict(turn) for turn in turns], reference_turns
    )
    raw_people = raw_hearing["metadados"]["envolvidos"]
    for person_index, (person, raw_person) in enumerate(
        zip(hearing.metadados.envolvidos, raw_people, strict=True)
    ):
        location = f"hearing {hearing.id} person {person_index}"
        matched, speech = resolve_person_speech(person.nome, turns)
        reference_matched, reference_speech = reference.resolve_person_speech(
            raw_person, reference_turns
        )
        report.compare(
            "matched_turns",
            location,
            [turn.turn_index for turn in matched],
            [turn["turn_index"] for turn in reference_matched],
        )
        report.compare("speech", location, speech, reference_speech)
        compare_sentences(
            report,
            reference,
            location,
            PersonSide(transcript, matched, speech),
            PersonSide(reference_transcript, reference_matched, reference_speech),
        )
        for opinion_index, opinion in enumerate(person.opinioes):
            compare_opinion(
                report,
                reference,
                f"{location} opinion {opinion_index}",
                (opinion, raw_person["opinioes"][opinion_index]),
                PersonSide(transcript, matched, speech),
                PersonSide(reference_transcript, reference_matched, reference_speech),
            )


def compare_sentences(
    report: DriftReport,
    reference: ModuleType,
    location: str,
    ours: PersonSide,
    theirs: PersonSide,
) -> None:
    units = split_turn_sentences(ours.matched)
    reference_units = reference.split_turn_sentences(theirs.matched)
    report.compare(
        "sentences",
        location,
        [{"text": unit.text, "turn_index": unit.turn_index} for unit in units],
        reference_units,
    )
    spans = [
        span_as_dict(
            locate_turn_sentence_span(unit.text, ours.transcript, ours.matched, unit.turn_index)
        )
        for unit in units
    ]
    reference_spans = [
        reference.locate_turn_sentence_span(
            unit["text"], theirs.transcript, theirs.matched, unit["turn_index"]
        )
        for unit in reference_units
    ]
    for sentence_index, (span, reference_span) in enumerate(
        zip_longest(spans, reference_spans, fillvalue=ABSENT)
    ):
        report.compare(
            "sentence_span", f"{location} sentence {sentence_index}", span, reference_span
        )
    report.compare(
        "concatenated_sentences",
        location,
        split_sentences(ours.speech) if ours.matched else [],
        reference.split_sentences(theirs.speech) if theirs.matched else [],
    )


def compare_opinion(
    report: DriftReport,
    reference: ModuleType,
    location: str,
    opinions: tuple[str, str],
    ours: PersonSide,
    theirs: PersonSide,
) -> None:
    opinion, raw_opinion = opinions
    report.compare(
        "extracted_quotes", location, extract_quotes(opinion), reference.extract_quotes(raw_opinion)
    )
    report.compare(
        "extracted_double_quotes",
        location,
        extract_quotes(opinion, DOUBLE_QUOTE_PATTERNS),
        reference.extract_quotes(raw_opinion, reference.DOUBLE_QUOTE_PATTERNS),
    )
    match = find_opinion_turn_quote_match(opinion, ours.matched)
    reference_match = reference.find_opinion_turn_quote_match(raw_opinion, theirs.matched)
    report.compare("turn_quote_match", location, turn_quote_match_as_dict(match), reference_match)
    report.compare(
        "trusted_quote",
        location,
        is_trusted_quote(match),
        reference.is_trusted_quote(reference_match),
    )
    if match is not None and reference_match is not None:
        report.compare(
            "quote_sentence_span",
            location,
            span_as_dict(
                locate_turn_sentence_span(
                    match.sentence, ours.transcript, ours.matched, match.turn_index
                )
            ),
            reference.locate_turn_sentence_span(
                reference_match["sentence"],
                theirs.transcript,
                theirs.matched,
                reference_match["turn_index"],
            ),
        )
    concatenated = find_opinion_quote_match(opinion, ours.speech)
    reference_concatenated = reference.find_opinion_quote_match(raw_opinion, theirs.speech)
    report.compare(
        "concatenated_quote_match",
        location,
        quote_match_as_dict(concatenated),
        reference_concatenated,
    )
    if concatenated is not None and reference_concatenated is not None:
        report.compare(
            "enclosing_sentence",
            location,
            enclosing_sentence(concatenated.prefix, ours.speech),
            reference.enclosing_sentence(reference_concatenated["prefix"], theirs.speech),
        )


@pytest.fixture(scope="module")
def reference(challenge_dir: Path, lds_path: Path) -> ModuleType:
    return load_reference(challenge_dir)


@pytest.fixture(scope="module")
def drift_report(
    lds_hearings: list[HearingRecord], lds_path: Path, reference: ModuleType
) -> DriftReport:
    raw_hearings = load_jsonl(lds_path)
    report = DriftReport()
    for hearing, raw_hearing in zip(lds_hearings, raw_hearings, strict=True):
        compare_hearing(report, reference, hearing, raw_hearing)
    print(json.dumps({"comparisons": report.comparisons, "mismatches": report.mismatches}))
    return report


def test_reference_is_the_per_turn_pipeline(reference: ModuleType) -> None:
    assert reference.SENTENCE_BOUNDARY_PATTERN.pattern == SENTENCE_BOUNDARY_PATTERN.pattern
    assert reference.STAGE_DIRECTION_PATTERN.pattern == STAGE_DIRECTION_PATTERN.pattern
    assert reference.TURN_HEADER_PATTERN.pattern == TURN_HEADER_PATTERN.pattern
    assert reference.WORD_TOKEN_PATTERN.pattern == WORD_TOKEN_PATTERN.pattern
    assert [pattern.pattern for pattern in reference.QUOTE_PATTERNS] == [
        pattern.pattern for pattern in QUOTE_PATTERNS
    ]
    assert [pattern.pattern for pattern in reference.DOUBLE_QUOTE_PATTERNS] == [
        pattern.pattern for pattern in DOUBLE_QUOTE_PATTERNS
    ]
    assert reference.PARTY_INFO_MARKERS == PARTY_INFO_MARKERS
    assert reference.MIN_SENTENCE_WORDS == MIN_SENTENCE_WORDS
    policy = DEFAULT_QUOTE_POLICY
    assert (
        policy.prefix_lengths,
        policy.trusted_prefix_words,
    ) == (reference.QUOTE_PREFIX_LENGTHS, reference.TRUSTED_PREFIX_WORDS)


def test_every_hearing_is_compared(drift_report: DriftReport) -> None:
    assert drift_report.comparisons["turns"] == HEARINGS
    assert drift_report.comparisons["matched_turns"] == PEOPLE
    assert drift_report.comparisons["sentences"] == PEOPLE
    assert drift_report.comparisons["sentence_span"] == SENTENCES
    assert drift_report.comparisons["turn_quote_match"] == OPINIONS
    assert drift_report.comparisons["concatenated_quote_match"] == OPINIONS


@pytest.mark.parametrize("stage", STAGES)
def test_stage_has_no_drift(drift_report: DriftReport, stage: str) -> None:
    assert drift_report.comparisons[stage] > 0
    assert drift_report.mismatches[stage] == 0, drift_report.examples.get(stage)


@pytest.mark.parametrize("fixture_name", ["lds_mini.jsonl", "lds_turns.jsonl"])
def test_synthetic_fixtures_have_no_drift(reference: ModuleType, fixture_name: str) -> None:
    path = FIXTURES_DIR / fixture_name
    report = DriftReport()
    for hearing, raw_hearing in zip(load_hearings(path), load_jsonl(path), strict=True):
        compare_hearing(report, reference, hearing, raw_hearing)
    assert report.comparisons["turn_quote_match"] > 0
    assert dict(report.mismatches) == {}, report.examples
