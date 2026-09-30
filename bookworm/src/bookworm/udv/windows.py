"""Candidate windows of consecutive sentences within a turn (``semantic_unit``)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from bookworm.transcript.offsets import Span, sentence_pattern
from bookworm.transcript.sentences import split_sentences, turn_text
from bookworm.transcript.text import normalize_whitespace
from bookworm.transcript.turns import Turn

SemanticUnit = Literal["sentence", "window2", "window3"]
SEMANTIC_UNITS: tuple[SemanticUnit, ...] = get_args(SemanticUnit)
UNIT_SIZES: dict[SemanticUnit, int] = {"sentence": 1, "window2": 2, "window3": 3}


@dataclass(frozen=True, slots=True)
class LocatedSentence:
    text: str
    turn_index: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class CandidateUnit:
    text: str
    evidence_text: str
    turn_index: int
    sentence_positions: tuple[int, ...]
    span: Span


def unit_size(unit: SemanticUnit) -> int:
    return UNIT_SIZES[unit]


def embedding_label(unit: SemanticUnit, hearing_id: int) -> str:
    return f"{unit}s_{hearing_id}"


def locate_turn_sentences(turn: Turn, transcript: str) -> list[LocatedSentence]:
    located: list[LocatedSentence] = []
    cursor = turn.start_char
    for text in split_sentences(turn_text(turn)):
        match = sentence_pattern(text).search(transcript, cursor, turn.end_char)
        if match is None:
            raise ValueError(f"sentence not found after its predecessor in turn {turn.turn_index}")
        located.append(LocatedSentence(text, turn.turn_index, match.start(), match.end()))
        cursor = match.end()
    return located


def turn_windows(
    sentences: Sequence[LocatedSentence], transcript: str, size: int, offset: int
) -> list[CandidateUnit]:
    width = min(size, len(sentences))
    units: list[CandidateUnit] = []
    for first in range(len(sentences) - width + 1):
        members = sentences[first : first + width]
        start, end = members[0].start_char, members[-1].end_char
        units.append(
            CandidateUnit(
                text=" ".join(member.text for member in members),
                evidence_text=normalize_whitespace(transcript[start:end]),
                turn_index=members[0].turn_index,
                sentence_positions=tuple(range(offset + first, offset + first + width)),
                span=Span(start_char=start, end_char=end, speaker_turn=members[0].turn_index),
            )
        )
    return units


def person_units(turns: Sequence[Turn], transcript: str, size: int) -> list[CandidateUnit]:
    units: list[CandidateUnit] = []
    offset = 0
    for turn in turns:
        sentences = locate_turn_sentences(turn, transcript)
        if not sentences:
            continue
        units.extend(turn_windows(sentences, transcript, size, offset))
        offset += len(sentences)
    return units
