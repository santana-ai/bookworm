"""Sentence boundaries, per-turn sentences and the sentence that encloses a span."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from bookworm.transcript.text import normalize_whitespace, prefix_pattern
from bookworm.transcript.turns import Turn

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[.!?]\))(?<!\(\.\.\.\))\s+")
STAGE_DIRECTION_PATTERN = re.compile(r"^(?:\([^()]*\)\s*)+$")
MIN_SENTENCE_WORDS = 4


@dataclass(frozen=True, slots=True)
class TurnSentence:
    text: str
    turn_index: int


def is_sentence(part: str) -> bool:
    return len(part.split()) >= MIN_SENTENCE_WORDS and not STAGE_DIRECTION_PATTERN.match(part)


def split_sentences(speech: str) -> list[str]:
    parts = SENTENCE_BOUNDARY_PATTERN.split(speech)
    return [normalize_whitespace(part) for part in parts if is_sentence(part)]


def turn_text(turn: Turn) -> str:
    return normalize_whitespace(turn.speech)


def split_turn_sentences(turns: Sequence[Turn]) -> list[TurnSentence]:
    """Candidate sentences of each turn, tagged with the index of their turn."""
    return [
        TurnSentence(text=sentence, turn_index=turn.turn_index)
        for turn in turns
        for sentence in split_sentences(turn_text(turn))
    ]


def sentence_part_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(text)))
    return spans


def sentences_agree(sentence: str, other_sentence: str) -> bool:
    return sentence == other_sentence or sentence in other_sentence or other_sentence in sentence


def enclosing_turn_sentence(text: str, start: int, end: int) -> str:
    covered = [text[a:b] for a, b in sentence_part_spans(text) if a < end and start < b]
    return normalize_whitespace(" ".join(covered))


def enclosing_sentence(prefix: str, speech: str) -> str:
    speech = normalize_whitespace(speech)
    hit = prefix_pattern(prefix).search(speech)
    if hit is None:
        raise ValueError(f"prefix not found in speech: {prefix!r}")
    return enclosing_turn_sentence(speech, hit.start(), hit.end())
