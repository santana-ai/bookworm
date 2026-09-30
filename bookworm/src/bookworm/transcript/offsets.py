"""Character span of a sentence in the turn it comes from."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from bookworm.transcript.turns import Turn


@dataclass(frozen=True, slots=True)
class Span:
    start_char: int
    end_char: int
    speaker_turn: int


def sentence_pattern(sentence: str) -> re.Pattern[str]:
    return re.compile(r"\s+".join(re.escape(token) for token in sentence.split()))


def locate_sentence_span(sentence: str, transcript: str, turns: Sequence[Turn]) -> Span | None:
    pattern = sentence_pattern(sentence)
    for turn in turns:
        match = pattern.search(transcript, turn.start_char, turn.end_char)
        if match is not None:
            return Span(
                start_char=match.start(),
                end_char=match.end(),
                speaker_turn=turn.turn_index,
            )
    return None


def locate_turn_sentence_span(
    sentence: str, transcript: str, turns: Sequence[Turn], turn_index: int
) -> Span | None:
    source_turns = [turn for turn in turns if turn.turn_index == turn_index]
    return locate_sentence_span(sentence, transcript, source_turns)
