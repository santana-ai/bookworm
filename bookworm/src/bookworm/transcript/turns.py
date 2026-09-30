"""Speaker turns, split at the ``O SR.`` and ``A SRA.`` headers."""

import re
from dataclasses import dataclass

TURN_HEADER_PATTERN = re.compile(
    r"(?:O\s+SR\.|A\s+SRA\.)\s*([A-ZÀ-Ü][A-ZÀ-Ü\.\s]{0,60}?)\s*(?:\(([^)]*)\))?\s*-\s?"
)


@dataclass(frozen=True, slots=True)
class Turn:
    turn_index: int
    raw_name: str
    party_info: str
    speech: str
    start_char: int
    end_char: int


def split_into_turns(transcript: str) -> list[Turn]:
    """Split a transcript into speaker turns at the ``O SR.``/``A SRA.`` headers."""
    matches = list(TURN_HEADER_PATTERN.finditer(transcript))
    turns: list[Turn] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        raw_speech = transcript[start:end]
        speech = raw_speech.strip()
        speech_start = start + (len(raw_speech) - len(raw_speech.lstrip()))
        turns.append(
            Turn(
                turn_index=index,
                raw_name=match.group(1).strip(),
                party_info=(match.group(2) or "").strip(),
                speech=speech,
                start_char=speech_start,
                end_char=speech_start + len(speech),
            )
        )
    return turns
