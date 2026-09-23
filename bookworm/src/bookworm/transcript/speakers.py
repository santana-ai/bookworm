from collections.abc import Sequence

from bookworm.transcript.text import normalize_name, normalize_whitespace
from bookworm.transcript.turns import Turn

PARTY_INFO_MARKERS = ("/", " - ")


def is_party_info(text: str) -> bool:
    return any(marker in text for marker in PARTY_INFO_MARKERS)


def resolve_turn_name(turn: Turn) -> str:
    party_info = turn.party_info
    if ". " in party_info:
        head, tail = party_info.rsplit(". ", 1)
        if is_party_info(tail):
            return head.strip()
    return turn.raw_name


def turn_name_candidates(turn: Turn) -> list[str]:
    candidates = [turn.raw_name]
    party_info = turn.party_info
    resolved_name = resolve_turn_name(turn)
    if resolved_name != turn.raw_name:
        candidates.append(resolved_name)
    elif party_info and not is_party_info(party_info):
        candidates.append(party_info)
    return candidates


def names_match(name_a: str, name_b: str) -> bool:
    tokens_a = set(normalize_name(name_a).split())
    tokens_b = set(normalize_name(name_b).split())
    if len(tokens_a) <= 1 or len(tokens_b) <= 1:
        return tokens_a == tokens_b
    return tokens_a <= tokens_b or tokens_b <= tokens_a


def matching_turns(name: str, turns: Sequence[Turn]) -> list[Turn]:
    return [
        turn
        for turn in turns
        if any(names_match(name, candidate) for candidate in turn_name_candidates(turn))
    ]


def single_token_matching_turns(name: str, turns: Sequence[Turn]) -> list[Turn]:
    token = normalize_name(name)
    speakers = {
        normalize_name(candidate)
        for turn in turns
        for candidate in turn_name_candidates(turn)
        if token in normalize_name(candidate).split()
    }
    if len(speakers) != 1:
        return []
    speaker = speakers.pop()
    return [
        turn
        for turn in turns
        if any(normalize_name(candidate) == speaker for candidate in turn_name_candidates(turn))
    ]


def resolve_person_speech(name: str, turns: Sequence[Turn]) -> tuple[list[Turn], str]:
    matched_turns = matching_turns(name, turns)
    if not matched_turns and len(normalize_name(name).split()) == 1:
        matched_turns = single_token_matching_turns(name, turns)
    speech = normalize_whitespace(" ".join(turn.speech for turn in matched_turns))
    return matched_turns, speech
