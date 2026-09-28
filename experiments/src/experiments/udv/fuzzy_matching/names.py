"""Fuzzy name rows: unresolved participants ranked against the hearing's speakers, and the
impostor control of the participants the exact rules resolved."""

from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from experiments.common.reporting import rounded
from experiments.common.transcript import (
    WORD_TOKEN_PATTERN,
    find_opinion_turn_quote_match,
    is_trusted_quote,
    normalize_name,
    turn_name_candidates,
    turn_text,
)
from experiments.udv.fuzzy_matching.config import MAX_SCORE, FuzzyConfig
from experiments.udv.fuzzy_matching.inputs import hearing_people

Record = dict[str, Any]


def normalized_person_name(name: str) -> str:
    return " ".join(WORD_TOKEN_PATTERN.findall(normalize_name(name)))


def hearing_speakers(turns: list[Record]) -> list[Record]:
    speakers: dict[str, Record] = {}
    for turn in turns:
        for candidate in turn_name_candidates(turn):
            key = normalized_person_name(candidate)
            if not key:
                continue
            entry = speakers.setdefault(
                key, {"key": key, "display": candidate, "turn_indices": set(), "party_info": []}
            )
            entry["turn_indices"].add(turn["turn_index"])
            if turn["party_info"] and turn["party_info"] not in entry["party_info"]:
                entry["party_info"].append(turn["party_info"])
    return list(speakers.values())


def name_score(metric: str, name: str, key: str) -> float:
    if metric == "token_set_ratio":
        return float(fuzz.token_set_ratio(name, key))
    return MAX_SCORE * float(JaroWinkler.normalized_similarity(name, key))


def ranked_speakers(name: str, speakers: list[Record], metric: str) -> list[tuple[float, int]]:
    scored = [
        (name_score(metric, name, speaker["key"]), position)
        for position, speaker in enumerate(speakers)
    ]
    return sorted(scored, key=lambda item: (-item[0], item[1]))


def assigned_participants(turn_indices: set[int], people: list[Record], skip: int) -> list[Record]:
    return [
        {"index": person["index"], "name": person["participant"]["nome"]}
        for person in people
        if person["index"] != skip
        and turn_indices & {turn["turn_index"] for turn in person["matched_turns"]}
    ]


def quote_support(opinions: list[str], speaker_turns: list[Record]) -> Record:
    matches = [find_opinion_turn_quote_match(opinion, speaker_turns) for opinion in opinions]
    return {
        "opinions": len(opinions),
        "with_trusted_prefix": sum(1 for match in matches if is_trusted_quote(match)),
        "with_any_prefix": sum(1 for match in matches if match is not None),
    }


def speaker_entry(
    speaker: Record, score: float, turns_by_index: dict[int, Record], config: FuzzyConfig
) -> Record:
    ordered = sorted(speaker["turn_indices"])
    return {
        "speaker": speaker["key"],
        "display": speaker["display"],
        "score": rounded(score),
        "turn_count": len(ordered),
        "turn_indices": ordered,
        "party_info": speaker["party_info"],
        "speech_excerpt": turn_text(turns_by_index[ordered[0]])[: config.speech_excerpt_chars],
    }


def disjoint_rival(
    ranking: list[tuple[float, int]], speakers: list[Record]
) -> tuple[float, int] | None:
    """The best-ranked speaker after the first that shares no turn with it."""
    best_turns = speakers[ranking[0][1]]["turn_indices"]
    return next(
        (
            (score, position)
            for score, position in ranking[1:]
            if not speakers[position]["turn_indices"] & best_turns
        ),
        None,
    )


def metric_result(
    name: str,
    speakers: list[Record],
    person: Record,
    people: list[Record],
    turns_by_index: dict[int, Record],
    config: FuzzyConfig,
    metric: str,
) -> Record | None:
    ranking = ranked_speakers(name, speakers, metric)
    if not ranking:
        return None
    best_score, best_position = ranking[0]
    best = speakers[best_position]
    rival = disjoint_rival(ranking, speakers)
    best_turns = [turns_by_index[index] for index in sorted(best["turn_indices"])]
    return {
        "best": speaker_entry(best, best_score, turns_by_index, config),
        "second": (
            speaker_entry(speakers[rival[1]], rival[0], turns_by_index, config) if rival else None
        ),
        "margin": rounded(best_score - rival[0]) if rival else None,
        "best_assigned_to": assigned_participants(best["turn_indices"], people, person["index"]),
        "best_quote_support": quote_support(person["participant"]["opinioes"], best_turns),
    }


def person_id(hearing_id: int, person: Record) -> str:
    return f"person-{hearing_id}-{person['index']}"


def unresolved_row(
    name: str,
    person: Record,
    people: list[Record],
    speakers: list[Record],
    turns_by_index: dict[int, Record],
    hearing_id: int,
    split: str,
    config: FuzzyConfig,
) -> Record:
    participant = person["participant"]
    return {
        "id": person_id(hearing_id, person),
        "split": split,
        "hearing_id": hearing_id,
        "person_index": person["index"],
        "name": participant["nome"],
        "normalized_name": name,
        "role": participant["cargo"],
        "opinions": participant["opinioes"],
        "speakers_in_hearing": len(speakers),
        "metrics": {
            metric: metric_result(name, speakers, person, people, turns_by_index, config, metric)
            for metric in config.name_metrics
        },
    }


def impostor_row(
    name: str,
    person: Record,
    speakers: list[Record],
    hearing_id: int,
    split: str,
    config: FuzzyConfig,
) -> Record:
    """The participant's best score on its own speakers and on the speakers sharing no turn."""
    own_turns = {turn["turn_index"] for turn in person["matched_turns"]}
    row: Record = {
        "id": person_id(hearing_id, person),
        "split": split,
        "hearing_id": hearing_id,
        "name": person["participant"]["nome"],
    }
    for metric in config.name_metrics:
        own = [
            name_score(metric, name, speaker["key"])
            for speaker in speakers
            if speaker["turn_indices"] & own_turns
        ]
        others = [
            (name_score(metric, name, speaker["key"]), speaker["display"])
            for speaker in speakers
            if not speaker["turn_indices"] & own_turns
        ]
        top_other = max(others, key=lambda item: item[0]) if others else None
        row[metric] = {
            "own_score": rounded(max(own)) if own else None,
            "impostor_score": rounded(top_other[0]) if top_other else None,
            "impostor_speaker": top_other[1] if top_other else None,
        }
    return row


def collect_names(
    hearings: list[Record], split_of: dict[int, str], config: FuzzyConfig
) -> tuple[list[Record], list[Record]]:
    unresolved: list[Record] = []
    impostors: list[Record] = []
    for hearing in hearings:
        hearing_id, split = hearing["id"], split_of[hearing["id"]]
        turns, people = hearing_people(hearing)
        speakers = hearing_speakers(turns)
        turns_by_index = {turn["turn_index"]: turn for turn in turns}
        for person in people:
            name = normalized_person_name(person["participant"]["nome"])
            if person["matched_turns"]:
                impostors.append(impostor_row(name, person, speakers, hearing_id, split, config))
                continue
            unresolved.append(
                unresolved_row(
                    name, person, people, speakers, turns_by_index, hearing_id, split, config
                )
            )
    return unresolved, impostors
