"""One pass over each transcript for the UDVs and, optionally, the actor speeches."""

import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from bookworm.actors.config import ActorsConfig
from bookworm.actors.schemas import UdvActorLink
from bookworm.actors.speeches import ActorCollector, ActorSpeeches
from bookworm.data.schemas import HearingRecord
from bookworm.features.encoders import CachedEncoder
from bookworm.transcript.turns import split_into_turns
from bookworm.udv.build import (
    EvidenceSettings,
    HearingProgress,
    PersonSpeech,
    UdvRun,
    build_hearing_udvs,
    udv_id,
)


@dataclass(frozen=True, slots=True)
class PendingLink:
    udv_id: str
    hearing_id: int
    matched_turns: int
    turn_keys: tuple[str, ...]


@dataclass
class PipelineRun:
    udv: UdvRun
    actors: ActorSpeeches | None = None
    links: list[UdvActorLink] = field(default_factory=list)


def pending_links(
    hearing_id: int, people: Sequence[PersonSpeech], kept_keys: Mapping[int, str]
) -> list[PendingLink]:
    return [
        PendingLink(
            udv_id=udv_id(hearing_id, person.index, opinion_index),
            hearing_id=hearing_id,
            matched_turns=len(person.matched_turns),
            turn_keys=tuple(
                kept_keys[turn.turn_index]
                for turn in person.matched_turns
                if turn.turn_index in kept_keys
            ),
        )
        for person in people
        for opinion_index in range(len(person.participant.opinioes))
    ]


def resolve_link(pending: PendingLink, display_names: Mapping[str, str]) -> UdvActorLink:
    counts = Counter(pending.turn_keys)
    if not counts:
        return UdvActorLink(
            udv_id=pending.udv_id,
            hearing_id=pending.hearing_id,
            actor_key=None,
            actor=None,
            matched_turns=pending.matched_turns,
            linked_turns=0,
        )
    key, linked = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return UdvActorLink(
        udv_id=pending.udv_id,
        hearing_id=pending.hearing_id,
        actor_key=key,
        actor=display_names[key],
        matched_turns=pending.matched_turns,
        linked_turns=linked,
    )


def run_pipeline(
    hearings: Sequence[HearingRecord],
    encoder: CachedEncoder,
    settings: EvidenceSettings,
    actors_config: ActorsConfig | None = None,
    *,
    on_hearing: HearingProgress | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PipelineRun:
    """Split each transcript once and build UDVs, actor speeches and links from it."""
    udv_run = UdvRun(hearings=tuple(hearings))
    collector = None if actors_config is None else ActorCollector(actors_config)
    pending: list[PendingLink] = []
    for number, hearing in enumerate(hearings, start=1):
        started = clock()
        turns = split_into_turns(hearing.transcricao)
        records, people = build_hearing_udvs(hearing, encoder, settings, turns)
        if collector is not None:
            kept_keys = collector.add_hearing(hearing.id, hearing.transcricao, turns)
            pending.extend(pending_links(hearing.id, people, kept_keys))
        udv_run.hearing_seconds.append(clock() - started)
        udv_run.records.extend(records)
        udv_run.people.extend(people)
        if on_hearing is not None:
            on_hearing(number, hearing, len(records), udv_run.hearing_seconds[-1])
    if collector is None:
        return PipelineRun(udv=udv_run)
    actors = collector.finish()
    names = actors.display_names()
    return PipelineRun(
        udv=udv_run, actors=actors, links=[resolve_link(link, names) for link in pending]
    )
