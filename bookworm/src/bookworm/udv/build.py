import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from bookworm.data.schemas import HearingRecord, Participant
from bookworm.features.encoders import CachedEncoder
from bookworm.transcript.sentences import split_turn_sentences
from bookworm.transcript.speakers import resolve_person_speech
from bookworm.transcript.turns import Turn, split_into_turns
from bookworm.udv.evidence import (
    build_quote_evidence,
    build_semantic_evidence,
    classify_tier,
    provenance_for,
)
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    QuotePolicy,
    find_opinion_turn_quote_match,
    is_trusted_quote,
)
from bookworm.udv.schemas import Actor, Evidence, Method, UdvRecord

HearingProgress = Callable[[int, HearingRecord, int, float], None]


@dataclass(frozen=True, slots=True)
class PersonSpeech:
    index: int
    participant: Participant
    matched_turns: tuple[Turn, ...]
    speech: str
    sentences: tuple[str, ...]
    sentence_turns: tuple[int, ...]

    @property
    def resolved(self) -> bool:
        return bool(self.matched_turns)


@dataclass(frozen=True, slots=True)
class EvidenceSettings:
    embedding_threshold: float | int
    quote_policy: QuotePolicy = DEFAULT_QUOTE_POLICY


@dataclass
class UdvRun:
    hearings: tuple[HearingRecord, ...]
    records: list[UdvRecord] = field(default_factory=list)
    people: list[PersonSpeech] = field(default_factory=list)
    hearing_seconds: list[float] = field(default_factory=list)


def udv_id(hearing_id: int, person_index: int, opinion_index: int) -> str:
    return f"udv-{hearing_id}-{person_index}-{opinion_index}"


def select_hearings(
    hearings: Sequence[HearingRecord], limit: int | None, ids: Iterable[int] | None
) -> list[HearingRecord]:
    if ids is not None:
        wanted = set(ids)
        return [hearing for hearing in hearings if hearing.id in wanted]
    if limit is not None:
        return list(hearings[:limit])
    return list(hearings)


def resolve_hearing_people(hearing: HearingRecord) -> list[PersonSpeech]:
    turns = split_into_turns(hearing.transcricao)
    people: list[PersonSpeech] = []
    for person_index, participant in enumerate(hearing.metadados.envolvidos):
        matched_turns, speech = resolve_person_speech(participant.nome, turns)
        units = split_turn_sentences(matched_turns)
        people.append(
            PersonSpeech(
                index=person_index,
                participant=participant,
                matched_turns=tuple(matched_turns),
                speech=speech,
                sentences=tuple(unit.text for unit in units),
                sentence_turns=tuple(unit.turn_index for unit in units),
            )
        )
    return people


def udv_corpus(hearings: Iterable[HearingRecord]) -> list[str]:
    corpus: list[str] = []
    for hearing in hearings:
        people = resolve_hearing_people(hearing)
        corpus.extend(sentence for person in people for sentence in person.sentences)
        corpus.extend(opinion for person in people for opinion in person.participant.opinioes)
    return corpus


def sentence_slices_by_person(people: Sequence[PersonSpeech]) -> dict[int, slice]:
    slices: dict[int, slice] = {}
    offset = 0
    for person in people:
        slices[person.index] = slice(offset, offset + len(person.sentences))
        offset += len(person.sentences)
    return slices


def build_udv_record(
    hearing: HearingRecord,
    person: PersonSpeech,
    opinion_index: int,
    opinion_text: str,
    evidence: Evidence | None,
    method: Method,
    settings: EvidenceSettings,
) -> UdvRecord:
    tier = classify_tier(evidence, person.resolved, settings.embedding_threshold)
    return UdvRecord(
        id=udv_id(hearing.id, person.index, opinion_index),
        hearing_id=hearing.id,
        actor=Actor(name=person.participant.nome, role=person.participant.cargo),
        proposition=opinion_text,
        evidence=evidence,
        tier=tier,
        provenance=provenance_for(tier),
        method=method,
    )


def build_hearing_udvs(
    hearing: HearingRecord, encoder: CachedEncoder, settings: EvidenceSettings
) -> tuple[list[UdvRecord], list[PersonSpeech]]:
    transcript = hearing.transcricao
    people = resolve_hearing_people(hearing)
    all_sentences = [sentence for person in people for sentence in person.sentences]
    sentence_embeddings = encoder.encode(all_sentences, f"sentences_{hearing.id}")
    opinions = [
        (person, opinion_index, opinion_text)
        for person in people
        for opinion_index, opinion_text in enumerate(person.participant.opinioes)
    ]
    opinion_embeddings = encoder.encode([text for _, _, text in opinions], f"opinions_{hearing.id}")
    slices = sentence_slices_by_person(people)
    method = Method(
        encoder=encoder.encoder.name,
        revision=encoder.encoder.revision,
        embedding_threshold=settings.embedding_threshold,
    )
    policy = settings.quote_policy

    records: list[UdvRecord] = []
    for position, (person, opinion_index, opinion_text) in enumerate(opinions):
        evidence: Evidence | None = None
        if person.resolved:
            quote_match = find_opinion_turn_quote_match(opinion_text, person.matched_turns, policy)
            if is_trusted_quote(quote_match, policy):
                evidence = build_quote_evidence(quote_match, person.matched_turns, transcript)
            elif person.sentences:
                evidence = build_semantic_evidence(
                    opinion_embeddings[position],
                    person.sentences,
                    person.sentence_turns,
                    sentence_embeddings[slices[person.index]],
                    person.matched_turns,
                    transcript,
                    quote_match,
                )
        records.append(
            build_udv_record(
                hearing, person, opinion_index, opinion_text, evidence, method, settings
            )
        )
    return records, people


def build_udvs(
    hearings: Sequence[HearingRecord],
    encoder: CachedEncoder,
    settings: EvidenceSettings,
    *,
    on_hearing: HearingProgress | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> UdvRun:
    run = UdvRun(hearings=tuple(hearings))
    for number, hearing in enumerate(hearings, start=1):
        started = clock()
        records, people = build_hearing_udvs(hearing, encoder, settings)
        run.hearing_seconds.append(clock() - started)
        run.records.extend(records)
        run.people.extend(people)
        if on_hearing is not None:
            on_hearing(number, hearing, len(records), run.hearing_seconds[-1])
    return run
