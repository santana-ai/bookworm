"""Construction of the UDVs of a hearing and of a run."""

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from bookworm.data.schemas import HearingRecord, Participant
from bookworm.features.encoders import CachedEncoder, FloatMatrix
from bookworm.transcript.sentences import split_turn_sentences
from bookworm.transcript.speakers import resolve_person_speech
from bookworm.transcript.turns import Turn, split_into_turns
from bookworm.udv.evidence import (
    build_full_quote_evidence,
    build_quote_evidence,
    build_semantic_evidence,
    build_unit_evidence,
    classify_tier,
    provenance_for,
)
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    QuoteExtentMode,
    QuotePolicy,
    TurnQuoteMatch,
    find_opinion_turn_quote_match,
    is_trusted_quote,
)
from bookworm.udv.schemas import Actor, Evidence, Method, UdvRecord
from bookworm.udv.windows import (
    CandidateUnit,
    SemanticUnit,
    embedding_label,
    person_units,
    unit_size,
)

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
    semantic_unit: SemanticUnit = "sentence"
    quote_extent: QuoteExtentMode = "prefix_sentence"

    @property
    def uses_windows(self) -> bool:
        return self.semantic_unit != "sentence"


@dataclass
class UdvRun:
    hearings: tuple[HearingRecord, ...]
    records: list[UdvRecord] = field(default_factory=list)
    people: list[PersonSpeech] = field(default_factory=list)
    hearing_seconds: list[float] = field(default_factory=list)

    def add_hearing(
        self, records: Sequence[UdvRecord], people: Sequence[PersonSpeech], seconds: float
    ) -> None:
        self.hearing_seconds.append(seconds)
        self.records.extend(records)
        self.people.extend(people)


def udv_id(hearing_id: int, person_index: int, opinion_index: int) -> str:
    return f"udv-{hearing_id}-{person_index}-{opinion_index}"


def select_hearings(
    hearings: Sequence[HearingRecord], limit: int | None, ids: Iterable[int] | None
) -> list[HearingRecord]:
    """Keep the hearings with the given ids or, without ids, the first ``limit``."""
    if ids is not None:
        wanted = set(ids)
        return [hearing for hearing in hearings if hearing.id in wanted]
    if limit is not None:
        return list(hearings[:limit])
    return list(hearings)


def resolve_hearing_people(
    hearing: HearingRecord, turns: Sequence[Turn] | None = None
) -> list[PersonSpeech]:
    if turns is None:
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
    """Candidate sentences and opinions of the hearings: the corpus TF-IDF is fitted on."""
    corpus: list[str] = []
    for hearing in hearings:
        people = resolve_hearing_people(hearing)
        corpus.extend(sentence for person in people for sentence in person.sentences)
        corpus.extend(opinion for person in people for opinion in person.participant.opinioes)
    return corpus


def consecutive_slices(lengths: Iterable[tuple[int, int]]) -> dict[int, slice]:
    """Slices of consecutive rows, one per ``(key, length)`` pair, in the given order."""
    slices: dict[int, slice] = {}
    offset = 0
    for key, length in lengths:
        slices[key] = slice(offset, offset + length)
        offset += length
    return slices


def sentence_slices_by_person(people: Sequence[PersonSpeech]) -> dict[int, slice]:
    return consecutive_slices((person.index, len(person.sentences)) for person in people)


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


@dataclass(frozen=True, slots=True)
class SemanticCandidates:
    embeddings: FloatMatrix
    slices: dict[int, slice]
    units: dict[int, list[CandidateUnit]]

    def person_embeddings(self, person: PersonSpeech) -> FloatMatrix:
        return self.embeddings[self.slices[person.index]]


def semantic_candidates(
    hearing: HearingRecord,
    people: Sequence[PersonSpeech],
    encoder: CachedEncoder,
    semantic_unit: SemanticUnit,
) -> SemanticCandidates:
    """Embeddings of the candidate sentences, or windows, of every person of a hearing."""
    if semantic_unit == "sentence":
        all_sentences = [sentence for person in people for sentence in person.sentences]
        return SemanticCandidates(
            embeddings=encoder.encode(all_sentences, f"sentences_{hearing.id}"),
            slices=sentence_slices_by_person(people),
            units={},
        )
    size = unit_size(semantic_unit)
    units = {
        person.index: person_units(person.matched_turns, hearing.transcricao, size)
        for person in people
    }
    texts = [unit.text for person in people for unit in units[person.index]]
    return SemanticCandidates(
        embeddings=encoder.encode(texts, embedding_label(semantic_unit, hearing.id)),
        slices=consecutive_slices((person.index, len(units[person.index])) for person in people),
        units=units,
    )


def quote_evidence(
    quote_match: TurnQuoteMatch,
    opinion_text: str,
    person: PersonSpeech,
    transcript: str,
    settings: EvidenceSettings,
) -> Evidence:
    if settings.quote_extent == "full_quote":
        return build_full_quote_evidence(
            quote_match, opinion_text, person.matched_turns, transcript, settings.quote_policy
        )
    return build_quote_evidence(quote_match, person.matched_turns, transcript)


def semantic_evidence(
    opinion_embedding: FloatMatrix,
    person: PersonSpeech,
    candidates: SemanticCandidates,
    transcript: str,
    quote_match: TurnQuoteMatch | None,
    settings: EvidenceSettings,
) -> Evidence | None:
    embeddings = candidates.person_embeddings(person)
    if settings.uses_windows:
        units = candidates.units[person.index]
        if not units:
            return None
        return build_unit_evidence(opinion_embedding, units, embeddings, quote_match)
    if not person.sentences:
        return None
    return build_semantic_evidence(
        opinion_embedding,
        person.sentences,
        person.sentence_turns,
        embeddings,
        person.matched_turns,
        transcript,
        quote_match,
    )


def opinion_evidence(
    opinion_text: str,
    opinion_embedding: FloatMatrix,
    person: PersonSpeech,
    candidates: SemanticCandidates,
    transcript: str,
    settings: EvidenceSettings,
) -> Evidence | None:
    """Quote evidence when a trusted quote is found, similarity evidence otherwise."""
    if not person.resolved:
        return None
    policy = settings.quote_policy
    quote_match = find_opinion_turn_quote_match(opinion_text, person.matched_turns, policy)
    if is_trusted_quote(quote_match, policy):
        return quote_evidence(quote_match, opinion_text, person, transcript, settings)
    return semantic_evidence(
        opinion_embedding, person, candidates, transcript, quote_match, settings
    )


def encoder_method(encoder: CachedEncoder, settings: EvidenceSettings) -> Method:
    return Method(
        encoder=encoder.encoder.name,
        revision=encoder.encoder.revision,
        embedding_threshold=settings.embedding_threshold,
    )


def build_hearing_udvs(
    hearing: HearingRecord,
    encoder: CachedEncoder,
    settings: EvidenceSettings,
    turns: Sequence[Turn] | None = None,
) -> tuple[list[UdvRecord], list[PersonSpeech]]:
    """Build the UDVs of one hearing and the resolved speech of each participant."""
    people = resolve_hearing_people(hearing, turns)
    candidates = semantic_candidates(hearing, people, encoder, settings.semantic_unit)
    opinions = [
        (person, opinion_index, opinion_text)
        for person in people
        for opinion_index, opinion_text in enumerate(person.participant.opinioes)
    ]
    opinion_embeddings = encoder.encode([text for _, _, text in opinions], f"opinions_{hearing.id}")
    method = encoder_method(encoder, settings)
    records = [
        build_udv_record(
            hearing,
            person,
            opinion_index,
            opinion_text,
            opinion_evidence(
                opinion_text,
                opinion_embeddings[position],
                person,
                candidates,
                hearing.transcricao,
                settings,
            ),
            method,
            settings,
        )
        for position, (person, opinion_index, opinion_text) in enumerate(opinions)
    ]
    return records, people


def build_udvs(
    hearings: Sequence[HearingRecord],
    encoder: CachedEncoder,
    settings: EvidenceSettings,
    *,
    on_hearing: HearingProgress | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> UdvRun:
    """Build the UDVs of every hearing, in order, with per-hearing timings."""
    run = UdvRun(hearings=tuple(hearings))
    for number, hearing in enumerate(hearings, start=1):
        started = clock()
        records, people = build_hearing_udvs(hearing, encoder, settings)
        run.add_hearing(records, people, clock() - started)
        if on_hearing is not None:
            on_hearing(number, hearing, len(records), run.hearing_seconds[-1])
    return run
