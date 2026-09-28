"""Demo JSON of one hearing (``export-hearing``)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bookworm.data.dates import article_date
from bookworm.data.io import JsonObject, is_json_integer_list
from bookworm.data.schemas import HearingRecord
from bookworm.data.splits import SPLIT_NAMES, SplitName
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder, FloatMatrix
from bookworm.transcript.offsets import Span, locate_sentence_span
from bookworm.transcript.sentences import split_sentences, turn_text
from bookworm.transcript.turns import Turn, split_into_turns
from bookworm.udv.build import (
    EvidenceSettings,
    PersonSpeech,
    resolve_hearing_people,
    sentence_slices_by_person,
    udv_id,
)
from bookworm.udv.coverage import pipeline_description
from bookworm.udv.evidence import sentence_similarities
from bookworm.udv.quotes import DEFAULT_QUOTE_POLICY, extract_quotes
from bookworm.udv.schemas import SEMANTIC_TIERS, Evidence, UdvRecord
from bookworm.udv.signals import SiteSignals
from bookworm.udv.windows import CandidateUnit, embedding_label, person_units, unit_size

DEFAULT_TOP_K = 8
CANDIDATE_SCORE_DECIMALS = 4


@dataclass(frozen=True, slots=True)
class LocatedSentence:
    text: str
    turn_index: int
    span: Span | None

    @property
    def start(self) -> int | None:
        return None if self.span is None else self.span.start_char

    @property
    def end(self) -> int | None:
        return None if self.span is None else self.span.end_char


def locate_turn_sentences(turn: Turn, transcript: str) -> list[LocatedSentence]:
    return [
        LocatedSentence(
            text=sentence,
            turn_index=turn.turn_index,
            span=locate_sentence_span(sentence, transcript, [turn]),
        )
        for sentence in split_sentences(turn_text(turn))
    ]


def split_of(manifest: Mapping[str, Any], hearing_id: int) -> SplitName:
    for name in SPLIT_NAMES:
        ids = manifest.get(name)
        if not is_json_integer_list(ids):
            raise ConfigError(f"split manifest: {name} is missing or not a list of integers")
        if hearing_id in ids:
            return name
    raise ConfigError(f"split manifest: hearing {hearing_id} is in no split")


def hearing_summary(hearing: HearingRecord, split: SplitName | None) -> JsonObject:
    published = article_date(hearing.materia)
    return {
        "id": hearing.id,
        "split": split,
        "article_date": None if published is None else published.isoformat(),
        "assunto": hearing.metadados.assunto,
        "materia": hearing.materia,
        "transcript_chars": len(hearing.transcricao),
        "transcript_words": len(hearing.transcricao.split()),
    }


def turn_entry(turn: Turn, sentences: Sequence[LocatedSentence]) -> JsonObject:
    return {
        "index": turn.turn_index,
        "speaker": turn.raw_name,
        "party": turn.party_info,
        "start": turn.start_char,
        "end": turn.end_char,
        "sentences": [
            {"text": sentence.text, "start": sentence.start, "end": sentence.end}
            for sentence in sentences
        ],
    }


def person_entry(person: PersonSpeech) -> JsonObject:
    return {
        "index": person.index,
        "name": person.participant.nome,
        "role": person.participant.cargo,
        "turns": [turn.turn_index for turn in person.matched_turns],
        "resolved": person.resolved,
    }


def person_sentences(
    person: PersonSpeech, by_turn: Mapping[int, Sequence[LocatedSentence]]
) -> list[LocatedSentence]:
    return [sentence for turn in person.matched_turns for sentence in by_turn[turn.turn_index]]


def top_candidates(
    opinion_embedding: FloatMatrix,
    sentence_embeddings: FloatMatrix,
    sentences: Sequence[LocatedSentence],
    top_k: int,
) -> list[JsonObject]:
    if not sentences:
        return []
    similarities = sentence_similarities(opinion_embedding, sentence_embeddings)
    order = np.argsort(-similarities, kind="stable")[:top_k]
    return [
        candidate_entry(
            sentences[index].text,
            float(similarities[index]),
            sentences[index].turn_index,
            sentences[index].span,
        )
        for index in (int(position) for position in order)
    ]


def check_run_records(
    hearing: HearingRecord,
    people: Sequence[PersonSpeech],
    records: Sequence[UdvRecord],
    encoder: CachedEncoder,
) -> None:
    expected = [
        udv_id(hearing.id, person.index, opinion_index)
        for person in people
        for opinion_index in range(len(person.participant.opinioes))
    ]
    found = [record.id for record in records]
    if found != expected:
        raise ConfigError(
            f"hearing {hearing.id}: run records {found[:3]}... do not match the LDS opinions "
            f"{expected[:3]}... ({len(found)} records, {len(expected)} opinions)"
        )
    identity = (encoder.encoder.name, encoder.encoder.revision)
    for record in records:
        if (record.method.encoder, record.method.revision) != identity:
            raise ConfigError(
                f"{record.id} was built with {record.method.encoder}@{record.method.revision}, "
                f"but the encoder is {identity[0]}@{identity[1]}"
            )


def check_run_pipeline(pipeline: object, settings: EvidenceSettings | None = None) -> None:
    """Refuse a run whose recorded pipeline is not the one ``settings`` describes."""
    if not isinstance(pipeline, Mapping):
        raise ConfigError(
            "coverage pipeline is missing or not an object, so the run cannot be tied to "
            "the pipeline of its config"
        )
    expected = (
        pipeline_description()
        if settings is None
        else pipeline_description(settings.quote_policy, settings)
    )
    differing = [key for key in expected if pipeline.get(key) != expected[key]]
    differing += [key for key in pipeline if key not in expected]
    if differing:
        raise ConfigError(
            f"coverage pipeline differs from the pipeline of the config in {', '.join(differing)}"
        )


def candidate_entry(text: str, score: float, turn: int, span: Span | None) -> JsonObject:
    return {
        "text": text,
        "score": round(score, CANDIDATE_SCORE_DECIMALS),
        "turn": turn,
        "start": None if span is None else span.start_char,
        "end": None if span is None else span.end_char,
    }


def top_unit_candidates(
    opinion_embedding: FloatMatrix,
    unit_embeddings: FloatMatrix,
    units: Sequence[CandidateUnit],
    top_k: int,
) -> list[JsonObject]:
    if not units:
        return []
    similarities = sentence_similarities(opinion_embedding, unit_embeddings)
    order = np.argsort(-similarities, kind="stable")[:top_k]
    return [
        candidate_entry(
            units[index].evidence_text,
            float(similarities[index]),
            units[index].turn_index,
            units[index].span,
        )
        for index in (int(position) for position in order)
    ]


@dataclass(frozen=True, slots=True)
class HearingCandidates:
    embeddings: FloatMatrix
    slices: dict[int, slice]
    sentences: dict[int, list[LocatedSentence]]
    units: dict[int, list[CandidateUnit]]

    def count(self, person: PersonSpeech) -> int:
        if self.units:
            return len(self.units[person.index])
        return len(person.sentences)

    def top(
        self, person: PersonSpeech, opinion_embedding: FloatMatrix, top_k: int
    ) -> list[JsonObject]:
        embeddings = self.embeddings[self.slices[person.index]]
        if self.units:
            return top_unit_candidates(
                opinion_embedding, embeddings, self.units[person.index], top_k
            )
        return top_candidates(opinion_embedding, embeddings, self.sentences[person.index], top_k)


def hearing_candidates(
    hearing: HearingRecord,
    people: Sequence[PersonSpeech],
    by_turn: Mapping[int, Sequence[LocatedSentence]],
    encoder: CachedEncoder,
    settings: EvidenceSettings | None,
) -> HearingCandidates:
    sentences = {person.index: person_sentences(person, by_turn) for person in people}
    if settings is None or not settings.uses_windows:
        return HearingCandidates(
            embeddings=encoder.encode(
                [sentence for person in people for sentence in person.sentences],
                f"sentences_{hearing.id}",
            ),
            slices=sentence_slices_by_person(people),
            sentences=sentences,
            units={},
        )
    size = unit_size(settings.semantic_unit)
    units = {
        person.index: person_units(person.matched_turns, hearing.transcricao, size)
        for person in people
    }
    slices: dict[int, slice] = {}
    offset = 0
    for person in people:
        slices[person.index] = slice(offset, offset + len(units[person.index]))
        offset += len(units[person.index])
    return HearingCandidates(
        embeddings=encoder.encode(
            [unit.text for person in people for unit in units[person.index]],
            embedding_label(settings.semantic_unit, hearing.id),
        ),
        slices=slices,
        sentences=sentences,
        units=units,
    )


def run_block(run_name: str, record: UdvRecord, settings: EvidenceSettings | None) -> JsonObject:
    block: JsonObject = {
        "name": run_name,
        "encoder": record.method.encoder,
        "revision": record.method.revision,
        "threshold": record.method.embedding_threshold,
    }
    if settings is not None and settings.uses_windows:
        block["semantic_unit"] = settings.semantic_unit
    if settings is not None and settings.quote_extent != "prefix_sentence":
        block["quote_extent"] = settings.quote_extent
    return block


def candidate_identity(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["text"],
        candidate["turn"],
        candidate["start"],
        candidate["end"],
        candidate["score"],
    )


def evidence_identity(evidence: Evidence) -> tuple[Any, ...] | None:
    if evidence.score is None:
        return None
    return (
        evidence.text,
        evidence.speaker_turn,
        evidence.start_char,
        evidence.end_char,
        round(evidence.score, CANDIDATE_SCORE_DECIMALS),
    )


def check_top_candidate(record: UdvRecord, candidates: Sequence[Mapping[str, Any]]) -> None:
    if record.tier not in SEMANTIC_TIERS:
        return
    recorded = None if record.evidence is None else evidence_identity(record.evidence)
    top = candidate_identity(candidates[0]) if candidates else None
    if recorded is None or top != recorded:
        raise ConfigError(
            f"{record.id}: the recorded evidence is not the top candidate sentence "
            "(text, turn, offsets and score) under the pipeline of the config and the encoder"
        )


def export_hearing(
    hearing: HearingRecord,
    records: Sequence[UdvRecord],
    encoder: CachedEncoder,
    *,
    run_name: str,
    pipeline: object,
    top_k: int = DEFAULT_TOP_K,
    split: SplitName | None = None,
    settings: EvidenceSettings | None = None,
    signals: SiteSignals | None = None,
) -> JsonObject:
    """Demo JSON of one hearing, with the ``top_k`` candidate units of each opinion.

    The candidate units are the sentences of the run, or its windows when ``settings`` has a
    window ``semantic_unit``; without ``settings`` the run must use the default pipeline.
    """
    if top_k < 1:
        raise ConfigError(f"top_k must be at least 1, got {top_k}")
    if not records:
        raise ConfigError(f"hearing {hearing.id}: the run has no records for it")
    check_run_pipeline(pipeline, settings)
    policy = DEFAULT_QUOTE_POLICY if settings is None else settings.quote_policy
    transcript = hearing.transcricao
    turns = split_into_turns(transcript)
    people = resolve_hearing_people(hearing)
    check_run_records(hearing, people, records, encoder)
    by_turn = {turn.turn_index: locate_turn_sentences(turn, transcript) for turn in turns}
    candidates_of = hearing_candidates(hearing, people, by_turn, encoder, settings)
    opinions = [(person, opinion) for person in people for opinion in person.participant.opinioes]
    opinion_embeddings = encoder.encode(
        [opinion for _, opinion in opinions], f"opinions_{hearing.id}"
    )
    udvs: list[JsonObject] = []
    for position, ((person, _), record) in enumerate(zip(opinions, records, strict=True)):
        candidates = candidates_of.top(person, opinion_embeddings[position], top_k)
        check_top_candidate(record, candidates)
        entry = {
            **record.to_dict(),
            "candidates": candidates,
            "n_candidates": candidates_of.count(person),
            "quotes": extract_quotes(record.proposition, policy.patterns),
        }
        if signals is not None:
            entry["signals"] = signals.for_record(record)
        udvs.append(entry)
    payload: JsonObject = {
        "hearing": hearing_summary(hearing, split),
        "transcript": transcript,
        "turns": [turn_entry(turn, by_turn[turn.turn_index]) for turn in turns],
        "people": [person_entry(person) for person in people],
        "udvs": udvs,
        "run": run_block(run_name, records[0], settings),
    }
    if signals is not None:
        payload["signals"] = signals.summary
    return payload
