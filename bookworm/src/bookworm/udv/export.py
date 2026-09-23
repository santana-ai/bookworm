from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bookworm.data.dates import article_date
from bookworm.data.io import is_json_integer_list
from bookworm.data.schemas import HearingRecord
from bookworm.data.splits import SPLIT_NAMES, SplitName
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder, FloatMatrix
from bookworm.transcript.offsets import Span, locate_sentence_span
from bookworm.transcript.sentences import split_sentences, turn_text
from bookworm.transcript.turns import Turn, split_into_turns
from bookworm.udv.build import (
    PersonSpeech,
    resolve_hearing_people,
    sentence_slices_by_person,
    udv_id,
)
from bookworm.udv.coverage import pipeline_description
from bookworm.udv.evidence import sentence_similarities
from bookworm.udv.quotes import DEFAULT_QUOTE_POLICY, QuotePolicy, extract_quotes
from bookworm.udv.schemas import SEMANTIC_TIERS, Evidence, UdvRecord

JsonObject = dict[str, Any]

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
        {
            "text": sentences[index].text,
            "score": round(float(similarities[index]), CANDIDATE_SCORE_DECIMALS),
            "turn": sentences[index].turn_index,
            "start": sentences[index].start,
            "end": sentences[index].end,
        }
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


def check_run_pipeline(pipeline: object, policy: QuotePolicy = DEFAULT_QUOTE_POLICY) -> None:
    if not isinstance(pipeline, Mapping):
        raise ConfigError(
            "coverage pipeline is missing or not an object, so the run cannot be tied to "
            "the current pipeline"
        )
    expected = pipeline_description(policy)
    differing = [key for key in expected if pipeline.get(key) != expected[key]]
    differing += [key for key in pipeline if key not in expected]
    if differing:
        raise ConfigError(
            f"coverage pipeline differs from the current pipeline in {', '.join(differing)}"
        )


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
            "(text, turn, offsets and score) under the current pipeline and encoder"
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
    quote_policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> JsonObject:
    if top_k < 1:
        raise ConfigError(f"top_k must be at least 1, got {top_k}")
    if not records:
        raise ConfigError(f"hearing {hearing.id}: the run has no records for it")
    check_run_pipeline(pipeline, quote_policy)
    transcript = hearing.transcricao
    turns = split_into_turns(transcript)
    people = resolve_hearing_people(hearing)
    check_run_records(hearing, people, records, encoder)
    by_turn = {turn.turn_index: locate_turn_sentences(turn, transcript) for turn in turns}
    sentence_embeddings = encoder.encode(
        [sentence for person in people for sentence in person.sentences],
        f"sentences_{hearing.id}",
    )
    opinions = [(person, opinion) for person in people for opinion in person.participant.opinioes]
    opinion_embeddings = encoder.encode(
        [opinion for _, opinion in opinions], f"opinions_{hearing.id}"
    )
    slices = sentence_slices_by_person(people)
    located = {person.index: person_sentences(person, by_turn) for person in people}
    udvs: list[JsonObject] = []
    for position, ((person, _), record) in enumerate(zip(opinions, records, strict=True)):
        candidates = top_candidates(
            opinion_embeddings[position],
            sentence_embeddings[slices[person.index]],
            located[person.index],
            top_k,
        )
        check_top_candidate(record, candidates)
        udvs.append(
            {
                **record.to_dict(),
                "candidates": candidates,
                "n_candidates": len(person.sentences),
                "quotes": extract_quotes(record.proposition, quote_policy.patterns),
            }
        )
    method = records[0].method
    return {
        "hearing": hearing_summary(hearing, split),
        "transcript": transcript,
        "turns": [turn_entry(turn, by_turn[turn.turn_index]) for turn in turns],
        "people": [person_entry(person) for person in people],
        "udvs": udvs,
        "run": {
            "name": run_name,
            "encoder": method.encoder,
            "revision": method.revision,
            "threshold": method.embedding_threshold,
        },
    }
