import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    DEFAULT_VECTOR,
    FIXTURE_VECTORS,
    MINI_THRESHOLD,
    SHORT_QUOTE_OPINION,
    THANKS_SENTENCE,
    TURNS_VECTORS,
    StubEncoder,
)

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    QuotePolicy,
    UdvRecord,
    UdvRun,
    build_hearing_udvs,
    build_udvs,
    cache_file_name,
    resolve_hearing_people,
    select_hearings,
)
from bookworm.udv.build import sentence_slices_by_person, udv_corpus, udv_id

EXPECTED_IDS = [
    "udv-1-0-0",
    "udv-1-0-1",
    "udv-1-1-0",
    "udv-1-1-1",
    "udv-1-1-2",
    "udv-1-2-0",
    "udv-2-0-0",
    "udv-2-1-0",
    "udv-2-2-0",
    "udv-2-3-0",
    "udv-2-4-0",
    "udv-2-5-0",
    "udv-2-6-0",
    "udv-2-7-0",
]
EXPECTED_TIERS = {
    "udv-1-0-0": "quote_found",
    "udv-1-0-1": "semantic_match_high",
    "udv-1-1-0": "quote_found",
    "udv-1-1-1": "semantic_match_high",
    "udv-1-1-2": "semantic_match_high",
    "udv-1-2-0": "semantic_match_high",
    "udv-2-0-0": "quote_found",
    "udv-2-1-0": "semantic_match_high",
    "udv-2-2-0": "quote_found",
    "udv-2-3-0": "person_not_resolved",
    "udv-2-4-0": "quote_found",
    "udv-2-5-0": "no_evidence",
    "udv-2-6-0": "person_not_resolved",
    "udv-2-7-0": "semantic_match_high",
}
PROVENANCE_BY_TIER = {
    "quote_found": "weak",
    "semantic_match_high": "model",
    "semantic_match_weak": "model",
    "no_evidence": None,
    "person_not_resolved": None,
}
STUB_METHOD = {"encoder": "stub-encoder", "revision": "stub-revision-1", "embedding_threshold": 0.6}
MARCOS = {"name": "Marcos Pereira", "role": "Presidente da Associação de Cooperativas do Interior"}
LEGACY_RECORDS: dict[str, dict[str, Any]] = {
    "udv-1-0-0": {
        "id": "udv-1-0-0",
        "hearing_id": 1,
        "actor": {"name": "João Silva", "role": "Deputado (PT-SP)"},
        "proposition": 'Defendeu que "o texto volte para a comissão de mérito".',
        "evidence": {
            "text": "Defendo que o texto volte para a comissão de mérito.",
            "support_type": "direct_quote",
            "score": None,
            "quote_prefix": "o texto volte para a comissão de mérito",
            "start_char": 690,
            "end_char": 742,
            "speaker_turn": 3,
        },
        "tier": "quote_found",
        "provenance": "weak",
        "method": STUB_METHOD,
    },
    "udv-1-1-1": {
        "id": "udv-1-1-1",
        "hearing_id": 1,
        "actor": MARCOS,
        "proposition": SHORT_QUOTE_OPINION,
        "evidence": {
            "text": "O setor precisa de prazos de transição mais longos e de regras claras.",
            "support_type": "semantic_with_short_quote",
            "score": 1.0,
            "quote_prefix": "prazos de transição",
            "start_char": 311,
            "end_char": 381,
            "speaker_turn": 1,
        },
        "tier": "semantic_match_high",
        "provenance": "model",
        "method": STUB_METHOD,
    },
    "udv-1-1-2": {
        "id": "udv-1-1-2",
        "hearing_id": 1,
        "actor": MARCOS,
        "proposition": "Agradeceu aos colegas da mesa pela oportunidade de falar.",
        "evidence": {
            "text": THANKS_SENTENCE,
            "support_type": "semantic_similarity",
            "score": 1.0,
            "quote_prefix": None,
            "start_char": 875,
            "end_char": 948,
            "speaker_turn": 5,
        },
        "tier": "semantic_match_high",
        "provenance": "model",
        "method": STUB_METHOD,
    },
    "udv-2-6-0": {
        "id": "udv-2-6-0",
        "hearing_id": 2,
        "actor": {"name": "Beatriz Nogueira", "role": "Secretária estadual de Cultura"},
        "proposition": "Defendeu a ampliação dos editais estaduais.",
        "evidence": None,
        "tier": "person_not_resolved",
        "provenance": None,
        "method": STUB_METHOD,
    },
}


def build_mini(
    hearings: list[HearingRecord],
    encoder: StubEncoder | None = None,
    threshold: float = MINI_THRESHOLD,
    policy: QuotePolicy | None = None,
) -> UdvRun:
    settings = (
        EvidenceSettings(threshold) if policy is None else EvidenceSettings(threshold, policy)
    )
    return build_udvs(hearings, CachedEncoder(encoder or StubEncoder()), settings)


def by_id(run: UdvRun) -> dict[str, UdvRecord]:
    return {record.id: record for record in run.records}


@pytest.fixture
def mini_run(mini_hearing_list: list[HearingRecord]) -> UdvRun:
    return build_mini(mini_hearing_list)


def test_resolve_hearing_people(mini_hearings: dict[int, HearingRecord]) -> None:
    people = resolve_hearing_people(mini_hearings[2])
    assert [(person.index, person.participant.nome, person.resolved) for person in people] == [
        (0, "Mestre Zezé", True),
        (1, "Rita Oliveira", True),
        (2, "Roberto", True),
        (3, "Ana", False),
        (4, "Ana Lima", True),
        (5, "Paulo Mendes", True),
        (6, "Beatriz Nogueira", False),
        (7, "Helena Prado", True),
    ]
    paulo = people[5]
    assert paulo.speech == "(Manifestação em LIBRAS.)"
    assert paulo.sentences == ()
    assert people[3].speech == ""
    assert people[3].sentences == ()


def test_sentence_slices_follow_person_order(mini_hearings: dict[int, HearingRecord]) -> None:
    people = resolve_hearing_people(mini_hearings[1])
    assert [len(person.sentences) for person in people] == [2, 5, 6]
    assert sentence_slices_by_person(people) == {0: slice(0, 2), 1: slice(2, 7), 2: slice(7, 13)}


def test_person_sentences_come_from_each_matched_turn(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    marcos = resolve_hearing_people(mini_hearings[1])[1]
    assert [turn.turn_index for turn in marcos.matched_turns] == [1, 5]
    assert marcos.sentence_turns == (1, 1, 1, 5, 5)
    assert marcos.sentences[3] == THANKS_SENTENCE
    assert len(marcos.sentences) == len(marcos.sentence_turns)


def test_udv_corpus_lists_sentences_then_opinions_per_hearing(
    mini_hearing_list: list[HearingRecord],
) -> None:
    corpus = udv_corpus(mini_hearing_list)
    first_people = resolve_hearing_people(mini_hearing_list[0])
    first_sentences = [sentence for person in first_people for sentence in person.sentences]
    first_opinions = [text for person in first_people for text in person.participant.opinioes]
    assert corpus[: len(first_sentences)] == first_sentences
    assert corpus[len(first_sentences) : len(first_sentences) + 6] == first_opinions
    assert len(corpus) == 13 + 6 + 12 + 8


def test_select_hearings(mini_hearing_list: list[HearingRecord]) -> None:
    assert [h.id for h in select_hearings(mini_hearing_list, None, None)] == [1, 2]
    assert [h.id for h in select_hearings(mini_hearing_list, 1, None)] == [1]
    assert [h.id for h in select_hearings(mini_hearing_list, None, [2, 1])] == [1, 2]
    assert [h.id for h in select_hearings(mini_hearing_list, 1, [2])] == [2]
    assert select_hearings(mini_hearing_list, None, [99]) == []


def test_udv_id() -> None:
    assert udv_id(12, 3, 0) == "udv-12-3-0"


def test_ids_follow_hearing_person_opinion_order(mini_run: UdvRun) -> None:
    assert [record.id for record in mini_run.records] == EXPECTED_IDS
    assert [hearing.id for hearing in mini_run.hearings] == [1, 2]
    assert len(mini_run.people) == 11
    assert len(mini_run.hearing_seconds) == 2


def test_tiers_and_provenance(mini_run: UdvRun) -> None:
    assert {record.id: record.tier for record in mini_run.records} == EXPECTED_TIERS
    for record in mini_run.records:
        assert record.provenance == PROVENANCE_BY_TIER[record.tier]
        assert (record.evidence is None) == (record.tier in ("no_evidence", "person_not_resolved"))


def test_trusted_quote_becomes_direct_quote_without_score(mini_run: UdvRun) -> None:
    for record in mini_run.records:
        if record.tier == "quote_found":
            assert record.evidence is not None
            assert record.evidence.support_type == "direct_quote"
            assert record.evidence.score is None
            assert record.evidence.quote_prefix is not None
            assert record.evidence.start_char is not None


@pytest.mark.parametrize("record_id", sorted(LEGACY_RECORDS))
def test_to_json_line_equals_the_legacy_serialization(mini_run: UdvRun, record_id: str) -> None:
    expected = json.dumps(LEGACY_RECORDS[record_id], ensure_ascii=False)
    assert by_id(mini_run)[record_id].to_json_line() == expected


def test_score_equal_to_threshold_is_high(mini_hearing_list: list[HearingRecord]) -> None:
    record = by_id(build_mini(mini_hearing_list, threshold=0.6))["udv-1-0-1"]
    assert record.evidence is not None
    assert record.evidence.score == 0.6
    assert record.tier == "semantic_match_high"


def test_score_just_below_threshold_is_weak(mini_hearing_list: list[HearingRecord]) -> None:
    threshold = 0.6 + 1e-9
    record = by_id(build_mini(mini_hearing_list, threshold=threshold))["udv-1-0-1"]
    assert record.evidence is not None
    score = record.evidence.score
    assert score is not None
    assert score == pytest.approx(threshold - 1e-9, abs=1e-15)
    assert score < threshold
    assert record.tier == "semantic_match_weak"
    assert record.provenance == "model"
    assert record.method.embedding_threshold == threshold


def test_agreeing_short_quote_keeps_its_prefix(mini_run: UdvRun) -> None:
    evidence = by_id(mini_run)["udv-1-1-1"].evidence
    assert evidence is not None
    assert evidence.support_type == "semantic_with_short_quote"
    assert evidence.quote_prefix == "prazos de transição"


def test_disagreeing_short_quote_drops_its_prefix(mini_hearing_list: list[HearingRecord]) -> None:
    vectors = {**FIXTURE_VECTORS, SHORT_QUOTE_OPINION: DEFAULT_VECTOR}
    evidence = by_id(build_mini(mini_hearing_list, StubEncoder(vectors)))["udv-1-1-1"].evidence
    assert evidence is not None
    assert evidence.text == (
        "A proposta atual de regulação cria custos altos para as pequenas cooperativas do interior."
    )
    assert evidence.support_type == "semantic_similarity"
    assert evidence.quote_prefix is None


def test_stricter_quote_policy_moves_a_quote_to_the_semantic_path(
    mini_hearing_list: list[HearingRecord],
) -> None:
    run = build_mini(mini_hearing_list, policy=QuotePolicy(trusted_prefix_words=10))
    record = by_id(run)["udv-1-0-0"]
    assert record.tier == "semantic_match_high"
    assert record.evidence is not None
    assert record.evidence.support_type == "semantic_with_short_quote"
    assert record.evidence.quote_prefix == "o texto volte para a comissão de mérito"


def test_encoder_receives_sentences_then_opinions_per_hearing(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    encoder = StubEncoder()
    hearing = mini_hearings[1]
    records, people = build_hearing_udvs(
        hearing, CachedEncoder(encoder), EvidenceSettings(MINI_THRESHOLD)
    )
    assert encoder.calls == [
        [sentence for person in people for sentence in person.sentences],
        [text for person in people for text in person.participant.opinioes],
    ]
    assert len(records) == 6


def test_method_comes_from_the_encoder_and_settings(mini_run: UdvRun) -> None:
    assert {record.method.model_dump_json() for record in mini_run.records} == {
        json.dumps(STUB_METHOD, separators=(",", ":"))
    }


def test_cached_build_reuses_embeddings(
    mini_hearing_list: list[HearingRecord], tmp_path: Path
) -> None:
    first_encoder = StubEncoder()
    first = build_udvs(
        mini_hearing_list, CachedEncoder(first_encoder, tmp_path), EvidenceSettings(0.6)
    )
    labels = ["sentences_1", "opinions_1", "sentences_2", "opinions_2"]
    expected_files = {
        cache_file_name(label, first_encoder.cache_identity, texts)
        for label, texts in zip(labels, first_encoder.calls, strict=True)
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    second_encoder = StubEncoder()
    second = build_udvs(
        mini_hearing_list, CachedEncoder(second_encoder, tmp_path), EvidenceSettings(0.6)
    )
    assert second_encoder.calls == []
    assert [record.to_json_line() for record in second.records] == [
        record.to_json_line() for record in first.records
    ]


def test_build_udvs_reports_progress_and_timing(mini_hearing_list: list[HearingRecord]) -> None:
    ticks: Iterator[float] = iter([10.0, 10.5, 20.0, 22.0])
    progress: list[tuple[int, int, int, float]] = []
    run = build_udvs(
        mini_hearing_list,
        CachedEncoder(StubEncoder()),
        EvidenceSettings(MINI_THRESHOLD),
        on_hearing=lambda number, hearing, records, seconds: progress.append(
            (number, hearing.id, records, seconds)
        ),
        clock=lambda: next(ticks),
    )
    assert run.hearing_seconds == [0.5, 2.0]
    assert progress == [(1, 1, 6, 0.5), (2, 2, 8, 2.0)]


TURNS_RECORDS = {
    "udv-3-0-0": ("quote_found", "direct_quote", "a reforma do ensino médio é necessária", 5),
    "udv-3-0-1": (
        "quote_found",
        "direct_quote",
        "o transporte escolar precisa chegar a todas as comunidades do",
        3,
    ),
    "udv-3-0-2": (
        "quote_found",
        "direct_quote",
        "as escolas rurais precisam de internet de qualidade",
        1,
    ),
    "udv-3-0-3": (
        "quote_found",
        "direct_quote",
        "faltam professores de física nas escolas rurais",
        1,
    ),
    "udv-3-0-4": ("semantic_match_high", "semantic_similarity", None, 1),
    "udv-3-1-0": ("semantic_match_high", "semantic_similarity", None, 2),
    "udv-3-1-1": ("quote_found", "direct_quote", "qual é o custo por aluno", 2),
    "udv-3-1-2": (
        "semantic_match_high",
        "semantic_with_short_quote",
        "o transporte escolar é",
        2,
    ),
    "udv-3-2-0": ("semantic_match_high", "semantic_similarity", None, 0),
}


def test_per_turn_quote_rules_on_the_turns_fixture(turns_hearing: HearingRecord) -> None:
    run = build_udvs(
        [turns_hearing], CachedEncoder(StubEncoder(TURNS_VECTORS)), EvidenceSettings(MINI_THRESHOLD)
    )
    found: dict[str, tuple[str, str, str | None, int | None]] = {}
    for record in run.records:
        assert record.evidence is not None
        found[record.id] = (
            record.tier,
            record.evidence.support_type,
            record.evidence.quote_prefix,
            record.evidence.speaker_turn,
        )
    assert found == TURNS_RECORDS
