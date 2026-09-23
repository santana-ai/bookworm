import numpy as np
import pytest
from conftest import SHORT_QUOTE_SENTENCE, THANKS_SENTENCE, TURNS_REFORM_TURN_5

from bookworm import (
    Evidence,
    HearingRecord,
    Span,
    Tier,
    TurnQuoteMatch,
    find_opinion_turn_quote_match,
    resolve_person_speech,
    split_into_turns,
    split_turn_sentences,
)
from bookworm.udv.evidence import (
    SentenceMatch,
    best_sentence_match,
    build_quote_evidence,
    build_semantic_evidence,
    classify_tier,
    evidence_from_span,
    provenance_for,
    short_quote_supports,
)

MARCOS_SHORT_QUOTE_SENTENCE = SHORT_QUOTE_SENTENCE


def semantic_evidence(score: float | None) -> Evidence:
    return evidence_from_span("Uma frase.", "semantic_similarity", score, None, None)


def test_evidence_from_span_copies_offsets() -> None:
    evidence = evidence_from_span(
        "Uma frase.",
        "direct_quote",
        None,
        "uma frase",
        Span(start_char=3, end_char=13, speaker_turn=2),
    )
    assert (evidence.start_char, evidence.end_char, evidence.speaker_turn) == (3, 13, 2)
    unlocated = evidence_from_span("Uma frase.", "direct_quote", None, "uma frase", None)
    assert (unlocated.start_char, unlocated.end_char, unlocated.speaker_turn) == (None, None, None)


@pytest.mark.parametrize(
    ("evidence", "resolved", "expected"),
    [
        (None, False, "person_not_resolved"),
        (semantic_evidence(0.9), False, "person_not_resolved"),
        (None, True, "no_evidence"),
        (
            evidence_from_span("Uma frase.", "direct_quote", None, "uma frase", None),
            True,
            "quote_found",
        ),
        (semantic_evidence(0.6), True, "semantic_match_high"),
        (semantic_evidence(0.6 - 1e-9), True, "semantic_match_weak"),
        (semantic_evidence(-0.2), True, "semantic_match_weak"),
    ],
)
def test_classify_tier(evidence: Evidence | None, resolved: bool, expected: Tier) -> None:
    assert classify_tier(evidence, resolved, 0.6) == expected


def test_classify_tier_rejects_semantic_evidence_without_score() -> None:
    with pytest.raises(ValueError, match="score"):
        classify_tier(semantic_evidence(None), True, 0.6)


@pytest.mark.parametrize(
    ("tier", "provenance"),
    [
        ("quote_found", "weak"),
        ("semantic_match_high", "model"),
        ("semantic_match_weak", "model"),
        ("no_evidence", None),
        ("person_not_resolved", None),
    ],
)
def test_provenance_for(tier: Tier, provenance: str | None) -> None:
    assert provenance_for(tier) == provenance


def test_best_sentence_match_returns_the_argmax_and_its_cosine() -> None:
    opinion = np.array([1.0, 0.0])
    sentences = np.array([[0.0, 1.0], [3.0, 4.0], [-1.0, 0.0]])
    assert best_sentence_match(opinion, sentences) == SentenceMatch(index=1, score=0.6)


def test_best_sentence_match_keeps_the_first_of_tied_sentences() -> None:
    opinion = np.array([1.0, 0.0])
    sentences = np.array([[0.0, 1.0], [2.0, 0.0], [5.0, 0.0]])
    assert best_sentence_match(opinion, sentences).index == 1


def test_best_sentence_match_keeps_float32_precision_of_the_embeddings() -> None:
    opinion = np.array([1.0, 0.0], dtype=np.float32)
    sentences = np.array([[3.0, 4.0]], dtype=np.float32)
    match = best_sentence_match(opinion, sentences)
    assert match.score == float(np.float32(0.6))
    assert match.score != 0.6


def short_match(sentence: str) -> TurnQuoteMatch:
    return TurnQuoteMatch(
        prefix="prazos de transição",
        words=3,
        quote_index=0,
        occurrence_count=1,
        occurrence_index=0,
        turn_index=1,
        start=19,
        end=38,
        sentence=sentence,
    )


@pytest.mark.parametrize(
    ("match", "sentence", "supported"),
    [
        (None, MARCOS_SHORT_QUOTE_SENTENCE, False),
        (short_match(MARCOS_SHORT_QUOTE_SENTENCE), MARCOS_SHORT_QUOTE_SENTENCE, True),
        (short_match(MARCOS_SHORT_QUOTE_SENTENCE), "de prazos de transição", True),
        (
            short_match(MARCOS_SHORT_QUOTE_SENTENCE),
            f"Antes. {MARCOS_SHORT_QUOTE_SENTENCE} Depois.",
            True,
        ),
        (
            short_match(MARCOS_SHORT_QUOTE_SENTENCE),
            "Espero que o relatório final considere as cooperativas menores.",
            False,
        ),
    ],
)
def test_short_quote_supports(match: TurnQuoteMatch | None, sentence: str, supported: bool) -> None:
    assert short_quote_supports(match, sentence) is supported


def test_build_quote_evidence_on_fixture(mini_hearings: dict[int, HearingRecord]) -> None:
    hearing = mini_hearings[1]
    turns = split_into_turns(hearing.transcricao)
    matched, _ = resolve_person_speech("João Silva", turns)
    match = find_opinion_turn_quote_match(hearing.metadados.envolvidos[0].opinioes[0], matched)
    assert match is not None
    assert build_quote_evidence(match, matched, hearing.transcricao) == Evidence(
        text="Defendo que o texto volte para a comissão de mérito.",
        support_type="direct_quote",
        score=None,
        quote_prefix="o texto volte para a comissão de mérito",
        start_char=690,
        end_char=742,
        speaker_turn=3,
    )


def test_build_quote_evidence_points_at_the_chosen_occurrence(
    turns_hearing: HearingRecord,
) -> None:
    turns = split_into_turns(turns_hearing.transcricao)
    matched, _ = resolve_person_speech("Clara Menezes", turns)
    match = find_opinion_turn_quote_match(
        turns_hearing.metadados.envolvidos[0].opinioes[0], matched
    )
    assert match is not None
    evidence = build_quote_evidence(match, matched, turns_hearing.transcricao)
    assert (evidence.text, evidence.speaker_turn) == (TURNS_REFORM_TURN_5, 5)
    assert evidence.start_char is not None
    assert turns_hearing.transcricao[evidence.start_char : evidence.end_char] == TURNS_REFORM_TURN_5


def marcos_semantic_evidence(
    hearing: HearingRecord, target: str, quote_match: TurnQuoteMatch | None
) -> Evidence:
    turns = split_into_turns(hearing.transcricao)
    matched, _ = resolve_person_speech("Marcos Pereira", turns)
    units = split_turn_sentences(matched)
    embeddings = np.array([[1.0, 0.0] if unit.text == target else [0.0, 1.0] for unit in units])
    return build_semantic_evidence(
        np.array([1.0, 0.0]),
        [unit.text for unit in units],
        [unit.turn_index for unit in units],
        embeddings,
        matched,
        hearing.transcricao,
        quote_match,
    )


def test_build_semantic_evidence_with_agreeing_short_quote(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    match = short_match(MARCOS_SHORT_QUOTE_SENTENCE)
    evidence = marcos_semantic_evidence(mini_hearings[1], MARCOS_SHORT_QUOTE_SENTENCE, match)
    assert evidence == Evidence(
        text=MARCOS_SHORT_QUOTE_SENTENCE,
        support_type="semantic_with_short_quote",
        score=1.0,
        quote_prefix="prazos de transição",
        start_char=311,
        end_char=381,
        speaker_turn=1,
    )


def test_build_semantic_evidence_with_disagreeing_short_quote(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    match = short_match(MARCOS_SHORT_QUOTE_SENTENCE)
    target = "Espero que o relatório final considere as cooperativas menores."
    evidence = marcos_semantic_evidence(mini_hearings[1], target, match)
    assert evidence.text == target
    assert evidence.support_type == "semantic_similarity"
    assert evidence.quote_prefix is None
    assert evidence.speaker_turn == 5


def test_build_semantic_evidence_locates_the_sentence_in_its_own_turn(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    evidence = marcos_semantic_evidence(hearing, THANKS_SENTENCE, None)
    assert evidence.text == THANKS_SENTENCE
    assert evidence.support_type == "semantic_similarity"
    assert evidence.speaker_turn == 5
    assert evidence.start_char is not None
    assert hearing.transcricao[evidence.start_char : evidence.end_char] == THANKS_SENTENCE
