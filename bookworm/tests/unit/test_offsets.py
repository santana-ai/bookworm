import pytest
from conftest import MINI_THRESHOLD, THANKS_SENTENCE, TURNS_VECTORS, StubEncoder

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    Span,
    build_udvs,
    locate_sentence_span,
    locate_turn_sentence_span,
    resolve_hearing_people,
    resolve_person_speech,
    split_into_turns,
)
from bookworm.transcript.sentences import turn_text
from bookworm.transcript.text import normalize_whitespace


def test_span_points_at_the_sentence_inside_the_turn(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    turns = split_into_turns(hearing.transcricao)
    matched, _ = resolve_person_speech("João Silva", turns)
    sentence = "Defendo que o texto volte para a comissão de mérito."
    span = locate_sentence_span(sentence, hearing.transcricao, matched)
    assert span == Span(start_char=690, end_char=742, speaker_turn=3)
    assert hearing.transcricao[span.start_char : span.end_char] == sentence


def test_span_tolerates_line_breaks_inside_the_turn() -> None:
    transcript = "A SRA. MARIA DIAS - Primeira linha da fala\n\ncontinua na linha seguinte. Fim."
    turns = split_into_turns(transcript)
    span = locate_sentence_span(
        "Primeira linha da fala continua na linha seguinte.", transcript, turns
    )
    assert span is not None
    assert span.speaker_turn == 0
    assert normalize_whitespace(transcript[span.start_char : span.end_char]) == (
        "Primeira linha da fala continua na linha seguinte."
    )


def test_span_search_is_limited_to_the_given_turns(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    turns = split_into_turns(hearing.transcricao)
    marcos, _ = resolve_person_speech("Marcos Pereira", turns)
    sentence = "Defendo que o texto volte para a comissão de mérito."
    assert locate_sentence_span(sentence, hearing.transcricao, marcos) is None


def test_span_uses_the_first_matching_turn() -> None:
    transcript = (
        "O SR. RUI - Repito a mesma frase de novo.\n\n"
        "A SRA. EVA - Outra fala.\n\n"
        "O SR. RUI - Repito a mesma frase de novo."
    )
    turns = split_into_turns(transcript)
    rui = [turn for turn in turns if turn.raw_name == "RUI"]
    span = locate_sentence_span("Repito a mesma frase de novo.", transcript, rui)
    assert span is not None
    assert span.speaker_turn == 0
    later = locate_sentence_span("Repito a mesma frase de novo.", transcript, rui[1:])
    assert later is not None
    assert later.speaker_turn == 2


def test_span_is_frozen() -> None:
    span = Span(start_char=1, end_char=2, speaker_turn=0)
    with pytest.raises(AttributeError):
        span.__setattr__("start_char", 5)


def test_turn_span_is_searched_only_in_the_source_turn() -> None:
    transcript = (
        "O SR. RUI - Repito a mesma frase de novo.\n\n"
        "A SRA. EVA - Outra fala.\n\n"
        "O SR. RUI - Repito a mesma frase de novo."
    )
    turns = split_into_turns(transcript)
    rui = [turn for turn in turns if turn.raw_name == "RUI"]
    sentence = "Repito a mesma frase de novo."
    later = locate_turn_sentence_span(sentence, transcript, rui, 2)
    assert later is not None
    assert later.speaker_turn == 2
    assert transcript[later.start_char : later.end_char] == sentence
    assert locate_turn_sentence_span(sentence, transcript, rui, 1) is None


def test_thanks_sentence_is_located_in_the_second_turn(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    matched, _ = resolve_person_speech("Marcos Pereira", split_into_turns(hearing.transcricao))
    span = locate_turn_sentence_span(THANKS_SENTENCE, hearing.transcricao, matched, 5)
    assert span is not None
    assert span.speaker_turn == 5
    assert hearing.transcricao[span.start_char : span.end_char] == THANKS_SENTENCE


@pytest.mark.parametrize("fixture_name", ["mini_hearing_list", "turns_hearing"])
def test_evidence_never_spans_turns(fixture_name: str, request: pytest.FixtureRequest) -> None:
    loaded = request.getfixturevalue(fixture_name)
    hearings: list[HearingRecord] = loaded if isinstance(loaded, list) else [loaded]
    for hearing in hearings:
        for person in resolve_hearing_people(hearing):
            turns = {turn.turn_index: turn for turn in person.matched_turns}
            for sentence, turn_index in zip(person.sentences, person.sentence_turns, strict=True):
                assert sentence in turn_text(turns[turn_index])
                span = locate_turn_sentence_span(
                    sentence, hearing.transcricao, person.matched_turns, turn_index
                )
                assert span is not None
                assert span.speaker_turn == turn_index
    run = build_udvs(
        hearings, CachedEncoder(StubEncoder(TURNS_VECTORS)), EvidenceSettings(MINI_THRESHOLD)
    )
    by_id = {hearing.id: hearing for hearing in hearings}
    turns_by_hearing = {hearing.id: split_into_turns(hearing.transcricao) for hearing in hearings}
    evidences = [record for record in run.records if record.evidence is not None]
    assert evidences
    for record in evidences:
        evidence = record.evidence
        assert evidence is not None
        assert evidence.start_char is not None and evidence.end_char is not None
        assert evidence.speaker_turn is not None
        turn = turns_by_hearing[record.hearing_id][evidence.speaker_turn]
        assert turn.start_char <= evidence.start_char < evidence.end_char <= turn.end_char
        transcript = by_id[record.hearing_id].transcricao
        span_text = transcript[evidence.start_char : evidence.end_char]
        assert normalize_whitespace(span_text) == evidence.text
