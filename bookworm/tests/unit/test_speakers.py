import pytest

from bookworm import HearingRecord, Turn, resolve_person_speech, split_into_turns
from bookworm.transcript.speakers import (
    is_party_info,
    matching_turns,
    names_match,
    resolve_turn_name,
    single_token_matching_turns,
    turn_name_candidates,
)


def make_turn(raw_name: str, party_info: str = "", index: int = 0) -> Turn:
    return Turn(
        turn_index=index,
        raw_name=raw_name,
        party_info=party_info,
        speech="fala",
        start_char=0,
        end_char=4,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PT - SP", True),
        ("Bloco/PT - BA", True),
        ("PSOL/RJ", True),
        ("MESTRE ZEZÉ", False),
        ("Manifestação em língua estrangeira. Tradução simultânea.", False),
        ("PT-SP", False),
    ],
)
def test_is_party_info(text: str, expected: bool) -> None:
    assert is_party_info(text) is expected


@pytest.mark.parametrize(
    ("party_info", "expected"),
    [
        ("Dep. Carlos Nunes. PL - RJ", "Dep. Carlos Nunes"),
        ("Maria Souza. Bloco/PSDB - RS", "Maria Souza"),
        ("PT - SP", "PRESIDENTE"),
        ("Tradução simultânea. Intérprete", "PRESIDENTE"),
        ("", "PRESIDENTE"),
    ],
)
def test_resolve_turn_name(party_info: str, expected: str) -> None:
    assert resolve_turn_name(make_turn("PRESIDENTE", party_info)) == expected


@pytest.mark.parametrize(
    ("raw_name", "party_info", "expected"),
    [
        ("PRESIDENTE", "Dep. Carlos Nunes. PL - RJ", ["PRESIDENTE", "Dep. Carlos Nunes"]),
        ("JOÃO SILVA", "PT - SP", ["JOÃO SILVA"]),
        ("JOSÉ ANTÔNIO DOS SANTOS", "MESTRE ZEZÉ", ["JOSÉ ANTÔNIO DOS SANTOS", "MESTRE ZEZÉ"]),
        ("RITA OLIVEIRA", "", ["RITA OLIVEIRA"]),
    ],
)
def test_turn_name_candidates(raw_name: str, party_info: str, expected: list[str]) -> None:
    assert turn_name_candidates(make_turn(raw_name, party_info)) == expected


@pytest.mark.parametrize(
    ("name_a", "name_b", "expected"),
    [
        ("João Silva", "JOÃO SILVA", True),
        ("Joao Silva", "JOÃO SILVA", True),
        ("Carlos Nunes", "Dep. Carlos Nunes", True),
        ("Marcos Antônio Pereira", "MARCOS PEREIRA", True),
        ("Marcos Pereira", "MARCOS SOUZA", False),
        ("Roberto", "ROBERTO ALVES", False),
        ("Roberto", "ROBERTO", True),
        ("Ana Lima", "ANA", False),
    ],
)
def test_names_match(name_a: str, name_b: str, expected: bool) -> None:
    assert names_match(name_a, name_b) is expected


def test_matching_turns_keeps_transcript_order() -> None:
    turns = [
        make_turn("MARCOS PEREIRA", index=0),
        make_turn("PRESIDENTE", "Dep. Carlos Nunes. PL - RJ", index=1),
        make_turn("MARCOS PEREIRA", index=2),
    ]
    assert [turn.turn_index for turn in matching_turns("Marcos Pereira", turns)] == [0, 2]


def test_single_token_requires_a_unique_speaker() -> None:
    turns = [
        make_turn("ANA LIMA", index=0),
        make_turn("ANA COSTA", index=1),
        make_turn("ROBERTO ALVES", index=2),
        make_turn("ROBERTO ALVES", index=3),
    ]
    assert single_token_matching_turns("Ana", turns) == []
    assert [turn.turn_index for turn in single_token_matching_turns("Roberto", turns)] == [2, 3]


@pytest.mark.parametrize(
    ("hearing_id", "person_index", "expected_turns"),
    [
        (1, 0, [3]),
        (1, 1, [1, 5]),
        (1, 2, [0, 2, 4, 6]),
        (2, 0, [1]),
        (2, 1, [2]),
        (2, 2, [3]),
        (2, 3, []),
        (2, 4, [4]),
        (2, 5, [6]),
        (2, 6, []),
        (2, 7, [0, 7]),
    ],
)
def test_resolve_person_speech_on_fixture(
    mini_hearings: dict[int, HearingRecord],
    hearing_id: int,
    person_index: int,
    expected_turns: list[int],
) -> None:
    hearing = mini_hearings[hearing_id]
    person = hearing.metadados.envolvidos[person_index]
    matched, speech = resolve_person_speech(person.nome, split_into_turns(hearing.transcricao))
    assert [turn.turn_index for turn in matched] == expected_turns
    assert speech == " ".join(" ".join(turn.speech.split()) for turn in matched)


def test_speech_joins_turns_with_single_spaces(mini_hearings: dict[int, HearingRecord]) -> None:
    hearing = mini_hearings[1]
    _, speech = resolve_person_speech("Marcos Pereira", split_into_turns(hearing.transcricao))
    assert "  " not in speech
    assert "\n" not in speech
    assert "sem exagero nenhum.(Palmas.) Agradeço, novamente," in speech


def test_unresolved_person_has_empty_speech(mini_hearings: dict[int, HearingRecord]) -> None:
    hearing = mini_hearings[2]
    matched, speech = resolve_person_speech(
        "Beatriz Nogueira", split_into_turns(hearing.transcricao)
    )
    assert matched == []
    assert speech == ""
