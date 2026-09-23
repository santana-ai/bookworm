import dataclasses

import pytest

from bookworm import HearingRecord, Turn, split_into_turns


def test_turns_cover_every_header_in_order(mini_hearings: dict[int, HearingRecord]) -> None:
    turns = split_into_turns(mini_hearings[1].transcricao)
    assert [(turn.turn_index, turn.raw_name, turn.party_info) for turn in turns] == [
        (0, "PRESIDENTE", "Dep. Carlos Nunes. PL - RJ"),
        (1, "MARCOS PEREIRA", ""),
        (2, "PRESIDENTE", "Dep. Carlos Nunes. PL - RJ"),
        (3, "JOÃO SILVA", "PT - SP"),
        (4, "PRESIDENTE", "Dep. Carlos Nunes. PL - RJ"),
        (5, "MARCOS PEREIRA", ""),
        (6, "PRESIDENTE", "Dep. Carlos Nunes. PL - RJ"),
    ]


def test_turn_offsets_point_at_the_stripped_speech(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    for hearing in mini_hearings.values():
        transcript = hearing.transcricao
        for turn in split_into_turns(transcript):
            assert transcript[turn.start_char : turn.end_char] == turn.speech
            assert turn.speech == turn.speech.strip()


def test_speech_runs_until_the_next_header(mini_hearings: dict[int, HearingRecord]) -> None:
    turns = split_into_turns(mini_hearings[1].transcricao)
    assert turns[0].speech.endswith("Marcos Pereira para abrir a rodada de falas.")
    assert "\n\nChamo" in turns[0].speech
    assert turns[1].speech.endswith("sem exagero nenhum.(Palmas.)")
    assert turns[5].speech.startswith("Agradeço, novamente,")


def test_headers_without_parentheses_or_space_before_hyphen(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    turns = split_into_turns(mini_hearings[2].transcricao)
    by_name = {turn.raw_name: turn for turn in turns}
    assert by_name["RITA OLIVEIRA"].party_info == ""
    assert by_name["RITA OLIVEIRA"].speech.startswith("Eu trabalho")
    assert by_name["ROBERTO ALVES"].party_info == ""


def test_parenthesized_social_name_is_kept_as_party_info(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    turns = split_into_turns(mini_hearings[2].transcricao)
    assert turns[1].raw_name == "JOSÉ ANTÔNIO DOS SANTOS"
    assert turns[1].party_info == "MESTRE ZEZÉ"


def test_stage_direction_after_the_hyphen_is_speech(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    turns = split_into_turns(mini_hearings[2].transcricao)
    libras = next(turn for turn in turns if turn.raw_name == "PAULO MENDES")
    assert libras.party_info == ""
    assert libras.speech == "(Manifestação em LIBRAS.)"


def test_text_before_the_first_header_is_not_a_turn() -> None:
    transcript = (
        "Abertura sem orador identificado.\n\nA SRA. MARIA DIAS - Primeira fala registrada."
    )
    turns = split_into_turns(transcript)
    assert len(turns) == 1
    assert turns[0].raw_name == "MARIA DIAS"
    assert turns[0].speech == "Primeira fala registrada."


def test_lowercase_titles_are_not_headers() -> None:
    assert split_into_turns("Convido o Sr. Marcos - por favor, e a Sra. Rita - também.") == []


def test_transcript_without_headers_has_no_turns() -> None:
    assert split_into_turns("") == []


def test_turn_is_frozen_and_slotted() -> None:
    turn = Turn(turn_index=0, raw_name="X", party_info="", speech="fala", start_char=0, end_char=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.__setattr__("speech", "outra")
    assert not hasattr(turn, "__dict__")
    assert dataclasses.asdict(turn) == {
        "turn_index": 0,
        "raw_name": "X",
        "party_info": "",
        "speech": "fala",
        "start_char": 0,
        "end_char": 4,
    }
