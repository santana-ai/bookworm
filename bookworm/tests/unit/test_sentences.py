import pytest
from conftest import CROSS_TURN_TEXT, PALMAS_SENTENCE, THANKS_SENTENCE

from bookworm import (
    HearingRecord,
    TurnSentence,
    enclosing_sentence,
    resolve_person_speech,
    sentences_agree,
    split_into_turns,
    split_sentences,
    split_turn_sentences,
)
from bookworm.transcript.sentences import (
    enclosing_turn_sentence,
    is_sentence,
    sentence_part_spans,
    turn_text,
)

MARCOS_TURN_SENTENCES = [
    TurnSentence(
        "A proposta atual de regulação cria custos altos para as pequenas cooperativas do "
        "interior.",
        1,
    ),
    TurnSentence("O setor precisa de prazos de transição mais longos e de regras claras.", 1),
    TurnSentence(PALMAS_SENTENCE, 1),
    TurnSentence(THANKS_SENTENCE, 5),
    TurnSentence("Espero que o relatório final considere as cooperativas menores.", 5),
]


def test_split_sentences_breaks_after_terminal_punctuation() -> None:
    speech = (
        "Primeira frase com quatro palavras. Segunda frase também serve! Terceira fica bem aqui?"
    )
    assert split_sentences(speech) == [
        "Primeira frase com quatro palavras.",
        "Segunda frase também serve!",
        "Terceira fica bem aqui?",
    ]


def test_split_sentences_drops_short_parts() -> None:
    assert split_sentences("Muito obrigado. Esta frase tem palavras suficientes.") == [
        "Esta frase tem palavras suficientes."
    ]


def test_split_sentences_drops_stage_directions() -> None:
    assert split_sentences("(Manifestação em LIBRAS.)") == []
    assert split_sentences("(Pausa longa na sessão.) (Palmas prolongadas no plenário.)") == []


@pytest.mark.parametrize(
    ("speech", "expected"),
    [
        (
            "Fim do argumento central aqui.(Palmas.) Retomo agora com outra ideia.",
            ["Fim do argumento central aqui.(Palmas.)", "Retomo agora com outra ideia."],
        ),
        (
            "Vamos seguir com a pauta. (Risos.) Então passo a palavra ao relator.",
            ["Vamos seguir com a pauta.", "Então passo a palavra ao relator."],
        ),
        (
            "Isso pode ser votado hoje?(Pausa.) Vou deixar registrado em ata.",
            ["Isso pode ser votado hoje?(Pausa.)", "Vou deixar registrado em ata."],
        ),
        (
            "Todos querem votar a proposta!(Manifestação na plateia!) Peço silêncio ao plenário.",
            [
                "Todos querem votar a proposta!(Manifestação na plateia!)",
                "Peço silêncio ao plenário.",
            ],
        ),
    ],
)
def test_split_sentences_breaks_after_a_closing_parenthesis_that_ends_a_sentence(
    speech: str, expected: list[str]
) -> None:
    assert split_sentences(speech) == expected


@pytest.mark.parametrize(
    "speech",
    [
        "O presidente da empresa, (...) Castello Branco, comemorou o resultado do ano.",
        "O texto lido diz que (...) a medida foi aprovada ontem pela comissão.",
        "O relatório (anexo) Continua sobre a mesa da comissão hoje.",
    ],
)
def test_split_sentences_keeps_elisions_and_plain_parentheses_inside_the_sentence(
    speech: str,
) -> None:
    assert split_sentences(speech) == [speech]


def test_split_sentences_breaks_after_abbreviations() -> None:
    assert split_sentences("Convido o Sr. Marcos Pereira para a sua exposição.") == [
        "Marcos Pereira para a sua exposição."
    ]


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        ("Uma frase com quatro palavras.", True),
        ("Três palavras só.", False),
        ("(Palmas prolongadas no plenário.)", False),
        ("(Pausa.) Retomo a fala agora mesmo.", True),
    ],
)
def test_is_sentence(part: str, expected: bool) -> None:
    assert is_sentence(part) is expected


def test_sentence_part_spans_cover_the_text_between_boundaries() -> None:
    text = "Primeira parte aqui. Segunda.(Palmas.) Terceira parte final"
    spans = sentence_part_spans(text)
    assert [text[start:end] for start, end in spans] == [
        "Primeira parte aqui.",
        "Segunda.(Palmas.)",
        "Terceira parte final",
    ]
    assert sentence_part_spans("") == [(0, 0)]


@pytest.mark.parametrize(
    ("sentence", "other", "expected"),
    [
        ("Frase igual.", "Frase igual.", True),
        ("parte da frase", "Uma parte da frase maior.", True),
        ("Uma parte da frase maior.", "parte da frase", True),
        ("Uma frase.", "Outra frase.", False),
    ],
)
def test_sentences_agree(sentence: str, other: str, expected: bool) -> None:
    assert sentences_agree(sentence, other) is expected


def test_enclosing_turn_sentence_joins_every_part_the_span_touches() -> None:
    text = "Começo da fala. Termina aqui. Continua depois disso."
    start = text.index("aqui")
    assert enclosing_turn_sentence(text, start, start + 4) == "Termina aqui."
    end = text.index("Continua") + len("Continua")
    assert enclosing_turn_sentence(text, start, end) == "Termina aqui. Continua depois disso."


def test_enclosing_sentence_returns_the_sentence_around_the_prefix() -> None:
    speech = "Primeira frase aqui. A proposta atual cria custos altos. Terceira frase final."
    assert enclosing_sentence("a proposta atual", speech) == "A proposta atual cria custos altos."


def test_enclosing_sentence_joins_every_part_the_prefix_touches() -> None:
    speech = "Começo da fala. Termina aqui. Continua depois disso."
    assert enclosing_sentence("aqui. Continua", speech) == "Termina aqui. Continua depois disso."


def test_enclosing_sentence_stops_at_a_stage_direction() -> None:
    speech = "Isso pode ser votado hoje?(Pausa.) Vou deixar registrado em ata."
    assert enclosing_sentence("vou deixar", speech) == "Vou deixar registrado em ata."


def test_enclosing_sentence_normalizes_whitespace_first() -> None:
    speech = "Primeira frase.\n\n  A proposta   atual\ncria custos."
    assert enclosing_sentence("A proposta atual", speech) == "A proposta atual cria custos."


def test_enclosing_sentence_keeps_short_parts() -> None:
    assert enclosing_sentence("Muito obrigado", "Muito obrigado. Outra frase.") == "Muito obrigado."


def test_enclosing_sentence_raises_when_prefix_is_absent() -> None:
    with pytest.raises(ValueError, match="prefix not found"):
        enclosing_sentence("texto ausente", "Uma fala sem o trecho procurado.")


def test_split_turn_sentences_keeps_each_sentence_inside_its_turn(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    matched, _ = resolve_person_speech("Marcos Pereira", split_into_turns(hearing.transcricao))
    assert [turn.turn_index for turn in matched] == [1, 5]
    units = split_turn_sentences(matched)
    assert units == MARCOS_TURN_SENTENCES
    texts = {turn.turn_index: turn_text(turn) for turn in matched}
    assert all(unit.text in texts[unit.turn_index] for unit in units)
    assert all(CROSS_TURN_TEXT not in text for text in texts.values())


def test_turn_text_normalizes_the_turn_speech(mini_hearings: dict[int, HearingRecord]) -> None:
    turn = split_into_turns(mini_hearings[1].transcricao)[0]
    assert "\n" in turn.speech
    assert turn_text(turn) == " ".join(turn.speech.split())


def test_split_turn_sentences_without_turns() -> None:
    assert split_turn_sentences([]) == []
