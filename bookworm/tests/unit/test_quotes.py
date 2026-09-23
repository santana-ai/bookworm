import pytest
from conftest import (
    TURNS_FAMILIES_OPINION,
    TURNS_PHYSICS_SENTENCE,
    TURNS_REFORM_TURN_1,
    TURNS_REFORM_TURN_5,
    TURNS_TRANSPORT_SENTENCE,
)

from bookworm import (
    HearingRecord,
    QuoteMatch,
    QuotePolicy,
    Turn,
    TurnQuoteMatch,
    enclosing_sentence,
    extract_quotes,
    find_opinion_quote_match,
    find_opinion_turn_quote_match,
    find_quote_match,
    is_trusted_quote,
    resolve_person_speech,
    split_into_turns,
)
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    DOUBLE_QUOTE_PATTERN,
    DOUBLE_QUOTE_PATTERNS,
    SINGLE_QUOTE_PATTERN,
    PrefixMatch,
    PrefixOccurrence,
    choose_occurrence_index,
    find_prefix_occurrences,
    find_turn_quote_match,
    quote_prefix_pattern,
    quote_prefixes,
    token_jaccard,
    word_tokens,
)

LEFT_QUOTE = "“"
RIGHT_QUOTE = "”"


def test_default_policy_values() -> None:
    expected = QuotePolicy(
        prefix_lengths=(10, 6, 4, 3),
        min_prefix_chars=6,
        trusted_prefix_words=6,
        patterns=(DOUBLE_QUOTE_PATTERN, SINGLE_QUOTE_PATTERN),
    )
    assert expected == DEFAULT_QUOTE_POLICY


def test_quote_patterns_are_the_published_ones() -> None:
    assert DOUBLE_QUOTE_PATTERN.pattern == '“([^”]{10,})”|"([^"]{10,})"'
    assert SINGLE_QUOTE_PATTERN.pattern == "(?<!\\w)'([^']{10,})'(?!\\w)"


def test_extract_quotes_reads_straight_and_curly_quotes_in_order() -> None:
    opinion = (
        f'Disse que "o setor precisa de prazo" e que {LEFT_QUOTE}a regra   atual\nprejudica '
        f"todos{RIGHT_QUOTE}."
    )
    assert extract_quotes(opinion) == ["o setor precisa de prazo", "a regra atual prejudica todos"]


def test_extract_quotes_ignores_quotes_shorter_than_ten_characters() -> None:
    assert extract_quotes('Chamou de "absurdo" a proposta.') == []
    assert extract_quotes("Chamou de 'absurdo' a proposta.") == []
    assert extract_quotes("Sem citação nenhuma.") == []


def test_extract_quotes_reads_single_quotes_between_word_boundaries() -> None:
    opinion = "Disse: 'as escolas rurais precisam de internet' e depois 'mais professores'."
    assert extract_quotes(opinion) == ["as escolas rurais precisam de internet", "mais professores"]
    assert extract_quotes(opinion, DOUBLE_QUOTE_PATTERNS) == []


def test_extract_quotes_ignores_apostrophes_inside_words() -> None:
    assert extract_quotes("Falou da caixa d'água e da gota d'orvalho na escola.") == []
    assert extract_quotes("Citou o'escândalo do orçamento'x hoje.") == []


def test_extract_quotes_lists_a_nested_single_quote_after_its_enclosing_quote() -> None:
    opinion = (
        "Relatou que \"o ministro disse 'a verba chegará em maio' aos prefeitos\" e "
        "'ninguém acreditou nisso'."
    )
    assert extract_quotes(opinion) == [
        "o ministro disse 'a verba chegará em maio' aos prefeitos",
        "a verba chegará em maio",
        "ninguém acreditou nisso",
    ]


def test_extract_quotes_ignores_curly_single_quotes() -> None:
    opinion = "Disse \u2018as escolas rurais precisam de internet\u2019."
    assert extract_quotes(opinion) == []


def test_quote_prefixes_follow_the_policy_lengths() -> None:
    quote = "um dois três quatro cinco seis sete oito nove dez onze"
    assert quote_prefixes(quote) == [
        ("um dois três quatro cinco seis sete oito nove dez", 10),
        ("um dois três quatro cinco seis", 6),
        ("um dois três quatro", 4),
        ("um dois três", 3),
    ]
    assert quote_prefixes("prazos longos") == [("prazos longos", 2)] * 4
    assert quote_prefixes("e a o") == []


@pytest.mark.parametrize(
    ("prefix", "text", "found"),
    [
        ("a proposta", "Primeiro. A proposta atual.", True),
        ("a proposta", "Primeiro. a proposta atual.", True),
        ("a proposta", "Primeiro. A PROPOSTA atual.", True),
        ("a proposta", "Primeiro.A proposta atual.", True),
        ("a proposta", "Primeiro.xa proposta atual.", False),
        ("A proposta", "e a proposta atual.", True),
        ("a proposta", "a propostas atuais.", False),
        ("2024 foi", "Em 2024 foi assim.", True),
        ("2024 foi", "Em x2024 foi assim.", False),
    ],
)
def test_quote_prefix_pattern(prefix: str, text: str, found: bool) -> None:
    assert (quote_prefix_pattern(prefix).search(text) is not None) is found


def test_uppercase_first_letter_matches_inside_a_word() -> None:
    assert quote_prefix_pattern("a proposta").search("xA proposta") is not None


def test_find_quote_match_prefers_the_longest_prefix() -> None:
    speech = "Hoje afirmo que a proposta atual de regulação cria custos altos para as cooperativas."
    quote = "a proposta atual de regulação cria custos altos para as cooperativas pequenas"
    assert find_quote_match(quote, speech) == QuoteMatch(
        prefix="a proposta atual de regulação cria custos altos para as", words=10
    )


def test_find_quote_match_falls_back_to_shorter_prefixes() -> None:
    speech = "O setor precisa de prazos de transição mais longos."
    assert find_quote_match("prazos de transição mais curtos e baratos", speech) == QuoteMatch(
        prefix="prazos de transição mais", words=4
    )
    assert find_quote_match("prazos de transição realmente longos", speech) == QuoteMatch(
        prefix="prazos de transição", words=3
    )


def test_find_quote_match_counts_words_of_short_quotes() -> None:
    speech = "Eu disse que a proposta atual cria custos."
    assert find_quote_match("a proposta atual cria custos", speech) == QuoteMatch(
        prefix="a proposta atual cria custos", words=5
    )


def test_find_quote_match_needs_prefixes_of_at_least_six_characters() -> None:
    assert find_quote_match("e a o", "e a o") is None
    assert find_quote_match("ab cde", "ab cde") == QuoteMatch(prefix="ab cde", words=2)
    assert find_quote_match("de a um", "de a um") == QuoteMatch(prefix="de a um", words=3)


def test_find_quote_match_returns_none_when_nothing_matches() -> None:
    assert find_quote_match("frase que ninguém disse aqui", "Uma fala sem relação.") is None


def test_find_opinion_quote_match_on_the_concatenated_speech_returns_the_first_matching_quote() -> (
    None
):
    speech = "Primeiro ponto importante da fala. Segundo ponto importante da fala."
    opinion = 'Citou "algo que nunca foi dito" e "segundo ponto importante da fala".'
    assert find_opinion_quote_match(opinion, speech) == QuoteMatch(
        prefix="segundo ponto importante da fala", words=5
    )


def test_find_opinion_quote_match_without_quotes() -> None:
    assert find_opinion_quote_match("Opinião sem citação.", "Qualquer fala aqui.") is None


@pytest.mark.parametrize(
    ("match", "trusted"),
    [
        (None, False),
        (QuoteMatch(prefix="a b c", words=3), False),
        (QuoteMatch(prefix="a b c d e", words=5), False),
        (QuoteMatch(prefix="a b c d e f", words=6), True),
        (QuoteMatch(prefix="a b c d e f g h i j", words=10), True),
    ],
)
def test_is_trusted_quote(match: QuoteMatch | None, trusted: bool) -> None:
    assert is_trusted_quote(match) is trusted


def test_is_trusted_quote_follows_the_policy() -> None:
    strict_policy = QuotePolicy(trusted_prefix_words=10)
    assert not is_trusted_quote(QuoteMatch(prefix="a b c d e f", words=6), strict_policy)


def turn(index: int, speech: str) -> Turn:
    return Turn(
        turn_index=index,
        raw_name="FALANTE",
        party_info="",
        speech=speech,
        start_char=0,
        end_char=len(speech),
    )


def test_word_tokens_and_token_jaccard() -> None:
    assert word_tokens("Educação, É já!") == {"educacao", "e", "ja"}
    assert token_jaccard("a b c", "b c d") == 0.5
    assert token_jaccard("", "...") == 0.0


def test_find_prefix_occurrences_searches_each_turn_and_keeps_the_turn_sentence() -> None:
    turns = [
        turn(4, "Primeira frase.  A reforma é urgente.\nOutra coisa."),
        turn(7, "Nada aqui. Repito: a reforma é urgente e necessária."),
        turn(9, "Fim da fala. A reforma"),
    ]
    assert find_prefix_occurrences("a reforma é urgente", turns) == [
        PrefixOccurrence(turn_index=4, start=16, end=35, sentence="A reforma é urgente."),
        PrefixOccurrence(
            turn_index=7,
            start=19,
            end=38,
            sentence="Repito: a reforma é urgente e necessária.",
        ),
    ]


def test_prefix_occurrences_never_join_two_turns() -> None:
    turns = [turn(0, "O texto termina com a reforma"), turn(1, "é urgente para todos.")]
    assert find_prefix_occurrences("a reforma é urgente", turns) == []


def test_find_turn_quote_match_keeps_every_occurrence_of_the_longest_prefix() -> None:
    turns = [turn(0, "A reforma é urgente hoje."), turn(1, "A reforma é urgente amanhã.")]
    match = find_turn_quote_match("a reforma é urgente agora mesmo", turns)
    assert match == PrefixMatch(
        prefix="a reforma é urgente",
        words=4,
        occurrences=(
            PrefixOccurrence(0, 0, 19, "A reforma é urgente hoje."),
            PrefixOccurrence(1, 0, 19, "A reforma é urgente amanhã."),
        ),
    )
    assert find_turn_quote_match("nada disso foi dito", turns) is None


def test_choose_occurrence_index_prefers_overlap_and_keeps_the_first_on_ties() -> None:
    occurrences = [
        PrefixOccurrence(0, 0, 5, "A reforma é urgente hoje."),
        PrefixOccurrence(1, 0, 5, "A reforma é urgente para os professores."),
        PrefixOccurrence(2, 0, 5, "A reforma é urgente para os professores."),
    ]
    assert (
        choose_occurrence_index(occurrences, "Disse que a reforma é urgente para os professores.")
        == 1
    )
    tied = [occurrences[0], occurrences[0]]
    assert choose_occurrence_index(tied, "Opinião sem nenhuma palavra em comum.") == 0


def clara_turns(hearing: HearingRecord) -> list[Turn]:
    matched, _ = resolve_person_speech("Clara Menezes", split_into_turns(hearing.transcricao))
    return matched


def clara_opinion(hearing: HearingRecord, index: int) -> str:
    return hearing.metadados.envolvidos[0].opinioes[index]


def test_trusted_prefix_takes_the_occurrence_closest_to_the_opinion(
    turns_hearing: HearingRecord,
) -> None:
    turns = clara_turns(turns_hearing)
    assert [item.turn_index for item in turns] == [1, 3, 5]
    match = find_opinion_turn_quote_match(clara_opinion(turns_hearing, 0), turns)
    assert match == TurnQuoteMatch(
        prefix="a reforma do ensino médio é necessária",
        words=7,
        quote_index=0,
        occurrence_count=2,
        occurrence_index=1,
        turn_index=5,
        start=15,
        end=53,
        sentence=TURNS_REFORM_TURN_5,
    )
    opinion = clara_opinion(turns_hearing, 0)
    assert token_jaccard(TURNS_REFORM_TURN_5, opinion) > token_jaccard(TURNS_REFORM_TURN_1, opinion)


def test_identical_occurrences_keep_the_first(turns_hearing: HearingRecord) -> None:
    match = find_opinion_turn_quote_match(
        clara_opinion(turns_hearing, 3), clara_turns(turns_hearing)
    )
    assert match is not None
    assert (match.occurrence_count, match.occurrence_index, match.turn_index) == (2, 0, 1)
    assert match.sentence == TURNS_PHYSICS_SENTENCE


def test_the_quote_with_most_prefix_words_wins(turns_hearing: HearingRecord) -> None:
    opinion = clara_opinion(turns_hearing, 1)
    turns = clara_turns(turns_hearing)
    match = find_opinion_turn_quote_match(opinion, turns)
    assert match is not None
    assert (match.quote_index, match.words, match.turn_index) == (1, 10, 3)
    assert match.prefix == "o transporte escolar precisa chegar a todas as comunidades do"
    _, speech = resolve_person_speech("Clara Menezes", split_into_turns(turns_hearing.transcricao))
    assert find_opinion_quote_match(opinion, speech) == QuoteMatch(
        prefix="faltam professores de", words=3
    )


def test_ties_between_quotes_keep_the_earliest_quote() -> None:
    turns = [turn(0, "A reforma é urgente. O transporte é caro.")]
    opinion = 'Disse "a reforma é urgente demais" e "o transporte é caro demais".'
    match = find_opinion_turn_quote_match(opinion, turns)
    assert match is not None
    assert (match.quote_index, match.prefix, match.words) == (0, "a reforma é urgente", 4)


def test_single_quotes_can_become_trusted(turns_hearing: HearingRecord) -> None:
    match = find_opinion_turn_quote_match(
        clara_opinion(turns_hearing, 2), clara_turns(turns_hearing)
    )
    assert match is not None
    assert is_trusted_quote(match)
    assert match.prefix == "as escolas rurais precisam de internet de qualidade"
    assert match.turn_index == 1
    double_only = QuotePolicy(patterns=DOUBLE_QUOTE_PATTERNS)
    assert (
        find_opinion_turn_quote_match(
            clara_opinion(turns_hearing, 2), clara_turns(turns_hearing), double_only
        )
        is None
    )


def test_short_prefixes_keep_the_first_occurrence(turns_hearing: HearingRecord) -> None:
    matched, _ = resolve_person_speech("Pedro Alves", split_into_turns(turns_hearing.transcricao))
    match = find_opinion_turn_quote_match(TURNS_FAMILIES_OPINION, matched)
    assert match is not None
    assert not is_trusted_quote(match)
    assert (match.prefix, match.occurrence_count, match.occurrence_index) == (
        "o transporte escolar é",
        2,
        0,
    )
    assert (match.turn_index, match.sentence) == (2, TURNS_TRANSPORT_SENTENCE)


def test_turn_quote_match_without_quotes_or_hits(turns_hearing: HearingRecord) -> None:
    turns = clara_turns(turns_hearing)
    assert find_opinion_turn_quote_match(clara_opinion(turns_hearing, 4), turns) is None
    assert find_opinion_turn_quote_match('Disse "algo que ninguém falou aqui".', turns) is None


def test_custom_prefix_lengths_change_the_match() -> None:
    speech = "O setor precisa de prazos de transição mais longos."
    policy = QuotePolicy(prefix_lengths=(4,))
    assert find_quote_match("prazos de transição realmente longos", speech, policy) is None


@pytest.mark.parametrize(
    ("hearing_id", "name", "opinion_index", "expected", "sentence"),
    [
        (
            1,
            "Marcos Pereira",
            0,
            QuoteMatch(prefix="a proposta atual de regulação cria custos altos para as", words=10),
            "A proposta atual de regulação cria custos altos para as pequenas cooperativas do "
            "interior.",
        ),
        (
            1,
            "Marcos Pereira",
            1,
            QuoteMatch(prefix="prazos de transição", words=3),
            "O setor precisa de prazos de transição mais longos e de regras claras.",
        ),
        (
            1,
            "João Silva",
            0,
            QuoteMatch(prefix="o texto volte para a comissão de mérito", words=8),
            "Defendo que o texto volte para a comissão de mérito.",
        ),
        (
            2,
            "Mestre Zezé",
            0,
            QuoteMatch(
                prefix="os mestres da cultura popular precisam de reconhecimento formal do",
                words=10,
            ),
            "Os mestres da cultura popular precisam de reconhecimento formal do Estado.",
        ),
    ],
)
def test_quote_matches_on_fixture(
    mini_hearings: dict[int, HearingRecord],
    hearing_id: int,
    name: str,
    opinion_index: int,
    expected: QuoteMatch,
    sentence: str,
) -> None:
    hearing = mini_hearings[hearing_id]
    person = next(person for person in hearing.metadados.envolvidos if person.nome == name)
    _, speech = resolve_person_speech(person.nome, split_into_turns(hearing.transcricao))
    match = find_opinion_quote_match(person.opinioes[opinion_index], speech)
    assert match == expected
    assert enclosing_sentence(expected.prefix, speech) == sentence
