import json
from datetime import date, datetime

import pytest

from bookworm import HearingRecord, check_article_date, summarize_date_extraction
from bookworm.data.dates import (
    MENTION_LOOKBACK_DAYS,
    ArticleDateCheck,
    MentionCheck,
    WeekdayMention,
    article_date,
    check_weekday_mention,
    published_at,
    resolve_mention_date,
    updated_at,
    weekday_mentions,
)


def test_published_at_reads_the_first_timestamp() -> None:
    article = "Título\n12/03/2024 - 15:30\nTexto citando 14/03/2024 - 10:00 depois."
    assert published_at(article) == datetime(2024, 3, 12, 15, 30)
    assert article_date(article) == date(2024, 3, 12)


def test_published_at_tolerates_missing_spaces_around_the_dash() -> None:
    assert published_at("05/04/2023-09:05") == datetime(2023, 4, 5, 9, 5)


def test_published_at_skips_the_update_timestamp_that_comes_first() -> None:
    article = "Atualizado em 03/03/2023 - 09:15\nTítulo\n01/03/2023 - 16:20\nTexto."
    assert published_at(article) == datetime(2023, 3, 1, 16, 20)
    assert updated_at(article) == datetime(2023, 3, 3, 9, 15)


def test_update_marker_is_case_insensitive() -> None:
    article = "ATUALIZADO   EM 20/06/2023 - 08:00\n19/06/2023 - 21:10"
    assert article_date(article) == date(2023, 6, 19)
    assert updated_at(article) == datetime(2023, 6, 20, 8, 0)


def test_publication_after_the_update_is_still_found() -> None:
    article = "Título\n10/10/2023 - 11:00\nAtualizado em 11/10/2023 - 12:00\nTexto."
    assert published_at(article) == datetime(2023, 10, 10, 11, 0)
    assert updated_at(article) == datetime(2023, 10, 11, 12, 0)


def test_article_without_timestamp() -> None:
    article = "Matéria sem carimbo; só cita 10/10/2023, sem horário."
    assert published_at(article) is None
    assert article_date(article) is None
    assert updated_at(article) is None


def test_only_an_update_timestamp_gives_no_publication_date() -> None:
    assert article_date("Atualizado em 03/03/2023 - 09:15") is None


def test_weekday_mentions() -> None:
    article = (
        "Debate nesta quarta-feira (1). Na terça-feira (28) houve outro; "
        "Neste Sexta-Feira (3) e nesta terca-feira ( 7 ) também; desta quinta-feira (9)."
    )
    assert weekday_mentions(article) == [
        WeekdayMention("nesta quarta-feira (1)", True, 2, 1),
        WeekdayMention("terça-feira (28)", False, 1, 28),
        WeekdayMention("Neste Sexta-Feira (3)", True, 4, 3),
        WeekdayMention("quinta-feira (9)", False, 3, 9),
    ]


def test_weekday_mention_serializes_in_field_order() -> None:
    mention = weekday_mentions("nesta terça-feira (14)")[0]
    assert list(mention.to_dict().items()) == [
        ("text", "nesta terça-feira (14)"),
        ("names_current_event", True),
        ("weekday", 1),
        ("day_of_month", 14),
    ]


def test_weekday_mention_needs_digits_right_after_the_parenthesis() -> None:
    assert weekday_mentions("nesta quarta-feira ( 7 ) e nesta quarta-feira (1º)") == []


@pytest.mark.parametrize(
    ("reference", "day_of_month", "expected"),
    [
        (date(2023, 3, 16), 16, date(2023, 3, 16)),
        (date(2023, 3, 16), 15, date(2023, 3, 15)),
        (date(2023, 3, 2), 28, date(2023, 2, 28)),
        (date(2024, 3, 1), 29, date(2024, 2, 29)),
        (date(2023, 3, 15), 31, None),
        (date(2023, 3, 30), 1, date(2023, 3, 1)),
    ],
)
def test_resolve_mention_date(reference: date, day_of_month: int, expected: date | None) -> None:
    assert resolve_mention_date(reference, day_of_month) == expected


def test_lookback_window_is_inclusive_at_thirty_days() -> None:
    assert MENTION_LOOKBACK_DAYS == 30
    assert resolve_mention_date(date(2023, 3, 1), 30) == date(2023, 1, 30)
    assert resolve_mention_date(date(2023, 3, 2), 30) is None


def test_check_weekday_mention_agrees_with_a_one_day_lag() -> None:
    mention = WeekdayMention("nesta quarta-feira (1)", True, 2, 1)
    assert check_weekday_mention(date(2023, 3, 2), mention) == MentionCheck(
        text="nesta quarta-feira (1)",
        names_current_event=True,
        resolved_date=date(2023, 3, 1),
        lag_days=1,
        weekday_agrees=True,
    )


def test_check_weekday_mention_detects_a_weekday_disagreement() -> None:
    mention = WeekdayMention("nesta sexta-feira (15)", True, 4, 15)
    check = check_weekday_mention(date(2023, 3, 16), mention)
    assert check.resolved_date == date(2023, 3, 15)
    assert check.lag_days == 1
    assert not check.weekday_agrees


def test_check_weekday_mention_without_a_resolvable_day() -> None:
    mention = WeekdayMention("nesta sexta-feira (31)", True, 4, 31)
    check = check_weekday_mention(date(2023, 3, 15), mention)
    assert (check.resolved_date, check.lag_days, check.weekday_agrees) == (None, None, False)
    assert check.to_dict() == {
        "text": "nesta sexta-feira (31)",
        "names_current_event": True,
        "resolved_date": None,
        "lag_days": None,
        "weekday_agrees": False,
    }


def test_check_article_date_serializes_in_the_reference_key_order() -> None:
    article = "Atualizado em 03/03/2023 - 09:15\n02/03/2023 - 10:05\nnesta quarta-feira (1)"
    check = check_article_date(article)
    assert check == ArticleDateCheck(
        article_date=date(2023, 3, 2),
        updated_at=datetime(2023, 3, 3, 9, 15),
        mentions=(MentionCheck("nesta quarta-feira (1)", True, date(2023, 3, 1), 1, True),),
    )
    payload = check.to_dict()
    assert json.dumps(payload) == json.dumps(
        {
            "article_date": "2023-03-02",
            "updated_at": "2023-03-03T09:15:00",
            "mentions": [
                {
                    "text": "nesta quarta-feira (1)",
                    "names_current_event": True,
                    "resolved_date": "2023-03-01",
                    "lag_days": 1,
                    "weekday_agrees": True,
                }
            ],
        }
    )


def test_check_article_date_without_timestamp_ignores_mentions() -> None:
    check = check_article_date("Sem data, mas nesta quarta-feira (1) houve debate.")
    assert check.to_dict() == {"article_date": None, "updated_at": None, "mentions": []}


def test_summarize_date_extraction_on_the_fixture(split_hearings: list[HearingRecord]) -> None:
    summary = summarize_date_extraction(split_hearings)
    assert summary.max_lag_days == 1
    assert summary.to_dict() == {
        "hearings": 10,
        "with_publication_timestamp": 10,
        "with_update_timestamp": 1,
        "with_current_event_mention": 7,
        "confirmed_by_at_least_one_mention": 6,
        "confirmed_by_no_mention": 1,
        "current_event_mentions": 7,
        "weekday_agrees": 6,
        "weekday_disagrees": 1,
        "lag_days_when_agreeing": {0: 4, 1: 2},
        "disagreements": [
            {
                "hearing_id": 6,
                "article_date": "2023-03-16",
                "mention": "nesta sexta-feira (15)",
                "resolved_date": "2023-03-15",
            }
        ],
        "hearings_without_any_agreeing_mention": [6],
    }


def test_summarize_date_extraction_of_nothing() -> None:
    summary = summarize_date_extraction([])
    assert summary.max_lag_days == 0
    assert summary.lag_days_when_agreeing == {}
    assert summary.hearings == 0


def test_fixture_dates_skip_the_update_stamp(split_hearings: list[HearingRecord]) -> None:
    by_id = {hearing.id: hearing for hearing in split_hearings}
    assert by_id[2].materia.startswith("Atualizado em 03/03/2023")
    assert article_date(by_id[2].materia) == date(2023, 3, 1)
