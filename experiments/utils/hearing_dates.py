import re
import unicodedata
from datetime import date, datetime
from typing import Any

Record = dict[str, Any]

TIMESTAMP_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})")
UPDATED_AT_PATTERN = re.compile(
    r"Atualizado\s+em\s+(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})", re.IGNORECASE
)
WEEKDAY_MENTION_PATTERN = re.compile(
    r"(nest[ae]\s+)?(segunda|ter[cç]a|quarta|quinta|sexta)-feira\s*\((\d{1,2})\)", re.IGNORECASE
)
WEEKDAY_INDEX = {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4}
MENTION_LOOKBACK_DAYS = 30


def strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def parse_timestamp(match: re.Match[str]) -> datetime:
    day, month, year, hour, minute = (int(group) for group in match.groups())
    return datetime(year, month, day, hour, minute)


def updated_spans(article: str) -> list[tuple[int, int]]:
    return [(match.start(1), match.end(5)) for match in UPDATED_AT_PATTERN.finditer(article)]


def published_at(article: str) -> datetime | None:
    skip = updated_spans(article)
    for match in TIMESTAMP_PATTERN.finditer(article):
        if any(start <= match.start() < end for start, end in skip):
            continue
        return parse_timestamp(match)
    return None


def updated_at(article: str) -> datetime | None:
    match = UPDATED_AT_PATTERN.search(article)
    return parse_timestamp(match) if match else None


def article_date(article: str) -> date | None:
    published = published_at(article)
    return published.date() if published else None


def weekday_mentions(article: str) -> list[Record]:
    return [
        {
            "text": match.group(0).strip(),
            "names_current_event": match.group(1) is not None,
            "weekday": WEEKDAY_INDEX[strip_accents(match.group(2)).lower()],
            "day_of_month": int(match.group(3)),
        }
        for match in WEEKDAY_MENTION_PATTERN.finditer(article)
    ]


def resolve_mention_date(reference: date, day_of_month: int) -> date | None:
    for offset in range(MENTION_LOOKBACK_DAYS + 1):
        candidate = date.fromordinal(reference.toordinal() - offset)
        if candidate.day == day_of_month:
            return candidate
    return None


def check_weekday_mention(reference: date, mention: Record) -> Record:
    resolved = resolve_mention_date(reference, mention["day_of_month"])
    return {
        "text": mention["text"],
        "names_current_event": mention["names_current_event"],
        "resolved_date": resolved.isoformat() if resolved else None,
        "lag_days": reference.toordinal() - resolved.toordinal() if resolved else None,
        "weekday_agrees": resolved is not None and resolved.weekday() == mention["weekday"],
    }


def check_article_date(article: str) -> Record:
    reference = article_date(article)
    if reference is None:
        return {"article_date": None, "updated_at": None, "mentions": []}
    updated = updated_at(article)
    return {
        "article_date": reference.isoformat(),
        "updated_at": updated.isoformat() if updated else None,
        "mentions": [check_weekday_mention(reference, m) for m in weekday_mentions(article)],
    }
