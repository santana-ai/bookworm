"""Publication date of each article and its check against the weekday mentions in the text."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime

from bookworm.data.io import JsonObject, json_dict_factory
from bookworm.data.schemas import HearingRecord
from bookworm.transcript.text import strip_accents

TIMESTAMP_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})")
UPDATED_AT_PATTERN = re.compile(
    r"Atualizado\s+em\s+(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})", re.IGNORECASE
)
WEEKDAY_MENTION_PATTERN = re.compile(
    r"(nest[ae]\s+)?(segunda|ter[cç]a|quarta|quinta|sexta)-feira\s*\((\d{1,2})\)", re.IGNORECASE
)
WEEKDAY_INDEX = {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4}
MENTION_LOOKBACK_DAYS = 30


@dataclass(frozen=True, slots=True)
class WeekdayMention:
    text: str
    names_current_event: bool
    weekday: int
    day_of_month: int

    def to_dict(self) -> JsonObject:
        return asdict(self, dict_factory=json_dict_factory)


@dataclass(frozen=True, slots=True)
class MentionCheck:
    text: str
    names_current_event: bool
    resolved_date: date | None
    lag_days: int | None
    weekday_agrees: bool

    def to_dict(self) -> JsonObject:
        return asdict(self, dict_factory=json_dict_factory)


@dataclass(frozen=True, slots=True)
class ArticleDateCheck:
    article_date: date | None
    updated_at: datetime | None
    mentions: tuple[MentionCheck, ...]

    def to_dict(self) -> JsonObject:
        return asdict(self, dict_factory=json_dict_factory)


@dataclass(frozen=True, slots=True)
class DateDisagreement:
    hearing_id: int
    article_date: date | None
    mention: str
    resolved_date: date | None


@dataclass(frozen=True, slots=True)
class DateExtractionSummary:
    hearings: int
    with_publication_timestamp: int
    with_update_timestamp: int
    with_current_event_mention: int
    confirmed_by_at_least_one_mention: int
    confirmed_by_no_mention: int
    current_event_mentions: int
    weekday_agrees: int
    weekday_disagrees: int
    lag_days_when_agreeing: dict[int, int]
    disagreements: tuple[DateDisagreement, ...]
    hearings_without_any_agreeing_mention: tuple[int, ...]

    @property
    def max_lag_days(self) -> int:
        return max(self.lag_days_when_agreeing, default=0)

    def to_dict(self) -> JsonObject:
        return asdict(self, dict_factory=json_dict_factory)


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
    """Date of the first timestamp of an article outside an ``Atualizado em``, if any."""
    published = published_at(article)
    return published.date() if published else None


def weekday_mentions(article: str) -> list[WeekdayMention]:
    return [
        WeekdayMention(
            text=match.group(0).strip(),
            names_current_event=match.group(1) is not None,
            weekday=WEEKDAY_INDEX[strip_accents(match.group(2)).lower()],
            day_of_month=int(match.group(3)),
        )
        for match in WEEKDAY_MENTION_PATTERN.finditer(article)
    ]


def resolve_mention_date(reference: date, day_of_month: int) -> date | None:
    for offset in range(MENTION_LOOKBACK_DAYS + 1):
        candidate = date.fromordinal(reference.toordinal() - offset)
        if candidate.day == day_of_month:
            return candidate
    return None


def check_weekday_mention(reference: date, mention: WeekdayMention) -> MentionCheck:
    resolved = resolve_mention_date(reference, mention.day_of_month)
    return MentionCheck(
        text=mention.text,
        names_current_event=mention.names_current_event,
        resolved_date=resolved,
        lag_days=reference.toordinal() - resolved.toordinal() if resolved else None,
        weekday_agrees=resolved is not None and resolved.weekday() == mention.weekday,
    )


def check_article_date(article: str) -> ArticleDateCheck:
    """Compare the publication date with the weekday mentions of the article."""
    reference = article_date(article)
    if reference is None:
        return ArticleDateCheck(article_date=None, updated_at=None, mentions=())
    return ArticleDateCheck(
        article_date=reference,
        updated_at=updated_at(article),
        mentions=tuple(check_weekday_mention(reference, m) for m in weekday_mentions(article)),
    )


def summarize_date_extraction(hearings: Sequence[HearingRecord]) -> DateExtractionSummary:
    checks = {hearing.id: check_article_date(hearing.materia) for hearing in hearings}
    current = [
        (hearing_id, mention)
        for hearing_id, check in checks.items()
        for mention in check.mentions
        if mention.names_current_event
    ]
    agreeing = [pair for pair in current if pair[1].weekday_agrees]
    mentioning_hearings = {hearing_id for hearing_id, _ in current}
    confirmed_hearings = {hearing_id for hearing_id, _ in agreeing}
    lag_counts = Counter(
        mention.lag_days for _, mention in agreeing if mention.lag_days is not None
    )
    return DateExtractionSummary(
        hearings=len(hearings),
        with_publication_timestamp=sum(1 for c in checks.values() if c.article_date),
        with_update_timestamp=sum(1 for c in checks.values() if c.updated_at),
        with_current_event_mention=len(mentioning_hearings),
        confirmed_by_at_least_one_mention=len(confirmed_hearings),
        confirmed_by_no_mention=len(mentioning_hearings - confirmed_hearings),
        current_event_mentions=len(current),
        weekday_agrees=len(agreeing),
        weekday_disagrees=len(current) - len(agreeing),
        lag_days_when_agreeing=dict(sorted(lag_counts.items())),
        disagreements=tuple(
            DateDisagreement(
                hearing_id=hearing_id,
                article_date=checks[hearing_id].article_date,
                mention=mention.text,
                resolved_date=mention.resolved_date,
            )
            for hearing_id, mention in current
            if not mention.weekday_agrees
        ),
        hearings_without_any_agreeing_mention=tuple(
            sorted(mentioning_hearings - confirmed_hearings)
        ),
    )
