import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeGuard, get_args

from bookworm.transcript.sentences import enclosing_turn_sentence, sentence_part_spans, turn_text
from bookworm.transcript.text import normalize_whitespace, strip_accents
from bookworm.transcript.text import prefix_pattern as quote_prefix_pattern
from bookworm.transcript.turns import Turn

__all__ = [
    "DEFAULT_QUOTE_POLICY",
    "DOUBLE_QUOTE_PATTERN",
    "DOUBLE_QUOTE_PATTERNS",
    "QUOTE_EXTENTS",
    "QUOTE_PATTERNS",
    "SINGLE_QUOTE_PATTERN",
    "PrefixMatch",
    "PrefixOccurrence",
    "QuoteExtent",
    "QuoteExtentMode",
    "QuoteMatch",
    "QuotePolicy",
    "TurnQuoteMatch",
    "choose_occurrence_index",
    "extend_quote_match",
    "extract_quotes",
    "find_opinion_quote_match",
    "find_opinion_turn_quote_match",
    "find_prefix_occurrences",
    "find_quote_end",
    "find_quote_match",
    "find_turn_quote_match",
    "is_trusted_quote",
    "quote_part_count",
    "quote_prefix_pattern",
    "quote_prefixes",
    "quote_suffixes",
    "token_jaccard",
    "word_tokens",
]

DOUBLE_QUOTE_PATTERN = re.compile(r'“([^”]{10,})”|"([^"]{10,})"')
SINGLE_QUOTE_PATTERN = re.compile(r"(?<!\w)'([^']{10,})'(?!\w)")
DOUBLE_QUOTE_PATTERNS: tuple[re.Pattern[str], ...] = (DOUBLE_QUOTE_PATTERN,)
QUOTE_PATTERNS: tuple[re.Pattern[str], ...] = (DOUBLE_QUOTE_PATTERN, SINGLE_QUOTE_PATTERN)
WORD_TOKEN_PATTERN = re.compile(r"\w+")
TRAILING_NON_WORD_PATTERN = re.compile(r"\W+$")
QuoteExtentMode = Literal["prefix_sentence", "full_quote"]
QUOTE_EXTENTS: tuple[QuoteExtentMode, ...] = get_args(QuoteExtentMode)
QuoteExtentRule = Literal["suffix", "sentence_count"]


@dataclass(frozen=True, slots=True)
class QuotePolicy:
    prefix_lengths: tuple[int, ...] = (10, 6, 4, 3)
    min_prefix_chars: int = 6
    trusted_prefix_words: int = 6
    patterns: tuple[re.Pattern[str], ...] = QUOTE_PATTERNS
    suffix_lengths: tuple[int, ...] = (6, 4, 3)
    max_span_ratio: float = 2.0


@dataclass(frozen=True, slots=True)
class QuoteMatch:
    prefix: str
    words: int


@dataclass(frozen=True, slots=True)
class PrefixOccurrence:
    turn_index: int
    start: int
    end: int
    sentence: str


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    prefix: str
    words: int
    occurrences: tuple[PrefixOccurrence, ...]


@dataclass(frozen=True, slots=True)
class TurnQuoteMatch:
    prefix: str
    words: int
    quote_index: int
    occurrence_count: int
    occurrence_index: int
    turn_index: int
    start: int
    end: int
    sentence: str


@dataclass(frozen=True, slots=True)
class QuoteExtent:
    text: str
    rule: QuoteExtentRule
    suffix_words: int | None
    start: int
    end: int


class CountedMatch(Protocol):
    @property
    def words(self) -> int: ...


DEFAULT_QUOTE_POLICY = QuotePolicy()


def quoted_text(match: re.Match[str]) -> str:
    return next(group for group in match.groups() if group is not None)


def extract_quotes(
    opinion_text: str, patterns: Sequence[re.Pattern[str]] = QUOTE_PATTERNS
) -> list[str]:
    matches = sorted(
        (match for pattern in patterns for match in pattern.finditer(opinion_text)),
        key=lambda match: (match.start(), -match.end()),
    )
    return [normalize_whitespace(quoted_text(match)) for match in matches]


def quote_prefixes(quote: str, policy: QuotePolicy = DEFAULT_QUOTE_POLICY) -> list[tuple[str, int]]:
    words = quote.split()
    prefixes: list[tuple[str, int]] = []
    for prefix_length in policy.prefix_lengths:
        prefix = " ".join(words[:prefix_length])
        if len(prefix) >= policy.min_prefix_chars:
            prefixes.append((prefix, min(prefix_length, len(words))))
    return prefixes


def find_quote_match(
    quote: str, person_speech: str, policy: QuotePolicy = DEFAULT_QUOTE_POLICY
) -> QuoteMatch | None:
    for prefix, words in quote_prefixes(quote, policy):
        if quote_prefix_pattern(prefix).search(person_speech):
            return QuoteMatch(prefix=prefix, words=words)
    return None


def find_opinion_quote_match(
    opinion_text: str, person_speech: str, policy: QuotePolicy = DEFAULT_QUOTE_POLICY
) -> QuoteMatch | None:
    for quote in extract_quotes(opinion_text, policy.patterns):
        match = find_quote_match(quote, person_speech, policy)
        if match is not None:
            return match
    return None


def is_trusted_quote[MatchT: CountedMatch](
    match: MatchT | None, policy: QuotePolicy = DEFAULT_QUOTE_POLICY
) -> TypeGuard[MatchT]:
    return match is not None and match.words >= policy.trusted_prefix_words


def find_prefix_occurrences(prefix: str, turns: Sequence[Turn]) -> list[PrefixOccurrence]:
    pattern = quote_prefix_pattern(prefix)
    occurrences: list[PrefixOccurrence] = []
    for turn in turns:
        text = turn_text(turn)
        occurrences.extend(
            PrefixOccurrence(
                turn_index=turn.turn_index,
                start=hit.start(),
                end=hit.end(),
                sentence=enclosing_turn_sentence(text, hit.start(), hit.end()),
            )
            for hit in pattern.finditer(text)
        )
    return occurrences


def find_turn_quote_match(
    quote: str, turns: Sequence[Turn], policy: QuotePolicy = DEFAULT_QUOTE_POLICY
) -> PrefixMatch | None:
    for prefix, words in quote_prefixes(quote, policy):
        occurrences = find_prefix_occurrences(prefix, turns)
        if occurrences:
            return PrefixMatch(prefix=prefix, words=words, occurrences=tuple(occurrences))
    return None


def word_tokens(text: str) -> set[str]:
    return set(WORD_TOKEN_PATTERN.findall(strip_accents(text).lower()))


def token_jaccard(text: str, other_text: str) -> float:
    tokens, other_tokens = word_tokens(text), word_tokens(other_text)
    union = tokens | other_tokens
    return len(tokens & other_tokens) / len(union) if union else 0.0


def choose_occurrence_index(occurrences: Sequence[PrefixOccurrence], opinion_text: str) -> int:
    return max(
        range(len(occurrences)),
        key=lambda index: token_jaccard(occurrences[index].sentence, opinion_text),
    )


def find_opinion_turn_quote_match(
    opinion_text: str, turns: Sequence[Turn], policy: QuotePolicy = DEFAULT_QUOTE_POLICY
) -> TurnQuoteMatch | None:
    best: tuple[int, PrefixMatch] | None = None
    for quote_index, quote in enumerate(extract_quotes(opinion_text, policy.patterns)):
        match = find_turn_quote_match(quote, turns, policy)
        if match is not None and (best is None or match.words > best[1].words):
            best = (quote_index, match)
    if best is None:
        return None
    quote_index, match = best
    occurrences = match.occurrences
    occurrence_index = (
        choose_occurrence_index(occurrences, opinion_text) if is_trusted_quote(match, policy) else 0
    )
    occurrence = occurrences[occurrence_index]
    return TurnQuoteMatch(
        prefix=match.prefix,
        words=match.words,
        quote_index=quote_index,
        occurrence_count=len(occurrences),
        occurrence_index=occurrence_index,
        turn_index=occurrence.turn_index,
        start=occurrence.start,
        end=occurrence.end,
        sentence=occurrence.sentence,
    )


def quote_suffixes(quote: str, policy: QuotePolicy = DEFAULT_QUOTE_POLICY) -> list[tuple[str, int]]:
    words = quote.split()
    suffixes: list[tuple[str, int]] = []
    for suffix_length in policy.suffix_lengths:
        if len(words) < suffix_length:
            continue
        suffix = TRAILING_NON_WORD_PATTERN.sub("", " ".join(words[-suffix_length:]))
        if len(suffix) >= policy.min_prefix_chars:
            suffixes.append((suffix, suffix_length))
    return suffixes


def find_quote_end(
    quote: str, text: str, start: int, prefix_end: int, policy: QuotePolicy = DEFAULT_QUOTE_POLICY
) -> tuple[int, int] | None:
    limit = start + policy.max_span_ratio * len(quote)
    for suffix, words in quote_suffixes(quote, policy):
        for hit in quote_prefix_pattern(suffix).finditer(text, start):
            if hit.end() > limit:
                break
            if hit.end() >= prefix_end:
                return hit.end(), words
    return None


def quote_part_count(quote: str) -> int:
    return sum(
        1
        for start, end in sentence_part_spans(quote)
        if WORD_TOKEN_PATTERN.search(quote[start:end])
    )


def sentence_count_end(text: str, start: int, prefix_end: int, parts_wanted: int) -> int:
    parts = sentence_part_spans(text)
    covering = [index for index, (a, b) in enumerate(parts) if a < prefix_end and start < b]
    last = max(covering[-1], min(covering[0] + parts_wanted - 1, len(parts) - 1))
    return parts[last][1]


def extend_quote_match(
    opinion_text: str,
    match: TurnQuoteMatch,
    turns: Sequence[Turn],
    policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> QuoteExtent:
    quote = extract_quotes(opinion_text, policy.patterns)[match.quote_index]
    turn = next(turn for turn in turns if turn.turn_index == match.turn_index)
    text = turn_text(turn)
    found = find_quote_end(quote, text, match.start, match.end, policy)
    if found is not None:
        end, words = found
        return QuoteExtent(
            text=enclosing_turn_sentence(text, match.start, end),
            rule="suffix",
            suffix_words=words,
            start=match.start,
            end=end,
        )
    end = sentence_count_end(text, match.start, match.end, quote_part_count(quote))
    return QuoteExtent(
        text=enclosing_turn_sentence(text, match.start, end),
        rule="sentence_count",
        suffix_words=None,
        start=match.start,
        end=end,
    )
