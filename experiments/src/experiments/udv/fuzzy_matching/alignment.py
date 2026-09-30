"""Approximate alignment of a quote prefix with a speaker's turns, by characters or by tokens."""

from dataclasses import dataclass
from functools import cache
from typing import Any

from rapidfuzz import fuzz

from experiments.common.transcript import WORD_TOKEN_PATTERN, strip_accents, turn_text
from experiments.udv.fuzzy_matching.config import FuzzyConfig

Record = dict[str, Any]


@dataclass(frozen=True)
class TurnView:
    turn_index: int
    text: str
    folded: str
    tokens: tuple[str, ...]
    token_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QuotePrefix:
    quote_index: int
    level: int
    words: int
    text: str
    folded: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Alignment:
    score: float
    turn_index: int
    start: int
    end: int


@cache
def fold_character(character: str) -> str:
    for candidate in (strip_accents(character).lower(), character.lower()):
        if len(candidate) == 1:
            return candidate
    return character


def fold_text(text: str) -> str:
    """Lower case without accents, one character for one, so offsets carry over."""
    return "".join(fold_character(character) for character in text)


def build_turn_view(turn: Record) -> TurnView:
    text = turn_text(turn)
    folded = fold_text(text)
    if len(folded) != len(text):
        raise SystemExit(f"folding changed the length of turn {turn['turn_index']}")
    hits = list(WORD_TOKEN_PATTERN.finditer(folded))
    return TurnView(
        turn_index=turn["turn_index"],
        text=text,
        folded=folded,
        tokens=tuple(hit.group() for hit in hits),
        token_spans=tuple(hit.span() for hit in hits),
    )


def quote_prefix(quote_index: int, level: int, text: str) -> QuotePrefix:
    folded = fold_text(text)
    return QuotePrefix(
        quote_index=quote_index,
        level=level,
        words=len(text.split()),
        text=text,
        folded=folded,
        tokens=tuple(WORD_TOKEN_PATTERN.findall(folded)),
    )


def quote_prefix_levels(quote: str, quote_index: int, config: FuzzyConfig) -> list[QuotePrefix]:
    words = quote.split()
    prefixes: list[QuotePrefix] = []
    seen: set[int] = set()
    for level in config.prefix_levels:
        count = min(level, len(words))
        if count < config.min_prefix_words or count in seen:
            continue
        seen.add(count)
        prefixes.append(quote_prefix(quote_index, level, " ".join(words[:count])))
    return prefixes


def quotes_prefixes(quotes: list[str], config: FuzzyConfig) -> list[QuotePrefix]:
    return [
        prefix
        for quote_index, quote in enumerate(quotes)
        for prefix in quote_prefix_levels(quote, quote_index, config)
    ]


def is_eligible_quote(quote: str, config: FuzzyConfig) -> bool:
    return len(quote.split()) >= config.min_prefix_words


def char_alignment(prefix: QuotePrefix, view: TurnView) -> Alignment:
    if len(view.folded) < len(prefix.folded):
        return Alignment(fuzz.ratio(prefix.folded, view.folded), view.turn_index, 0, len(view.text))
    hit = fuzz.partial_ratio_alignment(prefix.folded, view.folded)
    assert hit is not None
    return Alignment(hit.score, view.turn_index, hit.dest_start, hit.dest_end)


def token_alignment(prefix: QuotePrefix, view: TurnView) -> Alignment | None:
    if not prefix.tokens or not view.tokens:
        return None
    needle, haystack = list(prefix.tokens), list(view.tokens)
    if len(haystack) < len(needle):
        score, first, last = fuzz.ratio(needle, haystack), 0, len(haystack)
    else:
        hit = fuzz.partial_ratio_alignment(needle, haystack)
        assert hit is not None
        score, first, last = hit.score, hit.dest_start, hit.dest_end
    if last <= first:
        return Alignment(score, view.turn_index, 0, 0)
    return Alignment(
        score, view.turn_index, view.token_spans[first][0], view.token_spans[last - 1][1]
    )


def align(prefix: QuotePrefix, view: TurnView, method: str) -> Alignment | None:
    if method == "char":
        return char_alignment(prefix, view)
    return token_alignment(prefix, view)


def best_alignment(
    prefixes: list[QuotePrefix], views: list[TurnView], method: str, level: int
) -> tuple[QuotePrefix, Alignment] | None:
    """The highest-scoring alignment of the prefixes of one level; the first one wins ties."""
    best: tuple[QuotePrefix, Alignment] | None = None
    for prefix in prefixes:
        if prefix.level != level:
            continue
        for view in views:
            alignment = align(prefix, view, method)
            if alignment is not None and (best is None or alignment.score > best[1].score):
                best = (prefix, alignment)
    return best
