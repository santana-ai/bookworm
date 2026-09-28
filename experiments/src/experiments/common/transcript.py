"""Dict views of the bookworm transcript and quote rules, for the experiment scripts.

The experiment scripts pass turns, quote matches and spans around as plain dicts, the shape they
had before the rules moved into the library. Every rule here delegates to ``bookworm``; this
module only converts between those dicts and the library's dataclasses, so the rules exist in
one place. ``SOURCES`` lists the files a result computed through this module depends on.
"""

import dataclasses
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, TypeGuard

import bookworm.transcript.offsets
import bookworm.transcript.sentences
import bookworm.transcript.speakers
import bookworm.transcript.text
import bookworm.transcript.turns
import bookworm.udv.quotes
from bookworm.transcript.offsets import Span
from bookworm.transcript.sentences import (
    MIN_SENTENCE_WORDS,
    SENTENCE_BOUNDARY_PATTERN,
    STAGE_DIRECTION_PATTERN,
    enclosing_sentence,
    enclosing_turn_sentence,
    is_sentence,
    sentence_part_spans,
    sentences_agree,
    split_sentences,
)
from bookworm.transcript.speakers import PARTY_INFO_MARKERS, is_party_info, names_match
from bookworm.transcript.text import normalize_name, normalize_whitespace, strip_accents
from bookworm.transcript.turns import TURN_HEADER_PATTERN, Turn
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    DOUBLE_QUOTE_PATTERN,
    DOUBLE_QUOTE_PATTERNS,
    QUOTE_PATTERNS,
    SINGLE_QUOTE_PATTERN,
    WORD_TOKEN_PATTERN,
    extract_quotes,
    quote_prefix_pattern,
    quote_prefixes,
    quoted_text,
    token_jaccard,
    word_tokens,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = [
    "DOUBLE_QUOTE_PATTERN",
    "DOUBLE_QUOTE_PATTERNS",
    "MIN_SENTENCE_WORDS",
    "PARTY_INFO_MARKERS",
    "QUOTE_PATTERNS",
    "QUOTE_PREFIX_LENGTHS",
    "SENTENCE_BOUNDARY_PATTERN",
    "SINGLE_QUOTE_PATTERN",
    "SOURCES",
    "STAGE_DIRECTION_PATTERN",
    "TRUSTED_PREFIX_WORDS",
    "TURN_HEADER_PATTERN",
    "UDV_CONFIG_PATH",
    "WORD_TOKEN_PATTERN",
    "best_embedding_match",
    "best_semantic_match",
    "choose_occurrence",
    "enclosing_sentence",
    "enclosing_turn_sentence",
    "extract_quotes",
    "find_opinion_quote_evidence",
    "find_opinion_quote_match",
    "find_opinion_turn_quote_match",
    "find_prefix_occurrences",
    "find_quote_match",
    "find_turn_quote_match",
    "get_embedding_model",
    "is_party_info",
    "is_sentence",
    "is_trusted_quote",
    "load_encoder_spec",
    "locate_sentence_span",
    "locate_turn_sentence_span",
    "matching_turns",
    "names_match",
    "normalize_name",
    "normalize_whitespace",
    "quote_prefix_pattern",
    "quote_prefixes",
    "quoted_text",
    "resolve_person_speech",
    "resolve_turn_name",
    "sentence_part_spans",
    "sentences_agree",
    "single_token_matching_turns",
    "split_into_turns",
    "split_sentences",
    "split_turn_sentences",
    "strip_accents",
    "token_jaccard",
    "turn_name_candidates",
    "turn_text",
    "word_tokens",
]

Record = dict[str, Any]

UDV_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "udv.toml"
QUOTE_PREFIX_LENGTHS = DEFAULT_QUOTE_POLICY.prefix_lengths
TRUSTED_PREFIX_WORDS = DEFAULT_QUOTE_POLICY.trusted_prefix_words
TURN_FIELDS = tuple(field.name for field in dataclasses.fields(Turn))
SOURCES: tuple[ModuleType, ...] = (
    bookworm.transcript.text,
    bookworm.transcript.turns,
    bookworm.transcript.speakers,
    bookworm.transcript.sentences,
    bookworm.transcript.offsets,
    bookworm.udv.quotes,
)


def as_turn(turn: Record) -> Turn:
    return Turn(**{name: turn[name] for name in TURN_FIELDS})


def as_turns(turns: Sequence[Record]) -> list[Turn]:
    return [as_turn(turn) for turn in turns]


def original_turns(
    matched: Sequence[Turn], turns: Sequence[Turn], records: Sequence[Record]
) -> list[Record]:
    position = {id(turn): index for index, turn in enumerate(turns)}
    return [records[position[id(turn)]] for turn in matched]


def span_record(span: Span | None) -> Record | None:
    return None if span is None else dataclasses.asdict(span)


def split_into_turns(transcript: str) -> list[Record]:
    return [
        dataclasses.asdict(turn) for turn in bookworm.transcript.turns.split_into_turns(transcript)
    ]


def resolve_turn_name(turn: Record) -> str:
    return bookworm.transcript.speakers.resolve_turn_name(as_turn(turn))


def turn_name_candidates(turn: Record) -> list[str]:
    return bookworm.transcript.speakers.turn_name_candidates(as_turn(turn))


def matching_turns(name: str, turns: list[Record]) -> list[Record]:
    converted = as_turns(turns)
    matched = bookworm.transcript.speakers.matching_turns(name, converted)
    return original_turns(matched, converted, turns)


def single_token_matching_turns(name: str, turns: list[Record]) -> list[Record]:
    converted = as_turns(turns)
    matched = bookworm.transcript.speakers.single_token_matching_turns(name, converted)
    return original_turns(matched, converted, turns)


def resolve_person_speech(person: Record, turns: list[Record]) -> tuple[list[Record], str]:
    converted = as_turns(turns)
    matched, speech = bookworm.transcript.speakers.resolve_person_speech(person["nome"], converted)
    return original_turns(matched, converted, turns), speech


def turn_text(turn: Record) -> str:
    return bookworm.transcript.sentences.turn_text(as_turn(turn))


def split_turn_sentences(turns: list[Record]) -> list[Record]:
    return [
        dataclasses.asdict(unit)
        for unit in bookworm.transcript.sentences.split_turn_sentences(as_turns(turns))
    ]


def find_quote_match(quote: str, person_speech: str) -> Record | None:
    match = bookworm.udv.quotes.find_quote_match(quote, person_speech)
    return None if match is None else dataclasses.asdict(match)


def find_opinion_quote_match(opinion_text: str, person_speech: str) -> Record | None:
    match = bookworm.udv.quotes.find_opinion_quote_match(opinion_text, person_speech)
    return None if match is None else dataclasses.asdict(match)


def is_trusted_quote(match: Record | None) -> TypeGuard[Record]:
    return match is not None and match["words"] >= TRUSTED_PREFIX_WORDS


def find_opinion_quote_evidence(opinion_text: str, person_speech: str) -> str | None:
    match = find_opinion_quote_match(opinion_text, person_speech)
    return match["prefix"] if is_trusted_quote(match) else None


def find_prefix_occurrences(prefix: str, turns: list[Record]) -> list[Record]:
    return [
        dataclasses.asdict(occurrence)
        for occurrence in bookworm.udv.quotes.find_prefix_occurrences(prefix, as_turns(turns))
    ]


def find_turn_quote_match(quote: str, turns: list[Record]) -> Record | None:
    match = bookworm.udv.quotes.find_turn_quote_match(quote, as_turns(turns))
    if match is None:
        return None
    return {
        "prefix": match.prefix,
        "words": match.words,
        "occurrences": [dataclasses.asdict(occurrence) for occurrence in match.occurrences],
    }


def choose_occurrence(occurrences: list[Record], opinion_text: str) -> Record:
    return max(
        occurrences, key=lambda occurrence: token_jaccard(occurrence["sentence"], opinion_text)
    )


def find_opinion_turn_quote_match(
    opinion_text: str,
    turns: list[Record],
    patterns: tuple[re.Pattern[str], ...] = QUOTE_PATTERNS,
) -> Record | None:
    policy = dataclasses.replace(DEFAULT_QUOTE_POLICY, patterns=patterns)
    match = bookworm.udv.quotes.find_opinion_turn_quote_match(opinion_text, as_turns(turns), policy)
    return None if match is None else dataclasses.asdict(match)


def locate_sentence_span(sentence: str, transcript: str, turns: list[Record]) -> Record | None:
    return span_record(
        bookworm.transcript.offsets.locate_sentence_span(sentence, transcript, as_turns(turns))
    )


def locate_turn_sentence_span(
    sentence: str, transcript: str, turns: list[Record], turn_index: int
) -> Record | None:
    return span_record(
        bookworm.transcript.offsets.locate_turn_sentence_span(
            sentence, transcript, as_turns(turns), turn_index
        )
    )


def best_semantic_match(opinion_text: str, sentences: list[str]) -> tuple[str | None, float]:
    if not sentences:
        return None, 0.0
    corpus = [opinion_text] + sentences
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return None, 0.0
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])


_embedding_models: dict[tuple[str, str], Any] = {}


def load_encoder_spec(config_path: Path = UDV_CONFIG_PATH) -> tuple[str, str]:
    with open(config_path, "rb") as f:
        encoder = tomllib.load(f)["encoder"]
    return encoder["name"], encoder["revision"]


def get_embedding_model(config_path: Path = UDV_CONFIG_PATH) -> Any:
    spec = load_encoder_spec(config_path)
    if spec not in _embedding_models:
        from sentence_transformers import SentenceTransformer

        name, revision = spec
        _embedding_models[spec] = SentenceTransformer(name, revision=revision)
    return _embedding_models[spec]


def best_embedding_match(
    opinion_text: str, sentences: list[str], model: Any
) -> tuple[str | None, float]:
    if not sentences:
        return None, 0.0
    embeddings = model.encode([opinion_text] + sentences)
    similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])
