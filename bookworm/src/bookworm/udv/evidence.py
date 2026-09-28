from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics.pairwise import cosine_similarity

from bookworm.features.encoders import FloatMatrix
from bookworm.transcript.offsets import Span, locate_turn_sentence_span
from bookworm.transcript.sentences import sentences_agree
from bookworm.transcript.turns import Turn
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    QuotePolicy,
    TurnQuoteMatch,
    extend_quote_match,
)
from bookworm.udv.schemas import Evidence, Provenance, SupportType, Tier
from bookworm.udv.windows import CandidateUnit


@dataclass(frozen=True, slots=True)
class SentenceMatch:
    index: int
    score: float


def evidence_from_span(
    text: str,
    support_type: SupportType,
    score: float | None,
    quote_prefix: str | None,
    span: Span | None,
) -> Evidence:
    return Evidence(
        text=text,
        support_type=support_type,
        score=score,
        quote_prefix=quote_prefix,
        start_char=None if span is None else span.start_char,
        end_char=None if span is None else span.end_char,
        speaker_turn=None if span is None else span.speaker_turn,
    )


def build_quote_evidence(
    quote_match: TurnQuoteMatch, matched_turns: Sequence[Turn], transcript: str
) -> Evidence:
    sentence = quote_match.sentence
    span = locate_turn_sentence_span(sentence, transcript, matched_turns, quote_match.turn_index)
    return evidence_from_span(sentence, "direct_quote", None, quote_match.prefix, span)


def build_full_quote_evidence(
    quote_match: TurnQuoteMatch,
    opinion_text: str,
    matched_turns: Sequence[Turn],
    transcript: str,
    policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> Evidence:
    extent = extend_quote_match(opinion_text, quote_match, matched_turns, policy)
    span = locate_turn_sentence_span(extent.text, transcript, matched_turns, quote_match.turn_index)
    return evidence_from_span(extent.text, "direct_quote", None, quote_match.prefix, span)


def short_quote_supports(quote_match: TurnQuoteMatch | None, sentence: str) -> bool:
    if quote_match is None:
        return False
    return sentences_agree(quote_match.sentence, sentence)


def sentence_similarities(
    opinion_embedding: FloatMatrix, sentence_embeddings: FloatMatrix
) -> FloatMatrix:
    similarities: FloatMatrix = cosine_similarity(
        opinion_embedding.reshape(1, -1), sentence_embeddings
    ).flatten()
    return similarities


def best_sentence_match(
    opinion_embedding: FloatMatrix, sentence_embeddings: FloatMatrix
) -> SentenceMatch:
    similarities = sentence_similarities(opinion_embedding, sentence_embeddings)
    best_index = int(similarities.argmax())
    return SentenceMatch(index=best_index, score=float(similarities[best_index]))


def build_semantic_evidence(
    opinion_embedding: FloatMatrix,
    sentences: Sequence[str],
    sentence_turns: Sequence[int],
    sentence_embeddings: FloatMatrix,
    matched_turns: Sequence[Turn],
    transcript: str,
    quote_match: TurnQuoteMatch | None,
) -> Evidence:
    match = best_sentence_match(opinion_embedding, sentence_embeddings)
    sentence = sentences[match.index]
    supported = short_quote_supports(quote_match, sentence)
    span = locate_turn_sentence_span(
        sentence, transcript, matched_turns, sentence_turns[match.index]
    )
    return evidence_from_span(
        sentence,
        "semantic_with_short_quote" if supported else "semantic_similarity",
        match.score,
        quote_match.prefix if supported and quote_match is not None else None,
        span,
    )


def build_unit_evidence(
    opinion_embedding: FloatMatrix,
    units: Sequence[CandidateUnit],
    unit_embeddings: FloatMatrix,
    quote_match: TurnQuoteMatch | None,
) -> Evidence:
    match = best_sentence_match(opinion_embedding, unit_embeddings)
    unit = units[match.index]
    supported = short_quote_supports(quote_match, unit.evidence_text)
    return evidence_from_span(
        unit.evidence_text,
        "semantic_with_short_quote" if supported else "semantic_similarity",
        match.score,
        quote_match.prefix if supported and quote_match is not None else None,
        unit.span,
    )


def classify_tier(evidence: Evidence | None, resolved: bool, threshold: float) -> Tier:
    if not resolved:
        return "person_not_resolved"
    if evidence is None:
        return "no_evidence"
    if evidence.support_type == "direct_quote":
        return "quote_found"
    if evidence.score is None:
        raise ValueError("semantic evidence without a similarity score")
    return "semantic_match_high" if evidence.score >= threshold else "semantic_match_weak"


def provenance_for(tier: Tier) -> Provenance | None:
    if tier == "quote_found":
        return "weak"
    if tier in ("semantic_match_high", "semantic_match_weak"):
        return "model"
    return None
