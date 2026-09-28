"""Link of each profile claim to a passage of the actor and to one of the actor's UDVs."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from bookworm.data.io import JsonObject
from bookworm.profiles.profile_text import ProfileSection, trim_passage

UDV_MATCH_MIN = 0.3
PASSAGE_MATCH_MIN = 0.15
PASSAGE_MIN_WORDS = 5
SCORE_DECIMALS = 4
SAME_SENTENCE_RULE = "same_sentence"
SIMILAR_TEXT_RULE = "similar_text"
CLAIM_COUNT_KEYS = ("claims", "with_udv", "passage_only", "without_evidence")
MATCH_METHOD = (
    "TF-IDF cosine between each profile item and, separately, the actor's sentences in the "
    "hearings the profile read and the text (proposition and evidence) of the actor's UDVs in "
    "those hearings; vectorizer fitted per actor on those texts, accents removed, sublinear tf, "
    "fixed stop words. A UDV is linked when the item's passage overlaps the UDV evidence in the "
    "transcript (same_sentence) or, failing that, when the item is similar enough to the UDV "
    "text (similar_text)"
)
PORTUGUESE_STOP_WORDS = """
    a o e as os um uma uns umas de do da dos das em no na nos nas num numa por pelo pela pelos
    pelas para pra pro com sem sob sobre ate que se nao sim ja ha foi era sao ser sera estar esta
    estao este esse essa esses essas isso isto aquele aquela aquilo ele ela eles elas eu nos voce
    voces me te lhe lhes seu sua seus suas meu minha nosso nossa ao aos como mas ou mais muito
    tambem so entao aqui la quando onde qual quais quem porque tem ter dito disse falou falaram
    foram forma entre apos ainda cada todo toda todos todas outro outra outros outras
"""
STOP_WORDS = tuple(PORTUGUESE_STOP_WORDS.split())


@dataclass(frozen=True, slots=True)
class Sentence:
    hearing_id: int
    turn: int
    start: int | None
    end: int | None
    text: str


@dataclass(frozen=True, slots=True)
class MatchThresholds:
    udv: float = UDV_MATCH_MIN
    passage: float = PASSAGE_MATCH_MIN


def fit_vectorizer(corpus: Sequence[str]) -> TfidfVectorizer | None:
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        stop_words=list(STOP_WORDS),
        dtype=np.float64,
    )
    try:
        vectorizer.fit(list(corpus))
    except ValueError:
        return None
    return vectorizer


def cosine_rows(vectorizer: TfidfVectorizer, rows: Sequence[str], columns: Sequence[str]) -> Any:
    if not rows or not columns:
        return np.zeros((len(rows), len(columns)))
    left = vectorizer.transform(list(rows))
    right = vectorizer.transform(list(columns))
    return (left @ right.T).toarray()


def best_match(scores: Any, minimum: float) -> tuple[int, float] | None:
    if scores.size == 0:
        return None
    position = int(np.argmax(scores))
    value = float(scores[position])
    if value < minimum:
        return None
    return position, round(value, SCORE_DECIMALS)


def same_sentence(sentence: Sentence, udvs: Sequence[Mapping[str, Any]]) -> int | None:
    """Position of the first UDV whose evidence overlaps the sentence in the transcript."""
    if sentence.start is None or sentence.end is None:
        return None
    for position, udv in enumerate(udvs):
        evidence = udv["evidence"]
        if (
            udv["hearing_id"] == sentence.hearing_id
            and evidence["turn"] == sentence.turn
            and evidence["start"] < sentence.end
            and sentence.start < evidence["end"]
        ):
            return position
    return None


def passage_entry(sentence: Sentence, score: float) -> JsonObject:
    return {
        "hearing_id": sentence.hearing_id,
        "turn": sentence.turn,
        "start": sentence.start,
        "end": sentence.end,
        "text": trim_passage(sentence.text),
        "score": score,
    }


@dataclass(frozen=True)
class ClaimScores:
    """Claim-by-passage and claim-by-UDV similarities of one actor, claims in profile order."""

    sentences: Sequence[Sentence]
    udvs: Sequence[Mapping[str, Any]]
    passage_scores: Any
    udv_scores: Any

    @classmethod
    def compute(
        cls,
        claims: Sequence[str],
        sentences: Sequence[Sentence],
        udvs: Sequence[Mapping[str, Any]],
    ) -> "ClaimScores":
        sentence_texts = [sentence.text for sentence in sentences]
        udv_texts = [udv["match_text"] for udv in udvs]
        vectorizer = fit_vectorizer([*sentence_texts, *udv_texts, *claims])
        if vectorizer is None:
            return cls(sentences, udvs, np.zeros((len(claims), 0)), np.zeros((len(claims), 0)))
        return cls(
            sentences,
            udvs,
            cosine_rows(vectorizer, claims, sentence_texts),
            cosine_rows(vectorizer, claims, udv_texts),
        )

    def udv_link(
        self, row: int, passage: tuple[int, float] | None, minimum: float
    ) -> JsonObject | None:
        shared = None if passage is None else same_sentence(self.sentences[passage[0]], self.udvs)
        if shared is not None:
            return {
                "id": self.udvs[shared]["id"],
                "rule": SAME_SENTENCE_RULE,
                "score": round(float(self.udv_scores[row][shared]), SCORE_DECIMALS),
            }
        udv = best_match(self.udv_scores[row], minimum)
        if udv is None:
            return None
        return {"id": self.udvs[udv[0]]["id"], "rule": SIMILAR_TEXT_RULE, "score": udv[1]}

    def claim_item(self, row: int, claim: str, thresholds: MatchThresholds) -> JsonObject:
        passage = best_match(self.passage_scores[row], thresholds.passage)
        return {
            "text": claim,
            "passage": None
            if passage is None
            else passage_entry(self.sentences[passage[0]], passage[1]),
            "udv": self.udv_link(row, passage, thresholds.udv),
        }


def count_claims(items: Sequence[JsonObject]) -> JsonObject:
    counts = Counter[str]()
    for item in items:
        linked = item["udv"] is not None
        counts["claims"] += 1
        counts["with_udv"] += linked
        counts["passage_only"] += not linked and item["passage"] is not None
        counts["without_evidence"] += not linked and item["passage"] is None
    return {key: counts[key] for key in CLAIM_COUNT_KEYS}


def match_claims(
    sections: Sequence[ProfileSection],
    sentences: Sequence[Sentence],
    udvs: Sequence[Mapping[str, Any]],
    thresholds: MatchThresholds,
) -> tuple[list[JsonObject], JsonObject]:
    """Each claim with its best passage and linked UDV, grouped by section, and the counts."""
    claims = [claim for section in sections for claim in section.claims]
    scores = ClaimScores.compute(claims, sentences, udvs)
    out: list[JsonObject] = []
    items: list[JsonObject] = []
    row = 0
    for section in sections:
        section_items: list[JsonObject] = []
        for claim in section.claims:
            section_items.append(scores.claim_item(row, claim, thresholds))
            row += 1
        items.extend(section_items)
        out.append({"title": section.title, "claims": section_items})
    return out, count_claims(items)
