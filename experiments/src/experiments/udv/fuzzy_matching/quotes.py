"""Fuzzy quote rows: every opinion without a trusted exact quote, aligned at each prefix level,
against the rest of the hearing and against null quotes, plus the trusted positive control."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from experiments.common.reporting import rounded
from experiments.common.transcript import (
    enclosing_turn_sentence,
    extract_quotes,
    find_opinion_turn_quote_match,
    is_trusted_quote,
    locate_turn_sentence_span,
    sentences_agree,
)
from experiments.common.udv_run import UdvConfig, cache_key, sentence_slices_by_person
from experiments.udv.fuzzy_matching.alignment import (
    Alignment,
    QuotePrefix,
    TurnView,
    best_alignment,
    build_turn_view,
    is_eligible_quote,
    quote_prefix,
    quotes_prefixes,
)
from experiments.udv.fuzzy_matching.config import QUOTE_METHODS, SEMANTIC_TIERS, FuzzyConfig
from experiments.udv.fuzzy_matching.inputs import hearing_people, udv_id
from experiments.udv.fuzzy_matching.sampling import draw_rng

Record = dict[str, Any]

CACHE_KEY_CHARS = 16
SAME_HEARING_STREAM = 0
OTHER_HEARING_STREAM = 1


@dataclass(frozen=True)
class PoolQuote:
    source_id: str
    hearing_id: int
    person_index: int
    quote: str


@dataclass(frozen=True)
class HearingQuotes:
    """What every opinion of one hearing is aligned against."""

    hearing_id: int
    split: str
    transcript: str
    views: dict[int, TurnView]
    speaker_of: dict[int, str]
    pool: list[PoolQuote]
    cross_pool: list[PoolQuote]
    top1: dict[str, str]


@dataclass
class QuoteCollection:
    rows: list[Record]
    funnel: list[Record]
    controls: list[Record]
    cache_status: list[Record]


def describe_alignment(
    prefix: QuotePrefix,
    alignment: Alignment,
    views: dict[int, TurnView],
    matched_turns: list[Record],
    transcript: str,
    reference_text: str | None,
) -> Record:
    view = views[alignment.turn_index]
    has_span = alignment.score > 0 and alignment.end > alignment.start
    sentence = (
        enclosing_turn_sentence(view.text, alignment.start, alignment.end) if has_span else ""
    )
    span = (
        locate_turn_sentence_span(sentence, transcript, matched_turns, alignment.turn_index)
        if sentence
        else None
    )
    return {
        "level": prefix.level,
        "quote_index": prefix.quote_index,
        "prefix": prefix.text,
        "prefix_words": prefix.words,
        "prefix_tokens": len(prefix.tokens),
        "score": rounded(alignment.score),
        "turn_index": alignment.turn_index,
        "turn_text_start": alignment.start,
        "turn_text_end": alignment.end,
        "aligned_text": view.text[alignment.start : alignment.end],
        "sentence": sentence,
        "sentence_start_char": span["start_char"] if span else None,
        "sentence_end_char": span["end_char"] if span else None,
        "agrees_with_encoder": (
            sentences_agree(sentence, reference_text) if sentence and reference_text else None
        ),
    }


def fuzzy_quote_results(
    prefixes: list[QuotePrefix],
    views: dict[int, TurnView],
    matched_turns: list[Record],
    transcript: str,
    reference_text: str | None,
    config: FuzzyConfig,
) -> Record:
    ordered_views = [views[turn["turn_index"]] for turn in matched_turns]
    results: Record = {}
    for method in QUOTE_METHODS:
        results[method] = {}
        for level in config.prefix_levels:
            best = best_alignment(prefixes, ordered_views, method, level)
            results[method][str(level)] = (
                describe_alignment(
                    best[0], best[1], views, matched_turns, transcript, reference_text
                )
                if best is not None
                else None
            )
    return results


def level_scores(prefixes: list[QuotePrefix], views: list[TurnView], config: FuzzyConfig) -> Record:
    scores: Record = {}
    for method in QUOTE_METHODS:
        scores[method] = {}
        for level in config.prefix_levels:
            best = best_alignment(prefixes, views, method, level)
            scores[method][str(level)] = rounded(best[1].score) if best is not None else None
    return scores


def elsewhere_scores(
    prefixes: list[QuotePrefix],
    other_views: list[TurnView],
    speaker_of: dict[int, str],
    config: FuzzyConfig,
) -> Record:
    scores: Record = {}
    for method in QUOTE_METHODS:
        scores[method] = {}
        for level in config.prefix_levels:
            best = best_alignment(prefixes, other_views, method, level)
            scores[method][str(level)] = (
                {
                    "score": rounded(best[1].score),
                    "turn_index": best[1].turn_index,
                    "speaker": speaker_of[best[1].turn_index],
                }
                if best is not None
                else None
            )
    return scores


def null_draws(
    pool: list[PoolQuote],
    quote_count: int,
    views: list[TurnView],
    rng: np.random.Generator,
    config: FuzzyConfig,
) -> list[Record]:
    """Scores of quotes drawn from people who did not say them, against the person's turns."""
    if not pool or quote_count == 0:
        return []
    draws = []
    for _ in range(config.null_draws):
        picked = rng.choice(len(pool), size=min(quote_count, len(pool)), replace=False)
        chosen = [pool[int(index)] for index in sorted(picked)]
        prefixes = quotes_prefixes([entry.quote for entry in chosen], config)
        draws.append(
            {
                "sources": [entry.source_id for entry in chosen],
                **level_scores(prefixes, views, config),
            }
        )
    return draws


def cached_embeddings(
    texts: list[str], label: str, udv_config: UdvConfig, device_label: str
) -> np.ndarray | None:
    key = cache_key(texts, udv_config, device_label)[:CACHE_KEY_CHARS]
    path = udv_config.cache_dir / f"{label}_{key}.npy"
    return np.load(path) if path.exists() else None


def encoder_top1(
    hearing: Record, people: list[Record], udv_config: UdvConfig, device_label: str
) -> dict[str, str] | None:
    """The production encoder's top-1 sentence per opinion, from its cached embeddings only."""
    sentences = [sentence for person in people for sentence in person["sentences"]]
    opinions = [
        (person, opinion_index, opinion_text)
        for person in people
        for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"])
    ]
    if not sentences or not opinions:
        return None
    sentence_embeddings = cached_embeddings(
        sentences, f"sentences_{hearing['id']}", udv_config, device_label
    )
    opinion_embeddings = cached_embeddings(
        [text for _, _, text in opinions], f"opinions_{hearing['id']}", udv_config, device_label
    )
    if sentence_embeddings is None or opinion_embeddings is None:
        return None
    slices = sentence_slices_by_person(people)
    top: dict[str, str] = {}
    for position, (person, opinion_index, _) in enumerate(opinions):
        if not person["sentences"]:
            continue
        similarities = cosine_similarity(
            opinion_embeddings[position].reshape(1, -1),
            sentence_embeddings[slices[person["index"]]],
        ).flatten()
        top[udv_id(hearing["id"], person["index"], opinion_index)] = person["sentences"][
            int(similarities.argmax())
        ]
    return top


def build_quote_pools(hearings: list[Record], config: FuzzyConfig) -> dict[int, list[PoolQuote]]:
    pools: dict[int, list[PoolQuote]] = {}
    for hearing in hearings:
        entries = pools.setdefault(hearing["id"], [])
        for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
            for opinion_index, opinion_text in enumerate(participant["opinioes"]):
                source_id = udv_id(hearing["id"], person_index, opinion_index)
                entries.extend(
                    PoolQuote(source_id, hearing["id"], person_index, quote)
                    for quote in extract_quotes(opinion_text)
                    if is_eligible_quote(quote, config)
                )
    return pools


def reference_summary(record: Record | None) -> Record | None:
    if record is None:
        return None
    evidence = record["evidence"] or {}
    return {
        "tier": record["tier"],
        "support_type": evidence.get("support_type"),
        "score": evidence.get("score"),
        "text": evidence.get("text"),
        "speaker_turn": evidence.get("speaker_turn"),
    }


def control_result(
    prefix: QuotePrefix, ordered_views: list[TurnView], views: dict[int, TurnView], exact: Record
) -> dict[str, Record | None]:
    results: dict[str, Record | None] = {}
    for method in QUOTE_METHODS:
        best = best_alignment([prefix], ordered_views, method, prefix.level)
        if best is None:
            results[method] = None
            continue
        alignment = best[1]
        view = views[alignment.turn_index]
        sentence = (
            enclosing_turn_sentence(view.text, alignment.start, alignment.end)
            if alignment.end > alignment.start
            else ""
        )
        results[method] = {
            "score": rounded(alignment.score),
            "same_turn": alignment.turn_index == exact["turn_index"],
            "sentence_agrees": bool(sentence) and sentences_agree(sentence, exact["sentence"]),
        }
    return results


def positive_control_row(
    row_id: str,
    split: str,
    hearing_id: int,
    exact: Record,
    views: dict[int, TurnView],
    matched_turns: list[Record],
    encoder_sentence: str | None,
) -> Record:
    """The trusted exact prefix aligned with the same scorers, which should score 100."""
    prefix = quote_prefix(exact["quote_index"], exact["words"], exact["prefix"])
    ordered_views = [views[turn["turn_index"]] for turn in matched_turns]
    return {
        "id": row_id,
        "split": split,
        "hearing_id": hearing_id,
        "encoder_sentence": encoder_sentence,
        "encoder_agrees": (
            sentences_agree(exact["sentence"], encoder_sentence) if encoder_sentence else None
        ),
        **control_result(prefix, ordered_views, views, exact),
    }


def other_hearing_pool(
    pools: dict[int, list[PoolQuote]], split_of: dict[int, str], hearing_id: int
) -> list[PoolQuote]:
    return [
        entry
        for other_id, entries in pools.items()
        if other_id != hearing_id and split_of[other_id] == split_of[hearing_id]
        for entry in entries
    ]


def exact_match_summary(exact: Record | None) -> Record | None:
    if exact is None:
        return None
    return {
        "prefix": exact["prefix"],
        "words": exact["words"],
        "quote_index": exact["quote_index"],
        "sentence": exact["sentence"],
    }


def null_section(
    context: HearingQuotes,
    person: Record,
    opinion_index: int,
    eligible: int,
    ordered_views: list[TurnView],
    config: FuzzyConfig,
) -> Record:
    same_pool = [entry for entry in context.pool if entry.person_index != person["index"]]
    pools = {
        "same_hearing_other_person": (same_pool, SAME_HEARING_STREAM),
        "other_hearing_same_split": (context.cross_pool, OTHER_HEARING_STREAM),
    }
    return {
        kind: null_draws(
            pool,
            eligible,
            ordered_views,
            draw_rng(config.seed, stream, context.hearing_id, person["index"], opinion_index),
            config,
        )
        for kind, (pool, stream) in pools.items()
    }


def population_row(
    context: HearingQuotes,
    person: Record,
    opinion_index: int,
    quotes: list[str],
    exact: Record | None,
    reference: dict[str, Record],
    config: FuzzyConfig,
) -> Record:
    """One opinion with an extracted quote and no trusted exact prefix, with every score."""
    row_id = udv_id(context.hearing_id, person["index"], opinion_index)
    participant = person["participant"]
    matched_turns = person["matched_turns"]
    eligible = [quote for quote in quotes if is_eligible_quote(quote, config)]
    reference_record = reference_summary(reference.get(row_id))
    reference_text = (
        reference_record["text"]
        if reference_record and reference_record["tier"] in SEMANTIC_TIERS
        else None
    )
    prefixes = quotes_prefixes(quotes, config)
    ordered_views = [context.views[turn["turn_index"]] for turn in matched_turns]
    own_turns = {turn["turn_index"] for turn in matched_turns}
    other_views = [view for index, view in context.views.items() if index not in own_turns]
    top1 = context.top1
    return {
        "id": row_id,
        "split": context.split,
        "hearing_id": context.hearing_id,
        "person": {
            "index": person["index"],
            "name": participant["nome"],
            "role": participant["cargo"],
        },
        "opinion_index": opinion_index,
        "opinion": participant["opinioes"][opinion_index],
        "quotes": quotes,
        "eligible_quotes": len(eligible),
        "exact_match": exact_match_summary(exact),
        "reference": reference_record,
        "encoder_cache_top1": top1.get(row_id),
        "encoder_cache_matches_reference": (
            top1[row_id] == reference_text
            if row_id in top1 and reference_text is not None
            else None
        ),
        "fuzzy": fuzzy_quote_results(
            prefixes, context.views, matched_turns, context.transcript, reference_text, config
        ),
        "elsewhere": elsewhere_scores(prefixes, other_views, context.speaker_of, config),
        "null": null_section(context, person, opinion_index, len(eligible), ordered_views, config),
    }


def collect_opinion(
    context: HearingQuotes,
    person: Record,
    opinion_index: int,
    reference: dict[str, Record],
    collection: QuoteCollection,
    config: FuzzyConfig,
) -> None:
    opinion_text = person["participant"]["opinioes"][opinion_index]
    matched_turns = person["matched_turns"]
    quotes = extract_quotes(opinion_text) if matched_turns else []
    exact = find_opinion_turn_quote_match(opinion_text, matched_turns) if quotes else None
    trusted = is_trusted_quote(exact)
    collection.funnel.append(
        {
            "split": context.split,
            "resolved": bool(matched_turns),
            "with_extracted_quote": bool(quotes),
            "trusted_exact": trusted,
            "with_eligible_quote": any(is_eligible_quote(quote, config) for quote in quotes),
        }
    )
    if not quotes:
        return
    if trusted and exact is not None:
        row_id = udv_id(context.hearing_id, person["index"], opinion_index)
        collection.controls.append(
            positive_control_row(
                row_id,
                context.split,
                context.hearing_id,
                exact,
                context.views,
                matched_turns,
                context.top1.get(row_id),
            )
        )
        return
    collection.rows.append(
        population_row(context, person, opinion_index, quotes, exact, reference, config)
    )


def collect_quotes(
    hearings: list[Record],
    split_of: dict[int, str],
    reference: dict[str, Record],
    udv_config: UdvConfig,
    config: FuzzyConfig,
) -> QuoteCollection:
    pools = build_quote_pools(hearings, config)
    collection = QuoteCollection(rows=[], funnel=[], controls=[], cache_status=[])
    for hearing in hearings:
        hearing_id, split = hearing["id"], split_of[hearing["id"]]
        turns, people = hearing_people(hearing)
        top1 = encoder_top1(hearing, people, udv_config, config.cache_device_label)
        collection.cache_status.append(
            {"hearing_id": hearing_id, "split": split, "cached": top1 is not None}
        )
        context = HearingQuotes(
            hearing_id=hearing_id,
            split=split,
            transcript=hearing["transcricao"],
            views={turn["turn_index"]: build_turn_view(turn) for turn in turns},
            speaker_of={turn["turn_index"]: turn["raw_name"] for turn in turns},
            pool=pools[hearing_id],
            cross_pool=other_hearing_pool(pools, split_of, hearing_id),
            top1=top1 or {},
        )
        for person in people:
            for opinion_index in range(len(person["participant"]["opinioes"])):
                collect_opinion(context, person, opinion_index, reference, collection, config)
    return collection
