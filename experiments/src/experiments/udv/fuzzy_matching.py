import argparse
import json
import platform
import time
import tomllib
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import bookworm.data.io
import numpy as np
import rapidfuzz
from bookworm import load_gated_jsonl, load_jsonl, sha256_of_file, write_json, write_jsonl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.metrics.pairwise import cosine_similarity

from experiments.common import transcript, udv_run
from experiments.common.provenance import source_hashes
from experiments.common.transcript import (
    TRUSTED_PREFIX_WORDS,
    TURN_HEADER_PATTERN,
    WORD_TOKEN_PATTERN,
    enclosing_turn_sentence,
    extract_quotes,
    find_opinion_turn_quote_match,
    is_trusted_quote,
    locate_turn_sentence_span,
    normalize_name,
    sentences_agree,
    split_into_turns,
    strip_accents,
    turn_name_candidates,
    turn_text,
)
from experiments.common.udv_run import (
    UdvConfig,
    cache_key,
    resolve_hearing_people,
    sentence_slices_by_person,
)
from experiments.validation import generate_sample
from experiments.validation.generate_sample import (
    assign_item_ids,
    canonical_sha256,
    context_window,
    header_text,
)

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
CONFIG_SPLITS_ALLOWED = ("train", "validation")
QUOTE_METHODS = ("char", "token")
NULL_KINDS = ("same_hearing_other_person", "other_hearing_same_split")
NAME_METRICS = ("token_set_ratio", "jaro_winkler")
COMBINED_NAME_RULE = "both"
SEMANTIC_TIERS = ("semantic_match_high", "semantic_match_weak")
SCORE_DECIMALS = 4
REVIEW_KINDS = ("quotes", "names")
DISPLAY_FIELDS = {
    "quotes": (
        "hearing_id",
        "participant",
        "role",
        "opinion",
        "quotes",
        "turn_header",
        "context_before",
        "candidate",
        "context_after",
    ),
    "names": (
        "hearing_id",
        "participant",
        "role",
        "opinions",
        "candidate_speaker",
        "party_info",
        "turn_count",
        "speech_excerpt",
    ),
}
FILL_FIELDS = ("judgment", "note")


@dataclass(frozen=True)
class FuzzyConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    splits: tuple[str, ...]
    reference_runs: tuple[Path, ...]
    udv_config_path: Path
    cache_device_label: str
    prefix_levels: tuple[int, ...]
    min_prefix_words: int
    quote_thresholds: tuple[float, ...]
    null_draws: int
    quote_review_min_score: float
    score_bin_width: int
    name_metrics: tuple[str, ...]
    name_thresholds: tuple[float, ...]
    name_review_min_score: float
    speech_excerpt_chars: int
    impostor_examples: int
    bootstrap_samples: int
    confidence_level: float
    seed: int
    version: str
    output_dir: Path
    review_context_chars: int
    review_item_prefixes: dict[str, str]
    review_order_streams: dict[str, tuple[int, ...]]
    review_labels: dict[str, tuple[str, ...]]
    review_positive: dict[str, str]
    review_unsure: dict[str, str]
    source: Record = field(default_factory=dict)


@dataclass(frozen=True)
class HearingText:
    transcript: str
    turns: dict[int, Record]
    headers: tuple[str, ...]


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


@dataclass(frozen=True)
class PoolQuote:
    source_id: str
    hearing_id: int
    person_index: int
    quote: str


def validate_thresholds(values: list[float], key: str) -> tuple[float, ...]:
    if not values or any(not 0 <= value <= 100 for value in values):
        raise SystemExit(f"{key} must be a non-empty list of scores in [0, 100]")
    return tuple(sorted(float(value) for value in values))


def load_config(config_path: Path) -> FuzzyConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    splits = tuple(raw["splits"]["use"])
    if "test" in splits:
        raise SystemExit("splits.use must never list test; pass --final-test instead")
    if not splits or any(split not in CONFIG_SPLITS_ALLOWED for split in splits):
        raise SystemExit(f"splits.use must be a non-empty subset of {CONFIG_SPLITS_ALLOWED}")
    if len(set(splits)) != len(splits):
        raise SystemExit("splits.use must list distinct split names")
    quotes, names, bootstrap = raw["quotes"], raw["names"], raw["bootstrap"]
    levels = tuple(quotes["prefix_word_levels"])
    if not levels or list(levels) != sorted(set(levels), reverse=True):
        raise SystemExit("quotes.prefix_word_levels must be distinct and in decreasing order")
    if quotes["min_prefix_words"] < 1 or quotes["null_draws_per_opinion"] < 1:
        raise SystemExit("quotes.min_prefix_words and quotes.null_draws_per_opinion must be >= 1")
    if any(metric not in NAME_METRICS for metric in names["metrics"]):
        raise SystemExit(f"names.metrics must be among {NAME_METRICS}")
    if bootstrap["unit"] != "hearing":
        raise SystemExit(
            "bootstrap.unit must be hearing: opinions of one hearing are not independent"
        )
    if bootstrap["samples"] < 1 or not 0 < bootstrap["confidence_level"] < 1:
        raise SystemExit("bootstrap.samples must be >= 1 and confidence_level in (0, 1)")
    review = raw["review"]
    labels = {kind: tuple(review[kind]["labels"]) for kind in REVIEW_KINDS}
    for kind in REVIEW_KINDS:
        chosen = {review[kind]["positive_label"], review[kind]["unsure_label"]}
        if len(set(labels[kind])) != len(labels[kind]) or not chosen <= set(labels[kind]):
            raise SystemExit(f"review.{kind}: labels must be distinct and hold both chosen labels")
        if len(chosen) != 2 or len(labels[kind]) < 3:
            raise SystemExit(f"review.{kind}: needs a positive, a negative and an unsure label")
    prefixes = {kind: review["item_prefixes"][kind] for kind in REVIEW_KINDS}
    if len(set(prefixes.values())) != len(prefixes):
        raise SystemExit("review.item_prefixes must differ between sheets")
    streams = {kind: tuple(review["order_streams"][kind]) for kind in REVIEW_KINDS}
    if len(set(streams.values())) != len(streams) or any(
        stream[0] in (0, 1) for stream in streams.values()
    ):
        raise SystemExit("review.order_streams must differ and not reuse the null-draw streams")
    return FuzzyConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        splits=splits,
        reference_runs=tuple(Path(run) for run in raw["reference"]["udv_runs"]),
        udv_config_path=Path(raw["encoder_baseline"]["udv_config_path"]),
        cache_device_label=raw["encoder_baseline"]["cache_device_label"],
        prefix_levels=levels,
        min_prefix_words=quotes["min_prefix_words"],
        quote_thresholds=validate_thresholds(quotes["thresholds"], "quotes.thresholds"),
        null_draws=quotes["null_draws_per_opinion"],
        quote_review_min_score=float(quotes["review_min_score"]),
        score_bin_width=quotes["score_bin_width"],
        name_metrics=tuple(names["metrics"]),
        name_thresholds=validate_thresholds(names["thresholds"], "names.thresholds"),
        name_review_min_score=float(names["review_min_score"]),
        speech_excerpt_chars=names["speech_excerpt_chars"],
        impostor_examples=names["impostor_examples"],
        bootstrap_samples=bootstrap["samples"],
        confidence_level=bootstrap["confidence_level"],
        seed=raw["run"]["seed"],
        version=raw["run"]["version"],
        output_dir=Path(raw["run"]["output_dir"]),
        review_context_chars=review["context_chars"],
        review_item_prefixes=prefixes,
        review_order_streams=streams,
        review_labels=labels,
        review_positive={kind: review[kind]["positive_label"] for kind in REVIEW_KINDS},
        review_unsure={kind: review[kind]["unsure_label"] for kind in REVIEW_KINDS},
        source=raw,
    )


def load_split_lookup(manifest_path: Path, lds_sha256: str) -> tuple[dict[int, str], Record]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != lds_sha256:
        raise SystemExit(f"{manifest_path} was built from another LDS file")
    lookup: dict[int, str] = {}
    for name in SPLIT_NAMES:
        for hearing_id in manifest[name]:
            if hearing_id in lookup:
                raise SystemExit(f"hearing {hearing_id} is listed in two splits")
            lookup[hearing_id] = name
    source = {
        "path": str(manifest_path),
        "sha256": sha256_of_file(manifest_path),
        "split_version": manifest["split_version"],
    }
    return lookup, source


def select_hearings(
    lds: list[Record], split_of: dict[int, str], splits: tuple[str, ...]
) -> list[Record]:
    missing = [hearing["id"] for hearing in lds if hearing["id"] not in split_of]
    if missing:
        raise SystemExit(f"hearings missing from the split manifest: {missing}")
    return [hearing for hearing in lds if split_of[hearing["id"]] in splits]


def select_reference_run(runs: tuple[Path, ...]) -> tuple[dict[str, Record], Record]:
    for position, run in enumerate(runs):
        records_path = run.with_name(f"{run.name}.jsonl")
        coverage_path = run.with_name(f"{run.name}_coverage.json")
        if not (records_path.exists() and coverage_path.exists()):
            continue
        with open(coverage_path) as f:
            coverage = json.load(f)
        records = {record["id"]: record for record in load_jsonl(records_path)}
        encoder = coverage.get("config", {}).get("encoder", {})
        source = {
            "run_name": run.name,
            "path": str(records_path),
            "sha256": sha256_of_file(records_path),
            "coverage_path": str(coverage_path),
            "coverage_sha256": sha256_of_file(coverage_path),
            "coverage_created_at": coverage.get("created_at"),
            "encoder": encoder.get("name"),
            "encoder_revision": encoder.get("revision"),
            "embedding_threshold": coverage.get("config", {})
            .get("evidence", {})
            .get("embedding_threshold"),
            "has_pipeline_section": "pipeline" in coverage,
            "preference_position": position,
            "fallback_used": position > 0,
            "runs_skipped": [str(skipped) for skipped in runs[:position]],
            "records": len(records),
        }
        return records, source
    raise SystemExit(f"no reference UDV run found among {[str(run) for run in runs]}")


@cache
def fold_character(character: str) -> str:
    for candidate in (strip_accents(character).lower(), character.lower()):
        if len(candidate) == 1:
            return candidate
    return character


def fold_text(text: str) -> str:
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


def quote_prefix_levels(quote: str, quote_index: int, config: FuzzyConfig) -> list[QuotePrefix]:
    words = quote.split()
    prefixes: list[QuotePrefix] = []
    seen: set[int] = set()
    for level in config.prefix_levels:
        count = min(level, len(words))
        if count < config.min_prefix_words or count in seen:
            continue
        seen.add(count)
        text = " ".join(words[:count])
        folded = fold_text(text)
        prefixes.append(
            QuotePrefix(
                quote_index=quote_index,
                level=level,
                words=count,
                text=text,
                folded=folded,
                tokens=tuple(WORD_TOKEN_PATTERN.findall(folded)),
            )
        )
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
    best: tuple[QuotePrefix, Alignment] | None = None
    for prefix in prefixes:
        if prefix.level != level:
            continue
        for view in views:
            alignment = align(prefix, view, method)
            if alignment is not None and (best is None or alignment.score > best[1].score):
                best = (prefix, alignment)
    return best


def rounded(value: float) -> float:
    return round(float(value), SCORE_DECIMALS)


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


def draw_rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def null_draws(
    pool: list[PoolQuote],
    quote_count: int,
    views: list[TurnView],
    rng: np.random.Generator,
    config: FuzzyConfig,
) -> list[Record]:
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


def hearing_people(hearing: Record) -> tuple[list[Record], list[Record]]:
    return split_into_turns(hearing["transcricao"]), resolve_hearing_people(hearing)


def cached_embeddings(
    texts: list[str], label: str, udv_config: UdvConfig, device_label: str
) -> np.ndarray | None:
    key = cache_key(texts, udv_config, device_label)[:16]
    path = udv_config.cache_dir / f"{label}_{key}.npy"
    return np.load(path) if path.exists() else None


def encoder_top1(
    hearing: Record, people: list[Record], udv_config: UdvConfig, device_label: str
) -> dict[str, str] | None:
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


def udv_id(hearing_id: int, person_index: int, opinion_index: int) -> str:
    return f"udv-{hearing_id}-{person_index}-{opinion_index}"


def build_quote_pools(hearings: list[Record], config: FuzzyConfig) -> dict[int, list[PoolQuote]]:
    pools: dict[int, list[PoolQuote]] = {}
    for hearing in hearings:
        entries = pools.setdefault(hearing["id"], [])
        for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
            for opinion_index, opinion_text in enumerate(participant["opinioes"]):
                for quote in extract_quotes(opinion_text):
                    if is_eligible_quote(quote, config):
                        entries.append(
                            PoolQuote(
                                udv_id(hearing["id"], person_index, opinion_index),
                                hearing["id"],
                                person_index,
                                quote,
                            )
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


def positive_control_row(
    row_id: str,
    split: str,
    hearing_id: int,
    exact: Record,
    views: dict[int, TurnView],
    matched_turns: list[Record],
    encoder_sentence: str | None,
) -> Record:
    words = exact["prefix"].split()
    folded = fold_text(exact["prefix"])
    prefix = QuotePrefix(
        exact["quote_index"],
        exact["words"],
        len(words),
        exact["prefix"],
        folded,
        tuple(WORD_TOKEN_PATTERN.findall(folded)),
    )
    ordered_views = [views[turn["turn_index"]] for turn in matched_turns]
    row: Record = {
        "id": row_id,
        "split": split,
        "hearing_id": hearing_id,
        "encoder_sentence": encoder_sentence,
        "encoder_agrees": (
            sentences_agree(exact["sentence"], encoder_sentence) if encoder_sentence else None
        ),
    }
    for method in QUOTE_METHODS:
        best = best_alignment([prefix], ordered_views, method, prefix.level)
        if best is None:
            row[method] = None
            continue
        view = views[best[1].turn_index]
        sentence = (
            enclosing_turn_sentence(view.text, best[1].start, best[1].end)
            if best[1].end > best[1].start
            else ""
        )
        row[method] = {
            "score": rounded(best[1].score),
            "same_turn": best[1].turn_index == exact["turn_index"],
            "sentence_agrees": bool(sentence) and sentences_agree(sentence, exact["sentence"]),
        }
    return row


def other_hearing_pool(
    pools: dict[int, list[PoolQuote]], split_of: dict[int, str], hearing_id: int
) -> list[PoolQuote]:
    return [
        entry
        for other_id, entries in pools.items()
        if other_id != hearing_id and split_of[other_id] == split_of[hearing_id]
        for entry in entries
    ]


def collect_quotes(
    hearings: list[Record],
    split_of: dict[int, str],
    reference: dict[str, Record],
    udv_config: UdvConfig,
    config: FuzzyConfig,
) -> tuple[list[Record], list[Record], list[Record], list[Record]]:
    pools = build_quote_pools(hearings, config)
    rows: list[Record] = []
    funnel: list[Record] = []
    controls: list[Record] = []
    cache_status: list[Record] = []
    for hearing in hearings:
        hearing_id, split = hearing["id"], split_of[hearing["id"]]
        transcript = hearing["transcricao"]
        turns, people = hearing_people(hearing)
        top1 = encoder_top1(hearing, people, udv_config, config.cache_device_label)
        cache_status.append({"hearing_id": hearing_id, "split": split, "cached": top1 is not None})
        top1 = top1 or {}
        views = {turn["turn_index"]: build_turn_view(turn) for turn in turns}
        speaker_of = {turn["turn_index"]: turn["raw_name"] for turn in turns}
        cross_pool = other_hearing_pool(pools, split_of, hearing_id)
        for person in people:
            matched_turns = person["matched_turns"]
            participant = person["participant"]
            same_pool = [
                entry for entry in pools[hearing_id] if entry.person_index != person["index"]
            ]
            for opinion_index, opinion_text in enumerate(participant["opinioes"]):
                row_id = udv_id(hearing_id, person["index"], opinion_index)
                quotes = extract_quotes(opinion_text) if matched_turns else []
                exact = (
                    find_opinion_turn_quote_match(opinion_text, matched_turns) if quotes else None
                )
                trusted = is_trusted_quote(exact)
                eligible = [quote for quote in quotes if is_eligible_quote(quote, config)]
                funnel.append(
                    {
                        "split": split,
                        "resolved": bool(matched_turns),
                        "with_extracted_quote": bool(quotes),
                        "trusted_exact": trusted,
                        "with_eligible_quote": bool(eligible),
                    }
                )
                if not quotes:
                    continue
                if trusted and exact is not None:
                    controls.append(
                        positive_control_row(
                            row_id,
                            split,
                            hearing_id,
                            exact,
                            views,
                            matched_turns,
                            top1.get(row_id),
                        )
                    )
                    continue
                reference_record = reference_summary(reference.get(row_id))
                reference_text = (
                    reference_record["text"]
                    if reference_record and reference_record["tier"] in SEMANTIC_TIERS
                    else None
                )
                prefixes = quotes_prefixes(quotes, config)
                ordered_views = [views[turn["turn_index"]] for turn in matched_turns]
                own_turns = {turn["turn_index"] for turn in matched_turns}
                other_views = [view for index, view in views.items() if index not in own_turns]
                rows.append(
                    {
                        "id": row_id,
                        "split": split,
                        "hearing_id": hearing_id,
                        "person": {
                            "index": person["index"],
                            "name": participant["nome"],
                            "role": participant["cargo"],
                        },
                        "opinion_index": opinion_index,
                        "opinion": opinion_text,
                        "quotes": quotes,
                        "eligible_quotes": len(eligible),
                        "exact_match": (
                            {
                                "prefix": exact["prefix"],
                                "words": exact["words"],
                                "quote_index": exact["quote_index"],
                                "sentence": exact["sentence"],
                            }
                            if exact is not None
                            else None
                        ),
                        "reference": reference_record,
                        "encoder_cache_top1": top1.get(row_id),
                        "encoder_cache_matches_reference": (
                            top1[row_id] == reference_text
                            if row_id in top1 and reference_text is not None
                            else None
                        ),
                        "fuzzy": fuzzy_quote_results(
                            prefixes, views, matched_turns, transcript, reference_text, config
                        ),
                        "elsewhere": elsewhere_scores(prefixes, other_views, speaker_of, config),
                        "null": {
                            "same_hearing_other_person": null_draws(
                                same_pool,
                                len(eligible),
                                ordered_views,
                                draw_rng(
                                    config.seed, 0, hearing_id, person["index"], opinion_index
                                ),
                                config,
                            ),
                            "other_hearing_same_split": null_draws(
                                cross_pool,
                                len(eligible),
                                ordered_views,
                                draw_rng(
                                    config.seed, 1, hearing_id, person["index"], opinion_index
                                ),
                                config,
                            ),
                        },
                    }
                )
    return rows, funnel, controls, cache_status


def accepted_result(
    row: Record, method: str, threshold: float, levels: tuple[int, ...]
) -> Record | None:
    for level in levels:
        result: Record | None = row["fuzzy"][method].get(str(level))
        if result is not None and result["score"] >= threshold:
            return result
    return None


def max_level_score(scores: Record) -> float | None:
    values = [value for value in scores.values() if value is not None]
    return max(values) if values else None


def bootstrap_ratio(
    numerators: dict[int, float],
    denominators: dict[int, float],
    hearing_ids: list[int],
    config: FuzzyConfig,
    stream: str,
) -> Record:
    total = sum(denominators.values())
    if not hearing_ids or total == 0:
        return {"value": None, "low": None, "high": None}
    top = np.array([numerators.get(hearing_id, 0.0) for hearing_id in hearing_ids])
    bottom = np.array([denominators.get(hearing_id, 0.0) for hearing_id in hearing_ids])
    rng = draw_rng(config.seed, zlib.crc32(stream.encode()))
    picks = rng.integers(0, len(hearing_ids), size=(config.bootstrap_samples, len(hearing_ids)))
    sampled_top, sampled_bottom = top[picks].sum(axis=1), bottom[picks].sum(axis=1)
    valid = sampled_bottom > 0
    ratios = sampled_top[valid] / sampled_bottom[valid]
    tail = (1 - config.confidence_level) / 2
    return {
        "value": rounded(top.sum() / total),
        "low": rounded(float(np.quantile(ratios, tail))) if ratios.size else None,
        "high": rounded(float(np.quantile(ratios, 1 - tail))) if ratios.size else None,
        "replicates_with_denominator": int(valid.sum()),
    }


def score_histogram(values: list[float], width: int) -> dict[str, int]:
    bins = Counter("100" if value >= 100 else str(int(value // width) * width) for value in values)
    return dict(sorted(bins.items(), key=lambda item: float(item[0])))


def null_rate(
    rows: list[Record],
    method: str,
    kind: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    stream: str,
) -> Record:
    hits: dict[int, float] = {}
    draws: dict[int, float] = {}
    for row in rows:
        for draw in row["null"][kind]:
            score = max_level_score(draw[method])
            draws[row["hearing_id"]] = draws.get(row["hearing_id"], 0.0) + 1
            if score is not None and score >= threshold:
                hits[row["hearing_id"]] = hits.get(row["hearing_id"], 0.0) + 1
    rate = bootstrap_ratio(hits, draws, hearing_ids, config, stream)
    eligible = sum(1 for row in rows if row["eligible_quotes"] > 0)
    return {
        "draws": int(sum(draws.values())),
        "draws_at_or_above": int(sum(hits.values())),
        "rate": rate,
        "expected_chance_matches": (
            rounded(rate["value"] * eligible) if rate["value"] is not None else None
        ),
    }


def gains_at_threshold(
    rows: list[Record],
    method: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    accepted = [
        (row, result)
        for row in rows
        if (result := accepted_result(row, method, threshold, config.prefix_levels)) is not None
    ]
    agree: dict[int, float] = {}
    with_reference: dict[int, float] = {}
    for row, result in accepted:
        if result["agrees_with_encoder"] is None:
            continue
        with_reference[row["hearing_id"]] = with_reference.get(row["hearing_id"], 0.0) + 1
        if result["agrees_with_encoder"]:
            agree[row["hearing_id"]] = agree.get(row["hearing_id"], 0.0) + 1
    stream = f"{group}/{method}/{threshold}"
    elsewhere = [
        (row["elsewhere"][method][str(result["level"])] or {}).get("score")
        for row, result in accepted
    ]
    return {
        "opinions": len(accepted),
        "hearings": len({row["hearing_id"] for row, _ in accepted}),
        "elsewhere_at_or_above_threshold": sum(
            1 for score in elsewhere if score is not None and score >= threshold
        ),
        "elsewhere_higher_than_own": sum(
            1
            for score, (_, result) in zip(elsewhere, accepted, strict=True)
            if score is not None and score > result["score"]
        ),
        "by_level": dict(sorted(Counter(str(result["level"]) for _, result in accepted).items())),
        "score_100": sum(1 for _, result in accepted if result["score"] >= 100),
        "by_reference_tier": dict(
            Counter((row["reference"] or {}).get("tier") or "missing" for row, _ in accepted)
        ),
        "by_reference_support_type": dict(
            Counter((row["reference"] or {}).get("support_type") or "none" for row, _ in accepted)
        ),
        "by_exact_prefix_words": dict(
            sorted(
                Counter(
                    str((row["exact_match"] or {}).get("words", 0)) for row, _ in accepted
                ).items()
            )
        ),
        "with_encoder_reference": int(sum(with_reference.values())),
        "agrees_with_encoder": int(sum(agree.values())),
        "agreement_rate": bootstrap_ratio(
            agree, with_reference, hearing_ids, config, f"{stream}/agree"
        ),
        "null": {
            kind: null_rate(rows, method, kind, threshold, hearing_ids, config, f"{stream}/{kind}")
            for kind in NULL_KINDS
        },
    }


def count_by_hearing(rows: list[Record], key: str, expected: bool | None) -> dict[int, float]:
    counts: dict[int, float] = {}
    for row in rows:
        value = row[key]
        if value is None or (expected is not None and value is not expected):
            continue
        counts[row["hearing_id"]] = counts.get(row["hearing_id"], 0.0) + 1
    return counts


def summarize_controls(
    controls: list[Record], hearing_ids: list[int], config: FuzzyConfig, group: str
) -> Record:
    with_encoder = count_by_hearing(controls, "encoder_agrees", None)
    agrees = count_by_hearing(controls, "encoder_agrees", True)
    summary: Record = {
        "opinions": len(controls),
        "encoder_baseline": {
            "with_encoder_cache": int(sum(with_encoder.values())),
            "agrees": int(sum(agrees.values())),
            "rate": bootstrap_ratio(
                agrees, with_encoder, hearing_ids, config, f"{group}/control/encoder"
            ),
        },
    }
    for method in QUOTE_METHODS:
        results = [row[method] for row in controls if row[method] is not None]
        summary[method] = {
            "aligned": len(results),
            "score_100": sum(1 for result in results if result["score"] >= 100),
            "same_turn": sum(1 for result in results if result["same_turn"]),
            "sentence_agrees": sum(1 for result in results if result["sentence_agrees"]),
            "ids_below_100_or_other_sentence": [
                row["id"]
                for row in controls
                if row[method] is not None
                and (row[method]["score"] < 100 or not row[method]["sentence_agrees"])
            ],
        }
    return summary


def summarize_quote_funnel(entries: list[Record]) -> Record:
    resolved = [entry for entry in entries if entry["resolved"]]
    quoted = [entry for entry in resolved if entry["with_extracted_quote"]]
    population = [entry for entry in quoted if not entry["trusted_exact"]]
    return {
        "opinions": len(entries),
        "resolved_person": len(resolved),
        "with_extracted_quote": len(quoted),
        "trusted_exact_positive_control": sum(1 for entry in quoted if entry["trusted_exact"]),
        "population_without_trusted_exact": len(population),
        "population_with_eligible_quote": sum(
            1 for entry in population if entry["with_eligible_quote"]
        ),
    }


def score_bands(rows: list[Record], method: str, config: FuzzyConfig) -> list[Record]:
    thresholds = config.quote_thresholds
    bands = []
    for position, low in enumerate(thresholds):
        high = thresholds[position + 1] if position + 1 < len(thresholds) else None
        members = [
            result
            for row in rows
            if (result := accepted_result(row, method, low, config.prefix_levels)) is not None
            and (high is None or accepted_result(row, method, high, config.prefix_levels) is None)
        ]
        with_reference = [result for result in members if result["agrees_with_encoder"] is not None]
        bands.append(
            {
                "low": low,
                "high": high,
                "opinions": len(members),
                "by_level": dict(
                    sorted(Counter(str(result["level"]) for result in members).items())
                ),
                "with_encoder_reference": len(with_reference),
                "agrees_with_encoder": sum(
                    1 for result in with_reference if result["agrees_with_encoder"]
                ),
            }
        )
    return bands


def summarize_cache_consistency(rows: list[Record], cache_status: list[Record]) -> Record:
    values = [row["encoder_cache_matches_reference"] for row in rows]
    return {
        "hearings": len(cache_status),
        "hearings_with_encoder_cache": sum(1 for entry in cache_status if entry["cached"]),
        "population_without_cache_top1": sum(
            1 for row in rows if row["encoder_cache_top1"] is None
        ),
        "population_compared_with_reference": sum(1 for value in values if value is not None),
        "same_sentence_as_reference": sum(1 for value in values if value is True),
        "different_sentence_from_reference_ids": [
            row["id"] for row in rows if row["encoder_cache_matches_reference"] is False
        ],
    }


def summarize_quote_group(
    rows: list[Record],
    funnel: list[Record],
    controls: list[Record],
    cache_status: list[Record],
    config: FuzzyConfig,
    group: str,
) -> Record:
    hearing_ids = sorted(entry["hearing_id"] for entry in cache_status)
    return {
        "funnel": summarize_quote_funnel(funnel),
        "encoder_cache_consistency": summarize_cache_consistency(rows, cache_status),
        "population_exact_prefix_words": dict(
            sorted(Counter(str((row["exact_match"] or {}).get("words", 0)) for row in rows).items())
        ),
        "population_reference_tier": dict(
            Counter((row["reference"] or {}).get("tier") or "missing" for row in rows)
        ),
        "population_reference_support_type": dict(
            Counter((row["reference"] or {}).get("support_type") or "none" for row in rows)
        ),
        "positive_control": summarize_controls(controls, hearing_ids, config, group),
        "score_histograms": {
            method: {
                str(level): score_histogram(
                    [
                        row["fuzzy"][method][str(level)]["score"]
                        for row in rows
                        if row["fuzzy"][method][str(level)] is not None
                    ],
                    config.score_bin_width,
                )
                for level in config.prefix_levels
            }
            for method in QUOTE_METHODS
        },
        "gains": {
            method: {
                str(threshold): gains_at_threshold(
                    rows, method, threshold, hearing_ids, config, group
                )
                for threshold in config.quote_thresholds
            }
            for method in QUOTE_METHODS
        },
        "bands": {method: score_bands(rows, method, config) for method in QUOTE_METHODS},
    }


def hearing_text(hearing: Record) -> HearingText:
    transcript = hearing["transcricao"]
    return HearingText(
        transcript=transcript,
        turns={turn["turn_index"]: turn for turn in split_into_turns(transcript)},
        headers=tuple(header_text(match) for match in TURN_HEADER_PATTERN.finditer(transcript)),
    )


def quote_candidate_key(result: Record) -> tuple[Any, ...]:
    return (
        result["turn_index"],
        result["sentence_start_char"],
        result["sentence_end_char"],
        result["sentence"] or result["aligned_text"],
    )


def quote_display(
    row: Record, result: Record, text: HearingText, chars: int
) -> tuple[Record, bool]:
    start, end = result["sentence_start_char"], result["sentence_end_char"]
    located = start is not None and end is not None and bool(result["sentence"])
    before, after = (
        context_window(text.transcript, text.turns[result["turn_index"]], start, end, chars)
        if located
        else ("", "")
    )
    display = {
        "hearing_id": row["hearing_id"],
        "participant": row["person"]["name"],
        "role": row["person"]["role"],
        "opinion": row["opinion"],
        "quotes": row["quotes"],
        "turn_header": text.headers[result["turn_index"]],
        "context_before": before,
        "candidate": result["sentence"] or result["aligned_text"],
        "context_after": after,
    }
    return display, located


def quote_review_items(
    rows: list[Record], texts: dict[int, HearingText], config: FuzzyConfig
) -> list[Record]:
    items = []
    for row in rows:
        candidates: dict[tuple[Any, ...], Record] = {}
        for method in QUOTE_METHODS:
            for level in config.prefix_levels:
                result = row["fuzzy"][method][str(level)]
                if result is None or result["score"] < config.quote_review_min_score:
                    continue
                entry = candidates.setdefault(
                    quote_candidate_key(result), {"result": result, "proposals": []}
                )
                entry["proposals"].append(
                    {
                        "method": method,
                        "level": level,
                        "score": result["score"],
                        "quote_index": result["quote_index"],
                        "prefix": result["prefix"],
                        "aligned_text": result["aligned_text"],
                        "agrees_with_encoder": result["agrees_with_encoder"],
                        "best_elsewhere": row["elsewhere"][method][str(level)],
                    }
                )
        reference = row["reference"] or {}
        for entry in candidates.values():
            result = entry["result"]
            display, located = quote_display(
                row, result, texts[row["hearing_id"]], config.review_context_chars
            )
            items.append(
                {
                    "display": display,
                    "hidden": {
                        "row_id": row["id"],
                        "split": row["split"],
                        "hearing_id": row["hearing_id"],
                        "turn_index": result["turn_index"],
                        "sentence": result["sentence"],
                        "sentence_start_char": result["sentence_start_char"],
                        "sentence_end_char": result["sentence_end_char"],
                        "context_located": located,
                        "proposals": entry["proposals"],
                        "exact_match": row["exact_match"],
                        "encoder_sentence": reference.get("text"),
                        "encoder_tier": reference.get("tier"),
                        "encoder_support_type": reference.get("support_type"),
                    },
                }
            )
    return items


def normalized_person_name(name: str) -> str:
    return " ".join(WORD_TOKEN_PATTERN.findall(normalize_name(name)))


def hearing_speakers(turns: list[Record]) -> list[Record]:
    speakers: dict[str, Record] = {}
    for turn in turns:
        for candidate in turn_name_candidates(turn):
            key = normalized_person_name(candidate)
            if not key:
                continue
            entry = speakers.setdefault(
                key, {"key": key, "display": candidate, "turn_indices": set(), "party_info": []}
            )
            entry["turn_indices"].add(turn["turn_index"])
            if turn["party_info"] and turn["party_info"] not in entry["party_info"]:
                entry["party_info"].append(turn["party_info"])
    return list(speakers.values())


def name_score(metric: str, name: str, key: str) -> float:
    if metric == "token_set_ratio":
        return float(fuzz.token_set_ratio(name, key))
    return 100 * float(JaroWinkler.normalized_similarity(name, key))


def ranked_speakers(name: str, speakers: list[Record], metric: str) -> list[tuple[float, int]]:
    scored = [
        (name_score(metric, name, speaker["key"]), position)
        for position, speaker in enumerate(speakers)
    ]
    return sorted(scored, key=lambda item: (-item[0], item[1]))


def assigned_participants(turn_indices: set[int], people: list[Record], skip: int) -> list[Record]:
    return [
        {"index": person["index"], "name": person["participant"]["nome"]}
        for person in people
        if person["index"] != skip
        and turn_indices & {turn["turn_index"] for turn in person["matched_turns"]}
    ]


def quote_support(opinions: list[str], speaker_turns: list[Record]) -> Record:
    matches = [find_opinion_turn_quote_match(opinion, speaker_turns) for opinion in opinions]
    return {
        "opinions": len(opinions),
        "with_trusted_prefix": sum(1 for match in matches if is_trusted_quote(match)),
        "with_any_prefix": sum(1 for match in matches if match is not None),
    }


def speaker_entry(
    speaker: Record, score: float, turns_by_index: dict[int, Record], config: FuzzyConfig
) -> Record:
    ordered = sorted(speaker["turn_indices"])
    return {
        "speaker": speaker["key"],
        "display": speaker["display"],
        "score": rounded(score),
        "turn_count": len(ordered),
        "turn_indices": ordered,
        "party_info": speaker["party_info"],
        "speech_excerpt": turn_text(turns_by_index[ordered[0]])[: config.speech_excerpt_chars],
    }


def metric_result(
    name: str,
    speakers: list[Record],
    person: Record,
    people: list[Record],
    turns_by_index: dict[int, Record],
    config: FuzzyConfig,
    metric: str,
) -> Record | None:
    ranking = ranked_speakers(name, speakers, metric)
    if not ranking:
        return None
    best_score, best_position = ranking[0]
    best = speakers[best_position]
    rival = next(
        (
            (score, position)
            for score, position in ranking[1:]
            if not speakers[position]["turn_indices"] & best["turn_indices"]
        ),
        None,
    )
    best_turns = [turns_by_index[index] for index in sorted(best["turn_indices"])]
    return {
        "best": speaker_entry(best, best_score, turns_by_index, config),
        "second": (
            speaker_entry(speakers[rival[1]], rival[0], turns_by_index, config) if rival else None
        ),
        "margin": rounded(best_score - rival[0]) if rival else None,
        "best_assigned_to": assigned_participants(best["turn_indices"], people, person["index"]),
        "best_quote_support": quote_support(person["participant"]["opinioes"], best_turns),
    }


def collect_names(
    hearings: list[Record], split_of: dict[int, str], config: FuzzyConfig
) -> tuple[list[Record], list[Record]]:
    unresolved: list[Record] = []
    impostors: list[Record] = []
    for hearing in hearings:
        hearing_id, split = hearing["id"], split_of[hearing["id"]]
        turns, people = hearing_people(hearing)
        speakers = hearing_speakers(turns)
        turns_by_index = {turn["turn_index"]: turn for turn in turns}
        for person in people:
            participant = person["participant"]
            name = normalized_person_name(participant["nome"])
            if person["matched_turns"]:
                impostors.append(impostor_row(name, person, speakers, hearing_id, split, config))
                continue
            unresolved.append(
                {
                    "id": f"person-{hearing_id}-{person['index']}",
                    "split": split,
                    "hearing_id": hearing_id,
                    "person_index": person["index"],
                    "name": participant["nome"],
                    "normalized_name": name,
                    "role": participant["cargo"],
                    "opinions": participant["opinioes"],
                    "speakers_in_hearing": len(speakers),
                    "metrics": {
                        metric: metric_result(
                            name, speakers, person, people, turns_by_index, config, metric
                        )
                        for metric in config.name_metrics
                    },
                }
            )
    return unresolved, impostors


def impostor_row(
    name: str,
    person: Record,
    speakers: list[Record],
    hearing_id: int,
    split: str,
    config: FuzzyConfig,
) -> Record:
    own_turns = {turn["turn_index"] for turn in person["matched_turns"]}
    row: Record = {
        "id": f"person-{hearing_id}-{person['index']}",
        "split": split,
        "hearing_id": hearing_id,
        "name": person["participant"]["nome"],
    }
    for metric in config.name_metrics:
        own = [
            name_score(metric, name, speaker["key"])
            for speaker in speakers
            if speaker["turn_indices"] & own_turns
        ]
        others = [
            (name_score(metric, name, speaker["key"]), speaker["display"])
            for speaker in speakers
            if not speaker["turn_indices"] & own_turns
        ]
        top_other = max(others, key=lambda item: item[0]) if others else None
        row[metric] = {
            "own_score": rounded(max(own)) if own else None,
            "impostor_score": rounded(top_other[0]) if top_other else None,
            "impostor_speaker": top_other[1] if top_other else None,
        }
    return row


def name_acceptance(row: Record, rule: str, threshold: float, config: FuzzyConfig) -> Record | None:
    metrics = config.name_metrics if rule == COMBINED_NAME_RULE else (rule,)
    results = [row["metrics"].get(metric) for metric in metrics]
    if any(result is None for result in results):
        return None
    if any(result["best"]["score"] < threshold for result in results):
        return None
    if len({result["best"]["speaker"] for result in results}) != 1:
        return {"status": "metrics_disagree", "result": results[0]}
    ambiguous = any(
        result["second"] is not None and result["second"]["score"] >= threshold
        for result in results
    )
    if ambiguous:
        return {"status": "ambiguous", "result": results[0]}
    if results[0]["best_assigned_to"]:
        return {"status": "already_assigned", "result": results[0]}
    return {"status": "unique", "result": results[0]}


def contested_ids(accepted: list[tuple[Record, Record]]) -> set[str]:
    contested: set[str] = set()
    for position, (row, decision) in enumerate(accepted):
        turns = set(decision["result"]["best"]["turn_indices"])
        for other_row, other_decision in accepted[position + 1 :]:
            if other_row["hearing_id"] != row["hearing_id"]:
                continue
            if turns & set(other_decision["result"]["best"]["turn_indices"]):
                contested.update({row["id"], other_row["id"]})
    return contested


def name_decisions(
    rows: list[Record], rule: str, threshold: float, config: FuzzyConfig
) -> list[tuple[Record, Record]]:
    return [
        (row, decision)
        for row in rows
        if (decision := name_acceptance(row, rule, threshold, config)) is not None
    ]


def resolved_names(
    decisions: list[tuple[Record, Record]],
) -> tuple[list[tuple[Record, Record]], set[str]]:
    unique = [(row, decision) for row, decision in decisions if decision["status"] == "unique"]
    contested = contested_ids(unique)
    return [(row, decision) for row, decision in unique if row["id"] not in contested], contested


def names_at_threshold(
    rows: list[Record],
    impostors: list[Record],
    rule: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    decisions = name_decisions(rows, rule, threshold, config)
    resolved, contested = resolved_names(decisions)
    summary: Record = {
        "best_at_or_above": len(decisions),
        "by_status": dict(Counter(decision["status"] for _, decision in decisions)),
        "contested_among_unresolved": len(contested),
        "would_resolve": len(resolved),
        "would_resolve_opinions": sum(len(row["opinions"]) for row, _ in resolved),
        "would_resolve_with_trusted_quote_support": sum(
            1
            for _, decision in resolved
            if decision["result"]["best_quote_support"]["with_trusted_prefix"]
        ),
        "would_resolve_with_any_quote_prefix": sum(
            1
            for _, decision in resolved
            if decision["result"]["best_quote_support"]["with_any_prefix"]
        ),
        "would_resolve_ids": [row["id"] for row, _ in resolved],
    }
    if rule in NAME_METRICS:
        summary["impostor_control"] = impostor_rates(
            impostors, rule, threshold, hearing_ids, config, group
        )
    return summary


def impostor_rates(
    impostors: list[Record],
    metric: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    own_hits: dict[int, float] = {}
    impostor_hits: dict[int, float] = {}
    outranks: dict[int, float] = {}
    totals: dict[int, float] = {}
    for row in impostors:
        scores = row[metric]
        hearing_id = row["hearing_id"]
        totals[hearing_id] = totals.get(hearing_id, 0.0) + 1
        own, other = scores["own_score"], scores["impostor_score"]
        if own is not None and own >= threshold:
            own_hits[hearing_id] = own_hits.get(hearing_id, 0.0) + 1
        if other is not None and other >= threshold:
            impostor_hits[hearing_id] = impostor_hits.get(hearing_id, 0.0) + 1
            if own is None or other >= own:
                outranks[hearing_id] = outranks.get(hearing_id, 0.0) + 1
    stream = f"{group}/names/{metric}/{threshold}"
    return {
        "resolved_participants": int(sum(totals.values())),
        "own_speaker_at_or_above": bootstrap_ratio(
            own_hits, totals, hearing_ids, config, f"{stream}/own"
        ),
        "impostor_at_or_above": bootstrap_ratio(
            impostor_hits, totals, hearing_ids, config, f"{stream}/other"
        ),
        "impostor_at_or_above_and_not_below_own": int(sum(outranks.values())),
    }


def top_impostors(impostors: list[Record], config: FuzzyConfig) -> Record:
    return {
        metric: [
            {
                "id": row["id"],
                "split": row["split"],
                "name": row["name"],
                "own_score": row[metric]["own_score"],
                "impostor_speaker": row[metric]["impostor_speaker"],
                "impostor_score": row[metric]["impostor_score"],
            }
            for row in sorted(
                (row for row in impostors if row[metric]["impostor_score"] is not None),
                key=lambda row: (-row[metric]["impostor_score"], row["hearing_id"], row["id"]),
            )[: config.impostor_examples]
        ]
        for metric in config.name_metrics
    }


def name_rules(config: FuzzyConfig) -> tuple[str, ...]:
    if len(config.name_metrics) > 1:
        return (*config.name_metrics, COMBINED_NAME_RULE)
    return config.name_metrics


def summarize_name_group(
    rows: list[Record],
    impostors: list[Record],
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    return {
        "unresolved_participants": len(rows),
        "unresolved_opinions": sum(len(row["opinions"]) for row in rows),
        "hearings_with_unresolved": len({row["hearing_id"] for row in rows}),
        "without_any_speaker": sum(
            1 for row in rows if all(result is None for result in row["metrics"].values())
        ),
        "same_best_speaker_across_metrics": sum(
            1
            for row in rows
            if all(result is not None for result in row["metrics"].values())
            and len({result["best"]["speaker"] for result in row["metrics"].values()}) == 1
        ),
        "best_score_histograms": {
            metric: score_histogram(
                [
                    row["metrics"][metric]["best"]["score"]
                    for row in rows
                    if row["metrics"][metric] is not None
                ],
                config.score_bin_width,
            )
            for metric in config.name_metrics
        },
        "by_rule": {
            rule: {
                str(threshold): names_at_threshold(
                    rows, impostors, rule, threshold, hearing_ids, config, group
                )
                for threshold in config.name_thresholds
            }
            for rule in name_rules(config)
        },
        "top_impostors": top_impostors(impostors, config),
    }


def name_review_items(rows: list[Record], config: FuzzyConfig) -> list[Record]:
    items = []
    for row in rows:
        results = {metric: result for metric, result in row["metrics"].items() if result}
        if (
            not results
            or max(result["best"]["score"] for result in results.values())
            < config.name_review_min_score
        ):
            continue
        candidates: dict[str, Record] = {}
        for metric, result in results.items():
            for rank in ("best", "second"):
                entry = result[rank]
                if entry is None:
                    continue
                candidate = candidates.setdefault(entry["speaker"], {"entry": entry, "metrics": {}})
                candidate["metrics"][metric] = {
                    "rank": rank,
                    "score": entry["score"],
                    **(
                        {
                            "assigned_to": result["best_assigned_to"],
                            "quote_support": result["best_quote_support"],
                        }
                        if rank == "best"
                        else {}
                    ),
                }
        for speaker, candidate in candidates.items():
            entry = candidate["entry"]
            items.append(
                {
                    "display": {
                        "hearing_id": row["hearing_id"],
                        "participant": row["name"],
                        "role": row["role"],
                        "opinions": row["opinions"],
                        "candidate_speaker": entry["display"],
                        "party_info": entry["party_info"],
                        "turn_count": entry["turn_count"],
                        "speech_excerpt": entry["speech_excerpt"],
                    },
                    "hidden": {
                        "row_id": row["id"],
                        "split": row["split"],
                        "hearing_id": row["hearing_id"],
                        "speaker": speaker,
                        "turn_indices": entry["turn_indices"],
                        "metrics": candidate["metrics"],
                    },
                }
            )
    return items


def sheet_fields(kind: str) -> list[str]:
    return ["item_id", *DISPLAY_FIELDS[kind], *FILL_FIELDS]


def blind_sheet(kind: str, items: list[Record], config: FuzzyConfig) -> dict[str, Record]:
    labelled = assign_item_ids(
        items,
        draw_rng(config.seed, *config.review_order_streams[kind]),
        config.review_item_prefixes[kind],
    )
    for item_id, item in labelled.items():
        if list(item["display"]) != list(DISPLAY_FIELDS[kind]):
            raise SystemExit(f"{kind} review item {item_id} shows fields outside the sheet design")
    return labelled


def sheet_rows(kind: str, labelled: dict[str, Record]) -> list[Record]:
    rows = [
        {"item_id": item_id, **item["display"], **{name: None for name in FILL_FIELDS}}
        for item_id, item in labelled.items()
    ]
    if any(list(row) != sheet_fields(kind) for row in rows):
        raise SystemExit(f"the {kind} review sheet has fields outside its design")
    return rows


def refuse_filled_sheet(path: Path) -> None:
    if not path.exists():
        return
    filled = [row.get("item_id") for row in load_jsonl(path) if row.get("judgment") is not None]
    if filled:
        raise SystemExit(
            f"{path} already holds {len(filled)} judgments (e.g. {filled[:3]}); a filled review "
            "sheet is never overwritten, so nothing was written"
        )


def review_key(
    kind: str,
    labelled: dict[str, Record],
    run_name: str,
    splits: tuple[str, ...],
    final_test: bool,
    rows_path: Path,
    sheet_path: Path,
    config: FuzzyConfig,
) -> Record:
    return {
        "role": f"fuzzy_{kind}_review_key",
        "run_name": run_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "splits_used": list(splits),
        "final_test": final_test,
        "rows_file": {"path": str(rows_path), "sha256": sha256_of_file(rows_path)},
        "sheet": {
            "path": str(sheet_path),
            "rows": len(labelled),
            "sha256_at_creation": sha256_of_file(sheet_path),
            "fields": sheet_fields(kind),
        },
        "display_fields": list(DISPLAY_FIELDS[kind]),
        "fill_fields": list(FILL_FIELDS),
        "labels": list(config.review_labels[kind]),
        "positive_label": config.review_positive[kind],
        "unsure_label": config.review_unsure[kind],
        "items": {
            item_id: {**item["hidden"], "display_sha256": canonical_sha256(item["display"])}
            for item_id, item in labelled.items()
        },
    }


def review_summary(kind: str, key: Record, key_path: Path, config: FuzzyConfig) -> Record:
    items = list(key["items"].values())
    summary: Record = {
        "sheet": {
            "path": key["sheet"]["path"],
            "rows": key["sheet"]["rows"],
            "sha256": key["sheet"]["sha256_at_creation"],
        },
        "key": {"path": str(key_path), "sha256": sha256_of_file(key_path)},
        "shown_fields": key["display_fields"],
        "fill_fields": key["fill_fields"],
        "hidden_fields": sorted({name for item in items for name in item} - {"display_sha256"}),
        "labels": key["labels"],
        "items_by_split": dict(sorted(Counter(item["split"] for item in items).items())),
        "distinct_rows": len({item["row_id"] for item in items}),
    }
    if kind == "quotes":
        summary["proposals"] = sum(len(item["proposals"]) for item in items)
        summary["items_without_context"] = sum(1 for item in items if not item["context_located"])
        summary["items_per_opinion"] = dict(
            sorted(Counter(Counter(item["row_id"] for item in items).values()).items())
        )
    else:
        summary["items_per_participant"] = dict(
            sorted(Counter(Counter(item["row_id"] for item in items).values()).items())
        )
    return summary


def group_members(splits: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {name: (name,) for name in splits} | {"all": splits}


def artifact_entry(path: Path, rows: int) -> Record:
    return {"path": str(path), "rows": rows, "sha256": sha256_of_file(path)}


def build_report(
    run_name: str,
    splits: tuple[str, ...],
    final_test: bool,
    split_source: Record,
    reference_source: Record,
    encoder_source: Record,
    quote_rows: list[Record],
    funnel: list[Record],
    controls: list[Record],
    cache_status: list[Record],
    name_rows: list[Record],
    impostors: list[Record],
    artifacts: Record,
    review: Record,
    elapsed_seconds: float,
    config: FuzzyConfig,
) -> Record:
    groups = group_members(splits)
    return {
        "run_name": run_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": (
            "would approximate string matching recover quotes and participant names that the "
            "exact rules miss, and at what risk"
        ),
        "splits_used": list(splits),
        "final_test": final_test,
        "sources": {
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "splits": split_source,
            "reference_udv": reference_source,
            "encoder_baseline": encoder_source,
        },
        "definitions": {
            "quote_population": (
                "opinions of people with at least one matched turn, with at least one quote "
                "extracted by udv_pipeline.extract_quotes, whose exact match "
                "(find_opinion_turn_quote_match) has no trusted prefix of "
                f"{TRUSTED_PREFIX_WORDS}+ words"
            ),
            "eligible_quote": (
                f"an extracted quote with {config.min_prefix_words}+ whitespace words"
            ),
            "gain": (
                "a population opinion whose fuzzy score reaches the threshold at the first "
                "prefix level, in configured order, that reaches it; the sentence is the one "
                "enclosing the aligned span in that turn"
            ),
            "agrees_with_encoder": (
                "sentences_agree between the fuzzy sentence and the reference run's top-1 "
                "sentence, only for records whose reference tier is semantic; it is a weak "
                "precision proxy, because the encoder embeds the full opinion, quote included, "
                "so both signals see the quote text and are not independent"
            ),
            "null": (
                "quotes that should not occur in the person's speech, aligned against the same "
                "person's turns with the same scorer; the rate of draws at or above the "
                "threshold estimates how often a threshold is reached by chance; "
                "same_hearing_other_person is topic-matched and may include genuine repetitions"
            ),
            "expected_chance_matches": (
                "null rate times the population opinions with an eligible quote"
            ),
            "positive_control": (
                "opinions with a trusted exact prefix, the exact prefix aligned with the same "
                "scorer; score 100 and agreement with the exact sentence are expected"
            ),
            "encoder_baseline": (
                "on the positive control, how often the production encoder's top-1 sentence, "
                "recomputed from the cached embeddings of the production run (no encoding here), "
                "agrees with the sentence of the trusted exact quote; it is the agreement rate "
                "the proxy reaches on quote sentences known by exact matching, so an agreement "
                "rate among fuzzy gains is read against it, not against 1"
            ),
            "elsewhere": (
                "the same prefixes aligned against every turn of the hearing not matched to the "
                "person, at the accepted level; a gain whose quote scores higher elsewhere may "
                "belong to another speaker (misattribution, a quote read aloud, or the person "
                "quoting someone)"
            ),
            "encoder_cache_consistency": (
                "for population opinions, whether the cached top-1 equals the reference run's "
                "evidence; with the production run as reference every compared pair must match"
            ),
            "bands": (
                "opinions accepted at a threshold and not at the next one, with the sentence of "
                "the lower threshold"
            ),
            "names_population": "participants with no matched turn under resolve_person_speech",
            "names_status": (
                "unique: best at or above the threshold and the best disjoint rival below it; "
                "ambiguous: a disjoint rival also at or above it; already_assigned: the best "
                "speaker shares turns with a participant the exact rules resolved; "
                "metrics_disagree (rule both): the two metrics pick different speakers; "
                "contested: two unresolved participants of one hearing would take the same turns"
            ),
            "impostor_control": (
                "participants the exact rules resolved, scored against the hearing's speakers "
                "that share no turn with their own; the rate at or above the threshold "
                "estimates how often a name reaches it on a different speaker"
            ),
            "quote_support": (
                "opinions of the participant whose quote has an exact prefix match "
                "(find_opinion_turn_quote_match) inside the candidate speaker's turns"
            ),
            "bootstrap": (
                f"{config.bootstrap_samples} resamples of whole hearings, "
                f"{config.confidence_level:.0%} percentile interval, fixed seed"
            ),
        },
        "quotes": {
            name: summarize_quote_group(
                [row for row in quote_rows if row["split"] in members],
                [entry for entry in funnel if entry["split"] in members],
                [row for row in controls if row["split"] in members],
                [entry for entry in cache_status if entry["split"] in members],
                config,
                name,
            )
            for name, members in groups.items()
        },
        "names": {
            name: summarize_name_group(
                [row for row in name_rows if row["split"] in members],
                [row for row in impostors if row["split"] in members],
                sorted(entry["hearing_id"] for entry in cache_status if entry["split"] in members),
                config,
                name,
            )
            for name, members in groups.items()
        },
        "artifacts": artifacts,
        "review": review,
        "code": {
            **source_hashes(Path(__file__)),
            **source_hashes(transcript, *transcript.SOURCES),
            **source_hashes(udv_run),
            **source_hashes(bookworm.data.io),
            **source_hashes(generate_sample),
        },
        "timing": {"elapsed_seconds": round(elapsed_seconds, 1)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rapidfuzz": rapidfuzz.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "config": config.source,
    }


def print_summary(report: Record, config: FuzzyConfig) -> None:
    for group, summary in report["quotes"].items():
        funnel = summary["funnel"]
        print(
            f"quotes {group:10s} population={funnel['population_without_trusted_exact']} "
            f"eligible={funnel['population_with_eligible_quote']} "
            f"controls={summary['positive_control']['opinions']} "
            f"encoder_baseline={summary['positive_control']['encoder_baseline']['agrees']}/"
            f"{summary['positive_control']['encoder_baseline']['with_encoder_cache']} "
            f"cached_hearings={summary['encoder_cache_consistency']['hearings_with_encoder_cache']}/"
            f"{summary['encoder_cache_consistency']['hearings']}"
        )
        for method in QUOTE_METHODS:
            cells = []
            for threshold in (config.quote_thresholds[0], 90.0, 100.0):
                gains = summary["gains"][method].get(str(threshold))
                if gains is None:
                    continue
                null = gains["null"]["other_hearing_same_split"]["rate"]["value"]
                cells.append(
                    f"t={threshold:g}: gain={gains['opinions']} "
                    f"agree={gains['agrees_with_encoder']}/{gains['with_encoder_reference']} "
                    f"null={null}"
                )
            print(f"  {method:5s} " + " | ".join(cells))
    for group, summary in report["names"].items():
        print(
            f"names  {group:10s} unresolved={summary['unresolved_participants']} "
            f"opinions={summary['unresolved_opinions']}"
        )
        for rule, by_threshold in summary["by_rule"].items():
            cells = []
            for threshold in (config.name_thresholds[0], 90.0, 100.0):
                entry = by_threshold.get(str(threshold))
                if entry is not None:
                    cells.append(
                        f"t={threshold:g}: resolve={entry['would_resolve']} "
                        f"status={entry['by_status']}"
                    )
            print(f"  {rule:15s} " + " | ".join(cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure what approximate string matching (rapidfuzz) would recover over the "
        "exact quote and name rules, and at what risk, without changing the pipeline."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/fuzzy_matching.toml"))
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="also read the test split; only for the final, pre-registered evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    splits = (*config.splits, "test") if args.final_test else config.splits
    run_name = f"{config.version}_final_test" if args.final_test else config.version
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    hearings = select_hearings(lds, split_of, splits)
    reference, reference_source = select_reference_run(config.reference_runs)
    udv_config = udv_run.load_config(config.udv_config_path)
    encoder_source = {
        "udv_config_path": str(config.udv_config_path),
        "udv_config_sha256": sha256_of_file(config.udv_config_path),
        "encoder": udv_config.model_name,
        "encoder_revision": udv_config.model_revision,
        "cache_dir": str(udv_config.cache_dir),
        "cache_device_label": config.cache_device_label,
        "encoding_done_here": False,
    }
    print(
        f"{len(hearings)} hearings ({', '.join(splits)}) | reference {reference_source['run_name']}"
        f"{' (fallback)' if reference_source['fallback_used'] else ''}",
        flush=True,
    )
    quote_rows, funnel, controls, cache_status = collect_quotes(
        hearings, split_of, reference, udv_config, config
    )
    name_rows, impostors = collect_names(hearings, split_of, config)
    texts = {hearing["id"]: hearing_text(hearing) for hearing in hearings}
    sheets = {
        "quotes": blind_sheet("quotes", quote_review_items(quote_rows, texts, config), config),
        "names": blind_sheet("names", name_review_items(name_rows, config), config),
    }
    rows_paths = {
        "quotes": config.output_dir / f"{run_name}_quotes.jsonl",
        "names": config.output_dir / f"{run_name}_names.jsonl",
    }
    sheet_paths = {kind: config.output_dir / f"{run_name}_{kind}_review.jsonl" for kind in sheets}
    key_paths = {kind: config.output_dir / f"{run_name}_{kind}_review_key.json" for kind in sheets}
    for path in sheet_paths.values():
        refuse_filled_sheet(path)
    rows_by_kind = {"quotes": quote_rows, "names": name_rows}
    artifacts: Record = {}
    review: Record = {
        "blinding_rule": config.source["review"]["blinding_rule"],
        "known_leaks": config.source["review"]["known_leaks"],
        "reviewer_opens": [str(path) for path in sheet_paths.values()],
        "precision_command": "uv run --no-sync python -m "
        "experiments.validation.fuzzy_review_precision"
        + (" --final-test" if args.final_test else ""),
    }
    for kind in REVIEW_KINDS:
        write_jsonl(rows_by_kind[kind], rows_paths[kind])
        write_jsonl(sheet_rows(kind, sheets[kind]), sheet_paths[kind])
        key = review_key(
            kind,
            sheets[kind],
            run_name,
            splits,
            args.final_test,
            rows_paths[kind],
            sheet_paths[kind],
            config,
        )
        write_json(key, key_paths[kind])
        artifacts[kind] = artifact_entry(rows_paths[kind], len(rows_by_kind[kind]))
        artifacts[f"{kind}_review"] = artifact_entry(sheet_paths[kind], len(sheets[kind]))
        artifacts[f"{kind}_review_key"] = artifact_entry(key_paths[kind], len(sheets[kind]))
        review[kind] = review_summary(kind, key, key_paths[kind], config)
    report = build_report(
        run_name,
        splits,
        args.final_test,
        split_source,
        reference_source,
        encoder_source,
        quote_rows,
        funnel,
        controls,
        cache_status,
        name_rows,
        impostors,
        artifacts,
        review,
        time.perf_counter() - started,
        config,
    )
    write_json(report, config.output_dir / f"{run_name}_report.json")
    print_summary(report, config)
    print(f"elapsed {report['timing']['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
