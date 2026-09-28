"""Summaries of the quote rows: funnel, positive control, gains per threshold, bands, nulls."""

from collections import Counter
from typing import Any

from experiments.common.reporting import rounded
from experiments.udv.fuzzy_matching.config import MAX_SCORE, NULL_KINDS, QUOTE_METHODS, FuzzyConfig
from experiments.udv.fuzzy_matching.sampling import (
    HearingCounts,
    bootstrap_ratio,
    increment,
    score_histogram,
    sorted_counts,
)

Record = dict[str, Any]
Accepted = list[tuple[Record, Record]]


def accepted_result(
    row: Record, method: str, threshold: float, levels: tuple[int, ...]
) -> Record | None:
    """The result of the first prefix level, in configured order, that reaches the threshold."""
    for level in levels:
        result: Record | None = row["fuzzy"][method].get(str(level))
        if result is not None and result["score"] >= threshold:
            return result
    return None


def max_level_score(scores: Record) -> float | None:
    values = [value for value in scores.values() if value is not None]
    return max(values) if values else None


def null_rate(
    rows: list[Record],
    method: str,
    kind: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    stream: str,
) -> Record:
    hits: HearingCounts = {}
    draws: HearingCounts = {}
    for row in rows:
        for draw in row["null"][kind]:
            score = max_level_score(draw[method])
            increment(draws, row["hearing_id"])
            if score is not None and score >= threshold:
                increment(hits, row["hearing_id"])
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


def accepted_rows(
    rows: list[Record], method: str, threshold: float, config: FuzzyConfig
) -> Accepted:
    return [
        (row, result)
        for row in rows
        if (result := accepted_result(row, method, threshold, config.prefix_levels)) is not None
    ]


def encoder_agreement(accepted: Accepted) -> tuple[HearingCounts, HearingCounts]:
    agree: HearingCounts = {}
    with_reference: HearingCounts = {}
    for row, result in accepted:
        if result["agrees_with_encoder"] is None:
            continue
        increment(with_reference, row["hearing_id"])
        if result["agrees_with_encoder"]:
            increment(agree, row["hearing_id"])
    return agree, with_reference


def reference_counts(accepted: Accepted) -> Record:
    return {
        "by_reference_tier": dict(
            Counter((row["reference"] or {}).get("tier") or "missing" for row, _ in accepted)
        ),
        "by_reference_support_type": dict(
            Counter((row["reference"] or {}).get("support_type") or "none" for row, _ in accepted)
        ),
        "by_exact_prefix_words": sorted_counts(
            [str((row["exact_match"] or {}).get("words", 0)) for row, _ in accepted]
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
    accepted = accepted_rows(rows, method, threshold, config)
    agree, with_reference = encoder_agreement(accepted)
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
        "by_level": sorted_counts([str(result["level"]) for _, result in accepted]),
        "score_100": sum(1 for _, result in accepted if result["score"] >= MAX_SCORE),
        **reference_counts(accepted),
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


def count_by_hearing(rows: list[Record], key: str, expected: bool | None) -> HearingCounts:
    counts: HearingCounts = {}
    for row in rows:
        value = row[key]
        if value is None or (expected is not None and value is not expected):
            continue
        increment(counts, row["hearing_id"])
    return counts


def control_method_summary(controls: list[Record], method: str) -> Record:
    results = [row[method] for row in controls if row[method] is not None]
    return {
        "aligned": len(results),
        "score_100": sum(1 for result in results if result["score"] >= MAX_SCORE),
        "same_turn": sum(1 for result in results if result["same_turn"]),
        "sentence_agrees": sum(1 for result in results if result["sentence_agrees"]),
        "ids_below_100_or_other_sentence": [
            row["id"]
            for row in controls
            if row[method] is not None
            and (row[method]["score"] < MAX_SCORE or not row[method]["sentence_agrees"])
        ],
    }


def summarize_controls(
    controls: list[Record], hearing_ids: list[int], config: FuzzyConfig, group: str
) -> Record:
    with_encoder = count_by_hearing(controls, "encoder_agrees", None)
    agrees = count_by_hearing(controls, "encoder_agrees", True)
    return {
        "opinions": len(controls),
        "encoder_baseline": {
            "with_encoder_cache": int(sum(with_encoder.values())),
            "agrees": int(sum(agrees.values())),
            "rate": bootstrap_ratio(
                agrees, with_encoder, hearing_ids, config, f"{group}/control/encoder"
            ),
        },
        **{method: control_method_summary(controls, method) for method in QUOTE_METHODS},
    }


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


def band_members(
    rows: list[Record], method: str, low: float, high: float | None, config: FuzzyConfig
) -> list[Record]:
    """Results accepted at the low threshold and not at the high one."""
    return [
        result
        for row in rows
        if (result := accepted_result(row, method, low, config.prefix_levels)) is not None
        and (high is None or accepted_result(row, method, high, config.prefix_levels) is None)
    ]


def next_thresholds(thresholds: tuple[float, ...]) -> list[tuple[float, float | None]]:
    return list(zip(thresholds, [*thresholds[1:], None], strict=True))


def score_bands(rows: list[Record], method: str, config: FuzzyConfig) -> list[Record]:
    bands = []
    for low, high in next_thresholds(config.quote_thresholds):
        members = band_members(rows, method, low, high, config)
        with_reference = [result for result in members if result["agrees_with_encoder"] is not None]
        bands.append(
            {
                "low": low,
                "high": high,
                "opinions": len(members),
                "by_level": sorted_counts([str(result["level"]) for result in members]),
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


def level_histograms(rows: list[Record], method: str, config: FuzzyConfig) -> Record:
    return {
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
        "population_exact_prefix_words": sorted_counts(
            [str((row["exact_match"] or {}).get("words", 0)) for row in rows]
        ),
        "population_reference_tier": dict(
            Counter((row["reference"] or {}).get("tier") or "missing" for row in rows)
        ),
        "population_reference_support_type": dict(
            Counter((row["reference"] or {}).get("support_type") or "none" for row in rows)
        ),
        "positive_control": summarize_controls(controls, hearing_ids, config, group),
        "score_histograms": {
            method: level_histograms(rows, method, config) for method in QUOTE_METHODS
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
