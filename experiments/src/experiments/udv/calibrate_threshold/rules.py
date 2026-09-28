"""The calibration rules: three pair rules and the primary top-1 Youden rule."""

from functools import partial
from typing import Any

import numpy as np

from experiments.common.reporting import rounded
from experiments.udv.calibrate_threshold.config import (
    PAIR_RULES,
    PRIMARY_RULE,
    RULES,
    CalibrationConfig,
)
from experiments.udv.calibrate_threshold.negatives import (
    NegativeDraw,
    NegativeSampler,
    SentencePool,
    draw_hard_negatives,
    draw_pool_negatives,
    negatives_by_query,
)
from experiments.udv.calibrate_threshold.statistics import (
    describe,
    interval,
    resample_rows,
    rule_rng,
    unit_groups,
)

Record = dict[str, Any]

YOUDEN_TIE_TOLERANCE = 1e-12
PAIR_BOOTSTRAP_KEYS = ("threshold", "positive_quantile_value", "negative_quantile_value")
RULE_DEFINITIONS: Record = {
    "legacy_random": (
        "positives: unmasked opinion vs the sentence holding its trusted quote; negatives: the "
        "same opinion vs a sentence drawn uniformly from all sentences of the other calibration "
        "hearings; threshold: midpoint of the configured positive and negative quantiles"
    ),
    "hard_negative": (
        "positives as in legacy_random; negatives: the same opinion vs a sentence drawn uniformly "
        "from the speaker's own sentences that do not agree with the target; threshold as in "
        "legacy_random; opinions whose speaker has no such sentence are left out"
    ),
    "masked_hard_negative": (
        "positives: masked opinion (quoted spans removed, masked_quotes benchmark) vs its "
        "target; negatives: the masked opinion vs a non-target sentence of the same speaker; "
        "threshold as in legacy_random"
    ),
    "masked_top1_youden": (
        "for each masked opinion the top-1 sentence by cosine among the speaker's sentences "
        "(first index on ties, as in the UDV builder) is correct when it agrees with the "
        "target; the threshold maximizes Youden's J = TPR - FPR of 'top-1 score >= t predicts "
        "a correct top-1', the lowest threshold wins ties, and the reported value is the "
        "midpoint between the lowest top-1 score kept and the highest one excluded"
    ),
}


def pair_threshold(
    positives: np.ndarray, negatives: np.ndarray, config: CalibrationConfig
) -> Record:
    positive_value = float(np.quantile(positives, config.positive_quantile))
    negative_value = float(np.quantile(negatives, config.negative_quantile))
    return {
        "threshold": (positive_value + negative_value) / 2,
        "positive_quantile_value": positive_value,
        "negative_quantile_value": negative_value,
    }


def evaluate_pair_rule(
    queries: list[Record],
    sampler: NegativeSampler,
    config: CalibrationConfig,
    rng: np.random.Generator,
) -> tuple[Record, NegativeDraw]:
    positives = np.array([query["positive_score"] for query in queries], dtype=np.float64)
    point_draw = sampler(rng, np.arange(len(queries), dtype=np.int64))
    point = pair_threshold(positives, point_draw.scores, config)
    groups = unit_groups([query["hearing_id"] for query in queries], config.bootstrap_unit)
    replicates = []
    for _ in range(config.bootstrap_samples):
        rows = resample_rows(rng, groups)
        replicates.append(pair_threshold(positives[rows], sampler(rng, rows).scores, config))
    threshold = point["threshold"]
    result = {
        "pairs": {
            "queries": len(queries),
            "hearings": len({query["hearing_id"] for query in queries}),
            "positives": len(positives),
            "negatives": len(point_draw.scores),
        },
        "threshold": rounded(threshold),
        "threshold_exact": threshold,
        "threshold_rounded": round(threshold, config.threshold_decimals),
        "positive_quantile_value": rounded(point["positive_quantile_value"]),
        "negative_quantile_value": rounded(point["negative_quantile_value"]),
        "positives_below_threshold": int((positives < threshold).sum()),
        "negatives_at_or_above_threshold": int((point_draw.scores >= threshold).sum()),
        "positive_scores": describe(positives),
        "negative_scores": describe(point_draw.scores),
        "bootstrap": {
            key: interval([replicate[key] for replicate in replicates], config.confidence_level)
            for key in PAIR_BOOTSTRAP_KEYS
        },
    }
    return result, point_draw


def youden_optimum(scores: np.ndarray, correct: np.ndarray) -> Record | None:
    positives = int(correct.sum())
    negatives = len(correct) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered, labels = scores[order], correct[order]
    group_end = np.append(ordered[1:] != ordered[:-1], True)
    cut_scores = ordered[group_end]
    tpr = np.cumsum(labels)[group_end] / positives
    fpr = np.cumsum(~labels)[group_end] / negatives
    youden = tpr - fpr
    best = int(np.flatnonzero(youden >= youden.max() - YOUDEN_TIE_TOLERANCE)[-1])
    lowest_kept = float(cut_scores[best])
    highest_excluded = float(cut_scores[best + 1]) if best + 1 < len(cut_scores) else None
    threshold = lowest_kept if highest_excluded is None else (lowest_kept + highest_excluded) / 2
    return {
        "threshold": threshold,
        "youden_j": float(youden[best]),
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
        "lowest_kept_score": lowest_kept,
        "highest_excluded_score": highest_excluded,
    }


def operating_point(scores: np.ndarray, correct: np.ndarray, threshold: float) -> Record:
    kept = scores >= threshold
    correct_total = int(correct.sum())
    incorrect_total = len(correct) - correct_total
    return {
        "threshold": rounded(threshold),
        "kept": int(kept.sum()),
        "kept_correct": int((kept & correct).sum()),
        "coverage": rounded(kept.mean()) if len(scores) else None,
        "precision": rounded(correct[kept].mean()) if kept.any() else None,
        "tpr": rounded((kept & correct).sum() / correct_total) if correct_total else None,
        "fpr": rounded((kept & ~correct).sum() / incorrect_total) if incorrect_total else None,
    }


def top1_arrays(queries: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.array([query["top1_score"] for query in queries], dtype=np.float64)
    correct = np.array([query["top1_correct"] for query in queries], dtype=bool)
    return scores, correct


def primary_bootstrap(
    scores: np.ndarray,
    correct: np.ndarray,
    groups: list[np.ndarray],
    point_threshold: float | None,
    config: CalibrationConfig,
    rng: np.random.Generator,
) -> Record:
    """Hearing bootstrap of acc@1, the Youden optimum, and precision at the point threshold."""
    accuracy, thresholds, youden, precision, coverage = [], [], [], [], []
    without_optimum = 0
    for _ in range(config.bootstrap_samples):
        rows = resample_rows(rng, groups)
        sample_scores, sample_correct = scores[rows], correct[rows]
        accuracy.append(float(sample_correct.mean()))
        replicate = youden_optimum(sample_scores, sample_correct)
        if replicate is None:
            without_optimum += 1
        else:
            thresholds.append(replicate["threshold"])
            youden.append(replicate["youden_j"])
        if point_threshold is not None:
            kept = sample_scores >= point_threshold
            coverage.append(float(kept.mean()))
            if kept.any():
                precision.append(float(sample_correct[kept].mean()))
    level = config.confidence_level
    return {
        "acc_at_1": interval(accuracy, level),
        "threshold": interval(thresholds, level),
        "youden_j": interval(youden, level),
        "precision_at_point_threshold": interval(precision, level),
        "coverage_at_point_threshold": interval(coverage, level),
        "replicates_without_optimum": without_optimum,
    }


def optimum_fields(
    optimum: Record | None, scores: np.ndarray, correct: np.ndarray, decimals: int
) -> Record:
    if optimum is None:
        return {
            "threshold": None,
            "threshold_exact": None,
            "threshold_rounded": None,
            "optimum": None,
            "at_threshold": None,
        }
    return {
        "threshold": rounded(optimum["threshold"]),
        "threshold_exact": optimum["threshold"],
        "threshold_rounded": round(optimum["threshold"], decimals),
        "optimum": {
            key: None if value is None else rounded(value) for key, value in optimum.items()
        },
        "at_threshold": operating_point(scores, correct, optimum["threshold"]),
    }


def evaluate_primary_rule(
    queries: list[Record],
    unmasked_by_id: dict[str, Record],
    config: CalibrationConfig,
    rng: np.random.Generator,
) -> Record:
    scores, correct = top1_arrays(queries)
    optimum = youden_optimum(scores, correct)
    groups = unit_groups([query["hearing_id"] for query in queries], config.bootstrap_unit)
    bootstrap = primary_bootstrap(
        scores, correct, groups, None if optimum is None else optimum["threshold"], config, rng
    )
    n_candidates = np.array([len(query["candidates"]) for query in queries], dtype=np.float64)
    return {
        "pairs": {
            "queries": len(queries),
            "hearings": len({query["hearing_id"] for query in queries}),
            "top1_correct": int(correct.sum()),
            "top1_incorrect": int((~correct).sum()),
        },
        "acc_at_1": rounded(correct.mean()),
        "random_acc_at_1": rounded(
            np.mean([len(q["target_indices"]) / len(q["candidates"]) for q in queries])
        ),
        "unmasked_acc_at_1_same_queries": rounded(
            np.mean([unmasked_by_id[query["id"]]["top1_correct"] for query in queries])
        ),
        "candidates": describe(n_candidates),
        "top1_score_correct": describe(scores[correct]),
        "top1_score_incorrect": describe(scores[~correct]),
        **optimum_fields(optimum, scores, correct, config.threshold_decimals),
        "bootstrap": bootstrap,
    }


def with_hard_negatives(queries: list[Record]) -> list[Record]:
    return [query for query in queries if len(query["non_target_indices"])]


def hard_sampler(queries: list[Record], negatives_per_query: int) -> NegativeSampler:
    return partial(draw_hard_negatives, negatives_per_query=negatives_per_query, queries=queries)


def pair_rule_inputs(
    unmasked: list[Record], masked: list[Record], pool: SentencePool, negatives_per_query: int
) -> dict[str, tuple[list[Record], NegativeSampler]]:
    """The queries of each pair rule and the sampler that draws their negatives."""
    with_hard = with_hard_negatives(unmasked)
    masked_with_hard = with_hard_negatives(masked)
    pool_sampler = partial(
        draw_pool_negatives,
        negatives_per_query=negatives_per_query,
        query_hearing_ids=np.array([q["hearing_id"] for q in unmasked], dtype=np.int64),
        pool=pool,
    )
    return {
        "legacy_random": (unmasked, pool_sampler),
        "hard_negative": (with_hard, hard_sampler(with_hard, negatives_per_query)),
        "masked_hard_negative": (
            masked_with_hard,
            hard_sampler(masked_with_hard, negatives_per_query),
        ),
    }


def hard_negative_on_masked_ids(
    unmasked: list[Record], masked: list[Record], config: CalibrationConfig
) -> Record:
    masked_ids = {query["id"] for query in masked}
    same_ids = [query for query in with_hard_negatives(unmasked) if query["id"] in masked_ids]
    result, _ = evaluate_pair_rule(
        same_ids,
        hard_sampler(same_ids, config.negatives_per_query),
        config,
        rule_rng(config.seed, RULES.index("hard_negative"), 1),
    )
    return {
        "definition": (
            "hard_negative restricted to the opinions that are also masked queries, so that "
            "masked_hard_negative differs from it only by the masking"
        ),
        **result,
    }


def run_rules(
    unmasked: list[Record],
    masked: list[Record],
    pool: SentencePool,
    config: CalibrationConfig,
) -> tuple[Record, dict[str, dict[str, list[Record]]]]:
    """Every rule's result and, per pair rule, the negatives drawn for each query."""
    k = config.negatives_per_query
    inputs = pair_rule_inputs(unmasked, masked, pool, k)
    results: Record = {}
    draws: dict[str, dict[str, list[Record]]] = {}
    for stream, name in enumerate(PAIR_RULES):
        queries, sampler = inputs[name]
        result, draw = evaluate_pair_rule(queries, sampler, config, rule_rng(config.seed, stream))
        results[name] = {"definition": RULE_DEFINITIONS[name], **result}
        draws[name] = negatives_by_query(queries, draw, k)
    for name, queries in (("hard_negative", unmasked), ("masked_hard_negative", masked)):
        results[name]["queries_without_hard_negative"] = len(queries) - len(inputs[name][0])
    results["hard_negative"]["on_masked_query_ids"] = hard_negative_on_masked_ids(
        unmasked, masked, config
    )
    unmasked_by_id = {query["id"]: query for query in unmasked}
    results[PRIMARY_RULE] = {
        "definition": RULE_DEFINITIONS[PRIMARY_RULE],
        **evaluate_primary_rule(
            masked, unmasked_by_id, config, rule_rng(config.seed, RULES.index(PRIMARY_RULE))
        ),
    }
    return results, draws


def operating_points(results: Record, masked: list[Record], configured_threshold: float) -> Record:
    scores, correct = top1_arrays(masked)
    thresholds = {name: results[name]["threshold_exact"] for name in RULES}
    thresholds["configured_embedding_threshold"] = configured_threshold
    return {
        name: None if threshold is None else operating_point(scores, correct, threshold)
        for name, threshold in thresholds.items()
    }


def rule_query_hearings(unmasked: list[Record], masked: list[Record]) -> dict[str, set[int]]:
    return {
        "legacy_random": {query["hearing_id"] for query in unmasked},
        "hard_negative": {
            query["hearing_id"] for query in unmasked if len(query["non_target_indices"])
        },
        "masked_hard_negative": {
            query["hearing_id"] for query in masked if len(query["non_target_indices"])
        },
        PRIMARY_RULE: {query["hearing_id"] for query in masked},
    }
