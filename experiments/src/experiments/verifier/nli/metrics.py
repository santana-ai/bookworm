"""Ranking and thresholded metrics of a scorer against the benchmark labels."""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.common.reporting import rounded
from experiments.udv.calibrate_threshold import youden_optimum
from experiments.verifier.nli.config import VerifierConfig

Record = dict[str, Any]

FITTED_RULES = ("youden", "max_f1_not_inferable")
RANKING_METRICS = ("roc_auc", "average_precision_inferable", "average_precision_not_inferable")
BINARY_METRICS = (
    "accuracy",
    "cohen_kappa",
    "macro_f1",
    "f1_not_inferable",
    "precision_not_inferable",
    "recall_not_inferable",
)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if labels.all() or not labels.any():
        return {metric: float("nan") for metric in RANKING_METRICS}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision_inferable": float(average_precision_score(labels, scores)),
        "average_precision_not_inferable": float(average_precision_score(~labels, -scores)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    total = len(labels)
    inf_inf = float(np.sum(labels & predictions))
    inf_not = float(np.sum(labels & ~predictions))
    not_inf = float(np.sum(~labels & predictions))
    not_not = float(np.sum(~labels & ~predictions))
    accuracy = (inf_inf + not_not) / total
    expected = (
        (inf_inf + inf_not) * (inf_inf + not_inf) + (not_inf + not_not) * (inf_not + not_not)
    ) / total**2
    f1_not = safe_ratio(2 * not_not, 2 * not_not + not_inf + inf_not)
    f1_inf = safe_ratio(2 * inf_inf, 2 * inf_inf + inf_not + not_inf)
    return {
        "accuracy": accuracy,
        "cohen_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else float("nan"),
        "macro_f1": (f1_not + f1_inf) / 2,
        "f1_not_inferable": f1_not,
        "precision_not_inferable": safe_ratio(not_not, not_not + inf_not),
        "recall_not_inferable": safe_ratio(not_not, not_not + not_inf),
    }


def check_binary_metrics(fast: dict[str, float], reference: Record, where: str) -> None:
    pairs = {
        "accuracy": reference["accuracy"],
        "cohen_kappa": reference["cohen_kappa"],
        "macro_f1": reference["macro_f1"],
        "f1_not_inferable": reference["per_class"]["not_inferable"]["f1"],
    }
    for name, value in pairs.items():
        if not np.isnan(fast[name]) and abs(fast[name] - value) > 1e-4:
            raise SystemExit(f"{where}: {name} {fast[name]} != judge_metrics {value}")


def max_f1_not_inferable_optimum(scores: np.ndarray, labels: np.ndarray) -> Record | None:
    positives = int((~labels).sum())
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    ordered, ordered_labels = scores[order], labels[order]
    group_end = np.append(ordered[1:] != ordered[:-1], True)
    cut_scores = ordered[group_end]
    kept = np.concatenate([[0], np.flatnonzero(group_end) + 1])
    kept_inferable = np.concatenate([[0], np.cumsum(ordered_labels)[group_end]])
    kept_not = kept - kept_inferable
    true_not = positives - kept_not
    predicted_not = len(labels) - kept
    f1 = 2 * true_not / (predicted_not + positives)
    best = int(np.flatnonzero(f1 >= f1.max() - 1e-12)[-1])
    lowest_kept: float | None
    highest_excluded: float | None
    if best == 0:
        threshold = float(np.nextafter(cut_scores[0], np.inf))
        lowest_kept, highest_excluded = None, float(cut_scores[0])
    else:
        kept_score = float(cut_scores[best - 1])
        lowest_kept = kept_score
        highest_excluded = float(cut_scores[best]) if best < len(cut_scores) else None
        threshold = kept_score if highest_excluded is None else (kept_score + highest_excluded) / 2
    return {
        "threshold": threshold,
        "f1_not_inferable": float(f1[best]),
        "lowest_kept_score": lowest_kept,
        "highest_excluded_score": highest_excluded,
    }


def fit_rule(
    rule: str, scores: np.ndarray, labels: np.ndarray, config: VerifierConfig, probability: bool
) -> Record | None:
    if rule == "youden":
        return youden_optimum(scores, labels)
    if rule == "max_f1_not_inferable":
        return max_f1_not_inferable_optimum(scores, labels)
    return {"threshold": config.fixed_probability_threshold} if probability else None


def is_probability_system(system: str) -> bool:
    return not system.endswith(".cosine")


def rounded_record(values: dict[str, float]) -> Record:
    return {name: None if np.isnan(value) else rounded(value) for name, value in values.items()}
