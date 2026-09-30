"""Ranking and risk-coverage metrics of a confidence signal."""

import math
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.verifier.confidence.config import METRIC_CHECK_TOLERANCE

Record = dict[str, Any]


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(scores)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    n_positive = int(positive.sum())
    if n_positive == 0:
        return math.nan
    order = np.argsort(-scores, kind="stable")
    ordered, hits = scores[order], positive[order]
    last = np.append(np.flatnonzero(ordered[1:] != ordered[:-1]), len(ordered) - 1)
    true_positives = np.cumsum(hits)[last]
    precision = true_positives / (last + 1)
    recall = true_positives / n_positive
    return float(np.sum(np.diff(np.append(0.0, recall)) * precision))


def expected_cumulative_errors(scores: np.ndarray, errors: np.ndarray) -> np.ndarray:
    size = len(scores)
    if size == 0:
        return np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    ordered, wrong = scores[order], errors[order].astype(np.float64)
    starts = np.flatnonzero(np.append(True, ordered[1:] != ordered[:-1]))
    sizes = np.diff(np.append(starts, size))
    group_errors = np.add.reduceat(wrong, starts)
    before = np.append(0.0, np.cumsum(group_errors)[:-1])
    group = np.repeat(np.arange(len(starts)), sizes)
    position = np.arange(size) - starts[group] + 1
    return before[group] + position * group_errors[group] / sizes[group]


def coverage_count(coverage: float, size: int) -> int:
    return max(1, min(size, math.ceil(coverage * size - 1e-9)))


def signal_metrics(scores: np.ndarray, correct: np.ndarray, coverages: tuple[float, ...]) -> Record:
    size = len(scores)
    if size == 0:
        return {"queries": 0}
    cumulative = expected_cumulative_errors(scores, ~correct)
    counts = np.arange(1, size + 1)
    n_correct = int(correct.sum())
    oracle = np.maximum(0, counts - n_correct) / counts
    aurc = float((cumulative / counts).mean())
    metrics: Record = {
        "queries": size,
        "base_rate": n_correct / size,
        "roc_auc": roc_auc(scores, correct),
        "average_precision": average_precision(scores, correct),
        "aurc": aurc,
        "oracle_aurc": float(oracle.mean()),
        "e_aurc": aurc - float(oracle.mean()),
    }
    for coverage in coverages:
        count = coverage_count(coverage, size)
        metrics[f"precision_at_{coverage:g}"] = float(1 - cumulative[count - 1] / count)
    return metrics


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray, step: float) -> list[list[float]]:
    size = len(scores)
    if size == 0:
        return []
    cumulative = expected_cumulative_errors(scores, ~correct)
    grid = np.round(np.arange(step, 1 + 1e-9, step), 10)
    points = []
    for coverage in grid:
        count = coverage_count(float(coverage), size)
        points.append([float(coverage), round(float(cumulative[count - 1] / count), 6)])
    return points


def check_fast_metrics(scores: np.ndarray, correct: np.ndarray, where: str) -> None:
    if correct.all() or not correct.any():
        return
    pairs = (
        (roc_auc(scores, correct), roc_auc_score(correct, scores), "roc_auc"),
        (
            average_precision(scores, correct),
            average_precision_score(correct, scores),
            "average_precision",
        ),
    )
    for fast, reference, name in pairs:
        if abs(fast - float(reference)) > METRIC_CHECK_TOLERANCE:
            raise SystemExit(f"{where}: {name} {fast} differs from sklearn {reference}")


def policy_metrics(accepted: np.ndarray, correct: np.ndarray) -> Record:
    size = len(accepted)
    kept = int(accepted.sum())
    n_correct = int(correct.sum())
    kept_correct = int((accepted & correct).sum())
    return {
        "queries": size,
        "accepted": kept,
        "accepted_correct": kept_correct,
        "coverage": kept / size if size else math.nan,
        "precision": kept_correct / kept if kept else math.nan,
        "recall": kept_correct / n_correct if n_correct else math.nan,
    }
