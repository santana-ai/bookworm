"""Hearing-bootstrap replicates of the signal and policy metrics."""

import math
from typing import Any

import numpy as np

from experiments.verifier.confidence.config import (
    DIFFERENCE_METRICS,
    POLICY_METRICS,
    SUBSETS,
    ConfidenceConfig,
)
from experiments.verifier.confidence.metrics import policy_metrics, signal_metrics
from experiments.verifier.confidence.policies import Arrays, Policy, accepted_by
from experiments.verifier.confidence.records import rounded

Record = dict[str, Any]


def arrays_of(rows: list[Record], names: list[str]) -> Arrays:
    return Arrays(
        query_ids=[row["query_id"] for row in rows],
        hearing_ids=np.array([row["hearing_id"] for row in rows], dtype=np.int64),
        correct=np.array([row["correct"] for row in rows], dtype=bool),
        nontrivial=np.array([row["n_relevant"] < row["n_units"] for row in rows], dtype=bool),
        signals={
            name: np.array(
                [math.nan if row.get(name) is None else row[name] for row in rows],
                dtype=np.float64,
            )
            for name in names
        },
    )


def take(arrays: Arrays, index: np.ndarray) -> Arrays:
    return Arrays(
        query_ids=[arrays.query_ids[i] for i in index],
        hearing_ids=arrays.hearing_ids[index],
        correct=arrays.correct[index],
        nontrivial=arrays.nontrivial[index],
        signals={name: values[index] for name, values in arrays.signals.items()},
    )


def hearing_members(hearing_ids: np.ndarray, universe: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(hearing_ids == hearing) for hearing in universe]


def replicate_index(members: list[np.ndarray], draw: np.ndarray) -> np.ndarray:
    parts = [members[position] for position in draw]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def finite_interval(values: list[float], level: float) -> Record:
    array = np.array(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"low": None, "high": None, "replicates": 0}
    tail = (1 - level) / 2
    return {
        "low": rounded(float(np.quantile(finite, tail))),
        "high": rounded(float(np.quantile(finite, 1 - tail))),
        "replicates": int(len(finite)),
    }


def excludes_zero(interval: Record) -> bool | None:
    if interval["low"] is None or interval["high"] is None:
        return None
    return bool(interval["low"] > 0 or interval["high"] < 0)


def subset_index(arrays: Arrays, index: np.ndarray, subset: str) -> np.ndarray:
    if subset == "nontrivial":
        return index[arrays.nontrivial[index]]
    return index


class Replicates:
    def __init__(self) -> None:
        self.values: dict[tuple[str, ...], list[float]] = {}

    def add(self, key: tuple[str, ...], value: float) -> None:
        self.values.setdefault(key, []).append(value)

    def interval(self, key: tuple[str, ...], level: float) -> Record | None:
        if key not in self.values:
            return None
        return finite_interval(self.values[key], level)


def record_signal_replicates(
    store: Replicates,
    mode: str,
    signals: dict[str, np.ndarray],
    names: list[str],
    arrays: Arrays,
    index: np.ndarray,
    config: ConfidenceConfig,
) -> None:
    for subset in SUBSETS:
        chosen = subset_index(arrays, index, subset)
        if len(chosen) == 0:
            continue
        correct = arrays.correct[chosen]
        reference = signal_metrics(signals[config.reference_signal][chosen], correct, ())
        for name in names:
            values = signals[name][chosen]
            if not np.isfinite(values).all():
                continue
            metrics = signal_metrics(values, correct, config.coverages)
            for metric, value in metrics.items():
                if metric != "queries":
                    store.add((mode, subset, name, metric), value)
            if name == config.reference_signal:
                continue
            for metric in DIFFERENCE_METRICS:
                store.add(
                    (mode, subset, name, f"difference_{metric}"),
                    metrics[metric] - reference[metric],
                )


def record_policy_replicates(
    store: Replicates,
    mode: str,
    policies: list[Policy],
    signals: dict[str, np.ndarray],
    correct: np.ndarray,
    index: np.ndarray,
) -> None:
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal][index])
        if accepted is None:
            continue
        metrics = policy_metrics(accepted, correct[index])
        for metric in POLICY_METRICS:
            store.add((mode, policy.name, metric), metrics[metric])


def policy_threshold_record(policy: Policy) -> float | str | None:
    if policy.threshold is None:
        return None
    if math.isinf(policy.threshold):
        return "accept_all"
    return rounded(policy.threshold, 8)
