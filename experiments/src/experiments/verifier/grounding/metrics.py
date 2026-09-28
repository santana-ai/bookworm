"""Confidence metrics of each system and the paired comparisons against the reference."""

from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file

from experiments.common.reporting import rounded
from experiments.common.stats import holm
from experiments.udv.calibrate_threshold import interval
from experiments.verifier.confidence.metrics import signal_metrics
from experiments.verifier.grounding.config import (
    COMPARED_METRICS,
    COVERAGES,
    PRIMARY,
    REFERENCE,
    REPORTED_METRICS,
    Config,
)
from experiments.verifier.grounding.scorers import (
    normalized_ranks,
)
from experiments.verifier.stats import bootstrap_p_value, finite_interval, hearing_draws

Record = dict[str, Any]


def filled(values: list[float | None], probability: bool) -> np.ndarray:
    present = [value for value in values if value is not None]
    floor = 0.0 if probability else (min(present) if present else 0.0)
    return np.array([floor if value is None else value for value in values], dtype=np.float64)


def new_scores(
    config: Config, key: str, split: str, ids: list[str], pool: str
) -> tuple[np.ndarray, Record]:
    path = config.scores_dir / f"{key}_{split}.jsonl"
    rows = {row["id"]: row for row in load_jsonl(path)}
    if set(rows) != set(ids):
        raise SystemExit(f"{path}: ids differ from the evaluated set")
    values = [rows[unit]["scores"][pool] for unit in ids]
    spec = config.candidates[key]
    source = {"path": str(path), "sha256": sha256_of_file(path), "pool": pool}
    return filled(values, spec.probability), source


def with_combinations(systems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cosine = normalized_ranks(systems[REFERENCE])
    primary = normalized_ranks(systems[PRIMARY])
    return {
        **systems,
        "rank_mean": (cosine + primary) / 2,
        "rank_max": np.maximum(cosine, primary),
    }


def point_metrics(scores: np.ndarray, positive: np.ndarray) -> Record:
    metrics = signal_metrics(scores, positive, COVERAGES)
    return {name: metrics[name] for name in REPORTED_METRICS}


def replicate_metrics(
    systems: dict[str, np.ndarray], positive: np.ndarray, draws: list[np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in REPORTED_METRICS} for name in systems
    }
    for rows in draws:
        labels = positive[rows]
        single = labels.all() or not labels.any()
        for name, scores in systems.items():
            metrics = None if single else point_metrics(scores[rows], labels)
            for metric in REPORTED_METRICS:
                values[name][metric].append(np.nan if metrics is None else metrics[metric])
    return {
        name: {metric: np.array(series) for metric, series in by_metric.items()}
        for name, by_metric in values.items()
    }


def system_table(
    systems: dict[str, np.ndarray],
    positive: np.ndarray,
    replicates: dict[str, dict[str, np.ndarray]],
    level: float,
) -> Record:
    table: Record = {}
    for name, scores in systems.items():
        metrics = point_metrics(scores, positive)
        table[name] = {
            metric: {
                "point": rounded(metrics[metric]),
                "interval": finite_interval(replicates[name][metric], level),
            }
            for metric in REPORTED_METRICS
        }
        table[name]["ties"] = int(len(scores) - len(np.unique(scores)))
    return table


def paired_comparisons(
    systems: dict[str, np.ndarray],
    positive: np.ndarray,
    replicates: dict[str, dict[str, np.ndarray]],
    pairs: list[tuple[str, str]],
    level: float,
) -> list[Record]:
    entries: list[Record] = []
    for name, reference in pairs:
        metrics = point_metrics(systems[name], positive)
        base = point_metrics(systems[reference], positive)
        for metric in COMPARED_METRICS:
            deltas = replicates[name][metric] - replicates[reference][metric]
            valid = deltas[np.isfinite(deltas)]
            entries.append(
                {
                    "candidate": name,
                    "reference": reference,
                    "metric": metric,
                    "delta": rounded(metrics[metric] - base[metric]),
                    "interval": interval(valid.tolist(), level),
                    "p_value": bootstrap_p_value(valid),
                }
            )
    if not entries:
        return entries
    adjusted = holm([float(entry["p_value"]) for entry in entries])
    for entry, value in zip(entries, adjusted, strict=True):
        entry["p_holm"] = rounded(value)
        entry["p_value"] = rounded(entry["p_value"])
    return entries


def evaluate_set(
    systems: dict[str, np.ndarray],
    positive: np.ndarray,
    hearings: np.ndarray,
    samples: int,
    seed: int,
    level: float,
) -> Record:
    draws = hearing_draws(hearings, samples, np.random.default_rng(seed))
    replicates = replicate_metrics(systems, positive, draws)
    return {
        "items": int(len(positive)),
        "hearings": int(len(set(hearings.tolist()))),
        "positives": int(positive.sum()),
        "negatives": int((~positive).sum()),
        "systems": system_table(systems, positive, replicates, level),
        "bootstrap": f"{samples} hearing resamples, seed {seed}, percentile interval; replicates "
        "with a single class are skipped",
        "_replicates": replicates,
    }
