"""Calibration of the production rule and its checks outside the calibration hearings."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file

from experiments.common.stats import stream_rng
from experiments.common.udv_run import load_config as load_udv_config
from experiments.verifier.confidence.bootstrap import (
    Replicates,
    arrays_of,
    hearing_members,
    record_policy_replicates,
    replicate_index,
)
from experiments.verifier.confidence.config import POLICY_METRICS, ConfidenceConfig, Target
from experiments.verifier.confidence.evaluation import dotted_value
from experiments.verifier.confidence.metrics import policy_metrics
from experiments.verifier.confidence.policies import Policy, accepted_by
from experiments.verifier.confidence.records import rounded

Record = dict[str, Any]


def calibration_hearings(spec: Record) -> list[int]:
    path = Path(spec["hearings_source"])
    if path.suffix == ".jsonl":
        values = [row[spec["hearings_field"]] for row in load_jsonl(path)]
    else:
        with open(path) as f:
            values = list(dotted_value(json.load(f), spec["hearings_field"]))
    if not values or not all(isinstance(value, int) for value in values):
        raise SystemExit(f"{path}: {spec['hearings_field']} gives no hearing ids")
    return sorted(set(values))


def calibration_record(key: str, value: float, spec: Record, udv_evidence: Record) -> Record:
    threshold_path = Path(spec["threshold_source"])
    with open(threshold_path) as f:
        recorded = float(dotted_value(json.load(f), spec["threshold_field"]))
    if abs(recorded - value) > 1e-9:
        raise SystemExit(
            f"production_rule.calibration.{key}: {threshold_path} {spec['threshold_field']} is "
            f"{recorded}, udv config {key} is {value}"
        )
    if key == "embedding_threshold" and udv_evidence.get("calibration_source") != str(
        spec["hearings_source"]
    ):
        raise SystemExit(
            f"production_rule.calibration.{key}.hearings_source differs from the udv config "
            f"calibration_source {udv_evidence.get('calibration_source')!r}"
        )
    hearings = calibration_hearings(spec)
    return {
        "hearings_source": spec["hearings_source"],
        "hearings_source_sha256": sha256_of_file(Path(spec["hearings_source"])),
        "hearings_field": spec["hearings_field"],
        "threshold_source": spec["threshold_source"],
        "threshold_field": spec["threshold_field"],
        "threshold_recorded_there": recorded,
        "description": spec.get("description"),
        "role": spec.get("role"),
        "calibration_hearing_ids": hearings,
    }


def production_policies(config: ConfidenceConfig, target: Target) -> tuple[list[Policy], Record]:
    if target.name != config.production_target:
        return [], {}
    udv = load_udv_config(config.udv_config_path)
    evidence = udv.source["evidence"]
    thresholds = {key: float(evidence[key]) for key in config.threshold_keys}
    calibration = {
        key: calibration_record(key, value, config.threshold_calibration[key], evidence)
        for key, value in thresholds.items()
    }
    policies = [
        Policy(
            f"production:{key}",
            config.production_signal,
            "production",
            value,
            {
                "udv_config_key": key,
                "threshold": value,
                "calibration_source": calibration[key]["hearings_source"],
                "calibration_role": calibration[key]["role"],
                "calibration_hearings": len(calibration[key]["calibration_hearing_ids"]),
            },
        )
        for key, value in thresholds.items()
    ]
    source = {
        "udv_config": str(config.udv_config_path),
        "sha256": sha256_of_file(config.udv_config_path),
        "thresholds": thresholds,
        "calibration": calibration,
    }
    return policies, source


def subset_draws(
    rows: list[Record], label: str, config: ConfidenceConfig
) -> tuple[np.ndarray, np.ndarray]:
    hearings = np.array(sorted({row["hearing_id"] for row in rows}), dtype=np.int64)
    rng = stream_rng(config.seed, label)
    matrix = (
        rng.integers(0, len(hearings), size=(config.bootstrap_samples, len(hearings)))
        if len(hearings)
        else np.zeros((config.bootstrap_samples, 0), dtype=np.int64)
    )
    return hearings, matrix


def calibration_overlap(
    calibration: Record, key: str, rows: list[Record], population: list[Record]
) -> Record:
    hearing_ids = set(calibration[key]["calibration_hearing_ids"])
    split_hearings = sorted({row["hearing_id"] for row in rows})
    overlap = [hearing for hearing in split_hearings if hearing in hearing_ids]
    return {
        "calibration_hearings_in_split": overlap,
        "split_hearings": len(split_hearings),
        "queries_in_those_hearings": sum(1 for row in rows if row["hearing_id"] in hearing_ids),
        "multi_candidate_queries_in_those_hearings": sum(
            1 for row in population if row["hearing_id"] in hearing_ids
        ),
        "in_sample": bool(overlap),
    }


def production_outside_calibration(
    production: list[Policy],
    calibration: Record,
    rows: list[Record],
    population: list[Record],
    label: str,
    config: ConfidenceConfig,
) -> Record:
    excluded = set().union(
        *(set(entry["calibration_hearing_ids"]) for entry in calibration.values())
    )
    kept_rows = [row for row in rows if row["hearing_id"] not in excluded]
    kept_population = [row for row in population if row["hearing_id"] not in excluded]
    return {
        "rule": (
            "the queries of this split whose hearing is in no calibration set of any production "
            "threshold, so every cut is evaluated out of sample on the same queries; intervals "
            "resample these hearings only"
        ),
        "hearings_excluded": sorted(excluded & {row["hearing_id"] for row in rows}),
        "hearings_kept": len({row["hearing_id"] for row in kept_rows}),
        "multi_candidate": {
            "queries": len(kept_population),
            "policies": production_on_all(
                production,
                kept_population,
                subset_draws(kept_population, f"{label}|outside_calibration|multi", config),
                config,
            ),
        },
        "all_queries": {
            "queries": len(kept_rows),
            "policies": production_on_all(
                production,
                kept_rows,
                subset_draws(kept_rows, f"{label}|outside_calibration|all", config),
                config,
            ),
        },
    }


def universes(
    rows_by_split: dict[str, list[Record]],
    splits: tuple[str, ...],
    bench: str,
    seed: int,
    samples: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    draws = {}
    for split in splits:
        hearings = np.array(sorted({row["hearing_id"] for row in rows_by_split[split]}))
        rng = stream_rng(seed, f"{bench}|{split}|hearings")
        matrix = (
            rng.integers(0, len(hearings), size=(samples, len(hearings)))
            if len(hearings)
            else np.zeros((samples, 0), dtype=np.int64)
        )
        draws[split] = (hearings, matrix)
    return draws


def production_on_all(
    policies: list[Policy],
    rows: list[Record],
    draws: tuple[np.ndarray, np.ndarray],
    config: ConfidenceConfig,
) -> Record:
    arrays = arrays_of(rows, [config.production_signal])
    universe, matrix = draws
    members = hearing_members(arrays.hearing_ids, universe)
    store = Replicates()
    signals = arrays.signals
    for replicate in range(config.bootstrap_samples):
        index = replicate_index(members, matrix[replicate])
        record_policy_replicates(store, "fixed", policies, signals, arrays.correct, index)
    report = {}
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal])
        if accepted is None:
            continue
        point = policy_metrics(accepted, arrays.correct)
        report[policy.name] = {
            "threshold": policy.threshold,
            "evaluate": {k: rounded(v) if isinstance(v, float) else v for k, v in point.items()},
            "fixed": {
                metric: store.interval(("fixed", policy.name, metric), config.confidence_level)
                for metric in POLICY_METRICS
            },
        }
    return report
