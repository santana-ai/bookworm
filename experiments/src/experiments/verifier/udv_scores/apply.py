"""The apply command: verifier probabilities and decisions for every UDV."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from bookworm import sha256_of_file, write_json, write_jsonl

from experiments.common.reporting import file_record, rounded, utc_timestamp
from experiments.verifier.exploration.candidates import check_reading, feature_candidate
from experiments.verifier.exploration.config import ExplorationConfig, ScorerData
from experiments.verifier.exploration.scores import candidate_scores, load_scorer
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.provenance import environment
from experiments.verifier.udv_scores.analysis import (
    analysis,
    benchmark_overlap,
    evidence_cosine_feature,
    pool_agreement,
    score_reports,
    supported_shares,
    udv_semantic_unit,
)
from experiments.verifier.udv_scores.config import (
    UDV_SPLIT,
    UdvThreshold,
    UdvVerifierConfig,
    candidate_scorers,
    primary_candidate,
)
from experiments.verifier.udv_scores.primary import fit_primary, load_fit_data, refit_check
from experiments.verifier.udv_scores.provenance import code_hashes
from experiments.verifier.udv_scores.scoring import open_run
from experiments.verifier.udv_scores.units import load_udvs, support_type

Record = dict[str, Any]

THRESHOLD_MATCH_TOLERANCE = 1e-3


def translate_record(config: UdvVerifierConfig) -> Record | None:
    path = config.run_dir / "translate_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        report = json.load(f)
    return {
        "path": str(path),
        "sha256": sha256_of_file(path),
        "created_at": report["created_at"],
        "runs": {key: value["run"] for key, value in report["models"].items()},
        "still_missing": {key: value["still_missing"] for key, value in report["models"].items()},
    }


def load_udv_threshold(path: Path, train_threshold: float) -> UdvThreshold:
    with open(path) as f:
        record = json.load(f)
    if not record["leak_check"]["passed"]:
        raise SystemExit(f"{path}: the leak check of the UDV-premise cut did not pass")
    if abs(float(record["train_threshold"]) - train_threshold) > THRESHOLD_MATCH_TOLERANCE:
        raise SystemExit(f"{path}: train_threshold differs from the refitted primary threshold")
    return UdvThreshold(
        value=float(record["udv_threshold_exact"]),
        path=path,
        sha256=sha256_of_file(path),
        rule=record["adopted_rule"],
        record=record,
    )


def row_provenance(
    config: UdvVerifierConfig, train_threshold: float, udv_threshold: UdvThreshold
) -> Record:
    return {
        "verifier": config.primary,
        "premise": "evidence.text",
        "hypothesis": "proposition",
        "train_threshold": rounded(train_threshold),
        "udv_threshold": rounded(udv_threshold.value),
        "udv_threshold_rule": udv_threshold.rule,
        "udv_threshold_source": str(udv_threshold.path),
        "udv_source": str(config.udv_path),
    }


def output_rows(
    records: list[Record],
    split_of: dict[int, str],
    scored: dict[str, tuple[float, bool, float]],
    udv_threshold: float | None = None,
    provenance: Record | None = None,
) -> list[Record]:
    rows = []
    for record in records:
        values = scored.get(record["id"])
        row = {
            "udv_id": record["id"],
            "hearing_id": record["hearing_id"],
            "split": split_of[record["hearing_id"]],
            "tier": record["tier"],
            "support_type": None if not record.get("evidence") else support_type(record),
            "scored": values is not None,
            "primary_probability": None if values is None else values[0],
            "supported_at_train_threshold": None if values is None else values[1],
            "p4_supports": None if values is None else values[2],
        }
        if udv_threshold is not None:
            row["supported_at_udv_threshold"] = (
                None if values is None else values[0] >= udv_threshold
            )
            row["provenance"] = provenance
        rows.append(row)
    return rows


def check_new_outputs(config: UdvVerifierConfig) -> None:
    for path in (config.output_path, config.report_path):
        if path.exists():
            raise SystemExit(f"{path} exists: apply writes new files only")


def read_json(path: Path) -> Record:
    with open(path) as f:
        return cast(Record, json.load(f))


def load_udv_data(
    config: UdvVerifierConfig, exploration_config: ExplorationConfig, ids: list[str]
) -> dict[str, ScorerData]:
    udv_data: dict[str, ScorerData] = {}
    for key in config.scorers:
        kind, _ = exploration_config.scorers[key]
        scorer = load_scorer(key, kind, config.name, UDV_SPLIT, ids, exploration_config)
        if scorer is None:
            raise SystemExit(f"{key}: no UDV score file in {config.run_dir}; run score first")
        udv_data[key] = scorer
    return udv_data


def counts_record(
    records: list[Record],
    units: list[PremiseUnit],
    unscored: Counter[str],
    scored_records: list[Record],
    splits: list[str],
) -> Record:
    return {
        "udv_records": len(records),
        "scored": len(units),
        "unscored": sum(unscored.values()),
        "unscored_by_tier": dict(sorted(unscored.items())),
        "scored_by_tier": dict(Counter(r["tier"] for r in scored_records)),
        "scored_by_support_type": dict(Counter(support_type(r) for r in scored_records)),
        "scored_by_hearing_split": dict(sorted(Counter(splits).items())),
    }


def inputs_record(
    config: UdvVerifierConfig, split_source: Record, udv_data: dict[str, ScorerData]
) -> Record:
    return {
        "udv": file_record(config.udv_path),
        "splits": split_source,
        "selection": file_record(config.selection_path),
        "final_test": file_record(config.final_test_path),
        "udv_score_files": {key: file_record(s.path) for key, s in udv_data.items()},
    }


def config_record(config: UdvVerifierConfig) -> Record:
    return {
        **file_record(config.path),
        "echo": config.raw,
        "verifier_config": file_record(config.verifier_config),
        "exploration_config": file_record(config.exploration_config),
    }


def udv_threshold_sections(
    udv_threshold: UdvThreshold,
    primary: np.ndarray,
    scored_records: list[Record],
    splits: list[str],
) -> Record:
    udv_decisions = primary >= udv_threshold.value
    tiers = [record["tier"] for record in scored_records]
    return {
        "udv_threshold": {
            "value": rounded(udv_threshold.value),
            "exact": udv_threshold.value,
            "rule": udv_threshold.rule,
            "interval": udv_threshold.record["rules"][udv_threshold.rule]["bootstrap"]["threshold"],
            "path": str(udv_threshold.path),
            "sha256": udv_threshold.sha256,
        },
        "supported_at_udv_threshold": {
            "by_tier": supported_shares(udv_decisions, tiers),
            "by_support_type": supported_shares(
                udv_decisions, [support_type(r) for r in scored_records]
            ),
            "by_hearing_split": supported_shares(udv_decisions, splits),
        },
    }


def command_apply(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    started = time.perf_counter()
    check_new_outputs(config)
    verifier, exploration_config = open_run(config)
    records, units, unscored, split_of, split_source = load_udvs(config, verifier)
    candidate = primary_candidate(exploration_config, config.primary)
    selection = read_json(config.selection_path)
    final_test = read_json(config.final_test_path)
    if final_test["primary"] != config.primary:
        raise SystemExit("scoring.primary differs from the final_test primary")
    labels, hearings, fit_data, fit_files = load_fit_data(
        exploration_config, candidate_scorers(candidate), selection
    )
    fitted = fit_primary(candidate, labels, hearings, fit_data, exploration_config)
    refit = refit_check(fitted, final_test, config.primary)
    fit_reading = check_reading(exploration_config, fit_data)
    ids = [unit.unit_id for unit in units]
    udv_data = load_udv_data(config, exploration_config, ids)
    reading = check_reading(exploration_config, udv_data)
    pools = pool_agreement(candidate, udv_data)
    primary = fitted.probabilities(udv_data)
    secondary = candidate_scores(feature_candidate(config.secondary), udv_data)
    decisions = primary >= fitted.threshold
    semantic_unit = udv_semantic_unit(config.udv_path)
    recomputed_cosine = candidate_scores(
        feature_candidate(evidence_cosine_feature(semantic_unit)), udv_data
    )
    scored = {
        unit_id: (float(p), bool(d), float(s))
        for unit_id, p, d, s in zip(ids, primary, decisions, secondary, strict=True)
    }
    udv_threshold = (
        None
        if config.udv_threshold_path is None
        else load_udv_threshold(config.udv_threshold_path, fitted.threshold)
    )
    write_jsonl(
        output_rows(
            records,
            split_of,
            scored,
            None if udv_threshold is None else udv_threshold.value,
            None
            if udv_threshold is None
            else row_provenance(config, fitted.threshold, udv_threshold),
        ),
        config.output_path,
    )
    by_id = {record["id"]: record for record in records}
    scored_records = [by_id[unit_id] for unit_id in ids]
    splits = [unit.split for unit in units]
    report = {
        "experiment": "udv_verifier",
        "name": config.name,
        "created_at": utc_timestamp(),
        "declared": config.raw["declared"],
        "purpose": config.raw["purpose"],
        "caveats": {
            "domain_shift": config.raw["report"]["domain_shift"],
            "label_semantics": config.raw["report"]["label_semantics"],
            "benchmark_label_semantics": verifier.source["benchmark"]["label_semantics"],
        },
        "counts": counts_record(records, units, unscored, scored_records, splits),
        "benchmark_overlap": benchmark_overlap(verifier.source, units),
        "primary": {
            "candidate": config.primary,
            "features": len(candidate.features),
            "scorers": list(candidate_scorers(candidate)),
            "fit": fitted.record,
            "final_test_result": final_test["results"][config.primary],
            "decision_rule": config.raw["scoring"]["decision_rule"],
        },
        "secondary": {
            "signal": config.secondary,
            "reason": config.raw["scoring"]["secondary_reason"],
        },
        "refit_check": refit,
        "fit_files": fit_files,
        "fit_reading_check": fit_reading,
        "udv_reading_check": reading,
        "pool_check": pools,
        **analysis(
            config,
            scored_records,
            primary,
            secondary,
            decisions,
            recomputed_cosine,
            splits,
            semantic_unit,
        ),
        "outputs": file_record(config.output_path, rows=len(records)),
        "inputs": inputs_record(config, split_source, udv_data),
        "score_runs": score_reports(config),
        "translation": translate_record(config),
        "timing": {"apply_seconds": round(time.perf_counter() - started, 1)},
        "config": config_record(config),
        "code": code_hashes(),
        "environment": environment(),
    }
    if udv_threshold is not None:
        report.update(udv_threshold_sections(udv_threshold, primary, scored_records, splits))
    write_json(report, config.report_path)
    summary = {
        "refit_check": refit["checks"],
        "supported_by_tier": report["supported_at_train_threshold"]["by_tier"],
    }
    print(json.dumps(summary, indent=2), flush=True)
