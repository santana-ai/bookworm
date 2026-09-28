"""Derived systems, robustness checks and the declared comparison families."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl

from experiments.udv.calibrate_threshold import rounded
from experiments.verifier.decision.battery import (
    CONSENSUS,
    STACKED,
)
from experiments.verifier.decision.panel import (
    component_agreement,
    consensus_votes,
    fit_stacked,
    order_changes,
)
from experiments.verifier.nli.config import (
    COMPARED_RULE,
    DECISION_KINDS,
    REFERENCE_JUDGE,
    RunDeclaration,
    VerifierConfig,
)
from experiments.verifier.nli.decision import require_battery
from experiments.verifier.nli.evaluation import SplitData, judge_point, paired_delta, point_value
from experiments.verifier.nli.metrics import fit_rule
from experiments.verifier.nli.scoring import score_file
from experiments.verifier.stats import (
    MIN_HEARINGS_FOR_P_VALUE,
    apply_holm,
    bootstrap_p_value,
)

Record = dict[str, Any]

SUMMARY_OPTIONAL = ("language", "checks", "translation", "questions", "order_pairs_used")


def scorer_summaries(reports: Record) -> Record:
    return {
        key: {
            "model": report["model"],
            "label_probes": report["label_probes"],
            "counts": report["counts"],
            "truncation": report["truncation"],
            "timing": report["timing"],
            "premise": report["premise"],
            "files": report["files"],
            "created_at": report["created_at"],
            "code": report["code"],
            **{name: report[name] for name in SUMMARY_OPTIONAL if name in report},
        }
        for key, report in reports.items()
    }


def missing_detail(
    keys: list[str], run_dir: Path, imported: dict[str, str], splits: tuple[str, ...]
) -> Record:
    detail: Record = {}
    for key in keys:
        directory = run_dir.parent / imported[key] if key in imported else run_dir
        detail[key] = {
            "directory": str(directory),
            "splits_without_score_file": [
                split for split in splits if not score_file(directory, key, split).exists()
            ],
        }
    return detail


def run_declaration(config: VerifierConfig, args: argparse.Namespace) -> RunDeclaration | None:
    name = args.declaration if args.declaration is not None else args.run_name
    declaration = config.declarations.get(name)
    if args.declaration is not None and declaration is None:
        raise SystemExit(f"no [declarations.{name}] table in the config")
    return declaration


def add_derived_systems(
    fit: SplitData,
    evaluations: list[SplitData],
    systems_by_scorer: dict[str, list[str]],
    config: VerifierConfig,
) -> tuple[list[str], Record, dict[str, dict[str, float]]]:
    names: list[str] = []
    details: Record = {}
    thresholds_by_scorer: dict[str, dict[str, float]] = {}
    for key in systems_by_scorer:
        spec = config.scorers[key]
        if spec.kind not in DECISION_KINDS:
            continue
        battery = require_battery(config)
        entry: Record = {}
        thresholds: dict[str, float] = {}
        for component in battery.panel_components:
            system = f"{key}.max.{component}"
            optimum = fit_rule(battery.consensus_rule, fit.scores[system], fit.labels, config, True)
            if optimum is not None:
                thresholds[system] = float(optimum["threshold"])
        if len(thresholds) == len(battery.panel_components):
            for data in (fit, *evaluations):
                data.scores[f"{key}.{CONSENSUS}"] = consensus_votes(data.scores, thresholds)
            names.append(f"{key}.{CONSENSUS}")
            thresholds_by_scorer[key] = thresholds
            entry["consensus"] = {
                "components": list(thresholds),
                "rule": battery.consensus_rule,
                "fitted_on": fit.split,
                "thresholds": {system: rounded(t) for system, t in thresholds.items()},
            }
        else:
            entry["consensus"] = {
                "applicable": False,
                "reason": "a component has no fitted threshold",
            }
        features = [f"{key}.max.{q}" for q in battery.stacked_features if q in spec.questions]
        if fit.labels.all() or not fit.labels.any():
            entry["stacked"] = {"applicable": False, "reason": "the fit split has one label only"}
        else:
            model = fit_stacked(
                fit.scores, fit.labels, features, battery.stacked_c, battery.stacked_seed
            )
            for data in (fit, *evaluations):
                data.scores[f"{key}.{STACKED}"] = model.predict(data.scores)
            names.append(f"{key}.{STACKED}")
            entry["stacked"] = {
                "fitted_on": fit.split,
                "opinions": len(fit.ids),
                **model.describe(),
            }
        details[key] = entry
    return names, details, thresholds_by_scorer


def robustness_report(
    evaluations: list[SplitData],
    systems_by_scorer: dict[str, list[str]],
    directories: dict[str, Path],
    config: VerifierConfig,
    thresholds_by_scorer: dict[str, dict[str, float]],
) -> Record:
    result: Record = {}
    for key in systems_by_scorer:
        spec = config.scorers[key]
        if spec.kind not in DECISION_KINDS:
            continue
        battery = require_battery(config)
        thresholds = thresholds_by_scorer.get(key, {})
        entry: Record = {}
        for data in evaluations:
            wanted = set(data.ids)
            rows = [
                row
                for row in load_jsonl(score_file(directories[key], key, data.split))
                if row["id"] in wanted
            ]
            predictions = {
                system: data.scores[system] >= threshold for system, threshold in thresholds.items()
            }
            entry[data.split] = {
                "order_changes": order_changes(battery, spec.questions, rows),
                "panel_agreement": component_agreement(predictions) if predictions else None,
            }
        result[key] = entry
    return result


def delta_entry(values: np.ndarray, reference: np.ndarray, point: float, level: float) -> Record:
    return {
        **paired_delta(values, reference, point, level),
        "p_value": bootstrap_p_value(values - reference),
    }


def system_delta(
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    system: str,
    reference: str,
    split: str,
    key: tuple[str, str],
    level: float,
) -> Record | None:
    if key not in replicates[system][split] or key not in replicates[reference][split]:
        return None
    point = point_value(results, system, split, key) - point_value(results, reference, split, key)
    return delta_entry(
        replicates[system][split][key], replicates[reference][split][key], point, level
    )


def comparison_entry(
    comparison: Record,
    family: str,
    split: str,
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    judges: Record,
    judge_replicates: dict[str, dict[str, dict[str, np.ndarray]]],
    level: float,
    hearings: int,
) -> Record:
    judge = judges["reference_judge"]["key"]
    system, reference = comparison["system"], comparison["reference"]
    entry: Record = {
        "name": comparison["name"],
        "family": family,
        **{key: comparison[key] for key in ("system", "reference", "metric")},
    }
    needed = [system] + ([] if reference == REFERENCE_JUDGE else [reference])
    missing = [name for name in needed if name not in results]
    if missing:
        return {**entry, "missing": missing, "p_value": None}
    if comparison["metric"] == "roc_auc":
        primary = system_delta(
            results, replicates, system, reference, split, ("ranking", "roc_auc"), level
        )
        entry |= primary or {"p_value": None}
        entry["secondary_uncorrected"] = {
            "average_precision_not_inferable": system_delta(
                results,
                replicates,
                system,
                reference,
                split,
                ("ranking", "average_precision_not_inferable"),
                level,
            ),
            "cohen_kappa": system_delta(
                results, replicates, system, reference, split, (COMPARED_RULE, "cohen_kappa"), level
            ),
        }
    else:
        key = (COMPARED_RULE, "cohen_kappa")
        if key not in replicates[system][split]:
            return {**entry, "missing": [f"{system} {COMPARED_RULE}"], "p_value": None}
        point = point_value(results, system, split, key) - judge_point(
            judges, split, judge, "cohen_kappa"
        )
        entry["reference_judge"] = judge
        entry |= delta_entry(
            replicates[system][split][key],
            judge_replicates[split][judge]["cohen_kappa"],
            point,
            level,
        )
    if hearings < MIN_HEARINGS_FOR_P_VALUE:
        entry["p_value"] = None
        entry["p_value_reason"] = (
            f"the {split} split has {hearings} hearing; a hearing bootstrap needs at least "
            f"{MIN_HEARINGS_FOR_P_VALUE}"
        )
    return entry


def declared_comparisons(
    declaration: RunDeclaration,
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    judges: Record,
    judge_replicates: dict[str, dict[str, dict[str, np.ndarray]]],
    splits: list[str],
    config: VerifierConfig,
    hearings: dict[str, int],
) -> Record:
    level = config.confidence_level
    by_name = {comparison["name"]: comparison for comparison in declaration.comparisons}
    output: Record = {}
    for split in splits:
        families: Record = {}
        ordered: list[Record] = []
        for family in declaration.families:
            entries = [
                comparison_entry(
                    by_name[name],
                    family.name,
                    split,
                    results,
                    replicates,
                    judges,
                    judge_replicates,
                    level,
                    hearings[split],
                )
                for name in family.comparisons
            ]
            apply_holm(entries)
            missing = [entry["name"] for entry in entries if entry.get("missing")]
            families[family.name] = {
                "question": family.question,
                "comparisons": list(family.comparisons),
                "family_size": len(entries),
                "missing": missing,
                "complete": not missing,
            }
            ordered += entries
        superseded = "families" in declaration.source
        output[split] = {
            "rule": declaration.source["comparison_rule"],
            "rule_superseded_by": "families_amendment" if superseded else None,
            "holm_scope": "each_family" if superseded else "whole_list",
            "families_rule": declaration.source.get("families_amendment"),
            "family_order": [family.name for family in declaration.families],
            "families": families,
            "comparisons": ordered,
        }
    return output
