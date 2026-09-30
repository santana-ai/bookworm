"""The confirm command: evaluate the selected candidate once on validation."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import sha256_of_file
from sklearn.metrics import roc_auc_score

from experiments.common.reporting import file_record, rounded, utc_timestamp
from experiments.common.stats import holm
from experiments.verifier.exploration.candidates import (
    fixed_candidates,
    learned_candidates,
    resolve_reference,
)
from experiments.verifier.exploration.config import Candidate, ExplorationConfig, ScorerData
from experiments.verifier.exploration.cross_validation import feature_matrix, fit_learned
from experiments.verifier.exploration.provenance import code_hashes
from experiments.verifier.exploration.scores import (
    candidate_scores,
    load_all,
    output_dir,
    write_csv,
)
from experiments.verifier.nli.metrics import binary_metrics, max_f1_not_inferable_optimum
from experiments.verifier.nli.provenance import environment
from experiments.verifier.stats import bootstrap_p_value, finite_interval, hearing_draws

Record = dict[str, Any]


def selected_candidate(
    selection: Record, config: ExplorationConfig, data: dict[str, ScorerData]
) -> Candidate:
    key = selection["selected"]["candidate"]
    for candidate in [*fixed_candidates(config, data), *learned_candidates(config, data)]:
        if candidate.key == key:
            return candidate
    raise SystemExit(f"selected candidate {key} cannot be rebuilt")


def confirm_candidates(
    selection: Record, config: ExplorationConfig, data: dict[str, ScorerData], out: Path
) -> list[Candidate]:
    floor = config.raw["confirmation"].get("cv_floor")
    keys = [selection["selected"]["candidate"]]
    if floor is not None:
        with open(out / "cv_results.csv") as f:
            keys += [
                row["candidate"]
                for row in csv.DictReader(f)
                if float(row["cv_roc_auc_mean"]) >= float(floor) and row["candidate"] not in keys
            ]
    built = {c.key: c for c in [*fixed_candidates(config, data), *learned_candidates(config, data)]}
    missing = [key for key in keys if key not in built]
    if missing:
        raise SystemExit(f"candidates cannot be rebuilt: {missing}")
    return [built[key] for key in keys]


def system_scores(
    candidate: Candidate,
    fit_data: dict[str, ScorerData],
    data: dict[str, ScorerData],
    fit_labels: np.ndarray,
    fit_hearings: np.ndarray,
    config: ExplorationConfig,
) -> tuple[np.ndarray, np.ndarray, Record]:
    if candidate.kind != "learned":
        return candidate_scores(candidate, fit_data), candidate_scores(candidate, data), {}
    fit_matrix = feature_matrix(candidate, fit_data)
    scaler, model, c, means = fit_learned(fit_matrix, fit_labels, fit_hearings, config)
    learned = {
        "c": c,
        "inner_means": means,
        "coefficients": dict(zip(candidate.features, map(rounded, model.coef_[0]), strict=True)),
    }
    return (
        model.predict_proba(scaler.transform(fit_matrix))[:, 1],
        model.predict_proba(scaler.transform(feature_matrix(candidate, data)))[:, 1],
        learned,
    )


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, draws: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [
            float(roc_auc_score(labels[rows], scores[rows]))
            if labels[rows].any() and not labels[rows].all()
            else float("nan")
            for rows in draws
        ]
    )


def evaluate_split(
    systems: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_labels: np.ndarray,
    labels: np.ndarray,
    hearings: np.ndarray,
    config: ExplorationConfig,
) -> tuple[Record, dict[str, dict[str, np.ndarray]]]:
    draws = hearing_draws(
        hearings, config.bootstrap_samples, np.random.default_rng(config.bootstrap_seed)
    )
    level = float(config.verifier["evaluation"]["confidence_level"])
    results: Record = {}
    replicates: dict[str, dict[str, np.ndarray]] = {}
    for name, (fit_scores, scores) in systems.items():
        optimum = max_f1_not_inferable_optimum(fit_scores, fit_labels)
        threshold = None if optimum is None else optimum["threshold"]
        predictions = scores >= threshold if threshold is not None else np.ones(len(scores), bool)
        auc = bootstrap_auc(labels, scores, draws)
        kappa = np.array(
            [binary_metrics(labels[rows], predictions[rows])["cohen_kappa"] for rows in draws]
        )
        replicates[name] = {"roc_auc": auc, "cohen_kappa": kappa}
        metrics = binary_metrics(labels, predictions)
        results[name] = {
            "roc_auc": rounded(float(roc_auc_score(labels, scores))),
            "roc_auc_interval": finite_interval(auc, level),
            "threshold": threshold,
            "cohen_kappa": rounded(metrics["cohen_kappa"]),
            "cohen_kappa_interval": finite_interval(kappa, level),
            "f1_not_inferable": rounded(metrics["f1_not_inferable"]),
            "precision_not_inferable": rounded(metrics["precision_not_inferable"]),
            "recall_not_inferable": rounded(metrics["recall_not_inferable"]),
        }
    return results, replicates


def compare_systems(
    keys: list[str],
    references: tuple[str, ...],
    results: Record,
    replicates: dict[str, dict[str, np.ndarray]],
    config: ExplorationConfig,
) -> list[Record]:
    level = float(config.verifier["evaluation"]["confidence_level"])
    comparisons: list[Record] = []
    for key in keys:
        for reference in references:
            for metric in ("roc_auc", "cohen_kappa"):
                deltas = replicates[key][metric] - replicates[reference][metric]
                comparisons.append(
                    {
                        "candidate": key,
                        "reference": reference,
                        "metric": metric,
                        "delta": rounded(
                            float(results[key][metric]) - float(results[reference][metric])
                        ),
                        "bootstrap": finite_interval(deltas, level),
                        "p_value": bootstrap_p_value(deltas),
                    }
                )
    auc_entries = [entry for entry in comparisons if entry["metric"] == "roc_auc"]
    adjusted = holm([float(entry["p_value"]) for entry in auc_entries])
    for entry, value in zip(auc_entries, adjusted, strict=True):
        entry["p_holm"] = value
    return comparisons


Systems = dict[str, tuple[np.ndarray, np.ndarray]]


def read_cv_results(out: Path) -> dict[str, Record]:
    with open(out / "cv_results.csv") as f:
        return {row["candidate"]: row for row in csv.DictReader(f)}


def confirm_systems(
    candidates: list[Candidate],
    fit_data: dict[str, ScorerData],
    data: dict[str, ScorerData],
    fit_labels: np.ndarray,
    fit_hearings: np.ndarray,
    config: ExplorationConfig,
) -> tuple[Systems, Record]:
    """Fit-split and confirm-split scores of every candidate and reference."""
    systems: Systems = {}
    learned: Record = {}
    for candidate in candidates:
        fit_scores, scores, model = system_scores(
            candidate, fit_data, data, fit_labels, fit_hearings, config
        )
        systems[candidate.key] = (fit_scores, scores)
        if model:
            learned[candidate.key] = model
    for reference in config.references:
        systems[reference] = (
            resolve_reference(reference, config, fit_data),
            resolve_reference(reference, config, data),
        )
    return systems, learned


def system_role(name: str, selected: str, references: tuple[str, ...]) -> str:
    if name == selected:
        return "selected"
    if name in references:
        return "reference"
    return "above_cv_floor"


def annotate_results(
    results: Record,
    selected: str,
    cv: dict[str, Record],
    fit_split: str,
    config: ExplorationConfig,
) -> None:
    for name, result in results.items():
        result["role"] = system_role(name, selected, config.references)
        result["cv_roc_auc_mean"] = float(cv[name]["cv_roc_auc_mean"]) if name in cv else None
        result["threshold_fitted_on"] = fit_split


def confirmation_table(
    results: Record, comparisons: list[Record], references: tuple[str, ...]
) -> list[Record]:
    """One row per system, sorted by validation ROC AUC, with its Holm-adjusted AUC deltas."""
    by_candidate: dict[str, Record] = {}
    for entry in comparisons:
        by_candidate.setdefault(entry["candidate"], {})[
            f"{entry['metric']}|{entry['reference']}"
        ] = entry
    table: list[Record] = []
    for name, result in results.items():
        table_row: Record = {
            "system": name,
            "role": result["role"],
            "cv_roc_auc_mean": result["cv_roc_auc_mean"],
            "roc_auc": result["roc_auc"],
            "roc_auc_low": result["roc_auc_interval"]["low"],
            "roc_auc_high": result["roc_auc_interval"]["high"],
            "cohen_kappa": result["cohen_kappa"],
            "cohen_kappa_low": result["cohen_kappa_interval"]["low"],
            "cohen_kappa_high": result["cohen_kappa_interval"]["high"],
        }
        for reference in references:
            found = by_candidate.get(name, {}).get(f"roc_auc|{reference}")
            table_row[f"delta_auc_vs_{reference}"] = None if found is None else found["delta"]
            table_row[f"p_holm_vs_{reference}"] = None if found is None else found["p_holm"]
        table.append(table_row)
    table.sort(key=lambda row: -row["roc_auc"])
    return table


def command_confirm(args: argparse.Namespace, config: ExplorationConfig) -> None:
    out = output_dir(config, args.output_dir)
    target = out / "confirmation.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the confirmation runs once")
    with open(out / "selection.json") as f:
        selection = json.load(f)
    fit_split, confirm_split = config.raw["fit_split"], config.raw["confirm_split"]
    _, fit_labels, fit_hearings, fit_data, _ = load_all(config, fit_split)
    ids, labels, hearings, data, missing = load_all(config, confirm_split)
    candidates = confirm_candidates(selection, config, fit_data, out)
    cv = read_cv_results(out)
    systems, learned = confirm_systems(candidates, fit_data, data, fit_labels, fit_hearings, config)
    results, replicates = evaluate_split(systems, fit_labels, labels, hearings, config)
    annotate_results(results, candidates[0].key, cv, fit_split, config)
    comparisons = compare_systems(
        [c.key for c in candidates], config.references, results, replicates, config
    )
    report = {
        "experiment": "nli_verifier_exploration",
        "name": config.name,
        "created_at": utc_timestamp(),
        "splits_read": [fit_split, confirm_split],
        "selection": {
            "path": str(out / "selection.json"),
            "sha256": sha256_of_file(out / "selection.json"),
        },
        "scorers_missing": missing,
        "opinions": {
            "count": len(ids),
            "hearings": int(len(set(hearings.tolist()))),
            "not_inferable": int((~labels).sum()),
        },
        "candidates_evaluated": len(candidates),
        "scope_rule": config.raw["confirmation"].get("scope_amendment"),
        "learned": learned,
        "results": results,
        "comparisons": comparisons,
        "comparison_rule": config.raw["confirmation"]["comparison_rule"],
        "label_semantics": config.verifier["benchmark"]["label_semantics"],
        "config": file_record(config.path),
        "code": code_hashes(),
        "environment": environment(),
    }
    with open(target, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_csv(out / "confirmation.csv", confirmation_table(results, comparisons, config.references))
    print(f"{len(candidates)} candidates evaluated on {confirm_split}", flush=True)
