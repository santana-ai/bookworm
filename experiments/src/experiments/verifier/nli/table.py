"""Flat CSV table of the evaluation results."""

import csv
from pathlib import Path
from typing import Any

from experiments.verifier.nli.config import COMPARED_RULE, RunDeclaration, VerifierConfig

Record = dict[str, Any]

TABLE_COLUMNS = (
    "split",
    "system",
    "scorer",
    "score",
    "kind",
    "language",
    "role",
    "roc_auc",
    "roc_auc_low",
    "roc_auc_high",
    "average_precision_not_inferable",
    "threshold_max_f1_not_inferable",
    "cohen_kappa",
    "cohen_kappa_low",
    "cohen_kappa_high",
    "f1_not_inferable",
    "comparison",
    "family",
    "reference",
    "metric",
    "delta",
    "delta_low",
    "delta_high",
    "p_value",
    "p_holm",
    "missing",
)


def system_role(system: str, declaration: RunDeclaration | None, config: VerifierConfig) -> str:
    primary = declaration.primary_system if declaration else config.primary_system
    if system == primary:
        return "primary"
    compared = set()
    if declaration is not None:
        compared = {c["system"] for c in declaration.comparisons} | {
            c["reference"] for c in declaration.comparisons
        }
    return "compared" if system in compared else "secondary"


def interval_bounds(bootstrap: Record) -> tuple[Any, Any]:
    return bootstrap.get("low"), bootstrap.get("high")


def comparison_table(
    report: Record, config: VerifierConfig, declaration: RunDeclaration | None
) -> list[Record]:
    rows: list[Record] = []
    for split in report["splits"]["evaluate"]:
        for system, result in report["systems"].items():
            key, score = system.split(".", 1)
            spec = config.scorers[key]
            evaluation = result["evaluation"][split]
            rule = evaluation["rules"].get(COMPARED_RULE)
            auc_low, auc_high = interval_bounds(evaluation["ranking_bootstrap"]["roc_auc"])
            kappa_low, kappa_high = (
                interval_bounds(rule["bootstrap_fixed_threshold"]["cohen_kappa"])
                if rule
                else (None, None)
            )
            rows.append(
                {
                    "split": split,
                    "system": system,
                    "scorer": key,
                    "score": score,
                    "kind": spec.kind,
                    "language": spec.language,
                    "role": system_role(system, declaration, config),
                    "roc_auc": evaluation["ranking"]["roc_auc"],
                    "roc_auc_low": auc_low,
                    "roc_auc_high": auc_high,
                    "average_precision_not_inferable": evaluation["ranking"][
                        "average_precision_not_inferable"
                    ],
                    "threshold_max_f1_not_inferable": rule["threshold"] if rule else None,
                    "cohen_kappa": rule["metrics"]["cohen_kappa"] if rule else None,
                    "cohen_kappa_low": kappa_low,
                    "cohen_kappa_high": kappa_high,
                    "f1_not_inferable": (
                        rule["metrics"]["per_class"]["not_inferable"]["f1"] if rule else None
                    ),
                }
            )
        for judge, judged in report["judges"]["evaluation"][split].items():
            low, high = interval_bounds(judged["bootstrap"]["cohen_kappa"])
            rows.append(
                {
                    "split": split,
                    "system": f"judge.{judge}",
                    "scorer": "llm_judge",
                    "score": judge,
                    "kind": "llm_judge",
                    "language": "",
                    "role": (
                        "reference_judge"
                        if judge == report["judges"]["reference_judge"]["key"]
                        else "judge"
                    ),
                    "roc_auc": None,
                    "roc_auc_low": None,
                    "roc_auc_high": None,
                    "average_precision_not_inferable": None,
                    "threshold_max_f1_not_inferable": None,
                    "cohen_kappa": judged["metrics"]["cohen_kappa"],
                    "cohen_kappa_low": low,
                    "cohen_kappa_high": high,
                    "f1_not_inferable": judged["metrics"]["per_class"]["not_inferable"]["f1"],
                }
            )
        rows += comparison_rows(report, config, split)
    return [{name: row.get(name) for name in TABLE_COLUMNS} for row in rows]


def comparison_rows(report: Record, config: VerifierConfig, split: str) -> list[Record]:
    declared = report.get("declared_comparisons", {}).get(split)
    if declared is None:
        return []
    rows = []
    for entry in declared["comparisons"]:
        key, score = entry["system"].split(".", 1)
        low, high = interval_bounds(entry.get("bootstrap") or {})
        rows.append(
            {
                "split": split,
                "system": entry["system"],
                "scorer": key,
                "score": score,
                "kind": "declared_comparison",
                "language": config.scorers[key].language,
                "role": "comparison",
                "comparison": entry["name"],
                "family": entry["family"],
                "reference": entry["reference"],
                "metric": entry["metric"],
                "delta": entry.get("delta"),
                "delta_low": low,
                "delta_high": high,
                "p_value": entry.get("p_value"),
                "p_holm": entry["p_holm"],
                "missing": ";".join(entry.get("missing", [])) or None,
            }
        )
    return rows


def write_table(rows: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: "" if row.get(name) is None else row[name] for name in TABLE_COLUMNS}
            )


def declared_block(config: VerifierConfig, declaration: RunDeclaration | None) -> Record:
    evaluation = config.source["evaluation"]
    block = {
        "primary_system": config.primary_system,
        "primary_metric": config.primary_metric,
        "primary_declaration": evaluation["primary_declaration"],
        "reference_cosine_system": config.reference_cosine_system,
        "reference_judge_rule": evaluation["reference_judge_rule"],
        "threshold_rules": evaluation["threshold_rules_definition"],
        "bootstrap_intervals": evaluation["bootstrap_intervals"],
    }
    if declaration is not None:
        block |= {
            "run_declaration": declaration.name,
            "primary_system": declaration.primary_system,
            "primary_metric": declaration.primary_metric,
            "primary_declaration": declaration.source["primary_declaration"],
            "declared": declaration.source["declared"],
            "v1_primary_system": config.primary_system,
        }
    return block
