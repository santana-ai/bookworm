"""Evaluation of one target: entailment signals, predictions and every split."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file, write_jsonl

from experiments.verifier.confidence.bootstrap import arrays_of
from experiments.verifier.confidence.config import (
    ENTAILMENT_PREFIX,
    ConfidenceConfig,
    Target,
    entailment_file,
    pair_id_of,
    pairs_input_file,
    predictions_file,
)
from experiments.verifier.confidence.evaluation import evaluate_split, fit_split_description
from experiments.verifier.confidence.policies import (
    Fitted,
    Policy,
    apply_models,
    fit_all,
    logistic_record,
)
from experiments.verifier.confidence.production import (
    calibration_overlap,
    production_on_all,
    production_outside_calibration,
    production_policies,
)
from experiments.verifier.confidence.records import clean

Record = dict[str, Any]


def attach_entailment(
    rows: list[Record],
    target: Target,
    config: ConfidenceConfig,
    scores: dict[str, dict[str, Record]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for scorer in config.entailment_scorers:
            name = f"{ENTAILMENT_PREFIX}{scorer}"
            row[name] = None
            if target.unit not in config.entailment_units:
                continue
            entry = scores.get(scorer, {}).get(
                pair_id_of(row["bench"], row["query_id"], row["top1_text_sha256"])
            )
            if entry is None:
                counts[f"{scorer}.missing"] += 1
                continue
            if entry["split"] != row["split"] or entry["hearing_id"] != row["hearing_id"]:
                raise SystemExit(f"{entry['pair_id']}: split or hearing differs from the features")
            row[name] = float(entry["scores"][config.entailment_score_key])
            counts[f"{scorer}.scored"] += 1
            counts[f"{scorer}.truncated"] += int(any(i and i["truncated"] for i in entry["items"]))
    return counts


def load_entailment(
    run_dir: Path, config: ConfidenceConfig
) -> tuple[dict[str, dict[str, Record]], Record]:
    scores: dict[str, dict[str, Record]] = {}
    sources: Record = {}
    pairs_path = pairs_input_file(run_dir)
    pairs_sha = sha256_of_file(pairs_path) if pairs_path.exists() else None
    for scorer in config.entailment_scorers:
        path = entailment_file(run_dir, run_dir.name, scorer)
        if not path.exists():
            sources[scorer] = {"path": str(path), "available": False}
            continue
        rows = load_jsonl(path)
        scores[scorer] = {row["pair_id"]: row for row in rows}
        report_path = path.with_name(f"{scorer}_report.json")
        report: Record = {}
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
        scored_input = report.get("sources", {}).get("input", {}).get("sha256")
        sources[scorer] = {
            "available": True,
            "path": str(path),
            "sha256": sha256_of_file(path),
            "rows": len(rows),
            "model": rows[0]["model"] if rows else None,
            "revision": rows[0]["revision"] if rows else None,
            "report": str(report_path) if report else None,
            "report_created_at": report.get("created_at"),
            "scored_input_sha256": scored_input,
            "scored_input_matches_current_pairs_file": None
            if scored_input is None or pairs_sha is None
            else scored_input == pairs_sha,
            "label_probes": report.get("label_probes"),
        }
        if scored_input is not None and pairs_sha is not None and scored_input != pairs_sha:
            print(f"WARNING {scorer}: scores were computed on another pairs file", flush=True)
    return scores, sources


def prediction_rows(
    rows: list[Record],
    fitted: Fitted,
    raw_names: list[str],
    policies: list[Policy],
    config: ConfidenceConfig,
) -> list[Record]:
    arrays = arrays_of(rows, raw_names)
    signals = apply_models(fitted, arrays, raw_names)
    multi = np.array([row["n_units"] >= config.min_candidates for row in rows], dtype=bool)
    output = []
    for position, row in enumerate(rows):
        values = {name: clean(float(signals[name][position])) for name in signals}
        accepted: Record = {}
        for policy in policies:
            value = values.get(policy.signal)
            eligible = multi[position] or policy.rule == "production"
            if policy.threshold is None or value is None or not eligible:
                accepted[policy.name] = None
            else:
                accepted[policy.name] = bool(value >= policy.threshold)
        output.append(
            {
                "query_id": row["query_id"],
                "hearing_id": row["hearing_id"],
                "split": row["split"],
                "n_units": row["n_units"],
                "n_relevant": row["n_relevant"],
                "multi_candidate": bool(multi[position]),
                "nontrivial": bool(multi[position] and row["n_relevant"] < row["n_units"]),
                "correct": row["correct"],
                "correct_optimistic": row["correct_optimistic"],
                "correct_pessimistic": row["correct_pessimistic"],
                "zero_spread": row["zero_spread"],
                "top1_id": row["top1_id"],
                "signals": values,
                "accepted": accepted,
            }
        )
    return output


def split_counts(rows: list[Record], config: ConfidenceConfig) -> Record:
    multi = [row for row in rows if row["n_units"] >= config.min_candidates]
    return {
        "queries": len(rows),
        "hearings": len({row["hearing_id"] for row in rows}),
        "multi_candidate": len(multi),
        "single_candidate": len(rows) - len(multi),
        "nontrivial": sum(1 for row in multi if row["n_relevant"] < row["n_units"]),
        "correct_all": sum(1 for row in rows if row["correct"]),
        "correct_multi_candidate": sum(1 for row in multi if row["correct"]),
        "label_depends_on_ties_multi_candidate": sum(
            1 for row in multi if row["correct_optimistic"] != row["correct_pessimistic"]
        ),
        "zero_spread_multi_candidate": sum(1 for row in multi if row["zero_spread"]),
    }


def evaluate_target(
    bench: str,
    target: Target,
    rows_by_split: dict[str, list[Record]],
    eval_splits: tuple[str, ...],
    config: ConfidenceConfig,
    draws: dict[str, tuple[np.ndarray, np.ndarray]],
    run_dir: Path,
) -> tuple[Record, list[Record]]:
    population = {
        split: [row for row in rows if row["n_units"] >= config.min_candidates]
        for split, rows in rows_by_split.items()
    }
    fit_rows = [row for split in config.fit_splits for row in population[split]]
    decision_splits = (*config.fit_splits, *config.evaluate_splits)
    final_splits = [split for split in eval_splits if split not in decision_splits]
    entailment_names = [
        name
        for name in config.entailment_signals
        if target.unit in config.entailment_units
        and all(row.get(name) is not None for split in decision_splits for row in population[split])
    ]
    for split in final_splits:
        for name in entailment_names:
            unscored = sum(1 for row in population[split] if row.get(name) is None)
            if unscored:
                raise SystemExit(
                    f"{bench} {target.name}: {name} is scored on every "
                    f"{'+'.join(decision_splits)} query but missing for {unscored} {split} "
                    f"queries; the feature sets are fixed on {'+'.join(decision_splits)}, so the "
                    f"{split} evaluation is refused until its pairs are built and scored "
                    "(confidence_policies pairs --final-test, then nli_verifier_experiments "
                    "pairs --final-test)"
                )
    raw_names = [*config.base_signals, *entailment_names]
    missing_entailment = {
        name: sum(
            1 for split in decision_splits for row in population[split] if row.get(name) is None
        )
        for name in config.entailment_signals
        if name not in entailment_names
    }
    feature_sets = {
        name: features
        for name, features in config.feature_sets.items()
        if set(features) <= set(raw_names)
    }
    fit = arrays_of(fit_rows, raw_names)
    for name in raw_names:
        if not np.isfinite(fit.signals[name]).all():
            raise SystemExit(f"{bench} {target.name}: {name} is not finite on every fit row")
    fitted = fit_all(fit, raw_names, feature_sets, config)
    production, production_source = production_policies(config, target)
    fit_draws = draws[config.fit_splits[0]]
    report: Record = {
        "target": target.name,
        "retriever": target.retriever,
        "unit": target.unit,
        "counts": {split: split_counts(rows, config) for split, rows in rows_by_split.items()},
        "raw_signals": raw_names,
        "entailment_decision_splits": list(decision_splits),
        "entailment_missing_rows": missing_entailment,
        "feature_sets_used": {name: list(features) for name, features in feature_sets.items()},
        "feature_sets_skipped": sorted(set(config.feature_sets) - set(feature_sets)),
        "logistic": {
            name: logistic_record(model, fit.correct) for name, model in fitted.models.items()
        },
        "fit_split_description": fit_split_description(fitted, fit, config),
        "evaluate": {},
    }
    if production_source:
        report["production_rule"] = {
            "source": production_source,
            "all_queries": {},
            "calibration_overlap": {},
            "outside_calibration": {},
            "comparison_rule": config.source["production_rule"]["comparison_rule"],
        }
    for split in eval_splits:
        evaluation_rows = population[split]
        if not evaluation_rows:
            report["evaluate"][split] = {"queries": 0}
            continue
        evaluation = arrays_of(evaluation_rows, raw_names)
        report["evaluate"][split] = evaluate_split(
            fitted,
            fit,
            evaluation,
            raw_names,
            feature_sets,
            config,
            fit_draws,
            draws[split],
            production,
            f"{bench}|{target.name}|{split}",
        )
        if production:
            calibration = production_source["calibration"]
            overlap = {
                policy.name: calibration_overlap(
                    calibration,
                    policy.details["udv_config_key"],
                    rows_by_split[split],
                    evaluation_rows,
                )
                for policy in production
            }
            for name, entry in overlap.items():
                report["evaluate"][split]["policies"][name]["calibration_overlap"] = entry
            report["production_rule"]["calibration_overlap"][split] = overlap
            report["production_rule"]["all_queries"][split] = production_on_all(
                production, rows_by_split[split], draws[split], config
            )
            report["production_rule"]["outside_calibration"][split] = (
                production_outside_calibration(
                    production,
                    calibration,
                    rows_by_split[split],
                    evaluation_rows,
                    f"{bench}|{split}",
                    config,
                )
            )
    predictions: list[Record] = []
    for split, rows in rows_by_split.items():
        output = prediction_rows(rows, fitted, raw_names, fitted.policies + production, config)
        path = predictions_file(run_dir, bench, target.name, split)
        write_jsonl(output, path)
        predictions.append({"path": str(path), "rows": len(output), "sha256": sha256_of_file(path)})
    return report, predictions
