"""Per-split evaluation of systems and judges with hearing-level bootstrap."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl

from experiments.common.reporting import rounded
from experiments.common.splits import SPLIT_NAMES
from experiments.data.nli_benchmark import judge_metrics, parse_judge_key
from experiments.udv.calibrate_threshold import describe, interval
from experiments.verifier.nli.config import COMPARED_RULE, VerifierConfig
from experiments.verifier.nli.metrics import (
    BINARY_METRICS,
    FITTED_RULES,
    RANKING_METRICS,
    binary_metrics,
    check_binary_metrics,
    fit_rule,
    is_probability_system,
    ranking_metrics,
    rounded_record,
)
from experiments.verifier.nli.scoring import score_file, score_report_file
from experiments.verifier.stats import (
    finite_interval,
    hearing_draws,
)

Record = dict[str, Any]

COMPARED_RANKING_METRICS = ("roc_auc", "average_precision_not_inferable")
COMPARED_BINARY_METRICS = ("cohen_kappa", "f1_not_inferable")


@dataclass(frozen=True)
class SplitData:
    split: str
    ids: list[str]
    labels: np.ndarray
    hearing_ids: np.ndarray
    judges: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]


def replicate_binary(
    labels: np.ndarray, predictions: np.ndarray, draws: list[np.ndarray]
) -> dict[str, np.ndarray]:
    values = [binary_metrics(labels[rows], predictions[rows]) for rows in draws]
    return {metric: np.array([v[metric] for v in values]) for metric in BINARY_METRICS}


def replicate_refit(
    scores: np.ndarray,
    labels: np.ndarray,
    draws: list[np.ndarray],
    refits: list[Record | None],
) -> dict[str, np.ndarray]:
    values = [
        binary_metrics(labels[rows], scores[rows] >= refit["threshold"])
        if refit is not None
        else {metric: float("nan") for metric in BINARY_METRICS}
        for rows, refit in zip(draws, refits, strict=True)
    ]
    return {metric: np.array([v[metric] for v in values]) for metric in BINARY_METRICS}


def evaluate_binary(
    predictions: np.ndarray, data: SplitData, draws: list[np.ndarray], level: float, where: str
) -> tuple[Record, dict[str, np.ndarray]]:
    metrics = judge_metrics(data.labels.tolist(), predictions.tolist())
    check_binary_metrics(binary_metrics(data.labels, predictions), metrics, where)
    replicates = replicate_binary(data.labels, predictions, draws)
    intervals = {m: finite_interval(replicates[m], level) for m in BINARY_METRICS}
    return {"metrics": metrics, "bootstrap": intervals}, replicates


def fit_record(rule: str, optimum: Record | None, scores: np.ndarray, data: SplitData) -> Record:
    if optimum is None:
        return {"applicable": False}
    predictions = scores >= optimum["threshold"]
    return {
        "applicable": True,
        "threshold": optimum["threshold"],
        "optimum": {k: v if v is None else rounded(v) for k, v in optimum.items()},
        "in_sample_metrics": judge_metrics(data.labels.tolist(), predictions.tolist()),
        "in_sample_note": (
            f"measured on the fit split that chose the {rule} threshold"
            if rule in FITTED_RULES
            else "fixed threshold, not fitted; measured on the fit split"
        ),
    }


def evaluate_system_split(
    system: str,
    data: SplitData,
    fitted: dict[str, Record | None],
    refits: dict[str, list[Record | None]],
    draws: list[np.ndarray],
    config: VerifierConfig,
) -> tuple[Record, dict[tuple[str, str], np.ndarray]]:
    scores, labels, level = data.scores[system], data.labels, config.confidence_level
    ranking_values = [ranking_metrics(labels[rows], scores[rows]) for rows in draws]
    replicates = {("ranking", m): np.array([v[m] for v in ranking_values]) for m in RANKING_METRICS}
    rules: Record = {}
    for rule, optimum in fitted.items():
        if optimum is None:
            continue
        where = f"{system} {rule} {data.split}"
        predictions = scores >= optimum["threshold"]
        result, fixed = evaluate_binary(predictions, data, draws, level, where)
        refit = replicate_refit(scores, labels, draws, refits[rule])
        replicates |= {(rule, m): fixed[m] for m in BINARY_METRICS}
        rules[rule] = {
            "threshold": rounded(optimum["threshold"]),
            "metrics": result["metrics"],
            "bootstrap_fixed_threshold": result["bootstrap"],
            "bootstrap_refit_threshold": {
                m: finite_interval(refit[m], level) for m in BINARY_METRICS
            },
        }
    return {
        "ranking": rounded_record(ranking_metrics(labels, scores)),
        "ranking_bootstrap": {
            m: finite_interval(replicates[("ranking", m)], level) for m in RANKING_METRICS
        },
        "score_by_label": {
            "inferable": describe(scores[labels]),
            "not_inferable": describe(scores[~labels]),
        },
        "rules": rules,
    }, replicates


def evaluate_system(
    system: str,
    fit: SplitData,
    evaluations: list[SplitData],
    fit_draws: list[np.ndarray],
    eval_draws: dict[str, list[np.ndarray]],
    config: VerifierConfig,
) -> tuple[Record, dict[str, dict[tuple[str, str], np.ndarray]]]:
    probability = is_probability_system(system)
    scores, labels = fit.scores[system], fit.labels
    fitted = {
        rule: fit_rule(rule, scores, labels, config, probability) for rule in config.threshold_rules
    }
    refits = {
        rule: [
            fit_rule(rule, scores[rows], labels[rows], config, probability) for rows in fit_draws
        ]
        if rule in FITTED_RULES
        else [optimum] * len(fit_draws)
        for rule, optimum in fitted.items()
    }
    result: Record = {
        "probability_score": probability,
        "fit": {rule: fit_record(rule, fitted[rule], scores, fit) for rule in fitted},
        "threshold_bootstrap_on_fit_split": {
            rule: interval(
                [r["threshold"] for r in refits[rule] if r is not None], config.confidence_level
            )
            for rule in FITTED_RULES
            if rule in fitted and fitted[rule] is not None
        },
        "evaluation": {},
    }
    replicates: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    for data in evaluations:
        split_result, split_replicates = evaluate_system_split(
            system, data, fitted, refits, eval_draws[data.split], config
        )
        result["evaluation"][data.split] = split_result
        replicates[data.split] = split_replicates
    return result, replicates


def judge_keys_of(rows: list[Record]) -> list[str]:
    keys = list(rows[0]["judge_inferable"])
    if any(list(row["judge_inferable"]) != keys for row in rows):
        raise SystemExit("benchmark rows do not share the same judge keys")
    return keys


def load_scores(path: Path) -> dict[str, Record]:
    return {row["id"]: row for row in load_jsonl(path)}


def split_data(
    split: str,
    rows: list[Record],
    systems_by_scorer: dict[str, list[str]],
    directories: dict[str, Path],
    judge_keys: list[str],
    own: set[str],
) -> SplitData:
    scored = {
        key: load_scores(score_file(directories[key], key, split)) for key in systems_by_scorer
    }
    id_sets = {key: set(values) for key, values in scored.items()}
    base = {key: ids for key, ids in id_sets.items() if key in own} or id_sets
    reference = next(iter(base.values()))
    if any(ids != reference for ids in base.values()):
        sizes = {key: len(ids) for key, ids in base.items()}
        raise SystemExit(f"{split}: scorers were run on different opinions {sizes}")
    short = [key for key, ids in id_sets.items() if key not in base and not reference <= ids]
    if short:
        raise SystemExit(f"{split}: imported score files {short} lack opinions of this run")
    members = [row for row in rows if row["split"] == split and row["id"] in reference]
    if len(members) != len(reference):
        raise SystemExit(f"{split}: score files name opinions that are not benchmark rows")
    ids = [row["id"] for row in members]
    for key, names in systems_by_scorer.items():
        stale = sorted({n for i in ids for n in names if n not in scored[key][i]["scores"]})
        if stale:
            raise SystemExit(
                f"{key} {split}: the score file lacks {stale}; it was written before the score "
                f"names changed, so rerun `score --scorers {key}` (the NLI logits are read from "
                "the cache, no model forward pass is repeated)"
            )
    scores = {
        f"{key}.{name}": np.array([scored[key][i]["scores"][name] for i in ids], dtype=np.float64)
        for key, names in systems_by_scorer.items()
        for name in names
    }
    return SplitData(
        split=split,
        ids=ids,
        labels=np.array([row["label_inferable"] for row in members], dtype=bool),
        hearing_ids=np.array([row["hearing_id"] for row in members], dtype=np.int64),
        judges={
            k: np.array([r["judge_inferable"][k] for r in members], dtype=bool) for k in judge_keys
        },
        scores=scores,
    )


def available_scorers(
    config: VerifierConfig,
    run_dir: Path,
    splits: tuple[str, ...],
    own: tuple[str, ...],
    imported: dict[str, str],
) -> tuple[dict[str, list[str]], Record, dict[str, Path]]:
    systems: dict[str, list[str]] = {}
    reports: Record = {}
    directories: dict[str, Path] = {}
    for key in (*own, *imported):
        directory = run_dir if key in own else run_dir.parent / imported[key]
        if not all(score_file(directory, key, split).exists() for split in splits):
            continue
        with open(score_report_file(directory, key)) as f:
            reports[key] = json.load(f)
        systems[key] = list(config.scorers[key].scores)
        directories[key] = directory
    if not systems:
        raise SystemExit(f"no scorer has score files for {list(splits)} in {run_dir}")
    subsets = {
        json.dumps(report["subset"], sort_keys=True)
        for key, report in reports.items()
        if key in own
    }
    if len(subsets) > 1:
        raise SystemExit(f"score files come from different subsets: {subsets}")
    return systems, reports, directories


def split_summary(data: SplitData) -> Record:
    inferable = int(data.labels.sum())
    return {
        "opinions": len(data.ids),
        "hearings": len(set(data.hearing_ids.tolist())),
        "inferable": inferable,
        "not_inferable": len(data.ids) - inferable,
        "not_inferable_share": rounded((len(data.ids) - inferable) / len(data.ids)),
    }


def evaluate_judges(
    fit: SplitData,
    evaluations: list[SplitData],
    eval_draws: dict[str, list[np.ndarray]],
    level: float,
) -> tuple[Record, dict[str, dict[str, dict[str, np.ndarray]]]]:
    fit_kappa = {
        key: judge_metrics(fit.labels.tolist(), fit.judges[key].tolist())["cohen_kappa"]
        for key in fit.judges
    }
    reference = max(fit.judges, key=lambda key: fit_kappa[key])
    result: Record = {
        "reference_judge": {"key": reference, "fit_cohen_kappa": fit_kappa[reference]},
        "fit_cohen_kappa": fit_kappa,
        "evaluation": {},
    }
    replicates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for data in evaluations:
        judged: Record = {}
        replicates[data.split] = {}
        for key, predictions in data.judges.items():
            summary, values = evaluate_binary(predictions, data, eval_draws[data.split], level, key)
            judged[key] = {**parse_judge_key(key), **summary}
            replicates[data.split][key] = values
        result["evaluation"][data.split] = judged
    return result, replicates


def evaluate_constant(data: SplitData, draws: list[np.ndarray], level: float) -> Record:
    summary, _ = evaluate_binary(np.ones(len(data.ids), dtype=bool), data, draws, level, "always")
    prevalence = float(data.labels.mean())
    return {
        **summary,
        "ranking": {
            "roc_auc": 0.5,
            "average_precision_inferable": rounded(prevalence),
            "average_precision_not_inferable": rounded(1 - prevalence),
        },
    }


def paired_delta(values: np.ndarray, reference: np.ndarray, point: float, level: float) -> Record:
    delta = values - reference
    valid = delta[~np.isnan(delta)]
    return {
        "delta": None if np.isnan(point) else rounded(point),
        "bootstrap": interval(valid.tolist(), level),
        "share_of_replicates_above_zero": rounded(float(np.mean(valid > 0)))
        if len(valid)
        else None,
    }


def point_value(results: Record, system: str, split: str, key: tuple[str, str]) -> float:
    evaluation = results[system]["evaluation"][split]
    if key[0] == "ranking":
        value = evaluation["ranking"][key[1]]
    else:
        metrics = evaluation["rules"][key[0]]["metrics"]
        value = (
            metrics["per_class"]["not_inferable"]["f1"]
            if key[1] == "f1_not_inferable"
            else metrics[key[1]]
        )
    return float("nan") if value is None else float(value)


def judge_point(judges: Record, split: str, key: str, metric: str) -> float:
    metrics = judges["evaluation"][split][key]["metrics"]
    return float(
        metrics["per_class"]["not_inferable"]["f1"]
        if metric == "f1_not_inferable"
        else metrics[metric]
    )


def compare_systems(
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    judges: Record,
    judge_replicates: dict[str, dict[str, dict[str, np.ndarray]]],
    splits: list[str],
    config: VerifierConfig,
) -> Record:
    level = config.confidence_level
    reference_system = config.reference_cosine_system
    reference_judge = judges["reference_judge"]["key"]
    comparisons: Record = {}
    for split in splits:
        against_cosine: Record = {}
        against_judge: Record = {}
        for system in results:
            if reference_system in results and system != reference_system:
                against_cosine[system] = {
                    metric: paired_delta(
                        replicates[system][split][("ranking", metric)],
                        replicates[reference_system][split][("ranking", metric)],
                        point_value(results, system, split, ("ranking", metric))
                        - point_value(results, reference_system, split, ("ranking", metric)),
                        level,
                    )
                    for metric in COMPARED_RANKING_METRICS
                }
            if (COMPARED_RULE, "cohen_kappa") in replicates[system][split]:
                against_judge[system] = {
                    metric: paired_delta(
                        replicates[system][split][(COMPARED_RULE, metric)],
                        judge_replicates[split][reference_judge][metric],
                        point_value(results, system, split, (COMPARED_RULE, metric))
                        - judge_point(judges, split, reference_judge, metric),
                        level,
                    )
                    for metric in COMPARED_BINARY_METRICS
                }
        comparisons[split] = {
            "against_reference_cosine": {"reference": reference_system, "deltas": against_cosine},
            "against_reference_judge": {
                "reference": reference_judge,
                "rule": COMPARED_RULE,
                "deltas": against_judge,
            },
        }
    return comparisons


def draw_bootstraps(
    fit: SplitData, evaluations: list[SplitData], config: VerifierConfig
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    fit_rng = np.random.default_rng([config.evaluation_seed, 0])
    fit_draws = hearing_draws(fit.hearing_ids, config.bootstrap_samples, fit_rng)
    eval_draws = {
        data.split: hearing_draws(
            data.hearing_ids,
            config.bootstrap_samples,
            np.random.default_rng([config.evaluation_seed, 1 + SPLIT_NAMES.index(data.split)]),
        )
        for data in evaluations
    }
    return fit_draws, eval_draws


def merge_fit_splits(parts: list[SplitData]) -> SplitData:
    if len(parts) == 1:
        return parts[0]
    return SplitData(
        split="+".join(part.split for part in parts),
        ids=[i for part in parts for i in part.ids],
        labels=np.concatenate([part.labels for part in parts]),
        hearing_ids=np.concatenate([part.hearing_ids for part in parts]),
        judges={k: np.concatenate([p.judges[k] for p in parts]) for k in parts[0].judges},
        scores={k: np.concatenate([p.scores[k] for p in parts]) for k in parts[0].scores},
    )
