"""Evaluation of the signals and policies on one split."""

from typing import Any

import numpy as np

from experiments.verifier.confidence.bootstrap import (
    Replicates,
    excludes_zero,
    hearing_members,
    policy_threshold_record,
    record_policy_replicates,
    record_signal_replicates,
    replicate_index,
    subset_index,
    take,
)
from experiments.verifier.confidence.config import (
    DIFFERENCE_METRICS,
    LOGISTIC_PREFIX,
    POLICY_METRICS,
    SUBSETS,
    ConfidenceConfig,
)
from experiments.verifier.confidence.metrics import (
    check_fast_metrics,
    policy_metrics,
    risk_coverage_curve,
    signal_metrics,
)
from experiments.verifier.confidence.policies import (
    Arrays,
    Fitted,
    Policy,
    accepted_by,
    apply_models,
    fit_all,
)
from experiments.verifier.confidence.records import rounded

Record = dict[str, Any]


def evaluate_split(
    fitted: Fitted,
    fit: Arrays,
    evaluation: Arrays,
    raw_names: list[str],
    feature_sets: dict[str, tuple[str, ...]],
    config: ConfidenceConfig,
    fit_draws: tuple[np.ndarray, np.ndarray],
    eval_draws: tuple[np.ndarray, np.ndarray],
    production: list[Policy],
    where: str,
) -> Record:
    signals = apply_models(fitted, evaluation, raw_names)
    names = [name for name in signals if np.isfinite(signals[name]).all()]
    logistic_names = [name for name in names if name.startswith(LOGISTIC_PREFIX)]
    point: Record = {}
    curves: Record = {}
    for subset in SUBSETS:
        chosen = subset_index(evaluation, np.arange(len(evaluation)), subset)
        correct = evaluation.correct[chosen]
        point[subset], curves[subset] = {}, {}
        for name in names:
            values = signals[name][chosen]
            check_fast_metrics(values, correct, f"{where}|{subset}|{name}")
            point[subset][name] = signal_metrics(values, correct, config.coverages)
            curves[subset][name] = risk_coverage_curve(values, correct, config.grid_step)
    policies = fitted.policies + production
    policy_point = {}
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal])
        if accepted is not None:
            policy_point[policy.name] = policy_metrics(accepted, evaluation.correct)
    store = Replicates()
    fit_universe, fit_matrix = fit_draws
    eval_universe, eval_matrix = eval_draws
    fit_members = hearing_members(fit.hearing_ids, fit_universe)
    eval_members = hearing_members(evaluation.hearing_ids, eval_universe)
    for replicate in range(config.bootstrap_samples):
        eval_index = replicate_index(eval_members, eval_matrix[replicate])
        record_signal_replicates(store, "fixed", signals, names, evaluation, eval_index, config)
        record_policy_replicates(store, "fixed", policies, signals, evaluation.correct, eval_index)
        fit_index = replicate_index(fit_members, fit_matrix[replicate])
        refit = fit_all(take(fit, fit_index), raw_names, feature_sets, config)
        refit_signals = apply_models(refit, evaluation, raw_names)
        refit_names = [name for name in logistic_names if np.isfinite(refit_signals[name]).all()]
        record_signal_replicates(
            store, "refit", refit_signals, refit_names, evaluation, eval_index, config
        )
        record_policy_replicates(
            store, "refit", refit.policies, refit_signals, evaluation.correct, eval_index
        )
    level = config.confidence_level
    signal_report: Record = {}
    for subset in SUBSETS:
        signal_report[subset] = {}
        for name in names:
            metrics = point[subset][name]
            entry: Record = {"queries": metrics.get("queries", 0)}
            for metric, value in metrics.items():
                if metric == "queries":
                    continue
                entry[metric] = {"point": rounded(value)}
                for mode in ("fixed", "refit") if name in logistic_names else ("fixed",):
                    interval = store.interval((mode, subset, name, metric), level)
                    if interval is not None:
                        entry[metric][mode] = interval
            reference = point[subset][config.reference_signal]
            if name != config.reference_signal and "aurc" in reference and "aurc" in metrics:
                entry["difference_vs_reference"] = {}
                for metric in DIFFERENCE_METRICS:
                    difference: Record = {"point": rounded(metrics[metric] - reference[metric])}
                    for mode in ("fixed", "refit") if name in logistic_names else ("fixed",):
                        interval = store.interval(
                            (mode, subset, name, f"difference_{metric}"), level
                        )
                        if interval is not None:
                            difference[mode] = {
                                **interval,
                                "excludes_zero": excludes_zero(interval),
                            }
                    entry["difference_vs_reference"][metric] = difference
            signal_report[subset][name] = entry
    policy_report: Record = {}
    for policy in policies:
        fitted_policy = policy.rule != "production"
        entry = {
            "signal": policy.signal,
            "rule": policy.rule,
            "threshold": policy_threshold_record(policy),
            "fit_details": policy.details,
            "evaluate": {
                key: rounded(value) if isinstance(value, float) else value
                for key, value in policy_point.get(policy.name, {}).items()
            },
        }
        for mode in ("fixed", "refit") if fitted_policy else ("fixed",):
            entry[mode] = {
                metric: store.interval((mode, policy.name, metric), level)
                for metric in POLICY_METRICS
            }
        policy_report[policy.name] = entry
    return {
        "queries": len(evaluation),
        "hearings": int(len(np.unique(evaluation.hearing_ids))),
        "correct": int(evaluation.correct.sum()),
        "nontrivial": int(evaluation.nontrivial.sum()),
        "signals_evaluated": names,
        "signal_metrics": signal_report,
        "risk_coverage": curves,
        "policies": policy_report,
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "unit": "hearing",
            "evaluate_hearings_in_universe": int(len(eval_universe)),
            "fit_hearings_in_universe": int(len(fit_universe)),
        },
    }


def fit_split_description(fitted: Fitted, fit: Arrays, config: ConfidenceConfig) -> Record:
    description: Record = {}
    for name, values in fitted.fit_signals.items():
        if not np.isfinite(values).all():
            continue
        metrics = signal_metrics(values, fit.correct, config.coverages)
        description[name] = {
            key: rounded(value) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
    return description


def dotted_value(payload: Any, field_path: str) -> Any:
    value = payload
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"field {field_path!r} is missing")
        value = value[part]
    return value
