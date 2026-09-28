"""Evaluation of the signals and policies on one split."""

from typing import Any

import numpy as np

from experiments.verifier.confidence.bootstrap import (
    FIXED,
    REFIT,
    Replicates,
    bootstrap_modes,
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


def point_signal_metrics(
    signals: dict[str, np.ndarray],
    names: list[str],
    evaluation: Arrays,
    config: ConfidenceConfig,
    where: str,
) -> tuple[Record, Record]:
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
    return point, curves


def point_policy_metrics(
    policies: list[Policy], signals: dict[str, np.ndarray], evaluation: Arrays
) -> Record:
    point = {}
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal])
        if accepted is not None:
            point[policy.name] = policy_metrics(accepted, evaluation.correct)
    return point


def bootstrap_replicates(
    signals: dict[str, np.ndarray],
    names: list[str],
    logistic_names: list[str],
    policies: list[Policy],
    fit: Arrays,
    evaluation: Arrays,
    raw_names: list[str],
    feature_sets: dict[str, tuple[str, ...]],
    config: ConfidenceConfig,
    fit_draws: tuple[np.ndarray, np.ndarray],
    eval_draws: tuple[np.ndarray, np.ndarray],
) -> Replicates:
    """Fixed replicates resample the evaluated split; refit replicates also refit the models."""
    store = Replicates()
    fit_universe, fit_matrix = fit_draws
    eval_universe, eval_matrix = eval_draws
    fit_members = hearing_members(fit.hearing_ids, fit_universe)
    eval_members = hearing_members(evaluation.hearing_ids, eval_universe)
    for replicate in range(config.bootstrap_samples):
        eval_index = replicate_index(eval_members, eval_matrix[replicate])
        record_signal_replicates(store, FIXED, signals, names, evaluation, eval_index, config)
        record_policy_replicates(store, FIXED, policies, signals, evaluation.correct, eval_index)
        fit_index = replicate_index(fit_members, fit_matrix[replicate])
        refit = fit_all(take(fit, fit_index), raw_names, feature_sets, config)
        refit_signals = apply_models(refit, evaluation, raw_names)
        refit_names = [name for name in logistic_names if np.isfinite(refit_signals[name]).all()]
        record_signal_replicates(
            store, REFIT, refit_signals, refit_names, evaluation, eval_index, config
        )
        record_policy_replicates(
            store, REFIT, refit.policies, refit_signals, evaluation.correct, eval_index
        )
    return store


def difference_record(
    store: Replicates,
    subset: str,
    name: str,
    metrics: Record,
    reference: Record,
    modes: tuple[str, ...],
    level: float,
) -> Record:
    differences: Record = {}
    for metric in DIFFERENCE_METRICS:
        difference: Record = {"point": rounded(metrics[metric] - reference[metric])}
        for mode in modes:
            interval = store.interval((mode, subset, name, f"difference_{metric}"), level)
            if interval is not None:
                difference[mode] = {**interval, "excludes_zero": excludes_zero(interval)}
        differences[metric] = difference
    return differences


def signal_entry(
    store: Replicates,
    subset: str,
    name: str,
    metrics: Record,
    reference: Record,
    modes: tuple[str, ...],
    config: ConfidenceConfig,
) -> Record:
    level = config.confidence_level
    entry: Record = {"queries": metrics.get("queries", 0)}
    for metric, value in metrics.items():
        if metric == "queries":
            continue
        entry[metric] = {"point": rounded(value)}
        for mode in modes:
            interval = store.interval((mode, subset, name, metric), level)
            if interval is not None:
                entry[metric][mode] = interval
    if name != config.reference_signal and "aurc" in reference and "aurc" in metrics:
        entry["difference_vs_reference"] = difference_record(
            store, subset, name, metrics, reference, modes, level
        )
    return entry


def policy_entry(policy: Policy, store: Replicates, point: Record, level: float) -> Record:
    entry = {
        "signal": policy.signal,
        "rule": policy.rule,
        "threshold": policy_threshold_record(policy),
        "fit_details": policy.details,
        "evaluate": {
            key: rounded(value) if isinstance(value, float) else value
            for key, value in point.get(policy.name, {}).items()
        },
    }
    for mode in bootstrap_modes(policy.rule != "production"):
        entry[mode] = {
            metric: store.interval((mode, policy.name, metric), level) for metric in POLICY_METRICS
        }
    return entry


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
    point, curves = point_signal_metrics(signals, names, evaluation, config, where)
    policies = fitted.policies + production
    policy_point = point_policy_metrics(policies, signals, evaluation)
    store = bootstrap_replicates(
        signals,
        names,
        logistic_names,
        policies,
        fit,
        evaluation,
        raw_names,
        feature_sets,
        config,
        fit_draws,
        eval_draws,
    )
    signal_report: Record = {
        subset: {
            name: signal_entry(
                store,
                subset,
                name,
                point[subset][name],
                point[subset][config.reference_signal],
                bootstrap_modes(name in logistic_names),
                config,
            )
            for name in names
        }
        for subset in SUBSETS
    }
    policy_report = {
        policy.name: policy_entry(policy, store, policy_point, config.confidence_level)
        for policy in policies
    }
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
            "evaluate_hearings_in_universe": int(len(eval_draws[0])),
            "fit_hearings_in_universe": int(len(fit_draws[0])),
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
