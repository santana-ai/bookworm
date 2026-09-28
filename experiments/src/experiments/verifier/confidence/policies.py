"""Logistic combinations of the signals and the accept policies fitted on them."""

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from experiments.udv.calibrate_threshold import youden_optimum
from experiments.verifier.confidence.config import (
    DEGENERATE_YOUDEN,
    LOGISTIC_PREFIX,
    ConfidenceConfig,
)
from experiments.verifier.confidence.records import rounded

Record = dict[str, Any]


@dataclass(frozen=True)
class LogisticFit:
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    model: LogisticRegression
    positive_column: int


@dataclass(frozen=True)
class Policy:
    name: str
    signal: str
    rule: str
    threshold: float | None
    details: Record


@dataclass
class Arrays:
    query_ids: list[str]
    hearing_ids: np.ndarray
    correct: np.ndarray
    nontrivial: np.ndarray
    signals: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.query_ids)


@dataclass
class Fitted:
    models: dict[str, LogisticFit | None]
    policies: list[Policy]
    fit_signals: dict[str, np.ndarray]


def fit_logistic(
    signals: dict[str, np.ndarray],
    correct: np.ndarray,
    features: tuple[str, ...],
    params: Record,
    seed: int,
) -> LogisticFit | None:
    if len(correct) == 0 or correct.all() or not correct.any():
        return None
    matrix = np.column_stack([signals[name] for name in features])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    scale = np.where(std > 0, std, 1.0)
    model = LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=None if params["class_weight"] == "none" else params["class_weight"],
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit((matrix - mean) / scale, correct)
    return LogisticFit(features, mean, scale, model, list(model.classes_).index(True))


def predict_logistic(fit: LogisticFit | None, signals: dict[str, np.ndarray]) -> np.ndarray:
    size = len(next(iter(signals.values())))
    values = np.full(size, math.nan)
    if fit is None:
        return values
    matrix = np.column_stack([signals[name] for name in fit.features])
    finite = np.isfinite(matrix).all(axis=1)
    if finite.any():
        standardized = (matrix[finite] - fit.mean) / fit.scale
        values[finite] = fit.model.predict_proba(standardized)[:, fit.positive_column]
    return values


def logistic_record(fit: LogisticFit | None, correct: np.ndarray) -> Record:
    if fit is None:
        return {"fitted": False, "reason": "the fit rows hold one class only or no row"}
    return {
        "fitted": True,
        "features": list(fit.features),
        "coefficients_standardized": dict(
            zip(fit.features, fit.model.coef_[0].tolist(), strict=True)
        ),
        "intercept": float(fit.model.intercept_[0]),
        "feature_mean": dict(zip(fit.features, fit.mean.tolist(), strict=True)),
        "feature_scale": dict(zip(fit.features, fit.scale.tolist(), strict=True)),
        "iterations": int(fit.model.n_iter_[0]),
        "fit_rows": len(correct),
        "fit_positive_rate": float(correct.mean()),
    }


def youden_policy(signal: str, values: np.ndarray, correct: np.ndarray) -> Policy:
    optimum = youden_optimum(values, correct)
    name = f"youden:{signal}"
    if optimum is None:
        return Policy(name, signal, "youden", None, {"reason": "one class only in the fit rows"})
    details = {key: rounded(value) for key, value in optimum.items()}
    details["degenerate"] = optimum["youden_j"] <= DEGENERATE_YOUDEN
    return Policy(name, signal, "youden", float(optimum["threshold"]), details)


def conformal_policy(signal: str, values: np.ndarray, correct: np.ndarray, alpha: float) -> Policy:
    calibration = np.sort(values[correct])
    size = len(calibration)
    rank = math.floor(alpha * (size + 1))
    name = f"conformal:{signal}:alpha={alpha:g}"
    details = {"alpha": alpha, "target_recall": 1 - alpha, "calibration_queries": size, "k": rank}
    if size == 0:
        return Policy(name, signal, "conformal", None, {**details, "reason": "no correct fit row"})
    threshold = -math.inf if rank == 0 else float(calibration[rank - 1])
    return Policy(name, signal, "conformal", threshold, {**details, "accepts_all": rank == 0})


def fit_policies(
    signals: dict[str, np.ndarray], correct: np.ndarray, names: list[str], config: ConfidenceConfig
) -> list[Policy]:
    policies = [youden_policy(name, signals[name], correct) for name in names]
    for name in config.conformal_signals:
        if name in names:
            policies += [
                conformal_policy(name, signals[name], correct, alpha)
                for alpha in config.conformal_alphas
            ]
    policies += [
        Policy(
            f"probability:{name}",
            name,
            "probability",
            config.probability_threshold,
            {"threshold": config.probability_threshold},
        )
        for name in names
        if name.startswith(LOGISTIC_PREFIX)
    ]
    return policies


def accepted_by(policy: Policy, values: np.ndarray) -> np.ndarray | None:
    if policy.threshold is None:
        return None
    return np.asarray(values >= policy.threshold)


def fit_all(
    fit: Arrays,
    raw_names: list[str],
    feature_sets: dict[str, tuple[str, ...]],
    config: ConfidenceConfig,
) -> Fitted:
    raw = {name: fit.signals[name] for name in raw_names}
    models = {
        f"{LOGISTIC_PREFIX}{name}": fit_logistic(
            raw, fit.correct, features, config.logistic, config.seed
        )
        for name, features in feature_sets.items()
    }
    signals = dict(raw)
    for name, model in models.items():
        signals[name] = predict_logistic(model, raw)
    names = [
        name
        for name in signals
        if models.get(name, True) is not None and len(fit) and np.isfinite(signals[name]).all()
    ]
    return Fitted(models, fit_policies(signals, fit.correct, names, config), signals)


def apply_models(fitted: Fitted, arrays: Arrays, raw_names: list[str]) -> dict[str, np.ndarray]:
    raw = {name: arrays.signals[name] for name in raw_names}
    signals = dict(raw)
    for name, model in fitted.models.items():
        signals[name] = predict_logistic(model, raw)
    return signals
