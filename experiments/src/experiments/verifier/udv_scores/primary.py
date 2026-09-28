"""Refit of the primary candidate on train and the check against the final-test record."""

from typing import Any

import numpy as np
from bookworm import sha256_of_file

from experiments.udv.calibrate_threshold import rounded
from experiments.verifier.exploration.config import (
    MATCH_TOLERANCE,
    Candidate,
    ExplorationConfig,
    ScorerData,
)
from experiments.verifier.exploration.cross_validation import feature_matrix, fit_learned
from experiments.verifier.exploration.scores import load_labels, load_scorer
from experiments.verifier.nli.metrics import max_f1_not_inferable_optimum
from experiments.verifier.udv_scores.config import FittedPrimary

Record = dict[str, Any]


def load_fit_data(
    config: ExplorationConfig, scorers: tuple[str, ...], selection: Record
) -> tuple[np.ndarray, np.ndarray, dict[str, ScorerData], Record]:
    split = config.raw["fit_split"]
    ids, labels, hearings = load_labels(config, split)
    data: dict[str, ScorerData] = {}
    files: Record = {}
    for key in scorers:
        kind, run = config.scorers[key]
        scorer = load_scorer(key, kind, run, split, ids, config)
        if scorer is None:
            raise SystemExit(f"{key}: no {split} score file in run {run}")
        digest = sha256_of_file(scorer.path)
        recorded = selection["files_read"][key]["sha256"]
        if digest != recorded:
            raise SystemExit(f"{scorer.path}: sha256 differs from the file explore read")
        data[key] = scorer
        files[key] = {"path": str(scorer.path), "sha256": digest, "matches_selection": True}
    return labels, hearings, data, files


def fit_primary(
    candidate: Candidate,
    labels: np.ndarray,
    hearings: np.ndarray,
    data: dict[str, ScorerData],
    config: ExplorationConfig,
) -> FittedPrimary:
    matrix = feature_matrix(candidate, data)
    scaler, model, c, means = fit_learned(matrix, labels, hearings, config)
    fit_scores = model.predict_proba(scaler.transform(matrix))[:, 1]
    optimum = max_f1_not_inferable_optimum(fit_scores, labels)
    if optimum is None:
        raise SystemExit("the train split has a single label: no threshold")
    record = {
        "c": c,
        "inner_means": means,
        "coefficients": dict(zip(candidate.features, map(rounded, model.coef_[0]), strict=True)),
        "intercept": float(model.intercept_[0]),
        "threshold": float(optimum["threshold"]),
        "train_opinions": len(labels),
        "train_not_inferable": int((~labels).sum()),
    }
    return FittedPrimary(candidate, scaler, model, float(optimum["threshold"]), record)


def refit_check(fitted: FittedPrimary, final_test: Record, key: str) -> Record:
    stored = final_test["learned"][key]
    threshold = float(final_test["results"][key]["threshold"])
    checks = {
        "c": fitted.record["c"] == stored["c"],
        "inner_means": fitted.record["inner_means"] == stored["inner_means"],
        "coefficients": fitted.record["coefficients"] == stored["coefficients"],
        "threshold": abs(fitted.threshold - threshold) <= MATCH_TOLERANCE,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit(f"the refit differs from final_test.json in {failed}")
    return {
        "compared_with": "final_test.json learned and results entries of the primary",
        "checks": checks,
        "threshold_refit": fitted.threshold,
        "threshold_final_test": threshold,
        "threshold_abs_gap": abs(fitted.threshold - threshold),
    }
