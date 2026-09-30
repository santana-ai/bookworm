"""Hearing-grouped cross-validation of the candidates on train and the selection rule."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.common.reporting import rounded
from experiments.verifier.exploration.candidates import feature_candidate
from experiments.verifier.exploration.config import (
    KIND_ORDER,
    Candidate,
    ExplorationConfig,
    ScorerData,
)
from experiments.verifier.exploration.scores import candidate_scores
from experiments.verifier.nli.metrics import binary_metrics, max_f1_not_inferable_optimum

Record = dict[str, Any]


def cv_folds(
    config: ExplorationConfig, labels: np.ndarray, hearings: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    for repeat in range(config.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.cv_seed + repeat
        )
        folds += list(splitter.split(np.zeros(len(labels)), labels.astype(int), hearings))
    return folds


def fold_kappa(scores: np.ndarray, labels: np.ndarray, fit: np.ndarray, held: np.ndarray) -> float:
    optimum = max_f1_not_inferable_optimum(scores[fit], labels[fit])
    if optimum is None:
        return float("nan")
    return binary_metrics(labels[held], scores[held] >= optimum["threshold"])["cohen_kappa"]


def fit_learned(
    matrix: np.ndarray, labels: np.ndarray, hearings: np.ndarray, config: ExplorationConfig
) -> tuple[StandardScaler, LogisticRegression, float, Record]:
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_splits, shuffle=True, random_state=config.cv_seed
    )
    inner = list(splitter.split(matrix, labels.astype(int), hearings))
    means = {}
    for c in config.c_grid:
        values = []
        for fit, held in inner:
            scaler = StandardScaler().fit(matrix[fit])
            model = LogisticRegression(C=c, max_iter=2000, random_state=42)
            model.fit(scaler.transform(matrix[fit]), labels[fit].astype(int))
            values.append(
                roc_auc_score(
                    labels[held], model.predict_proba(scaler.transform(matrix[held]))[:, 1]
                )
            )
        means[c] = float(np.mean(values))
    chosen = max(config.c_grid, key=lambda c: (means[c], -c))
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(C=chosen, max_iter=2000, random_state=42)
    model.fit(scaler.transform(matrix), labels.astype(int))
    return scaler, model, chosen, {str(c): rounded(v) for c, v in means.items()}


def feature_matrix(candidate: Candidate, data: dict[str, ScorerData]) -> np.ndarray:
    return np.column_stack(
        [candidate_scores(feature_candidate(f), data) for f in candidate.features]
    )


def evaluate_fixed(
    scores: np.ndarray, labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[list[float], list[float]]:
    aucs = [float(roc_auc_score(labels[held], scores[held])) for _, held in folds]
    kappas = [fold_kappa(scores, labels, fit, held) for fit, held in folds]
    return aucs, kappas


def evaluate_learned(
    candidate: Candidate,
    data: dict[str, ScorerData],
    labels: np.ndarray,
    hearings: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ExplorationConfig,
) -> tuple[list[float], list[float], list[float]]:
    matrix = feature_matrix(candidate, data)
    aucs, kappas, chosen = [], [], []
    for fit, held in folds:
        scaler, model, c, _ = fit_learned(matrix[fit], labels[fit], hearings[fit], config)
        fit_scores = model.predict_proba(scaler.transform(matrix[fit]))[:, 1]
        held_scores = model.predict_proba(scaler.transform(matrix[held]))[:, 1]
        aucs.append(float(roc_auc_score(labels[held], held_scores)))
        optimum = max_f1_not_inferable_optimum(fit_scores, labels[fit])
        kappas.append(
            float("nan")
            if optimum is None
            else binary_metrics(labels[held], held_scores >= optimum["threshold"])["cohen_kappa"]
        )
        chosen.append(c)
    return aucs, kappas, chosen


def summary_row(
    candidate: Candidate, aucs: list[float], kappas: list[float], n_splits: int
) -> Record:
    return {
        "candidate": candidate.key,
        "kind": candidate.kind,
        "scorer": candidate.scorer,
        "components": "+".join(candidate.components),
        "component_count": len(candidate.components)
        if candidate.kind != "learned"
        else len(candidate.features),
        "pool": candidate.pool,
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_roc_auc_se": float(np.std(aucs, ddof=1) / np.sqrt(n_splits)),
        "cv_roc_auc_min": float(np.min(aucs)),
        "cv_roc_auc_max": float(np.max(aucs)),
        "cv_kappa_mean": float(np.nanmean(kappas)),
        "cv_kappa_se": float(np.nanstd(kappas, ddof=1) / np.sqrt(n_splits)),
    }


def select(rows: list[Record]) -> tuple[Record, Record, list[Record]]:
    best = max(rows, key=lambda row: row["cv_roc_auc_mean"])
    floor = best["cv_roc_auc_mean"] - best["cv_roc_auc_se"]
    eligible = [row for row in rows if row["cv_roc_auc_mean"] >= floor]
    selected = min(
        eligible,
        key=lambda row: (
            KIND_ORDER[row["kind"]],
            row["component_count"] if row["kind"] == "panel" else 0,
            -row["cv_roc_auc_mean"],
        ),
    )
    return best, selected, eligible
