"""Per-opinion signals, agreement between components and the stacked combiner."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.verifier.decision.battery import (
    PANEL,
    Battery,
    against_signals,
    order_pairs_used,
    score_names,
    support_signals,
)
from experiments.verifier.decision.questions import (
    DecisionAnswer,
)

Record = dict[str, Any]
STACKED_MAX_ITER = 1000


def item_signals(
    battery: Battery, selected: Sequence[str], answers: Mapping[str, DecisionAnswer]
) -> dict[str, float]:
    values = {key: battery.questions[key].support(answers[key]) for key in selected}
    for name, members in order_pairs_used(battery, selected).items():
        values[name] = float(np.mean([values[member] for member in members]))
    values[PANEL] = float(np.mean([values[c] for c in battery.panel_components]))
    for key in selected:
        spec = battery.questions[key]
        if spec.against_signal is not None:
            values[spec.against_signal] = spec.against(answers[key])
    return values


def opinion_scores(
    battery: Battery,
    selected: Sequence[str],
    items: Sequence[Mapping[str, float]],
    concatenated: Mapping[str, float] | None,
    with_concatenated: bool,
) -> dict[str, float]:
    support = support_signals(battery, selected)
    against = against_signals(battery, selected)
    if not items:
        return {name: 0.0 for name in score_names(battery, selected, with_concatenated)}
    scores = {f"max.{s}": max(item[s] for item in items) for s in support}
    scores |= {f"min.{a}": min(item[a] for item in items) for a in against}
    if concatenated is not None:
        scores |= {f"concatenated.{n}": float(concatenated[n]) for n in (*support, *against)}
    return scores


def order_changes(battery: Battery, selected: Sequence[str], rows: Sequence[Record]) -> Record:
    result: Record = {}
    for name, members in order_pairs_used(battery, selected).items():
        if len(members) < 2:
            result[name] = {"applicable": False, "reason": f"only {members} selected"}
            continue
        forward, reverse = members
        counts = {"chunk": [0, 0], "concatenated": [0, 0]}
        for row in rows:
            entries = [("chunk", item) for item in row["items"] if item is not None]
            if row.get("concatenated") is not None:
                entries.append(("concatenated", row["concatenated"]))
            for mode, entry in entries:
                answers = entry["answers"]
                counts[mode][0] += 1
                counts[mode][1] += answers[forward]["choice"] != answers[reverse]["choice"]
        pooled = [sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())]
        result[name] = {
            "applicable": True,
            "questions": list(members),
            "rule": "share of scored premises whose argmax option differs between the two orders",
            **{
                mode: {
                    "premises": total,
                    "changed": changed,
                    "change_rate": changed / total if total else None,
                }
                for mode, (total, changed) in {**counts, "pooled": pooled}.items()
            },
        }
    return result


def binary_kappa(first: np.ndarray, second: np.ndarray) -> float:
    total = len(first)
    if total == 0:
        return float("nan")
    observed = float(np.mean(first == second))
    p_first, p_second = float(first.mean()), float(second.mean())
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def component_agreement(predictions: Mapping[str, np.ndarray]) -> Record:
    names = list(predictions)
    stacked = np.vstack([predictions[name] for name in names])
    kappas = {
        f"{a}|{b}": binary_kappa(predictions[a], predictions[b]) for a, b in combinations(names, 2)
    }
    pairwise = {pair: None if np.isnan(value) else value for pair, value in kappas.items()}
    all_agree = np.all(stacked == stacked[0], axis=0) if len(names) else np.array([])
    return {
        "components": names,
        "pairwise_cohen_kappa": pairwise,
        "share_all_agree": float(all_agree.mean()) if len(all_agree) else None,
        "share_predicted_inferable": {name: float(predictions[name].mean()) for name in names},
    }


def consensus_votes(
    scores: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> np.ndarray:
    votes = np.vstack([scores[name] >= thresholds[name] for name in thresholds])
    return votes.mean(axis=0)


@dataclass(frozen=True)
class StackedModel:
    features: tuple[str, ...]
    scaler: StandardScaler
    model: LogisticRegression

    def predict(self, scores: Mapping[str, np.ndarray]) -> np.ndarray:
        matrix = np.column_stack([scores[name] for name in self.features])
        return self.model.predict_proba(self.scaler.transform(matrix))[:, 1]

    def describe(self) -> Record:
        return {
            "features": list(self.features),
            "coefficients_standardized": dict(
                zip(self.features, (float(c) for c in self.model.coef_[0]), strict=True)
            ),
            "intercept": float(self.model.intercept_[0]),
            "feature_mean": dict(
                zip(self.features, (float(m) for m in self.scaler.mean_), strict=True)
            ),
            "feature_scale": dict(
                zip(self.features, (float(s) for s in self.scaler.scale_), strict=True)
            ),
        }


def fit_stacked(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    features: Sequence[str],
    c: float,
    seed: int,
) -> StackedModel:
    matrix = np.column_stack([scores[name] for name in features])
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(C=c, random_state=seed, max_iter=STACKED_MAX_ITER)
    model.fit(scaler.transform(matrix), labels.astype(int))
    return StackedModel(tuple(features), scaler, model)
