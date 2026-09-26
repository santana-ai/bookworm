from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.decision_models import (
    DecisionAnswer,
    DecisionQuestion,
    choice_question,
    noul_question,
    score_question,
)
from utils.retrieval_stats import holm

Record = dict[str, Any]

PANEL = "panel"
CONSENSUS = "max.consensus"
STACKED = "max.stacked"
DERIVED_SCORES = (CONSENSUS, STACKED)
MISSING_P_VALUE = 1.0
MIN_HEARINGS_FOR_P_VALUE = 2


class BatteryError(ValueError):
    pass


@dataclass(frozen=True)
class BatteryQuestion:
    question: DecisionQuestion
    support_option: str | None = None
    against_option: str | None = None
    against_signal: str | None = None
    reverses: str | None = None

    def support(self, answer: DecisionAnswer) -> float:
        question = self.question
        if question.type == "choice":
            return float(answer.probabilities[str(self.support_option)])
        if question.type == "noul":
            return float(answer.probabilities["true"])
        return answer.expected_level() / (len(question.option_names) - 1)

    def against(self, answer: DecisionAnswer) -> float:
        return 1.0 - float(answer.probabilities[str(self.against_option)])


@dataclass(frozen=True)
class Battery:
    questions: dict[str, BatteryQuestion]
    order_pairs: dict[str, tuple[str, ...]]
    panel_components: tuple[str, ...]
    stacked_features: tuple[str, ...]
    stacked_c: float
    stacked_seed: int
    consensus_rule: str
    state_keys: tuple[str, ...]
    source: Record = field(default_factory=dict)


def battery_question(key: str, raw: Record, tables: Record) -> BatteryQuestion:
    base = tables[raw["same_as"]] if "same_as" in raw else raw
    kind = base["type"]
    if raw.get("type", kind) != kind:
        raise BatteryError(f"decision_battery.{key}: type differs from {raw['same_as']}")
    instructions = base["instructions"]
    if kind == "choice":
        options = raw.get("options", base.get("options"))
        criteria = base["criteria"]
        if set(options) != set(criteria):
            raise BatteryError(f"decision_battery.{key}: options {options} != criteria keys")
        question = choice_question(key, instructions, [(name, criteria[name]) for name in options])
        support = raw.get("support_option")
        if support not in options:
            raise BatteryError(f"decision_battery.{key}: support_option must be one of {options}")
        against = raw.get("against_option")
        if against is not None and against not in options:
            raise BatteryError(f"decision_battery.{key}: against_option must be one of {options}")
        if (against is None) != (raw.get("against_signal") is None):
            raise BatteryError(f"decision_battery.{key}: against_option needs against_signal")
        return BatteryQuestion(
            question, support, against, raw.get("against_signal"), raw.get("same_as")
        )
    if kind == "noul":
        return BatteryQuestion(noul_question(key, instructions), reverses=raw.get("same_as"))
    if kind == "score":
        return BatteryQuestion(score_question(key, instructions, base["levels"]))
    raise BatteryError(f"decision_battery.{key}: unknown type {kind!r}")


def parse_battery(raw: Record) -> Battery:
    keys = tuple(raw["questions"])
    missing = [key for key in keys if key not in raw]
    if missing:
        raise BatteryError(f"decision_battery lacks the tables {missing}")
    questions = {key: battery_question(key, raw[key], raw) for key in keys}
    aggregation = raw["aggregation"]
    pairs = {name: tuple(members) for name, members in aggregation["order_pairs"].items()}
    for name, members in pairs.items():
        if len(members) != 2 or not set(members) <= set(keys):
            raise BatteryError(f"order pair {name} must name two battery questions")
        forward, reverse = (questions[member] for member in members)
        if reverse.reverses != members[0]:
            raise BatteryError(f"order pair {name}: {members[1]} is not same_as {members[0]}")
        if forward.support_option != reverse.support_option:
            raise BatteryError(f"order pair {name}: the two questions support different options")
        if forward.question.option_names != tuple(reversed(reverse.question.option_names)):
            raise BatteryError(f"order pair {name}: the options are not in reversed order")
    components = tuple(aggregation["panel_components"])
    unknown = [c for c in components if c not in keys and c not in pairs]
    if unknown:
        raise BatteryError(f"panel_components names unknown signals {unknown}")
    features = tuple(aggregation["stacked_features"])
    if not set(features) <= set(keys):
        raise BatteryError("stacked_features must be battery questions")
    state_keys = tuple(raw["state_keys"])
    if state_keys != ("premise", "hypothesis"):
        raise BatteryError("decision_battery.state_keys must be [premise, hypothesis]")
    return Battery(
        questions=questions,
        order_pairs=pairs,
        panel_components=components,
        stacked_features=features,
        stacked_c=float(aggregation["stacked_c"]),
        stacked_seed=int(aggregation["stacked_seed"]),
        consensus_rule=aggregation["consensus_rule"],
        state_keys=state_keys,
        source=raw,
    )


def check_selection(battery: Battery, selected: Sequence[str]) -> tuple[str, ...]:
    unknown = [key for key in selected if key not in battery.questions]
    if unknown:
        raise BatteryError(f"questions {unknown} are not in the battery")
    ordered = tuple(key for key in battery.questions if key in selected)
    for key in ordered:
        reverses = battery.questions[key].reverses
        if reverses is not None and reverses not in ordered:
            raise BatteryError(f"{key} reverses {reverses}, which is not selected")
    for component in battery.panel_components:
        base = battery.order_pairs[component][0] if component in battery.order_pairs else component
        if base not in ordered:
            raise BatteryError(f"the panel needs {base}, which is not selected")
    return ordered


def order_pairs_used(battery: Battery, selected: Sequence[str]) -> dict[str, list[str]]:
    return {
        name: [member for member in members if member in selected]
        for name, members in battery.order_pairs.items()
        if members[0] in selected
    }


def support_signals(battery: Battery, selected: Sequence[str]) -> list[str]:
    return [*selected, *order_pairs_used(battery, selected), PANEL]


def against_signals(battery: Battery, selected: Sequence[str]) -> list[str]:
    names = [battery.questions[key].against_signal for key in selected]
    return [name for name in names if name is not None]


def score_names(battery: Battery, selected: Sequence[str], concatenated: bool) -> tuple[str, ...]:
    support = support_signals(battery, selected)
    against = against_signals(battery, selected)
    names = [*(f"max.{s}" for s in support), *(f"min.{a}" for a in against)]
    if concatenated:
        names += [f"concatenated.{signal}" for signal in (*support, *against)]
    return tuple(names)


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
    model = LogisticRegression(C=c, random_state=seed, max_iter=1000)
    model.fit(scaler.transform(matrix), labels.astype(int))
    return StackedModel(tuple(features), scaler, model)


def bootstrap_p_value(deltas: np.ndarray) -> float | None:
    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        return None
    below = int(np.sum(valid <= 0))
    above = int(np.sum(valid >= 0))
    return float(min(1.0, 2 * (min(below, above) + 1) / (len(valid) + 1)))


def apply_holm(entries: list[Record]) -> None:
    p_values = [
        MISSING_P_VALUE if entry.get("p_value") is None else float(entry["p_value"])
        for entry in entries
    ]
    for entry, adjusted in zip(entries, holm(p_values), strict=True):
        entry["p_holm"] = adjusted
        entry["p_holm_counts_missing_as_one"] = entry.get("p_value") is None
