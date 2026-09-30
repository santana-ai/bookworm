"""Question battery parsing and the names of the signals it produces."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from experiments.verifier.decision.questions import (
    DecisionAnswer,
    DecisionQuestion,
    choice_question,
    noul_question,
    score_question,
)

Record = dict[str, Any]

PANEL = "panel"
CONSENSUS = "max.consensus"
STACKED = "max.stacked"
DERIVED_SCORES = (CONSENSUS, STACKED)


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
        options = cast(list[str], raw.get("options", base.get("options")))
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
