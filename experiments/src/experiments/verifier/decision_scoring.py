"""Decision question batteries and the per-opinion scores derived from their answers."""

from types import ModuleType

from experiments.verifier.decision import battery, panel
from experiments.verifier.decision.battery import (
    CONSENSUS,
    DERIVED_SCORES,
    PANEL,
    STACKED,
    Battery,
    BatteryError,
    BatteryQuestion,
    check_selection,
    order_pairs_used,
    parse_battery,
    score_names,
)
from experiments.verifier.decision.panel import (
    StackedModel,
    component_agreement,
    consensus_votes,
    fit_stacked,
    item_signals,
    opinion_scores,
    order_changes,
)

SOURCES: tuple[ModuleType, ...] = (
    battery,
    panel,
)

__all__ = [
    "Battery",
    "BatteryError",
    "BatteryQuestion",
    "CONSENSUS",
    "DERIVED_SCORES",
    "PANEL",
    "STACKED",
    "StackedModel",
    "check_selection",
    "component_agreement",
    "consensus_votes",
    "fit_stacked",
    "item_signals",
    "opinion_scores",
    "order_changes",
    "order_pairs_used",
    "parse_battery",
    "score_names",
    "SOURCES",
]
