"""E6: what approximate string matching would recover over the exact quote and name rules.

``python -m experiments.udv.fuzzy_matching`` runs it; the blind review sheets it writes are
scored by ``experiments.validation.fuzzy_review_precision``.
"""

from experiments.common.reporting import rounded
from experiments.udv.fuzzy_matching.config import (
    NAME_METRICS,
    QUOTE_METHODS,
    REVIEW_KINDS,
    FuzzyConfig,
    load_config,
    run_name_for,
)
from experiments.udv.fuzzy_matching.name_summary import name_decisions, name_rules, resolved_names
from experiments.udv.fuzzy_matching.quote_summary import accepted_result
from experiments.udv.fuzzy_matching.report import group_members
from experiments.udv.fuzzy_matching.review import DISPLAY_FIELDS, FILL_FIELDS
from experiments.udv.fuzzy_matching.sampling import bootstrap_ratio

__all__ = [
    "DISPLAY_FIELDS",
    "FILL_FIELDS",
    "NAME_METRICS",
    "QUOTE_METHODS",
    "REVIEW_KINDS",
    "FuzzyConfig",
    "accepted_result",
    "bootstrap_ratio",
    "group_members",
    "load_config",
    "name_decisions",
    "name_rules",
    "resolved_names",
    "rounded",
    "run_name_for",
]
