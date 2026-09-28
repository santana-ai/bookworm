"""udv_v2 against udv_v1: the diff, the annotation plan, the supplementary sheet and precision.

``python -m experiments.udv.v2_analysis`` runs ``cli.main`` with the subcommands ``analyze``,
``score-annotation`` and ``supplement-sheet``.
"""

from experiments.udv.v2_analysis.analysis import change_kind, closing_words_found, command_analyze
from experiments.udv.v2_analysis.cli import main
from experiments.udv.v2_analysis.plan import annotation_plan, key_strata, stratum_population
from experiments.udv.v2_analysis.relations import (
    evidence_relation,
    item_relation,
    item_relations,
    load_inheritance,
)
from experiments.udv.v2_analysis.scoring import NO_JUDGMENTS, command_score, partial_judgments
from experiments.udv.v2_analysis.supplement import (
    SUPPLEMENT_KIND,
    SUPPLEMENT_NAME,
    command_supplement,
    load_supplement,
    reannotated_records,
    supplement_items,
)

__all__ = [
    "NO_JUDGMENTS",
    "SUPPLEMENT_KIND",
    "SUPPLEMENT_NAME",
    "annotation_plan",
    "change_kind",
    "closing_words_found",
    "command_analyze",
    "command_score",
    "command_supplement",
    "evidence_relation",
    "item_relation",
    "item_relations",
    "key_strata",
    "load_inheritance",
    "load_supplement",
    "main",
    "partial_judgments",
    "reannotated_records",
    "stratum_population",
    "supplement_items",
]
