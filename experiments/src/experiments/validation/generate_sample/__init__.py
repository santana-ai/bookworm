"""The stratified human validation sample of a UDV run and its blind repeat sheet.

``python -m experiments.validation.generate_sample`` runs ``cli.main``. The sheet and key helpers
are shared with the precision report, the udv_v2 analysis and the fuzzy matching review.
"""

from experiments.validation.generate_sample.blinding import blinding_section
from experiments.validation.generate_sample.cli import main
from experiments.validation.generate_sample.config import (
    Question,
    Stratum,
    ValidationConfig,
    load_validation_config,
    stream_rng,
)
from experiments.validation.generate_sample.items import (
    assign_item_ids,
    context_window,
    evidence_item,
    header_text,
    hearing_view,
    write_transcripts,
)
from experiments.validation.generate_sample.sheets import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    CSV_COLUMNS,
    REANNOTATION_CSV,
    REANNOTATION_KEY,
    canonical_sha256,
    ensure_new_dir,
    existing_precision_reports,
    load_key,
    read_annotation_csv,
    require_final_test,
    sheet_rows,
    validate_annotation,
    write_annotation_csv,
)
from experiments.validation.generate_sample.statistics import (
    criteria_feasibility,
    min_successes_to_pass,
    wilson_interval,
)

__all__ = [
    "ANNOTATION_CSV",
    "ANNOTATION_KEY",
    "CSV_COLUMNS",
    "REANNOTATION_CSV",
    "REANNOTATION_KEY",
    "Question",
    "Stratum",
    "ValidationConfig",
    "assign_item_ids",
    "blinding_section",
    "canonical_sha256",
    "context_window",
    "criteria_feasibility",
    "ensure_new_dir",
    "evidence_item",
    "existing_precision_reports",
    "header_text",
    "hearing_view",
    "load_key",
    "load_validation_config",
    "main",
    "min_successes_to_pass",
    "read_annotation_csv",
    "require_final_test",
    "sheet_rows",
    "stream_rng",
    "validate_annotation",
    "wilson_interval",
    "write_annotation_csv",
    "write_transcripts",
]
