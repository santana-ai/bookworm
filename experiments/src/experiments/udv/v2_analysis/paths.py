"""Default inputs and outputs of the udv_v2 analysis, relative to ``experiments/``."""

from pathlib import Path
from typing import Any

Record = dict[str, Any]

PACKAGE_DIR = Path(__file__).parent
DEFAULT_PATHS: Record = {
    "v1": "artifacts/udv/udv_v1.jsonl",
    "v2": "artifacts/udv/udv_v2.jsonl",
    "v1_coverage": "artifacts/udv/udv_v1_coverage.json",
    "v2_coverage": "artifacts/udv/udv_v2_coverage.json",
    "v1_verifier": "artifacts/udv/udv_v1_verifier.jsonl",
    "v2_verifier": "artifacts/udv/udv_v2_verifier.jsonl",
    "v2_verifier_report": "artifacts/udv/udv_v2_verifier_report.json",
    "cosine_calibration": "artifacts/calibration/threshold_v2.json",
    "verifier_calibration": "artifacts/calibration/udv_verifier_threshold_v1.json",
    "splits": "artifacts/splits/temporal_v1.json",
    "sample_dir": "artifacts/validation/human_validation_v1_udv_v1",
}
DEFAULT_ANALYSIS = "artifacts/udv/udv_v2_analysis.json"
DEFAULT_PLAN = "artifacts/udv/udv_v2_annotation_plan.json"
DEFAULT_CONFIG = Path("configs/validation_sample.toml")
PRECISION_OUTPUT_PATTERN = "artifacts/udv/udv_v2_precision_{status}_{stamp}.json"
