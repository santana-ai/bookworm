"""The ``[calibration]`` section of a UDV config and the rows it reads."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bookworm import load_jsonl

from experiments.common.udv_run import UdvConfig

Record = dict[str, Any]

CALIBRATION_SPLITS_ALLOWED = ("train", "validation")
PAIR_RULES = ("legacy_random", "hard_negative", "masked_hard_negative")
PRIMARY_RULE = "masked_top1_youden"
RULES = (*PAIR_RULES, PRIMARY_RULE)
RULE_QUERY_KIND = {
    "legacy_random": "opinion",
    "hard_negative": "opinion",
    "masked_hard_negative": "masked_opinion",
    "masked_top1_youden": "masked_opinion",
}
BOOTSTRAP_UNITS = ("hearing", "query")
QUANTILE_KEYS = ("positive_quantile", "negative_quantile")


@dataclass(frozen=True)
class CalibrationConfig:
    version: str
    splits: tuple[str, ...]
    manifest_path: Path
    masked_benchmark_path: Path
    output_dir: Path
    primary_rule: str
    seed: int
    bootstrap_samples: int
    bootstrap_unit: str
    confidence_level: float
    negatives_per_query: int
    positive_quantile: float
    negative_quantile: float
    threshold_decimals: int
    source: Record = field(default_factory=dict)


def check_calibration_section(raw: Record) -> None:
    splits = tuple(raw["splits"])
    if "test" in splits:
        raise SystemExit("calibration.splits must never include test")
    if not splits or len(set(splits)) != len(splits):
        raise SystemExit("calibration.splits must list distinct split names")
    if any(split not in CALIBRATION_SPLITS_ALLOWED for split in splits):
        raise SystemExit(f"calibration.splits must be among {CALIBRATION_SPLITS_ALLOWED}")
    if raw["primary_rule"] not in RULES:
        raise SystemExit(f"calibration.primary_rule must be one of {RULES}")
    if raw["bootstrap_unit"] not in BOOTSTRAP_UNITS:
        raise SystemExit(f"calibration.bootstrap_unit must be one of {BOOTSTRAP_UNITS}")
    if raw["bootstrap_samples"] < 1 or raw["negatives_per_query"] < 1:
        raise SystemExit("calibration.bootstrap_samples and negatives_per_query must be >= 1")
    if not 0 < raw["confidence_level"] < 1:
        raise SystemExit("calibration.confidence_level must be in (0, 1)")
    if not all(0 <= raw[key] <= 1 for key in QUANTILE_KEYS):
        raise SystemExit("calibration quantiles must be in [0, 1]")


def load_calibration_config(udv_config: UdvConfig) -> CalibrationConfig:
    raw = udv_config.source.get("calibration")
    if raw is None:
        raise SystemExit("the [calibration] section is missing from the UDV config")
    check_calibration_section(raw)
    return CalibrationConfig(
        version=raw["version"],
        splits=tuple(raw["splits"]),
        manifest_path=Path(raw["manifest_path"]),
        masked_benchmark_path=Path(raw["masked_benchmark_path"]),
        output_dir=Path(raw["output_dir"]),
        primary_rule=raw["primary_rule"],
        seed=raw["seed"],
        bootstrap_samples=raw["bootstrap_samples"],
        bootstrap_unit=raw["bootstrap_unit"],
        confidence_level=raw["confidence_level"],
        negatives_per_query=raw["negatives_per_query"],
        positive_quantile=raw["positive_quantile"],
        negative_quantile=raw["negative_quantile"],
        threshold_decimals=raw["threshold_decimals"],
        source=raw,
    )


def select_calibration_hearings(
    lds: list[Record], split_of: dict[int, str], splits: tuple[str, ...]
) -> list[Record]:
    missing = [hearing["id"] for hearing in lds if hearing["id"] not in split_of]
    if missing:
        raise SystemExit(f"hearings missing from the split manifest: {missing}")
    return [hearing for hearing in lds if split_of[hearing["id"]] in splits]


def load_masked_rows(
    path: Path, split_of: dict[int, str], splits: tuple[str, ...]
) -> tuple[dict[int, list[Record]], Record]:
    """Masked-quote benchmark rows of the calibration splits, by hearing, and their counts."""
    by_hearing: dict[int, list[Record]] = {}
    read = 0
    outside = 0
    for row in load_jsonl(path):
        read += 1
        manifest_split = split_of.get(row["hearing_id"])
        if manifest_split != row["split"]:
            raise SystemExit(
                f"{row['id']}: split {row['split']!r} differs from the manifest {manifest_split!r}"
            )
        if row["split"] not in splits:
            outside += 1
            continue
        by_hearing.setdefault(row["hearing_id"], []).append(row)
    counts = {
        "rows_read": read,
        "rows_outside_calibration_splits": outside,
        "rows_in_calibration_splits": read - outside,
    }
    return by_hearing, counts
