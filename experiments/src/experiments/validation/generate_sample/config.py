"""The validation sample settings of ``configs/validation_sample.toml``."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.splits import SPLIT_NAMES
from experiments.common.udv_run import SUPPORT_TYPES, TIERS

Record = dict[str, Any]

QUESTIONS = ("trecho_sustenta", "pessoa_falou")
NO_EVIDENCE_SUPPORT = "none"
SEED_STREAMS = ("stratum_draw", "row_order", "repeat_draw", "repeat_order")
CSV_DELIMITERS = (";", ",", "\t")


@dataclass(frozen=True)
class Stratum:
    name: str
    question: str
    tiers: tuple[str, ...]
    support_types: tuple[str, ...]
    target: int


@dataclass(frozen=True)
class Question:
    name: str
    judgments: tuple[str, ...]
    better_passage_required: bool
    prompt: str


@dataclass(frozen=True)
class ValidationConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    sample_splits: tuple[str, ...]
    udv_dir: Path
    version: str
    seed: int
    seed_streams: dict[str, int]
    output_dir: Path
    transcripts_dir: Path
    item_id_prefix: str
    context_chars: int
    csv_delimiter: str
    csv_encoding: str
    strata: tuple[Stratum, ...]
    questions: dict[str, Question]
    better_passage_values: tuple[str, ...]
    repeat_items: int
    repeat_item_id_prefix: str
    repeat_questions: tuple[str, ...]
    repeat_min_hours: float
    confidence_level: float
    source: Record = field(default_factory=dict)


def check_split_names(splits: tuple[str, ...]) -> None:
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"splits must be distinct names among {SPLIT_NAMES}, got {splits}")


def check_stratum(stratum: Stratum) -> None:
    if stratum.question not in QUESTIONS:
        raise SystemExit(f"{stratum.name}: question must be one of {QUESTIONS}")
    if not set(stratum.tiers) <= set(TIERS):
        raise SystemExit(f"{stratum.name}: tiers must be among {TIERS}")
    if not set(stratum.support_types) <= {*SUPPORT_TYPES, NO_EVIDENCE_SUPPORT}:
        raise SystemExit(f"{stratum.name}: unknown support type in {stratum.support_types}")
    if stratum.target < 1:
        raise SystemExit(f"{stratum.name}: target must be >= 1")


def check_strata(strata: tuple[Stratum, ...]) -> None:
    """Refuse strata that repeat a name or claim the same (tier, support type) cell twice."""
    names = [stratum.name for stratum in strata]
    if len(set(names)) != len(names):
        raise SystemExit(f"stratum names must be unique: {names}")
    claimed: dict[tuple[str, str], str] = {}
    for stratum in strata:
        check_stratum(stratum)
        for cell in ((t, s) for t in stratum.tiers for s in stratum.support_types):
            if cell in claimed:
                raise SystemExit(f"{cell} is claimed by {claimed[cell]} and {stratum.name}")
            claimed[cell] = stratum.name


def parse_questions(annotation: Record) -> dict[str, Question]:
    questions = {
        name: Question(
            name=name,
            judgments=tuple(spec["judgments"]),
            better_passage_required=spec["better_passage_required"],
            prompt=spec["prompt"],
        )
        for name, spec in annotation["questions"].items()
    }
    if set(questions) != set(QUESTIONS):
        raise SystemExit(f"annotation.questions must define exactly {QUESTIONS}")
    return questions


def parse_strata(sample: Record) -> tuple[Stratum, ...]:
    strata = tuple(
        Stratum(
            name=spec["name"],
            question=spec["question"],
            tiers=tuple(spec["tiers"]),
            support_types=tuple(spec["support_types"]),
            target=spec["target"],
        )
        for spec in sample["strata"]
    )
    check_strata(strata)
    return strata


def check_sample_settings(sample: Record, repeat: Record, report: Record) -> None:
    streams = sample["seed_streams"]
    if set(streams) != set(SEED_STREAMS) or len(set(streams.values())) != len(streams):
        raise SystemExit(f"sample.seed_streams must give distinct values to {SEED_STREAMS}")
    repeat_questions = repeat["eligible_questions"]
    if not repeat_questions or not set(repeat_questions) <= set(QUESTIONS):
        raise SystemExit(f"repeat.eligible_questions must be among {QUESTIONS}")
    if sample["csv_delimiter"] not in CSV_DELIMITERS:
        raise SystemExit(f"sample.csv_delimiter must be one of {CSV_DELIMITERS}")
    if sample["item_id_prefix"] == repeat["item_id_prefix"]:
        raise SystemExit("sample and repeat item id prefixes must differ")
    if not 0 < report["confidence_level"] < 1:
        raise SystemExit("report.confidence_level must be in (0, 1)")


def load_validation_config(path: Path) -> ValidationConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sample, repeat, annotation = raw["sample"], raw["repeat"], raw["annotation"]
    questions = parse_questions(annotation)
    strata = parse_strata(sample)
    sample_splits = tuple(raw["splits"]["sample_splits"])
    check_split_names(sample_splits)
    check_sample_settings(sample, repeat, raw["report"])
    return ValidationConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        sample_splits=sample_splits,
        udv_dir=Path(raw["source"]["udv_dir"]),
        version=sample["version"],
        seed=sample["seed"],
        seed_streams=dict(sample["seed_streams"]),
        output_dir=Path(sample["output_dir"]),
        transcripts_dir=Path(sample["transcripts_dir"]),
        item_id_prefix=sample["item_id_prefix"],
        context_chars=sample["context_chars"],
        csv_delimiter=sample["csv_delimiter"],
        csv_encoding=sample["csv_encoding"],
        strata=strata,
        questions=questions,
        better_passage_values=tuple(annotation["better_passage_values"]),
        repeat_items=repeat["items"],
        repeat_item_id_prefix=repeat["item_id_prefix"],
        repeat_questions=tuple(repeat["eligible_questions"]),
        repeat_min_hours=float(repeat["min_hours_after_first_pass"]),
        confidence_level=raw["report"]["confidence_level"],
        source=raw,
    )


def stream_rng(config: ValidationConfig, stream: str) -> np.random.Generator:
    return np.random.default_rng([config.seed, config.seed_streams[stream]])
