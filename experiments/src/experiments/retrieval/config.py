"""The retrieval harness config (``configs/retrieval_experiments.toml``), validated on load."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.retrieval.data import BENCHES, SPLIT_NAMES, UNIT_KINDS
from experiments.retrieval.specs import (
    FIT_SCOPES,
    RETRIEVER_KINDS,
    SPARSE_KINDS,
    decision_rerank_spec,
    retriever_ids,
)

Record = dict[str, Any]

WINDOW_KINDS = {"window2", "window3"}
MIN_WINDOW_SIZE = 2
QUEUE_STEP_KEYS = {"retrievers", "units", "benches", "device"}
VALIDATION_DEVICE = "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    default_splits: tuple[str, ...]
    benches: dict[str, Record]
    unit_kinds: tuple[str, ...]
    window_sizes: dict[str, int]
    retrievers: dict[str, Record]
    evaluation: Record
    cache: Record
    run: Record
    queue_steps: list[Record]
    source: Record = field(default_factory=dict)


def validate_retrievers(retrievers: dict[str, Record], evaluation: Record) -> None:
    for name, spec in retrievers.items():
        if spec["kind"] not in RETRIEVER_KINDS:
            raise SystemExit(f"retrievers.{name}.kind must be one of {RETRIEVER_KINDS}")
        if spec["kind"] in SPARSE_KINDS and not set(spec["fit_scopes"]) <= set(FIT_SCOPES):
            raise SystemExit(f"retrievers.{name}.fit_scopes must be among {FIT_SCOPES}")
    ids = retriever_ids(retrievers)
    for name, spec in retrievers.items():
        references = spec.get("components", []) + ([spec["base"]] if "base" in spec else [])
        unknown = [reference for reference in references if reference not in ids]
        if unknown:
            raise SystemExit(f"retrievers.{name} references unknown retrievers {unknown}")
    if evaluation["baseline_retriever"] not in ids:
        raise SystemExit("evaluation.baseline_retriever is not a configured retriever")


def validate_decision_retrievers(retrievers: dict[str, Record], cache: Record) -> None:
    for name, spec in retrievers.items():
        if spec["kind"] == "decision_rerank":
            decision_rerank_spec(name, spec, VALIDATION_DEVICE, Path(cache["decision_dir"]))


def validate_queue(
    steps: list[Record],
    retrievers: dict[str, Record],
    kinds: tuple[str, ...],
    benches: tuple[str, ...],
) -> None:
    ids = retriever_ids(retrievers)
    names = set(ids) | {base for base, _ in ids.values()}
    for number, step in enumerate(steps):
        if set(step) - QUEUE_STEP_KEYS or not step.get("retrievers"):
            raise SystemExit(f"queue.steps[{number}] takes retrievers, units, benches, device")
        unknown = [name for name in step["retrievers"] if name not in names]
        unknown += [kind for kind in step.get("units", []) if kind not in kinds]
        unknown += [bench for bench in step.get("benches", []) if bench not in benches]
        if unknown:
            raise SystemExit(f"queue.steps[{number}] names unknown items {unknown}")


def validate_splits(splits: tuple[str, ...]) -> None:
    if "test" in splits:
        raise SystemExit("splits.default must never include test")
    if not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"splits.default must be among {SPLIT_NAMES}")


def validate_units(units: Record) -> tuple[tuple[str, ...], dict[str, int]]:
    kinds = tuple(units["kinds"])
    if not set(kinds) <= set(UNIT_KINDS) or "sentence" not in kinds:
        raise SystemExit(f"units.kinds must be among {UNIT_KINDS} and include sentence")
    if units["stride"] != 1:
        raise SystemExit("units.stride must be 1")
    window_sizes = {kind: int(size) for kind, size in units["window_sizes"].items()}
    if set(window_sizes) != WINDOW_KINDS or any(
        size < MIN_WINDOW_SIZE for size in window_sizes.values()
    ):
        raise SystemExit("units.window_sizes must define window2 and window3 with sizes >= 2")
    return kinds, window_sizes


def load_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    default_splits = tuple(raw["splits"]["default"])
    validate_splits(default_splits)
    kinds, window_sizes = validate_units(raw["units"])
    if not set(raw["benchmarks"]) <= set(BENCHES):
        raise SystemExit(f"benchmarks must be among {BENCHES}")
    validate_retrievers(raw["retrievers"], raw["evaluation"])
    validate_decision_retrievers(raw["retrievers"], raw["cache"])
    if raw["evaluation"]["baseline_unit"] not in kinds:
        raise SystemExit("evaluation.baseline_unit must be one of units.kinds")
    validate_queue(raw["queue"]["steps"], raw["retrievers"], kinds, tuple(raw["benchmarks"]))
    return ExperimentConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        default_splits=default_splits,
        benches=raw["benchmarks"],
        unit_kinds=kinds,
        window_sizes=window_sizes,
        retrievers=raw["retrievers"],
        evaluation=raw["evaluation"],
        cache=raw["cache"],
        run=raw["run"],
        queue_steps=raw["queue"]["steps"],
        source=raw,
    )


@dataclass(frozen=True)
class RunPlan:
    """What a ``run`` command scores: resolved from the command line and the config."""

    run_name: str
    run_dir: Path
    splits: tuple[str, ...]
    retrievers: list[str]
    kinds: tuple[str, ...]
    benches: tuple[str, ...]
    final_test: bool
    limit_hearings: int | None
    limit_queries: int | None


@dataclass(frozen=True)
class SummaryPlan:
    """What a ``summarize`` command reads and which baselines it tests against."""

    run_name: str
    run_dir: Path
    splits: tuple[str, ...]
    final_test: bool
    baseline_unit: str
    baseline_retriever: str

    @property
    def baseline(self) -> tuple[str, str]:
        return self.baseline_unit, self.baseline_retriever


def parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_splits(
    requested: list[str] | None, final_test: bool, config: ExperimentConfig
) -> tuple[str, ...]:
    splits = tuple(requested) if requested else config.default_splits
    if not set(splits) <= set(SPLIT_NAMES) or len(set(splits)) != len(splits):
        raise SystemExit(f"--splits must list distinct names among {SPLIT_NAMES}")
    if "test" in splits and not final_test:
        raise SystemExit("the test split is refused without --final-test")
    return splits


def resolve_retrievers(requested: list[str] | None, config: ExperimentConfig) -> list[str]:
    ids = retriever_ids(config.retrievers)
    if not requested:
        return list(ids)
    selected: list[str] = []
    for name in requested:
        matches = [rid for rid, (base, _) in ids.items() if rid == name or base == name]
        if not matches:
            raise SystemExit(f"unknown retriever {name!r}; known: {sorted(ids)}")
        selected.extend(match for match in matches if match not in selected)
    return selected


def resolve_choice(
    requested: list[str] | None, allowed: tuple[str, ...], flag: str
) -> tuple[str, ...]:
    if not requested:
        return allowed
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise SystemExit(f"{flag} {unknown} not among {allowed}")
    return tuple(item for item in allowed if item in requested)
