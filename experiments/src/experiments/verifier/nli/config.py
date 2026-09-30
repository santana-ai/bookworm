"""Verifier run configuration: scorers, declared comparisons and split selection."""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.common.splits import SPLIT_NAMES
from experiments.common.udv_run import UdvConfig
from experiments.common.udv_run import load_config as load_udv_config
from experiments.verifier.decision.battery import (
    DERIVED_SCORES,
    Battery,
    BatteryError,
    check_selection,
    parse_battery,
    score_names,
)
from experiments.verifier.decision.questions import DecisionModelError

Record = dict[str, Any]

DEVICES = ("auto", "cpu", "mps", "cuda")
SCORER_KINDS = ("nli", "cosine", "laya", "jev")
DECISION_KINDS = ("laya", "jev")
LANGUAGES = ("pt", "en")
COMPARISON_METRICS = ("roc_auc", "cohen_kappa")
REFERENCE_JUDGE = "reference_judge"
TEXT_SOURCES = ("nli_chunks", "lds_offsets")
THRESHOLD_RULES = ("youden", "max_f1_not_inferable", "fixed_probability")
COMPARED_RULE = "max_f1_not_inferable"
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
CHUNK_AGGREGATES = {"entailment": "max", "not_contradiction": "min"}
COSINE_SCORES = ("max.cosine", "sentence_max.cosine")


@dataclass(frozen=True)
class ScorerSpec:
    key: str
    kind: str
    name: str
    revision: str
    labels: tuple[str, ...]
    max_length: int
    dtype: str
    batch_size: int
    probe_expected: tuple[str, ...]
    scores: tuple[str, ...]
    source: Record = field(default_factory=dict)
    language: str = "pt"
    questions: tuple[str, ...] = ()
    concatenated: bool = True
    derived: tuple[str, ...] = ()
    translation_model: str | None = None


@dataclass(frozen=True)
class ComparisonFamily:
    name: str
    question: str
    comparisons: tuple[str, ...]


@dataclass(frozen=True)
class RunDeclaration:
    name: str
    scorers: tuple[str, ...]
    imported: dict[str, str]
    primary_system: str
    primary_metric: str
    comparisons: tuple[Record, ...]
    source: Record = field(default_factory=dict)
    families: tuple[ComparisonFamily, ...] = ()
    twins: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierConfig:
    lds_path: Path
    lds_sha256: str
    nli_path: Path
    nli_sha256: str
    benchmark_path: Path
    benchmark_report_path: Path
    manifest_path: Path
    fit_splits: tuple[str, ...]
    evaluate_splits: tuple[str, ...]
    final_test_splits: tuple[str, ...]
    text_source: str
    concat_separator: str
    udv: UdvConfig
    cache_dir: Path
    scorers: dict[str, ScorerSpec]
    probe_premises: tuple[str, ...]
    probe_hypotheses: tuple[str, ...]
    primary_system: str
    primary_metric: str
    reference_cosine_system: str
    threshold_rules: tuple[str, ...]
    fixed_probability_threshold: float
    bootstrap_samples: int
    confidence_level: float
    evaluation_seed: int
    pairs_default_scorers: tuple[str, ...]
    progress_every: int
    seed: int
    device: str
    hf_hub_offline: bool
    output_dir: Path
    source: Record = field(default_factory=dict)
    battery: Battery | None = None
    translation_config_path: Path | None = None
    decision_cache_dir: Path = Path("artifacts/cache/decision_models")
    evaluation_scorers: tuple[str, ...] = ()
    declarations: dict[str, RunDeclaration] = field(default_factory=dict)


def expected_nli_scores(labels: tuple[str, ...]) -> tuple[str, ...]:
    names = [ENTAILMENT] + (["not_contradiction"] if CONTRADICTION in labels else [])
    per_chunk = [f"{CHUNK_AGGREGATES[name]}.{name}" for name in names]
    return (*per_chunk, *(f"concatenated.{name}" for name in names))


JEV_FIELDS = (
    "timeout_seconds",
    "max_retries",
    "backoff_initial_seconds",
    "backoff_max_seconds",
    "requests_per_minute",
    "max_concurrency",
    "price_usd_per_million_input_tokens",
)


def scorer_translation_model(key: str, raw: Record, language: str) -> str | None:
    model = raw.get("translation_model")
    if language == "en" and not isinstance(model, str):
        raise SystemExit(f"scorers.{key}: language en needs translation_model (nllb or m2m100)")
    if language != "en" and model is not None:
        raise SystemExit(f"scorers.{key}: translation_model is only for language en")
    return model


def parse_decision_scorer(
    key: str, raw: Record, battery: Battery | None, language: str
) -> ScorerSpec:
    if battery is None:
        raise SystemExit(f"scorers.{key}: kind {raw['kind']} needs a [decision_battery] table")
    try:
        selected = check_selection(battery, raw["questions"])
    except BatteryError as error:
        raise SystemExit(f"scorers.{key}.questions: {error}") from error
    concatenated = bool(raw["concatenated"])
    scores = score_names(battery, selected, concatenated)
    if "scores" in raw and set(raw["scores"]) != set(scores):
        raise SystemExit(f"scorers.{key}.scores must be {scores}")
    probe_question = battery.questions[selected[0]].question
    expected = tuple(raw["probe_expected"])
    if probe_question.type != "choice" or not set(expected) <= set(probe_question.option_names):
        raise SystemExit(
            f"scorers.{key}.probe_expected must name options of {probe_question.key}, "
            "the first selected question"
        )
    if raw["kind"] == "laya":
        max_length, dtype, batch_size = raw["max_length"], "float32", raw["batch_size"]
        if not isinstance(raw.get("subfolder"), str) or "head_max_length" not in raw:
            raise SystemExit(f"scorers.{key} needs subfolder and head_max_length")
    else:
        missing = [name for name in JEV_FIELDS if name not in raw]
        if missing:
            raise SystemExit(f"scorers.{key} lacks {missing}")
        max_length, dtype, batch_size = 0, "", raw["max_concurrency"]
    return ScorerSpec(
        key,
        raw["kind"],
        raw["name"],
        raw["revision"],
        (),
        max_length,
        dtype,
        batch_size,
        expected,
        scores,
        raw,
        language=language,
        questions=selected,
        concatenated=concatenated,
        derived=DERIVED_SCORES,
        translation_model=scorer_translation_model(key, raw, language),
    )


def parse_scorer(key: str, raw: Record, udv: UdvConfig, battery: Battery | None) -> ScorerSpec:
    kind = raw["kind"]
    if kind not in SCORER_KINDS:
        raise SystemExit(f"scorers.{key}.kind must be one of {SCORER_KINDS}")
    language = raw.get("language", "pt")
    if language not in LANGUAGES:
        raise SystemExit(f"scorers.{key}.language must be one of {LANGUAGES}")
    if kind in DECISION_KINDS:
        return parse_decision_scorer(key, raw, battery, language)
    scores = tuple(raw["scores"])
    if kind == "cosine":
        if language != "pt":
            raise SystemExit(f"scorers.{key}: the cosine scorer reads the Portuguese text only")
        if (raw["name"], raw["revision"]) != (udv.model_name, udv.model_revision):
            raise SystemExit(f"scorers.{key} differs from the production encoder in udv.toml")
        if set(scores) != set(COSINE_SCORES):
            raise SystemExit(f"scorers.{key}.scores must be {COSINE_SCORES}")
        return ScorerSpec(
            key,
            kind,
            raw["name"],
            raw["revision"],
            (),
            raw["max_seq_length"],
            "",
            udv.batch_size,
            (),
            scores,
            raw,
        )
    labels = tuple(raw["labels"])
    if ENTAILMENT not in labels or len(set(labels)) != len(labels):
        raise SystemExit(f"scorers.{key}.labels must be distinct and include {ENTAILMENT!r}")
    if set(scores) != set(expected_nli_scores(labels)):
        raise SystemExit(f"scorers.{key}.scores must be {expected_nli_scores(labels)}")
    expected = tuple(raw["probe_expected"])
    if not set(expected) <= set(labels):
        raise SystemExit(f"scorers.{key}.probe_expected names a label outside {labels}")
    return ScorerSpec(
        key,
        kind,
        raw["name"],
        raw["revision"],
        labels,
        raw["max_length"],
        raw["dtype"],
        raw["batch_size"],
        expected,
        scores,
        raw,
        language=language,
        translation_model=scorer_translation_model(key, raw, language),
    )


def check_splits(splits: Record) -> None:
    named = [*splits["fit"], *splits["evaluate"], *splits["final_test"]]
    if not set(named) <= set(SPLIT_NAMES) or len(set(named)) != len(named):
        raise SystemExit("splits.fit, evaluate and final_test must be distinct known split names")
    if "test" in splits["fit"] or "test" in splits["evaluate"]:
        raise SystemExit("test may only appear in splits.final_test")
    if not splits["fit"] or not splits["evaluate"]:
        raise SystemExit("splits.fit and splits.evaluate must not be empty")


def declared_systems(scorers: dict[str, ScorerSpec]) -> set[str]:
    return {
        f"{key}.{score}" for key, spec in scorers.items() for score in (*spec.scores, *spec.derived)
    }


def parse_declaration(
    name: str, raw: Record, scorers: dict[str, ScorerSpec], evaluation: Record
) -> RunDeclaration:
    own = tuple(raw["scorers"])
    imported = dict(raw.get("imported", {}))
    unknown = [key for key in (*own, *imported) if key not in scorers]
    if unknown or set(own) & set(imported):
        raise SystemExit(f"declarations.{name}: unknown or doubly listed scorers {unknown}")
    systems = declared_systems({key: scorers[key] for key in (*own, *imported)})
    named = [raw["primary_system"], evaluation["reference_cosine_system"]]
    comparisons = tuple(raw.get("comparisons", ()))
    for comparison in comparisons:
        if comparison["metric"] not in COMPARISON_METRICS:
            raise SystemExit(f"declarations.{name}.{comparison['name']}: unknown metric")
        if (comparison["reference"] == REFERENCE_JUDGE) != (comparison["metric"] == "cohen_kappa"):
            raise SystemExit(
                f"declarations.{name}.{comparison['name']}: cohen_kappa is compared with the "
                "reference judge and roc_auc with another system"
            )
        named.append(comparison["system"])
        if comparison["reference"] != REFERENCE_JUDGE:
            named.append(comparison["reference"])
    missing = [system for system in named if system not in systems]
    if missing:
        raise SystemExit(f"declarations.{name} names undeclared systems {missing}")
    if len({c["name"] for c in comparisons}) != len(comparisons):
        raise SystemExit(f"declarations.{name}: comparison names must be unique")
    return RunDeclaration(
        name=name,
        scorers=own,
        imported=imported,
        primary_system=raw["primary_system"],
        primary_metric=raw["primary_metric"],
        comparisons=comparisons,
        source=raw,
        families=parse_families(name, raw, [c["name"] for c in comparisons]),
        twins=parse_twins(name, raw, scorers, own),
    )


def parse_twins(
    name: str, raw: Record, scorers: dict[str, ScorerSpec], own: tuple[str, ...]
) -> dict[str, str]:
    twins = dict(raw.get("twins", {}))
    for twin, first in twins.items():
        if twin not in own or first not in own or twin == first:
            raise SystemExit(
                f"declarations.{name}.twins: {twin} and {first} must be two scorers of this run"
            )
        a, b = scorers[twin], scorers[first]
        same = dataclasses.replace(
            a, key=b.key, translation_model=b.translation_model, source=b.source
        )
        if a.translation_model == b.translation_model or same != b:
            raise SystemExit(
                f"declarations.{name}.twins: {twin} must equal {first} in every value except "
                "its translation model"
            )
    if len(set(twins.values())) != len(twins) or set(twins) & set(twins.values()):
        raise SystemExit(f"declarations.{name}.twins: each scorer has at most one twin")
    local = raw.get("compute", {}).get("local_scorers")
    if local is not None:
        uneven = [twin for twin, first in twins.items() if (twin in local) != (first in local)]
        if uneven:
            raise SystemExit(
                f"declarations.{name}.compute.local_scorers must list both scorers of the twins "
                f"{uneven} or neither"
            )
    return twins


def parse_families(name: str, raw: Record, comparisons: list[str]) -> tuple[ComparisonFamily, ...]:
    if "families" not in raw:
        return (ComparisonFamily("all", "every declared comparison", tuple(comparisons)),)
    order = tuple(raw["family_order"])
    tables = raw["families"]
    if set(order) != set(tables) or len(set(order)) != len(order):
        raise SystemExit(f"declarations.{name}: family_order must list every family once")
    members = [member for family in order for member in tables[family]["comparisons"]]
    if len(set(members)) != len(members) or set(members) != set(comparisons):
        raise SystemExit(f"declarations.{name}: every comparison must be in exactly one family")
    return tuple(
        ComparisonFamily(family, tables[family]["question"], tuple(tables[family]["comparisons"]))
        for family in order
    )


def check_evaluation(raw: Record, scorers: dict[str, ScorerSpec]) -> None:
    systems = declared_systems(scorers)
    for name in ("primary_system", "reference_cosine_system"):
        if raw[name] not in systems:
            raise SystemExit(f"evaluation.{name} {raw[name]!r} is not a declared system")
    if not set(raw["threshold_rules"]) <= set(THRESHOLD_RULES):
        raise SystemExit(f"evaluation.threshold_rules must be among {THRESHOLD_RULES}")
    if COMPARED_RULE not in raw["threshold_rules"]:
        raise SystemExit(f"evaluation.threshold_rules must include {COMPARED_RULE}")
    if raw["bootstrap_unit"] != "hearing" or raw["bootstrap_samples"] < 1:
        raise SystemExit("evaluation.bootstrap_unit must be hearing, with at least 1 sample")
    if not 0 < raw["confidence_level"] < 1:
        raise SystemExit("evaluation.confidence_level must be in (0, 1)")


def load_config(config_path: Path) -> VerifierConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    check_splits(raw["splits"])
    if raw["premise"]["text_source"] not in TEXT_SOURCES:
        raise SystemExit(f"premise.text_source must be one of {TEXT_SOURCES}")
    udv = load_udv_config(Path(raw["encoder"]["udv_config_path"]))
    if udv.expected_sha256 != raw["dataset"]["lds_sha256"]:
        raise SystemExit("udv.toml and nli_verifier.toml gate different LDS files")
    try:
        battery = parse_battery(raw["decision_battery"]) if "decision_battery" in raw else None
    except (BatteryError, DecisionModelError) as error:
        raise SystemExit(f"decision_battery: {error}") from error
    scorers = {key: parse_scorer(key, spec, udv, battery) for key, spec in raw["scorers"].items()}
    evaluation = raw["evaluation"]
    check_evaluation(evaluation, scorers)
    evaluation_scorers = tuple(evaluation.get("scorers", scorers))
    if not set(evaluation_scorers) <= set(scorers):
        raise SystemExit("evaluation.scorers names an unknown scorer")
    declarations = {
        name: parse_declaration(name, table, scorers, evaluation)
        for name, table in raw.get("declarations", {}).items()
    }
    probes = raw["label_probes"]
    if len(probes["premises"]) != len(probes["hypotheses"]):
        raise SystemExit("label_probes.premises and hypotheses must have the same length")
    for spec in scorers.values():
        if spec.kind != "cosine" and len(spec.probe_expected) != len(probes["premises"]):
            raise SystemExit(f"scorers.{spec.key}.probe_expected must have one label per probe")
    if any(spec.language == "en" for spec in scorers.values()) and "translation" not in raw:
        raise SystemExit("scorers with language = en need a [translation] table")
    defaults = tuple(raw["pairs"]["default_scorers"])
    if not set(defaults) <= set(scorers):
        raise SystemExit("pairs.default_scorers names an unknown scorer")
    return VerifierConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        nli_path=Path(raw["dataset"]["nli_path"]),
        nli_sha256=raw["dataset"]["nli_sha256"],
        benchmark_path=Path(raw["benchmark"]["path"]),
        benchmark_report_path=Path(raw["benchmark"]["report_path"]),
        manifest_path=Path(raw["splits"]["manifest_path"]),
        fit_splits=tuple(raw["splits"]["fit"]),
        evaluate_splits=tuple(raw["splits"]["evaluate"]),
        final_test_splits=tuple(raw["splits"]["final_test"]),
        text_source=raw["premise"]["text_source"],
        concat_separator=raw["premise"]["concat_separator"],
        udv=udv,
        cache_dir=Path(raw["encoder"]["cache_dir"]),
        scorers=scorers,
        probe_premises=tuple(probes["premises"]),
        probe_hypotheses=tuple(probes["hypotheses"]),
        primary_system=evaluation["primary_system"],
        primary_metric=evaluation["primary_metric"],
        reference_cosine_system=evaluation["reference_cosine_system"],
        threshold_rules=tuple(evaluation["threshold_rules"]),
        fixed_probability_threshold=evaluation["fixed_probability_threshold"],
        bootstrap_samples=evaluation["bootstrap_samples"],
        confidence_level=evaluation["confidence_level"],
        evaluation_seed=evaluation["seed"],
        pairs_default_scorers=defaults,
        progress_every=raw["run"]["progress_every_batches"],
        seed=raw["run"]["seed"],
        device=raw["run"]["device"],
        hf_hub_offline=bool(raw["run"]["hf_hub_offline"]),
        output_dir=Path(raw["run"]["output_dir"]),
        source=raw,
        battery=battery,
        translation_config_path=(
            Path(raw["translation"]["config_path"]) if "translation" in raw else None
        ),
        decision_cache_dir=Path(
            raw.get("decision_models", {}).get("cache_dir", "artifacts/cache/decision_models")
        ),
        evaluation_scorers=evaluation_scorers,
        declarations=declarations,
    )


def scored_splits(config: VerifierConfig, final_test: bool) -> tuple[str, ...]:
    extra = config.final_test_splits if final_test else ()
    return (*config.fit_splits, *config.evaluate_splits, *extra)


def narrowed_splits(
    config: VerifierConfig, final_test: bool, requested: list[str] | None
) -> tuple[str, ...]:
    allowed = scored_splits(config, final_test)
    if requested is None:
        return allowed
    refused = [split for split in requested if split not in allowed]
    if refused:
        raise SystemExit(
            f"--splits {refused} are outside {list(allowed)} (test splits need --final-test)"
        )
    return tuple(split for split in allowed if split in requested)


def evaluated_splits(config: VerifierConfig, final_test: bool) -> tuple[str, ...]:
    extra = config.final_test_splits if final_test else ()
    return (*config.evaluate_splits, *extra)


def selected_scorers(config: VerifierConfig, requested: list[str] | None) -> list[ScorerSpec]:
    keys = requested if requested is not None else list(config.scorers)
    unknown = [key for key in keys if key not in config.scorers]
    if unknown:
        raise SystemExit(f"unknown scorers {unknown}; configured: {list(config.scorers)}")
    return [config.scorers[key] for key in keys]
