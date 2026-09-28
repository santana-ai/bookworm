import argparse
import csv
import dataclasses
import functools
import gc
import hashlib
import json
import os
import platform
import re
import time
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import laya
import numpy as np
import sentence_transformers
import sklearn
import torch
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedConfig,
)

from utils import (
    build_nli_benchmark,
    build_udvs,
    cache_lock,
    calibrate_threshold,
    dataset_io,
    decision_models,
    decision_scoring,
    hub_offline,
    retrieval_stats,
    translation,
    udv_pipeline,
)
from utils.build_nli_benchmark import iter_opinions, judge_metrics, parse_judge_key
from utils.build_udvs import (
    UdvConfig,
    encode_with_cache,
    load_encoder,
    seed_everything,
    select_device,
)
from utils.build_udvs import load_config as load_udv_config
from utils.calibrate_threshold import (
    describe,
    interval,
    load_split_lookup,
    resample_rows,
    rounded,
    unit_groups,
    youden_optimum,
)
from utils.dataset_io import (
    load_gated_jsonl,
    load_jsonl,
    module_path,
    sha256_of_file,
    write_json,
    write_jsonl,
)
from utils.decision_models import (
    DEFAULT_API_KEY_ENV,
    JEV_ENDPOINT,
    DecisionAnswer,
    DecisionModel,
    DecisionModelError,
    DecisionQuestion,
    JevDecisionModel,
    JevSpec,
    LayaDecisionModel,
    LayaSpec,
    LayaTokenizer,
    ReplayDecisionModel,
    fetch_laya,
    laya_checkpoint_dir,
    missing_key_message,
)
from utils.decision_scoring import (
    CONSENSUS,
    DERIVED_SCORES,
    MIN_HEARINGS_FOR_P_VALUE,
    STACKED,
    Battery,
    BatteryError,
    apply_holm,
    bootstrap_p_value,
    check_selection,
    component_agreement,
    consensus_votes,
    fit_stacked,
    item_signals,
    opinion_scores,
    order_changes,
    order_pairs_used,
    parse_battery,
    score_names,
)
from utils.hub_offline import enforce_offline, offline_state, pinned_weights_file
from utils.translation import TranslationConfig, TranslationStore
from utils.udv_pipeline import normalize_whitespace, split_sentences

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
DEVICES = ("auto", "cpu", "mps", "cuda")
SCORER_KINDS = ("nli", "cosine", "laya", "jev")
DECISION_KINDS = ("laya", "jev")
LOCAL_KINDS = ("nli", "laya")
LANGUAGES = ("pt", "en")
SUBSET_RULES = ("random", "first")
DECISION_MODES = ("live", "replay")
COMPARISON_METRICS = ("roc_auc", "cohen_kappa")
REFERENCE_JUDGE = "reference_judge"
TRUNCATED_KEY = "premise"
DECISION_CHUNK = 256
TEXT_SOURCES = ("nli_chunks", "lds_offsets")
THRESHOLD_RULES = ("youden", "max_f1_not_inferable", "fixed_probability")
FITTED_RULES = ("youden", "max_f1_not_inferable")
RANKING_METRICS = ("roc_auc", "average_precision_inferable", "average_precision_not_inferable")
BINARY_METRICS = (
    "accuracy",
    "cohen_kappa",
    "macro_f1",
    "f1_not_inferable",
    "precision_not_inferable",
    "recall_not_inferable",
)
COMPARED_RANKING_METRICS = ("roc_auc", "average_precision_not_inferable")
COMPARED_BINARY_METRICS = ("cohen_kappa", "f1_not_inferable")
COMPARED_RULE = "max_f1_not_inferable"
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
SCORE_MINIMUM = {"entailment": 0.0, "not_contradiction": 0.0, "cosine": -1.0}
CHUNK_AGGREGATES = {"entailment": "max", "not_contradiction": "min"}
COSINE_SCORES = ("max.cosine", "sentence_max.cosine")
WHITESPACE_PATTERN = re.compile(r"\s+")
PAIR_STRING_FIELDS = ("pair_id", "query_id", "proposition")


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


@dataclass(frozen=True)
class PremiseUnit:
    unit_id: str
    hearing_id: int
    split: str
    hypothesis: str
    items: tuple[str, ...]


@dataclass(frozen=True)
class NliResult:
    logits: np.ndarray
    probabilities: np.ndarray
    tokens: int
    premise_tokens: int
    premise_tokens_kept: int
    truncated: bool


@dataclass
class LogitCache:
    path: Path
    signature: str
    entries: dict[str, list[float]]
    hits: int = 0
    computed: int = 0


@dataclass
class NliModel:
    spec: ScorerSpec
    tokenizer: Any
    model: Any
    device: str
    info: Record
    cache: LogitCache | None = None


@dataclass(frozen=True)
class TextVectors:
    vectors: dict[str, np.ndarray]
    sentence_lists: dict[str, tuple[list[str], bool]]


@dataclass(frozen=True)
class SplitData:
    split: str
    ids: list[str]
    labels: np.ndarray
    hearing_ids: np.ndarray
    judges: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]


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


def split_records(splits: tuple[str, ...], final_test: bool) -> Record:
    return {"splits": list(splits), "final_test_flag": final_test, "test_read": "test" in splits}


def check_benchmark_file(config: VerifierConfig) -> Record:
    with open(config.benchmark_report_path) as f:
        recorded = json.load(f)["artifact"]["sha256"]
    actual = sha256_of_file(config.benchmark_path)
    if actual != recorded:
        raise SystemExit(f"{config.benchmark_path}: sha256 {actual} != its report {recorded}")
    return {"path": str(config.benchmark_path), "sha256": actual}


def load_benchmark_rows(
    config: VerifierConfig, split_of: dict[int, str], splits: tuple[str, ...]
) -> list[Record]:
    rows = []
    for row in load_jsonl(config.benchmark_path):
        manifest_split = split_of.get(row["hearing_id"])
        if manifest_split != row["split"]:
            raise SystemExit(f"{row['id']}: split {row['split']!r} != manifest {manifest_split!r}")
        if row["split"] in splits:
            rows.append(row)
    return rows


def first_rows(rows: list[Record], limit: int) -> list[Record]:
    taken: Counter[str] = Counter()
    kept = []
    for row in rows:
        if taken[row["split"]] < limit:
            taken[row["split"]] += 1
            kept.append(row)
    return kept


def sample_rows(
    rows: list[Record], limit: int | None, seed: int, rule: str = "random"
) -> list[Record]:
    if limit is None:
        return rows
    if rule == "first":
        return first_rows(rows, limit)
    kept: set[str] = set()
    for index, split in enumerate(SPLIT_NAMES):
        members = [row["id"] for row in rows if row["split"] == split]
        if len(members) <= limit:
            kept.update(members)
            continue
        rng = np.random.default_rng([seed, index])
        kept.update(members[pick] for pick in rng.choice(len(members), size=limit, replace=False))
    return [row for row in rows if row["id"] in kept]


def load_nli_chunks(config: VerifierConfig, rows: list[Record]) -> dict[str, list[str]]:
    wanted = {row["id"]: row for row in rows}
    chunks: dict[str, list[str]] = {}
    for hearing, person_index, _, opinion_index, opinion in iter_opinions(
        load_gated_jsonl(config.nli_path, config.nli_sha256)
    ):
        row_id = f"nli-{hearing['id']}-{person_index}-{opinion_index}"
        if row_id not in wanted:
            continue
        if opinion["opiniao"] != wanted[row_id]["opinion"]:
            raise SystemExit(f"{row_id}: the NLI opinion differs from the benchmark row")
        chunks[row_id] = list(opinion["chunks_proximos"])
    missing = sorted(set(wanted) - set(chunks))
    if missing:
        raise SystemExit(f"benchmark rows missing from the NLI file: {missing[:10]}")
    return chunks


def load_transcripts(config: VerifierConfig, hearing_ids: set[int]) -> dict[int, str]:
    return {
        record["id"]: record["transcricao"]
        for record in load_gated_jsonl(config.lds_path, config.lds_sha256)
        if record["id"] in hearing_ids
    }


def rebuild_chunk(chunk: Record, transcript: str) -> str:
    if not chunk["located"]:
        return ""
    return "\n".join(transcript[s["start_char"] : s["end_char"]] for s in chunk["segments"])


def without_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub("", text)


def chunk_source_problem(chunk: Record, nli_text: str, rebuilt: str) -> str | None:
    if not chunk["located"]:
        return None if not nli_text.split() else "not_located_but_text"
    if without_whitespace(rebuilt) != without_whitespace(nli_text):
        return "text_differs"
    return None


def verify_chunk_sources(
    rows: list[Record], nli_chunks: dict[str, list[str]], transcripts: dict[int, str]
) -> Record:
    counts: Counter[str] = Counter()
    problems: list[Record] = []
    for row in rows:
        texts = nli_chunks[row["id"]]
        if len(texts) != len(row["chunks"]):
            problems.append({"id": row["id"], "problem": "chunk_count"})
            continue
        for chunk, text in zip(row["chunks"], texts, strict=True):
            rebuilt = rebuild_chunk(chunk, transcripts[row["hearing_id"]])
            counts["chunks"] += 1
            problem = chunk_source_problem(chunk, text, rebuilt)
            if problem is not None:
                problems.append(
                    {"id": row["id"], "position": chunk["position"], "problem": problem}
                )
            elif not chunk["located"]:
                counts["empty_not_located"] += 1
            elif normalize_whitespace(rebuilt) == normalize_whitespace(text):
                counts["identical_after_whitespace_normalization"] += 1
            else:
                counts["differ_only_in_whitespace"] += 1
    if problems:
        raise SystemExit(f"chunk text sources disagree: {problems[:10]}")
    return {
        "method": (
            "each located chunk is rebuilt from its benchmark offsets in the LDS transcript "
            "(segments joined by a line break) and compared with chunks_proximos after removing "
            "all whitespace; any other difference stops the run"
        ),
        "opinions": len(rows),
        "opinions_without_chunks": sum(1 for row in rows if not row["chunks"]),
        **dict(sorted(counts.items())),
    }


def benchmark_units(
    rows: list[Record],
    nli_chunks: dict[str, list[str]],
    transcripts: dict[int, str],
    text_source: str,
) -> list[PremiseUnit]:
    units = []
    for row in rows:
        if text_source == "nli_chunks":
            texts = nli_chunks[row["id"]]
        else:
            texts = [
                rebuild_chunk(chunk, transcripts[row["hearing_id"]]) for chunk in row["chunks"]
            ]
        units.append(
            PremiseUnit(
                unit_id=row["id"],
                hearing_id=row["hearing_id"],
                split=row["split"],
                hypothesis=normalize_whitespace(row["opinion"]),
                items=tuple(normalize_whitespace(text) for text in texts),
            )
        )
    return units


def pair_evidence(evidence: Any, line_number: int) -> tuple[str, ...]:
    if isinstance(evidence, str):
        return (normalize_whitespace(evidence),)
    if isinstance(evidence, list) and evidence and all(isinstance(e, str) for e in evidence):
        return tuple(normalize_whitespace(text) for text in evidence)
    raise SystemExit(f"line {line_number}: evidence must be a string or a non-empty string list")


def check_pair_row(row: Record, line_number: int) -> None:
    for name in PAIR_STRING_FIELDS:
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise SystemExit(f"line {line_number}: {name} must be a non-empty string")
    if type(row.get("hearing_id")) is not int:
        raise SystemExit(f"line {line_number}: hearing_id must be an integer")
    if "meta" in row and not isinstance(row["meta"], dict):
        raise SystemExit(f"line {line_number}: meta must be an object")


def load_pair_units(
    path: Path, split_of: dict[int, str], allowed: tuple[str, ...]
) -> tuple[list[PremiseUnit], list[Record]]:
    rows = load_jsonl(path)
    units: list[PremiseUnit] = []
    refused: Counter[str] = Counter()
    for line_number, row in enumerate(rows, start=1):
        check_pair_row(row, line_number)
        split = split_of.get(row["hearing_id"])
        if split is None:
            raise SystemExit(
                f"line {line_number}: hearing {row['hearing_id']} is not in the manifest"
            )
        if split not in allowed:
            refused[split] += 1
            continue
        units.append(
            PremiseUnit(
                unit_id=row["pair_id"],
                hearing_id=row["hearing_id"],
                split=split,
                hypothesis=normalize_whitespace(row["proposition"]),
                items=pair_evidence(row["evidence"], line_number),
            )
        )
    if refused:
        raise SystemExit(
            f"pairs from splits outside {list(allowed)} refused: {dict(refused)} "
            "(test pairs need --final-test)"
        )
    duplicated = [key for key, count in Counter(u.unit_id for u in units).items() if count > 1]
    if duplicated:
        raise SystemExit(f"pair_id values are not unique: {duplicated[:10]}")
    return units, rows


def distinct_items(items: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def concatenated_premise(unit: PremiseUnit, separator: str, concatenate_single: bool) -> str | None:
    nonempty = [item for item in unit.items if item]
    if not nonempty or (len(nonempty) == 1 and not concatenate_single):
        return None
    return separator.join(nonempty)


def label_values(labels: tuple[str, ...], values: np.ndarray) -> Record:
    return {label: float(value) for label, value in zip(labels, values, strict=True)}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def load_model_config(spec: ScorerSpec) -> tuple[Any, Record]:
    config_dict, _ = PreTrainedConfig.get_config_dict(
        spec.name, revision=spec.revision, local_files_only=True
    )
    raw = dict(config_dict.get("id2label") or {})
    cast = {str(index): str(name) for index, name in raw.items()}
    if cast != raw:
        config_dict["id2label"] = cast
        config_dict["label2id"] = {name: int(index) for index, name in cast.items()}
    model_type = config_dict.pop("model_type")
    return AutoConfig.for_model(model_type, **config_dict), {
        "raw_id2label": raw,
        "id2label_values_cast_to_str": cast != raw,
        "resolved_commit": config_dict.get("_commit_hash"),
    }


def check_label_names(raw_id2label: Record, spec: ScorerSpec) -> Record:
    id2label = {int(index): str(name) for index, name in raw_id2label.items()}
    if sorted(id2label) != list(range(len(spec.labels))):
        raise SystemExit(f"{spec.key}: model labels {id2label} do not match {spec.labels}")
    names = [id2label[index] for index in range(len(spec.labels))]
    named = not any(name.isdigit() or name.upper().startswith("LABEL_") for name in names)
    if named and [name.lower() for name in names] != [label.lower() for label in spec.labels]:
        raise SystemExit(f"{spec.key}: model id2label {names} contradicts configured {spec.labels}")
    return {
        "model_config_id2label": {str(index): name for index, name in sorted(id2label.items())},
        "configured_labels": list(spec.labels),
        "verified_against_model_config": named,
    }


def load_nli_model(spec: ScorerSpec, device: str) -> NliModel:
    started = time.perf_counter()
    model_config, loaded = load_model_config(spec)
    if loaded["resolved_commit"] != spec.revision:
        raise SystemExit(f"{spec.key}: resolved {loaded['resolved_commit']} != {spec.revision}")
    weights = pinned_weights_file(spec.name, spec.revision)
    if not weights["snapshot_is_pinned_revision"]:
        raise SystemExit(
            f"{spec.key}: weights file {weights['path']} is not in the pinned snapshot"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, config=model_config, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.name,
        revision=spec.revision,
        config=model_config,
        dtype=getattr(torch, spec.dtype),
        local_files_only=True,
        use_safetensors=weights["format"] == "safetensors",
    )
    model.to(device).eval()
    info = {
        "name": spec.name,
        "revision": spec.revision,
        "config_loading": loaded,
        "weights_file": {
            **weights,
            "loading": (
                "the first of model.safetensors, model.safetensors.index.json, pytorch_model.bin, "
                "pytorch_model.bin.index.json present in the pinned snapshot, loaded with "
                "local_files_only and use_safetensors set to its format"
            ),
        },
        "labels": check_label_names(loaded["raw_id2label"], spec),
        "max_length": spec.max_length,
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "tokenizer_class": type(tokenizer).__name__,
        "special_tokens_per_pair": tokenizer.num_special_tokens_to_add(pair=True),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "load_seconds": round(time.perf_counter() - started, 1),
    }
    return NliModel(spec, tokenizer, model, device, info)


def token_counts(tokenizer: Any, texts: list[str], special: bool) -> dict[str, int]:
    distinct = list(dict.fromkeys(texts))
    if not distinct:
        return {}
    encoded = tokenizer(distinct, add_special_tokens=special, truncation=False)["input_ids"]
    return {text: len(ids) for text, ids in zip(distinct, encoded, strict=True)}


def pair_token_arrays(
    nli: NliModel, requests: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    premise_counts = token_counts(nli.tokenizer, [premise for premise, _ in requests], False)
    hypothesis_counts = token_counts(
        nli.tokenizer, [hypothesis for _, hypothesis in requests], False
    )
    premises = np.array([premise_counts[p] for p, _ in requests], dtype=np.int64)
    hypotheses = np.array([hypothesis_counts[h] for _, h in requests], dtype=np.int64)
    totals = premises + hypotheses + nli.info["special_tokens_per_pair"]
    return premises, hypotheses, totals


def cache_signature(spec: ScorerSpec, device: str) -> str:
    return f"{spec.name}@{spec.revision}@{spec.dtype}@{spec.max_length}@{device}"


def request_key(signature: str, premise: str, hypothesis: str) -> str:
    return hashlib.sha256(f"{signature}\x1e{premise}\x1e{hypothesis}".encode()).hexdigest()


def open_logit_cache(config: VerifierConfig, spec: ScorerSpec, device: str) -> LogitCache:
    path = config.cache_dir / f"nli_{spec.key}_{spec.revision[:12]}_{device}.jsonl"
    try:
        cache_lock.acquire_writer_lock(path)
    except cache_lock.CacheLockedError as error:
        raise SystemExit(str(error)) from error
    signature = cache_signature(spec, device)
    entries: dict[str, list[float]] = {}
    if path.exists():
        for entry in load_jsonl(path):
            if entry["signature"] == signature and len(entry["logits"]) == len(spec.labels):
                entries[entry["key"]] = entry["logits"]
    return LogitCache(path=path, signature=signature, entries=entries)


def append_to_cache(cache: LogitCache, keys: list[str], logits: np.ndarray) -> None:
    cache.path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache.path, "a") as f:
        for key, values in zip(keys, logits, strict=True):
            record = {"signature": cache.signature, "key": key, "logits": values.tolist()}
            f.write(json.dumps(record) + "\n")
            cache.entries[key] = values.tolist()


def forward_batch(nli: NliModel, batch: list[tuple[str, str]]) -> np.ndarray:
    encoded = nli.tokenizer(
        [premise for premise, _ in batch],
        [hypothesis for _, hypothesis in batch],
        truncation="only_first",
        max_length=nli.spec.max_length,
        padding=True,
        return_tensors="pt",
    ).to(nli.device)
    with torch.inference_mode():
        output = nli.model(**encoded).logits
    return output.float().cpu().numpy().astype(np.float64)


def print_progress(nli: NliModel, done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0.0
    remaining = (total - done) / rate / 60 if rate else float("nan")
    print(
        f"  {nli.spec.key}: {done}/{total} pairs, {rate:.2f}/s, about {remaining:.1f} min left",
        flush=True,
    )


def forward_logits(
    nli: NliModel, requests: list[tuple[str, str]], order: np.ndarray, progress_every: int
) -> np.ndarray:
    cache = nli.cache
    if cache is None:
        raise SystemExit(f"{nli.spec.key}: no logit cache opened")
    keys = [request_key(cache.signature, premise, hypothesis) for premise, hypothesis in requests]
    missing = [int(row) for row in order if keys[row] not in cache.entries]
    started = time.perf_counter()
    for number, start in enumerate(range(0, len(missing), nli.spec.batch_size), start=1):
        rows = missing[start : start + nli.spec.batch_size]
        logits = forward_batch(nli, [requests[row] for row in rows])
        append_to_cache(cache, [keys[row] for row in rows], logits)
        if progress_every and number % progress_every == 0:
            print_progress(nli, start + len(rows), len(missing), started)
    cache.computed += len(missing)
    cache.hits += len(requests) - len(missing)
    return np.array([cache.entries[key] for key in keys], dtype=np.float64).reshape(
        len(requests), len(nli.spec.labels)
    )


def run_nli(
    nli: NliModel, requests: list[tuple[str, str]], progress_every: int = 0
) -> tuple[list[NliResult], Record]:
    premises, hypotheses, totals = pair_token_arrays(nli, requests)
    order = np.argsort(-totals, kind="stable")
    computed_before = nli.cache.computed if nli.cache is not None else 0
    started = time.perf_counter()
    logits = forward_logits(nli, requests, order, progress_every)
    seconds = time.perf_counter() - started
    computed = (nli.cache.computed if nli.cache is not None else 0) - computed_before
    probabilities = softmax(logits)
    room = np.maximum(nli.spec.max_length - hypotheses - nli.info["special_tokens_per_pair"], 0)
    kept = np.minimum(premises, room)
    results = [
        NliResult(
            logits=logits[row],
            probabilities=probabilities[row],
            tokens=int(totals[row]),
            premise_tokens=int(premises[row]),
            premise_tokens_kept=int(kept[row]),
            truncated=bool(totals[row] > nli.spec.max_length),
        )
        for row in range(len(requests))
    ]
    model_tokens = int(np.minimum(totals, nli.spec.max_length).sum())
    timing = {
        "requests": len(requests),
        "computed": computed,
        "cache_hits": len(requests) - computed,
        "seconds": round(seconds, 2),
        "computed_per_second": round(computed / seconds, 2) if seconds and computed else None,
        "model_input_tokens": model_tokens,
    }
    return results, timing


def portuguese_probes(config: VerifierConfig) -> list[tuple[str, str]]:
    return list(zip(config.probe_premises, config.probe_hypotheses, strict=True))


def run_label_probes(nli: NliModel, requests: list[tuple[str, str]]) -> Record:
    results, _ = run_nli(nli, requests)
    probes = []
    for (premise, hypothesis), expected, result in zip(
        requests, nli.spec.probe_expected, results, strict=True
    ):
        predicted = nli.spec.labels[int(result.probabilities.argmax())]
        probes.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "expected": expected,
                "predicted": predicted,
                "probabilities": label_values(nli.spec.labels, result.probabilities),
                "agrees": predicted == expected,
            }
        )
    return {
        "probes": probes,
        "agreeing": sum(1 for probe in probes if probe["agrees"]),
        "total": len(probes),
    }


def nli_requests(
    units: list[PremiseUnit], separator: str, concatenate_single: bool
) -> list[tuple[str, str]]:
    requests: dict[tuple[str, str], None] = {}
    for unit in units:
        for item in distinct_items(unit.items):
            requests[(item, unit.hypothesis)] = None
        concatenated = concatenated_premise(unit, separator, concatenate_single)
        if concatenated is not None:
            requests[(concatenated, unit.hypothesis)] = None
    return list(requests)


def per_item_scores(result: NliResult, labels: tuple[str, ...]) -> Record:
    scores = {ENTAILMENT: float(result.probabilities[labels.index(ENTAILMENT)])}
    if CONTRADICTION in labels:
        scores["not_contradiction"] = 1.0 - float(result.probabilities[labels.index(CONTRADICTION)])
    return scores


def chunk_scores(results: list[NliResult], labels: tuple[str, ...]) -> Record:
    per_item = [per_item_scores(result, labels) for result in results]
    scores: Record = {}
    for name, aggregate in CHUNK_AGGREGATES.items():
        if name not in per_item[0]:
            continue
        values = [item[name] for item in per_item]
        scores[f"{aggregate}.{name}"] = max(values) if aggregate == "max" else min(values)
    return scores


def concatenated_scores(result: NliResult, labels: tuple[str, ...]) -> Record:
    return {
        f"concatenated.{name}": value for name, value in per_item_scores(result, labels).items()
    }


def minimum_scores(names: list[str]) -> Record:
    return {name: SCORE_MINIMUM[name.split(".", 1)[1]] for name in names}


def unit_header(unit: PremiseUnit) -> Record:
    nonempty = sum(1 for item in unit.items if item)
    return {
        "id": unit.unit_id,
        "hearing_id": unit.hearing_id,
        "split": unit.split,
        "item_count": len(unit.items),
        "nonempty_items": nonempty,
        "no_premise": nonempty == 0,
    }


def nli_item_record(result: NliResult, labels: tuple[str, ...]) -> Record:
    return {
        "probabilities": label_values(labels, result.probabilities),
        "logits": label_values(labels, result.logits),
        "tokens": result.tokens,
        "truncated": result.truncated,
    }


def nli_unit_row(
    unit: PremiseUnit,
    results: dict[tuple[str, str], NliResult],
    spec: ScorerSpec,
    separator: str,
    concatenate_single: bool,
) -> Record:
    items = [
        nli_item_record(results[(item, unit.hypothesis)], spec.labels) if item else None
        for item in unit.items
    ]
    item_results = [results[(item, unit.hypothesis)] for item in distinct_items(unit.items)]
    text = concatenated_premise(unit, separator, concatenate_single)
    joined = results[(text, unit.hypothesis)] if text is not None else None
    concatenated = None
    if joined is not None:
        concatenated = {
            **nli_item_record(joined, spec.labels),
            "premise_tokens": joined.premise_tokens,
            "premise_tokens_kept": joined.premise_tokens_kept,
            "items_joined": sum(1 for item in unit.items if item),
        }
    if item_results:
        scores = chunk_scores(item_results, spec.labels)
        if joined is not None:
            scores |= concatenated_scores(joined, spec.labels)
    else:
        names = [n for n in spec.scores if concatenate_single or not n.startswith("concatenated.")]
        scores = minimum_scores(names)
    return {**unit_header(unit), "items": items, "concatenated": concatenated, "scores": scores}


def release_device(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def score_units_nli(
    units: list[PremiseUnit],
    spec: ScorerSpec,
    config: VerifierConfig,
    device: str,
    concatenate_single: bool,
    probe_pairs: list[tuple[str, str]],
) -> tuple[list[Record], Record]:
    nli = load_nli_model(spec, device)
    nli.cache = open_logit_cache(config, spec, device)
    nli.info["logit_cache"] = {
        "path": str(nli.cache.path),
        "entries_at_start": len(nli.cache.entries),
    }
    probes = run_label_probes(nli, probe_pairs)
    if probes["agreeing"] != probes["total"]:
        print(f"WARNING {spec.key}: {probes['agreeing']}/{probes['total']} label probes agree")
    requests = nli_requests(units, config.concat_separator, concatenate_single)
    results_list, timing = run_nli(nli, requests, config.progress_every)
    results = dict(zip(requests, results_list, strict=True))
    rows = [
        nli_unit_row(unit, results, spec, config.concat_separator, concatenate_single)
        for unit in units
    ]
    details = {"model": nli.info, "label_probes": probes, "timing": timing}
    del nli
    release_device(device)
    return rows, details


def cosine_encoder(config: VerifierConfig, spec: ScorerSpec, device: str) -> tuple[Any, UdvConfig]:
    udv = dataclasses.replace(config.udv, cache_dir=config.cache_dir)
    encoder = load_encoder(udv, device)
    if not huggingface_hub.is_offline_mode():
        raise SystemExit(f"{spec.key}: the encoder must be loaded in offline mode")
    if encoder.max_seq_length != spec.max_length:
        raise SystemExit(
            f"{spec.key}: max_seq_length {encoder.max_seq_length} != {spec.max_length}"
        )
    return encoder, udv


def encode_texts(
    encoder: SentenceTransformer, texts: list[str], udv: UdvConfig, device: str, label: str
) -> dict[str, np.ndarray]:
    distinct = list(dict.fromkeys(texts))
    embeddings = encode_with_cache(encoder, distinct, udv, device, label)
    return {text: embeddings[index] for index, text in enumerate(distinct)}


def item_sentences(item: str) -> tuple[list[str], bool]:
    sentences = split_sentences(item)
    return (sentences, False) if sentences else ([item], True)


def cosines(vector: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    return cosine_similarity(vector.reshape(1, -1), np.vstack(others)).flatten()


def cosine_item_record(
    item: str,
    hypothesis_vector: np.ndarray,
    vectors: TextVectors,
    tokens: dict[str, int],
    max_length: int,
) -> Record:
    sentences, fallback = vectors.sentence_lists[item]
    sentence_scores = cosines(hypothesis_vector, [vectors.vectors[s] for s in sentences])
    return {
        "cosine": float(cosines(hypothesis_vector, [vectors.vectors[item]])[0]),
        "tokens": tokens[item],
        "truncated": tokens[item] > max_length,
        "sentences": len(sentences),
        "sentences_truncated": sum(1 for s in sentences if tokens[s] > max_length),
        "sentence_max": float(sentence_scores.max()),
        "sentence_fallback": fallback,
    }


def cosine_unit_row(
    unit: PremiseUnit, vectors: TextVectors, tokens: dict[str, int], spec: ScorerSpec
) -> Record:
    hypothesis_vector = vectors.vectors[unit.hypothesis]
    records = {
        item: cosine_item_record(item, hypothesis_vector, vectors, tokens, spec.max_length)
        for item in distinct_items(unit.items)
    }
    if records:
        scores = {
            "max.cosine": max(record["cosine"] for record in records.values()),
            "sentence_max.cosine": max(record["sentence_max"] for record in records.values()),
        }
    else:
        scores = minimum_scores(list(spec.scores))
    return {
        **unit_header(unit),
        "items": [records[item] if item else None for item in unit.items],
        "concatenated": None,
        "scores": scores,
    }


def encode_units(
    units: list[PremiseUnit],
    encoder: SentenceTransformer,
    udv: UdvConfig,
    device: str,
    label: str,
) -> TextVectors:
    items = list(dict.fromkeys(item for unit in units for item in distinct_items(unit.items)))
    sentence_lists = {item: item_sentences(item) for item in items}
    sentence_texts = [s for listed, _ in sentence_lists.values() for s in listed]
    texts = [*(unit.hypothesis for unit in units), *items, *sentence_texts]
    return TextVectors(
        vectors=encode_texts(encoder, texts, udv, device, label),
        sentence_lists=sentence_lists,
    )


def score_units_cosine(
    units: list[PremiseUnit], spec: ScorerSpec, config: VerifierConfig, device: str, prefix: str
) -> tuple[list[Record], Record]:
    started = time.perf_counter()
    encoder, udv = cosine_encoder(config, spec, device)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = encode_units(units, encoder, udv, device, f"{prefix}_all")
    seconds = time.perf_counter() - started
    tokens = token_counts(encoder.tokenizer, list(vectors.vectors), True)
    rows = [cosine_unit_row(unit, vectors, tokens, spec) for unit in units]
    details = {
        "model": {
            "name": spec.name,
            "revision": spec.revision,
            "device": device,
            "max_seq_length": encoder.max_seq_length,
            "weights_file": pinned_weights_file(spec.name, spec.revision),
            "embedding_dimension": encoder.get_embedding_dimension(),
            "batch_size": udv.batch_size,
            "cache_dir": str(udv.cache_dir),
            "cache_label": f"{prefix}_all: distinct opinions, chunks and chunk sentences",
            "load_seconds": round(load_seconds, 1),
        },
        "label_probes": None,
        "timing": {
            "distinct_texts": len(vectors.vectors),
            "seconds": round(seconds, 2),
            "texts_per_second": round(len(vectors.vectors) / seconds, 1) if seconds else None,
        },
    }
    del encoder
    release_device(device)
    return rows, details


@dataclass(frozen=True)
class Translations:
    config: TranslationConfig
    stores: dict[str, TranslationStore]

    def store(self, spec: ScorerSpec) -> TranslationStore:
        if spec.translation_model not in self.stores:
            raise SystemExit(f"{spec.key}: no store opened for {spec.translation_model}")
        return self.stores[spec.translation_model]


def load_translation_config(config: VerifierConfig) -> TranslationConfig:
    if config.translation_config_path is None:
        raise SystemExit("language en needs [translation] config_path")
    translation_config = translation.load_config(config.translation_config_path)
    same_probes = (translation_config.probe_premises, translation_config.probe_hypotheses) == (
        config.probe_premises,
        config.probe_hypotheses,
    )
    if not same_probes or translation_config.nli_sha256 != config.nli_sha256:
        raise SystemExit(
            f"{config.translation_config_path} reads another E3 config: its NLI file or label "
            "probes differ from this one"
        )
    return translation_config


def open_translations(
    config: VerifierConfig, specs: list[ScorerSpec], cache_dir: Path | None
) -> Translations | None:
    models = list(dict.fromkeys(s.translation_model for s in specs if s.language == "en"))
    if not models:
        return None
    translation_config = load_translation_config(config)
    unknown = [
        f"{spec.key}: {spec.translation_model}"
        for spec in specs
        if spec.language == "en" and spec.translation_model not in translation_config.models
    ]
    if unknown:
        raise SystemExit(
            f"translation_model must be a condition of {config.translation_config_path} "
            f"{list(translation_config.models)}: {unknown}"
        )
    stores = {
        key: translation.open_store(translation_config, key, cache_dir=cache_dir) for key in models
    }
    return Translations(translation_config, cast(dict[str, TranslationStore], stores))


def translation_summary(
    config: VerifierConfig, translations: Translations, spec: ScorerSpec
) -> Record:
    path = config.translation_config_path
    model = translations.config.models[str(spec.translation_model)]
    return {
        "config": {"path": str(path), "sha256": sha256_of_file(path) if path else None},
        "model": translation.model_record(translations.config, model),
        "store": translations.store(spec).summary(),
        "rule": config.source["translation"]["rule"],
    }


def translation_texts(
    units: list[PremiseUnit], probes: list[tuple[str, str]], store: TranslationStore
) -> list[str]:
    opinions = [*(unit.hypothesis for unit in units), *(hypothesis for _, hypothesis in probes)]
    chunks = dict.fromkeys(
        [*(item for unit in units for item in unit.items if item), *(p for p, _ in probes)]
    )
    segments = [segment for chunk in chunks for segment in store.segmenter.segments(chunk)]
    return list(dict.fromkeys(normalize_whitespace(text) for text in [*opinions, *segments]))


def check_translations(
    spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig, store: TranslationStore
) -> Record:
    texts = translation_texts(units, portuguese_probes(config), store)
    missing = store.missing(texts)
    model = spec.translation_model
    if missing:
        raise SystemExit(
            f"{spec.key} (language en, translation model {model}, "
            f"{store.signature.get('model')}): {len(missing)} of {len(texts)} distinct texts of "
            f"the selected opinions and label probes have no translation in {store.path} "
            f"(signature {store.digest[:16]}); run python -m utils.translation translate --model "
            f"{model} for these splits, or point --translation-cache-dir at a cache that has them"
        )
    return {"translation_model": model, "distinct_texts": len(texts), "missing": 0}


def english_units(units: list[PremiseUnit], store: TranslationStore) -> list[PremiseUnit]:
    english = []
    for unit in units:
        items = tuple(
            normalize_whitespace(store.translate_chunk(item)) if item else "" for item in unit.items
        )
        if any(bool(pt) != bool(en) for pt, en in zip(unit.items, items, strict=True)):
            raise SystemExit(f"{unit.unit_id}: a chunk and its translation differ in emptiness")
        hypothesis = normalize_whitespace(store.translate_opinion(unit.hypothesis))
        english.append(PremiseUnit(unit.unit_id, unit.hearing_id, unit.split, hypothesis, items))
    return english


def english_probes(config: VerifierConfig, store: TranslationStore) -> list[tuple[str, str]]:
    return [
        (
            normalize_whitespace(store.translate_chunk(premise)),
            normalize_whitespace(store.translate_opinion(hypothesis)),
        )
        for premise, hypothesis in portuguese_probes(config)
    ]


def require_battery(config: VerifierConfig) -> Battery:
    if config.battery is None:
        raise SystemExit("decision-model scorers need a [decision_battery] table")
    return config.battery


def battery_questions(config: VerifierConfig, spec: ScorerSpec) -> list[DecisionQuestion]:
    battery = require_battery(config)
    return [battery.questions[key].question for key in spec.questions]


def laya_spec(spec: ScorerSpec, config: VerifierConfig, device: str) -> LayaSpec:
    return LayaSpec(
        repo_id=spec.name,
        revision=spec.revision,
        subfolder=spec.source["subfolder"],
        device=device,
        batch_size=spec.batch_size,
        max_length=spec.max_length,
        head_max_length=int(spec.source["head_max_length"]),
        truncate_key=TRUNCATED_KEY,
        cache_dir=config.decision_cache_dir,
    )


def api_key_env(spec: ScorerSpec) -> str:
    return str(spec.source.get("api_key_env", DEFAULT_API_KEY_ENV))


def jev_spec(spec: ScorerSpec, config: VerifierConfig) -> JevSpec:
    raw = spec.source
    return JevSpec(
        model_version=spec.revision,
        endpoint=raw.get("endpoint", JEV_ENDPOINT),
        api_key_env=api_key_env(spec),
        timeout_seconds=float(raw["timeout_seconds"]),
        max_retries=int(raw["max_retries"]),
        backoff_initial_seconds=float(raw["backoff_initial_seconds"]),
        backoff_max_seconds=float(raw["backoff_max_seconds"]),
        requests_per_minute=float(raw["requests_per_minute"]),
        max_concurrency=int(raw["max_concurrency"]),
        price_usd_per_million_input_tokens=float(raw["price_usd_per_million_input_tokens"]),
        cache_dir=config.decision_cache_dir,
    )


def load_decision_model(
    spec: ScorerSpec, config: VerifierConfig, device: str, mode: str
) -> DecisionModel:
    try:
        if spec.kind == "laya":
            return LayaDecisionModel(laya_spec(spec, config, device))
        if mode == "replay":
            return ReplayDecisionModel(jev_spec(spec, config))
        return JevDecisionModel(jev_spec(spec, config))
    except DecisionModelError as error:
        raise SystemExit(f"{spec.key}: {error}") from error


def check_decision_scorer(spec: ScorerSpec, config: VerifierConfig, mode: str) -> Record:
    if spec.kind == "laya":
        try:
            directory = laya_checkpoint_dir(spec.name, spec.revision, spec.source["subfolder"])
        except DecisionModelError as error:
            raise SystemExit(f"{spec.key}: {error}") from error
        return {"checkpoint_dir": str(directory)}
    env = api_key_env(spec)
    if mode == "live" and not os.environ.get(env):
        raise SystemExit(f"{spec.key}: {missing_key_message(env)}")
    return {"mode": mode, "api_key_env": env, "api_key_set": bool(os.environ.get(env))}


def decision_state(premise: str, hypothesis: str) -> Record:
    return {"premise": premise, "hypothesis": hypothesis}


def decision_requests(
    units: list[PremiseUnit], separator: str, with_concatenated: bool, concatenate_single: bool
) -> list[tuple[str, str]]:
    requests: dict[tuple[str, str], None] = {}
    for unit in units:
        for item in distinct_items(unit.items):
            requests[(item, unit.hypothesis)] = None
        if with_concatenated:
            text = concatenated_premise(unit, separator, concatenate_single)
            if text is not None:
                requests[(text, unit.hypothesis)] = None
    return list(requests)


def run_decision(
    model: DecisionModel,
    questions: list[DecisionQuestion],
    requests: list[tuple[str, str]],
    label: str,
    progress: bool,
) -> tuple[dict[tuple[str, str], dict[str, DecisionAnswer]], float]:
    order = sorted(range(len(requests)), key=lambda row: -len(requests[row][0]))
    answers: dict[tuple[str, str], dict[str, DecisionAnswer]] = {}
    started = time.perf_counter()
    for start in range(0, len(order), DECISION_CHUNK):
        rows = [requests[row] for row in order[start : start + DECISION_CHUNK]]
        try:
            results = model.predict_batch([decision_state(p, h) for p, h in rows], questions)
        except DecisionModelError as error:
            raise SystemExit(f"{label}: {error}") from error
        answers.update(zip(rows, results, strict=True))
        if progress:
            done = start + len(rows)
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            left = (len(order) - done) / rate / 60 if rate else float("nan")
            print(
                f"  {label}: {done}/{len(order)} premise texts x {len(questions)} questions, "
                f"{rate:.2f} texts/s, about {left:.1f} min left",
                flush=True,
            )
    return answers, time.perf_counter() - started


def run_decision_probes(
    model: DecisionModel,
    questions: list[DecisionQuestion],
    spec: ScorerSpec,
    pairs: list[tuple[str, str]],
) -> Record:
    try:
        results = model.predict_batch([decision_state(p, h) for p, h in pairs], questions)
    except DecisionModelError as error:
        raise SystemExit(f"{spec.key} label probes: {error}") from error
    question = spec.questions[0]
    probes = []
    for (premise, hypothesis), expected, answers in zip(
        pairs, spec.probe_expected, results, strict=True
    ):
        predicted = answers[question].choice
        probes.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "answers": {key: answer.record() for key, answer in answers.items()},
                "agrees": predicted == expected,
            }
        )
    return {
        "probes": probes,
        "agreeing": sum(1 for probe in probes if probe["agrees"]),
        "total": len(probes),
    }


def decision_item_record(answers: dict[str, DecisionAnswer], signals: Record) -> Record:
    values = list(answers.values())
    truncated = [key for key, answer in answers.items() if answer.truncated]
    record: Record = {
        "answers": {key: answer.record() for key, answer in answers.items()},
        "signals": signals,
        "tokens": max(answer.input_tokens or 0 for answer in values),
        "state_tokens": values[0].state_tokens,
        "truncated": bool(truncated),
        "truncated_questions": truncated,
        "overflow": any(answer.overflow for answer in values),
    }
    if truncated:
        record["premise_chars"] = answers[truncated[0]].premise_chars
        record["premise_chars_kept"] = min(
            int(answers[key].premise_chars_kept or 0) for key in truncated
        )
    return record


def decision_unit_row(
    unit: PremiseUnit,
    answers: dict[tuple[str, str], dict[str, DecisionAnswer]],
    spec: ScorerSpec,
    battery: Battery,
    separator: str,
    concatenate_single: bool,
) -> Record:
    signals: dict[str, Record] = {}
    items: list[Record | None] = []
    for item in unit.items:
        if not item:
            items.append(None)
            continue
        answered = answers[(item, unit.hypothesis)]
        values = signals.setdefault(item, item_signals(battery, spec.questions, answered))
        items.append(decision_item_record(answered, values))
    item_values = [signals[item] for item in distinct_items(unit.items)]
    text = concatenated_premise(unit, separator, concatenate_single) if spec.concatenated else None
    concatenated = None
    joined_values = None
    if text is not None:
        joined = answers[(text, unit.hypothesis)]
        joined_values = item_signals(battery, spec.questions, joined)
        concatenated = {
            **decision_item_record(joined, joined_values),
            "items_joined": sum(1 for item in unit.items if item),
        }
    scores = opinion_scores(
        battery,
        spec.questions,
        item_values,
        joined_values,
        spec.concatenated and concatenate_single,
    )
    return {**unit_header(unit), "items": items, "concatenated": concatenated, "scores": scores}


def counter_delta(after: Record, before: Record) -> Record:
    return {
        key: round(value - before.get(key, 0), 4)
        if isinstance(value, float)
        else value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and key != "max_rounding_gap"
    }


def decision_timing(
    spec: ScorerSpec, before: Record, after: Record, texts: int, seconds: float
) -> Record:
    counters = after.get("counters", {})
    delta = counter_delta(counters, before.get("counters", {}))
    timing: Record = {
        "premise_texts": texts,
        "questions": len(spec.questions),
        "wall_seconds": round(seconds, 2),
        **delta,
    }
    if spec.kind == "laya" and "computed_input_tokens" in delta:
        tokens, forward = delta["computed_input_tokens"], delta["forward_seconds"]
        timing["seconds_per_input_token"] = forward / tokens if tokens else None
        timing["computed_per_second"] = round(delta["computed"] / forward, 2) if forward else None
        timing["max_rounding_gap"] = counters.get("max_rounding_gap")
    return timing


def question_record(battery: Battery, key: str) -> Record:
    spec = battery.questions[key]
    return {
        "payload": spec.question.payload(),
        "support_option": spec.support_option,
        "against_option": spec.against_option,
        "against_signal": spec.against_signal,
        "reverses": spec.reverses,
    }


def score_units_decision(
    units: list[PremiseUnit],
    spec: ScorerSpec,
    config: VerifierConfig,
    device: str,
    concatenate_single: bool,
    probe_pairs: list[tuple[str, str]],
    mode: str,
) -> tuple[list[Record], Record]:
    battery = require_battery(config)
    questions = battery_questions(config, spec)
    model = load_decision_model(spec, config, device, mode)
    probes = run_decision_probes(model, questions, spec, probe_pairs)
    if probes["agreeing"] != probes["total"]:
        print(f"WARNING {spec.key}: {probes['agreeing']}/{probes['total']} label probes agree")
    requests = decision_requests(
        units, config.concat_separator, spec.concatenated, concatenate_single
    )
    before = model.describe()
    answers, seconds = run_decision(model, questions, requests, spec.key, True)
    after = model.describe()
    rows = [
        decision_unit_row(unit, answers, spec, battery, config.concat_separator, concatenate_single)
        for unit in units
    ]
    details = {
        "model": after,
        "label_probes": probes,
        "questions": {key: question_record(battery, key) for key in spec.questions},
        "order_pairs_used": order_pairs_used(battery, spec.questions),
        "timing": decision_timing(spec, before, after, len(requests), seconds),
    }
    del model
    gc.collect()
    release_device(device)
    return rows, details


def decision_truncation_summary(rows: list[Record], questions: tuple[str, ...]) -> Record:
    items = [item for row in rows for item in row["items"] if item is not None]
    joined = [row["concatenated"] for row in rows if row["concatenated"] is not None]

    def by_question(entries: list[Record]) -> Record:
        counts = Counter(key for entry in entries for key in entry["truncated_questions"])
        return {key: counts.get(key, 0) for key in questions}

    summary: Record = {
        "items_scored": len(items),
        "items_truncated": sum(1 for item in items if item["truncated"]),
        "items_truncated_by_question": by_question(items),
        "concatenated_premises": len(joined),
        "concatenated_truncated": sum(1 for entry in joined if entry["truncated"]),
        "concatenated_truncated_by_question": by_question(joined),
        "overflow": sum(1 for entry in [*items, *joined] if entry["overflow"]),
    }
    for name, entries in (("item_state_tokens", items), ("concatenated_state_tokens", joined)):
        tokens = [entry["state_tokens"] for entry in entries if entry["state_tokens"] is not None]
        if tokens:
            summary[name] = describe(np.array(tokens, dtype=np.float64))
    kept = [
        entry["premise_chars_kept"] / entry["premise_chars"]
        for entry in [*items, *joined]
        if entry["truncated"] and entry.get("premise_chars")
    ]
    if kept:
        summary["kept_premise_char_fraction_when_truncated"] = describe(
            np.array(kept, dtype=np.float64)
        )
    return summary


def score_units(
    units: list[PremiseUnit],
    spec: ScorerSpec,
    config: VerifierConfig,
    device: str,
    concatenate_single: bool,
    prefix: str,
    probe_pairs: list[tuple[str, str]] | None = None,
    decision_mode: str = "live",
) -> tuple[list[Record], Record]:
    pairs = portuguese_probes(config) if probe_pairs is None else probe_pairs
    if spec.kind == "cosine":
        return score_units_cosine(units, spec, config, device, prefix)
    if spec.kind in DECISION_KINDS:
        return score_units_decision(
            units, spec, config, device, concatenate_single, pairs, decision_mode
        )
    return score_units_nli(units, spec, config, device, concatenate_single, pairs)


def truncation_summary(rows: list[Record]) -> Record:
    items = [item for row in rows for item in row["items"] if item is not None]
    concatenated = [row["concatenated"] for row in rows if row["concatenated"] is not None]
    truncated = [entry for entry in concatenated if entry["truncated"]]
    summary: Record = {
        "items_scored": len(items),
        "items_truncated": sum(1 for item in items if item["truncated"]),
        "item_tokens": describe(np.array([item["tokens"] for item in items], dtype=np.float64)),
        "concatenated_premises": len(concatenated),
        "concatenated_truncated": len(truncated),
    }
    if concatenated:
        summary["concatenated_tokens"] = describe(
            np.array([entry["tokens"] for entry in concatenated], dtype=np.float64)
        )
    if truncated:
        summary["kept_premise_fraction_when_truncated"] = describe(
            np.array([e["premise_tokens_kept"] / e["premise_tokens"] for e in truncated])
        )
    if items and "sentences" in items[0]:
        summary["sentences"] = sum(item["sentences"] for item in items)
        summary["sentences_truncated"] = sum(item["sentences_truncated"] for item in items)
        summary["items_without_sentence"] = sum(1 for item in items if item["sentence_fallback"])
    return summary


def unit_counts(rows: list[Record]) -> Record:
    return {
        "units": len(rows),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "hearings": len({row["hearing_id"] for row in rows}),
        "no_premise": sum(1 for row in rows if row["no_premise"]),
        "no_premise_ids": [row["id"] for row in rows if row["no_premise"]],
        "item_count_histogram": dict(sorted(Counter(row["item_count"] for row in rows).items())),
    }


def check_score_names(rows: list[Record], spec: ScorerSpec) -> None:
    for row in rows:
        if set(row["scores"]) != set(spec.scores):
            raise SystemExit(f"{spec.key} {row['id']}: scores {sorted(row['scores'])}")


@functools.cache
def code_hashes() -> Record:
    modules = (
        udv_pipeline,
        build_udvs,
        build_nli_benchmark,
        cache_lock,
        calibrate_threshold,
        dataset_io,
        hub_offline,
        decision_models,
        decision_scoring,
        retrieval_stats,
        translation,
    )
    hashes = {f"utils/{module_path(m).name}": sha256_of_file(module_path(m)) for m in modules}
    return {"utils/nli_verifier_experiments.py": sha256_of_file(Path(__file__)), **hashes}


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "laya": laya.__version__,
        "torch_threads": torch.get_num_threads(),
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "mps_available": torch.backends.mps.is_available(),
        "platform": platform.platform(),
        "hub_offline": offline_state(),
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def score_file(run_dir: Path, scorer: str, split: str) -> Path:
    return run_dir / "scores" / f"{scorer}_{split}.jsonl"


def score_report_file(run_dir: Path, scorer: str) -> Path:
    return run_dir / "scores" / f"{scorer}_report.json"


def run_directory(config: VerifierConfig, args: argparse.Namespace) -> Path:
    base = args.output_dir if args.output_dir is not None else config.output_dir
    return base / args.run_name


def subset_record(limit: int | None, rule: str, seed: int) -> Record | None:
    if limit is None:
        return None
    if rule == "first":
        return {"limit_per_split": limit, "rule": "first rows of each split in file order"}
    return {"limit_per_split": limit, "seed": seed}


def prepare_benchmark_units(
    config: VerifierConfig, splits: tuple[str, ...], limit: int | None, rule: str = "random"
) -> tuple[list[PremiseUnit], Record]:
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    benchmark = check_benchmark_file(config)
    rows = sample_rows(load_benchmark_rows(config, split_of, splits), limit, config.seed, rule)
    nli_chunks = load_nli_chunks(config, rows)
    transcripts = load_transcripts(config, {row["hearing_id"] for row in rows})
    verification = verify_chunk_sources(rows, nli_chunks, transcripts)
    units = benchmark_units(rows, nli_chunks, transcripts, config.text_source)
    context = {
        "sources": {
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "nli": {"path": str(config.nli_path), "sha256": config.nli_sha256},
            "benchmark": benchmark,
            "splits": split_source,
        },
        "premise": {
            "text_source": config.text_source,
            "concat_separator": config.concat_separator,
            "chunk_source_verification": verification,
        },
        "subset": subset_record(limit, rule, config.seed),
    }
    return units, context


def score_report(
    mode: str,
    run_name: str,
    spec: ScorerSpec,
    rows: list[Record],
    details: Record,
    context: Record,
    splits: Record,
    config: VerifierConfig,
    files: Record,
) -> Record:
    return {
        "experiment": "nli_verifier",
        "mode": mode,
        "run_name": run_name,
        "created_at": now(),
        "scorer": spec.key,
        "kind": spec.kind,
        "splits": splits,
        "language": spec.language,
        "counts": unit_counts(rows),
        "truncation": (
            decision_truncation_summary(rows, spec.questions)
            if spec.kind in DECISION_KINDS
            else truncation_summary(rows)
        ),
        **details,
        **context,
        "files": files,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }


def write_split_scores(rows: list[Record], run_dir: Path, scorer: str) -> Record:
    files: Record = {}
    for split in SPLIT_NAMES:
        members = [row for row in rows if row["split"] == split]
        if not members:
            continue
        path = score_file(run_dir, scorer, split)
        write_jsonl(members, path)
        files[split] = {"path": str(path), "rows": len(members), "sha256": sha256_of_file(path)}
    return files


def scorer_checks(
    spec: ScorerSpec,
    units: list[PremiseUnit],
    config: VerifierConfig,
    translations: Translations | None,
    mode: str,
) -> Record:
    checks: Record = {}
    if spec.kind in DECISION_KINDS:
        checks |= check_decision_scorer(spec, config, mode)
    if spec.language == "en":
        if translations is None:
            raise SystemExit(f"{spec.key}: no translation store opened")
        checks["translation"] = check_translations(spec, units, config, translations.store(spec))
    return checks


def device_of(args: argparse.Namespace, config: VerifierConfig) -> str:
    requested = getattr(args, "device", None)
    return select_device(requested if requested is not None else config.device)


def command_score(args: argparse.Namespace, config: VerifierConfig) -> None:
    splits = narrowed_splits(config, args.final_test, args.splits)
    specs = selected_scorers(config, args.scorers)
    device = device_of(args, config)
    units, context = prepare_benchmark_units(config, splits, args.limit_per_split, args.subset_rule)
    run_dir = run_directory(config, args)
    translations = open_translations(config, specs, args.translation_cache_dir)
    checks = {
        spec.key: scorer_checks(spec, units, config, translations, args.decision_mode)
        for spec in specs
    }
    print(f"{len(units)} opinions ({', '.join(splits)}) on {device} -> {run_dir}", flush=True)
    for spec in specs:
        spec_units, probes, extra = units, portuguese_probes(config), {}
        if checks[spec.key]:
            extra["checks"] = checks[spec.key]
        if spec.language == "en" and translations is not None:
            store = translations.store(spec)
            spec_units, probes = english_units(units, store), english_probes(config, store)
            extra["translation"] = translation_summary(config, translations, spec)
        rows, details = score_units(
            spec_units, spec, config, device, True, "b2", probes, args.decision_mode
        )
        details |= extra
        check_score_names(rows, spec)
        files = write_split_scores(rows, run_dir, spec.key)
        report = score_report(
            "benchmark",
            args.run_name,
            spec,
            rows,
            details,
            context,
            split_records(splits, args.final_test),
            config,
            files,
        )
        write_json(report, score_report_file(run_dir, spec.key))
        print(
            f"{spec.key}: {len(rows)} opinions, timing {json.dumps(details['timing'])}, "
            f"truncation {json.dumps(report['truncation'])}",
            flush=True,
        )


def cosine_plan(units: list[PremiseUnit]) -> Record:
    items = {item for unit in units for item in distinct_items(unit.items)}
    sentences = {s for item in items for s in item_sentences(item)[0]}
    return {"distinct_items": len(items), "distinct_sentences": len(sentences)}


def nli_plan(spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig) -> Record:
    model_config, _ = load_model_config(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, config=model_config, local_files_only=True
    )
    nli = NliModel(
        spec,
        tokenizer,
        None,
        "cpu",
        {"special_tokens_per_pair": tokenizer.num_special_tokens_to_add(pair=True)},
    )
    requests = nli_requests(units, config.concat_separator, True)
    _, _, totals = pair_token_arrays(nli, requests)
    concatenated = {
        (concatenated_premise(unit, config.concat_separator, True), unit.hypothesis)
        for unit in units
    }
    truncated = [
        total > spec.max_length
        for request, total in zip(requests, totals, strict=True)
        if request in concatenated
    ]
    return {
        "requests": len(requests),
        "model_input_tokens": int(np.minimum(totals, spec.max_length).sum()),
        "concatenated_requests": len(truncated),
        "concatenated_truncated": int(sum(truncated)),
        "item_requests_truncated": int(
            sum(
                total > spec.max_length
                for request, total in zip(requests, totals, strict=True)
                if request not in concatenated
            )
        ),
    }


def request_modes(
    units: list[PremiseUnit], separator: str, with_concatenated: bool
) -> dict[str, list[tuple[str, str]]]:
    chunks = decision_requests(units, separator, False, True)
    joined = [
        request
        for request in decision_requests(units, separator, with_concatenated, True)
        if request not in set(chunks)
    ]
    return {"chunk": chunks, "concatenated": joined}


def laya_plan(spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig) -> Record:
    try:
        tokenizer = LayaTokenizer(spec.name, spec.revision, spec.source["subfolder"])
    except DecisionModelError as error:
        raise SystemExit(f"{spec.key}: {error}") from error
    questions = battery_questions(config, spec)
    modes = request_modes(units, config.concat_separator, spec.concatenated)
    by_question: Record = {}
    for question in questions:
        room = tokenizer.room(question)
        prefix = tokenizer.max_length - room
        by_question[question.key] = {"state_room": room}
        for mode, requests in modes.items():
            tokens = np.array(
                [
                    tokenizer.count(json.dumps(decision_state(p, h), ensure_ascii=False))
                    for p, h in requests
                ],
                dtype=np.int64,
            )
            by_question[question.key][mode] = {
                "requests": len(requests),
                "model_input_tokens": int((prefix + np.minimum(tokens, room)).sum()),
                "truncated": int((tokens > room).sum()),
            }
    return {
        "premise_texts": {mode: len(requests) for mode, requests in modes.items()},
        "questions": len(questions),
        "requests": sum(len(r) for r in modes.values()) * len(questions),
        "model_input_tokens": sum(
            entry[mode]["model_input_tokens"] for entry in by_question.values() for mode in modes
        ),
        "max_len": tokenizer.max_length,
        "head_max_len": tokenizer.head_max_length,
        "by_question": by_question,
    }


def jev_plan(
    spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig, proxy: LayaTokenizer
) -> Record:
    questions = battery_questions(config, spec)
    modes = request_modes(units, config.concat_separator, spec.concatenated)
    question_tokens = sum(proxy.count(question.payload_json()) for question in questions)
    state_tokens = sum(
        proxy.count(json.dumps(decision_state(p, h), ensure_ascii=False))
        for requests in modes.values()
        for p, h in requests
    )
    requests = sum(len(r) for r in modes.values())
    tokens = state_tokens + requests * question_tokens
    rate = float(spec.source["requests_per_minute"])
    return {
        "requests": requests,
        "premise_texts": {mode: len(r) for mode, r in modes.items()},
        "questions_per_request": len(questions),
        "proxy_input_tokens": tokens,
        "proxy_rule": (
            "state and question JSON counted with the laya multilingual tokenizer (Gemma "
            "vocabulary); Jev's own tokenizer is not public, so the cost is an estimate"
        ),
        "proxy_cost_usd": round(
            tokens * float(spec.source["price_usd_per_million_input_tokens"]) / 1e6, 3
        ),
        "minutes_at_rate_limit": round(requests / rate, 1) if rate else None,
    }


def proxy_tokenizer(config: VerifierConfig) -> LayaTokenizer:
    for spec in config.scorers.values():
        if spec.kind == "laya" and spec.source["subfolder"] == "multilingual":
            try:
                return LayaTokenizer(spec.name, spec.revision, "multilingual")
            except DecisionModelError as error:
                raise SystemExit(f"{spec.key}: {error}") from error
    raise SystemExit("the Jev token proxy needs a laya scorer with subfolder multilingual")


def plan_units(
    spec: ScorerSpec,
    units: list[PremiseUnit],
    config: VerifierConfig,
    translations: Translations | None,
) -> tuple[list[PremiseUnit], Record]:
    if spec.language != "en":
        return units, {"texts": "portuguese"}
    if translations is None:
        return units, {
            "texts": "portuguese_proxy",
            "translation_model": spec.translation_model,
            "reason": "no translation store",
        }
    store = translations.store(spec)
    texts = translation_texts(units, portuguese_probes(config), store)
    missing = store.missing(texts)
    source = {"translation_model": spec.translation_model, "store": str(store.path)}
    if missing:
        return units, {
            "texts": "portuguese_proxy",
            **source,
            "missing_translations": len(missing),
            "distinct_texts": len(texts),
            "rule": "the English texts of this translation model are not all translated yet, so "
            "the Portuguese texts are counted with this scorer's tokenizer; the counts are a proxy",
        }
    return english_units(units, store), {
        "texts": "english",
        **source,
        "distinct_texts": len(texts),
    }


def smoke_rate(report: Record) -> Record:
    timing = report["timing"]
    device = report["model"].get("device")
    if report["kind"] == "laya":
        rate = timing.get("seconds_per_input_token")
        basis = "forward_seconds / computed_input_tokens"
    elif timing.get("computed") and timing.get("model_input_tokens"):
        share = timing["computed"] / timing["requests"]
        rate = timing["seconds"] / (timing["model_input_tokens"] * share)
        basis = "seconds / (model_input_tokens x computed share)"
    else:
        rate, basis = None, "no computed request in the smoke"
    return {"device": device, "seconds_per_input_token": rate, "basis": basis}


def scorer_tokens(plan: Record, dropped: set[tuple[str, str]]) -> int:
    if "by_question" not in plan:
        return int(plan["model_input_tokens"])
    return sum(
        entry[mode]["model_input_tokens"]
        for key, entry in plan["by_question"].items()
        for mode in ("chunk", "concatenated")
        if (key, mode) not in dropped
    )


def check_estimate_scorers(planned: list[str], declaration: RunDeclaration) -> None:
    local = list(declaration.source["compute"]["local_scorers"])
    missing = [key for key in local if key not in planned]
    if missing:
        raise SystemExit(
            f"--timing-from estimates every local scorer of declarations.{declaration.name}; "
            f"the plan lacks {missing}: run plan without --scorers, or with all of {local}"
        )


def compute_estimate(plans: Record, timing_dir: Path, declaration: RunDeclaration) -> Record:
    check_estimate_scorers(list(plans), declaration)
    compute = declaration.source["compute"]
    budget = float(compute["budget_hours"])
    partners = {**declaration.twins, **{first: twin for twin, first in declaration.twins.items()}}
    rates: Record = {}
    for key in compute["local_scorers"]:
        path = score_report_file(timing_dir, key)
        if not path.exists():
            rates[key] = {"report": str(path), "seconds_per_input_token": None}
            continue
        with open(path) as f:
            report = json.load(f)
        rates[key] = {"report": str(path), **smoke_rate(report)}
    known = all(rates[key]["seconds_per_input_token"] for key in rates)
    devices = sorted({str(rates[key].get("device")) for key in rates})
    dropped: dict[str, set[tuple[str, str]]] = {key: set() for key in rates}

    def hours() -> dict[str, float | None]:
        return {
            key: (
                rates[key]["seconds_per_input_token"]
                * scorer_tokens(plans[key], dropped[key])
                / 3600
                if rates[key]["seconds_per_input_token"]
                else None
            )
            for key in rates
        }

    first = hours()
    steps = []
    total = sum(v for v in first.values() if v is not None) if known else None
    for name in compute["drop_order"]:
        if total is None or total <= budget:
            break
        step = compute["drop_steps"][name]
        named = list(step["scorers"])
        affected = list(dict.fromkeys([*named, *(partners[k] for k in named if k in partners)]))
        for key in affected:
            dropped[key] |= {(q, mode) for q in step["questions"] for mode in step["modes"]}
        after = hours()
        total = sum(v for v in after.values() if v is not None)
        steps.append(
            {
                "step": name,
                "text": step["text"],
                "scorers": affected,
                "total_hours_after": round(total, 2),
            }
        )
    last = hours()
    return {
        "budget_hours": budget,
        "timing_from": str(timing_dir),
        "devices": devices,
        "rates": rates,
        "hours_full": {k: None if v is None else round(v, 2) for k, v in first.items()},
        "total_hours_full": None
        if not known
        else round(sum(cast(Iterable[float], first.values())), 2),
        "steps_applied": steps,
        "hours_after_steps": {k: None if v is None else round(v, 2) for k, v in last.items()},
        "total_hours_after_steps": None if total is None else round(total, 2),
        "fits": None if total is None else total <= budget,
        "rule": compute["drop_rule"],
        "twin_rule": compute.get("twin_rule"),
    }


def command_plan(args: argparse.Namespace, config: VerifierConfig) -> None:
    splits = scored_splits(config, args.final_test)
    units, context = prepare_benchmark_units(config, splits, args.limit_per_split, args.subset_rule)
    specs = selected_scorers(config, args.scorers)
    declaration = config.declarations.get(args.run_name)
    if args.timing_from is not None:
        if declaration is None or "compute" not in declaration.source:
            raise SystemExit(f"--timing-from needs a compute table in declarations.{args.run_name}")
        check_estimate_scorers([spec.key for spec in specs], declaration)
    translations = open_translations(config, specs, args.translation_cache_dir)
    proxy = None
    plans: Record = {}
    for spec in specs:
        spec_units, texts = plan_units(spec, units, config, translations)
        if spec.kind == "cosine":
            plans[spec.key] = cosine_plan(spec_units)
        elif spec.kind == "nli":
            plans[spec.key] = nli_plan(spec, spec_units, config)
        elif spec.kind == "laya":
            plans[spec.key] = laya_plan(spec, spec_units, config)
        else:
            if proxy is None:
                proxy = proxy_tokenizer(config)
            plans[spec.key] = jev_plan(spec, spec_units, config, proxy)
        if spec.language == "en":
            plans[spec.key]["texts"] = texts
        print(f"planned {spec.key}", flush=True)
    estimate = None
    if args.timing_from is not None and declaration is not None:
        estimate = compute_estimate(plans, args.timing_from, declaration)
    report = {
        "created_at": now(),
        "splits": split_records(splits, args.final_test),
        "opinions": len(units),
        "plans": plans,
        "compute_estimate": estimate,
        "translation": (
            None
            if translations is None
            else {
                spec.key: translation_summary(config, translations, spec)
                for spec in specs
                if spec.language == "en"
            }
        ),
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    path = run_directory(config, args) / "plan.json"
    write_json(report, path)
    print(json.dumps({"plans": plans, "compute_estimate": estimate}, indent=2))


def command_pairs(args: argparse.Namespace, config: VerifierConfig) -> None:
    splits = scored_splits(config, args.final_test)
    keys = args.scorers if args.scorers is not None else list(config.pairs_default_scorers)
    specs = selected_scorers(config, keys)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    units, inputs = load_pair_units(args.input, split_of, splits)
    english = [spec.key for spec in specs if spec.language != "pt"]
    if english:
        raise SystemExit(f"pairs reads Portuguese propositions only; {english} have language en")
    for spec in specs:
        if spec.kind in DECISION_KINDS:
            check_decision_scorer(spec, config, args.decision_mode)
    device = device_of(args, config)
    out_dir = (args.output_dir or config.output_dir) / "pairs" / args.run_name
    context = {
        "sources": {
            "input": {"path": str(args.input), "sha256": sha256_of_file(args.input)},
            "splits": split_source,
        },
        "format": {
            "input": config.source["pairs"]["input_format"],
            "output": config.source["pairs"]["output_format"],
        },
        "premise": {"concat_separator": config.concat_separator},
        "subset": None,
    }
    by_id = {row["pair_id"]: row for row in inputs}
    print(f"{len(units)} pairs ({', '.join(splits)}) on {device} -> {out_dir}", flush=True)
    for spec in specs:
        rows, details = score_units(
            units,
            spec,
            config,
            device,
            False,
            f"pairs_{args.run_name}",
            decision_mode=args.decision_mode,
        )
        output = [pair_output_row(row, by_id[row["id"]], spec) for row in rows]
        path = out_dir / f"{spec.key}.jsonl"
        write_jsonl(output, path)
        files = {"pairs": {"path": str(path), "rows": len(output), "sha256": sha256_of_file(path)}}
        report = score_report(
            "pairs",
            args.run_name,
            spec,
            rows,
            details,
            context,
            split_records(splits, args.final_test),
            config,
            files,
        )
        write_json(report, out_dir / f"{spec.key}_report.json")
        print(
            f"{spec.key}: {len(output)} pairs, timing {json.dumps(details['timing'])}", flush=True
        )


def pair_output_row(row: Record, source: Record, spec: ScorerSpec) -> Record:
    header = {key: value for key, value in row.items() if key != "id"}
    return {
        "pair_id": row["id"],
        "query_id": source["query_id"],
        **header,
        "scorer": spec.key,
        "model": spec.name,
        "revision": spec.revision,
        "meta": source.get("meta"),
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if labels.all() or not labels.any():
        return {metric: float("nan") for metric in RANKING_METRICS}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision_inferable": float(average_precision_score(labels, scores)),
        "average_precision_not_inferable": float(average_precision_score(~labels, -scores)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    total = len(labels)
    inf_inf = float(np.sum(labels & predictions))
    inf_not = float(np.sum(labels & ~predictions))
    not_inf = float(np.sum(~labels & predictions))
    not_not = float(np.sum(~labels & ~predictions))
    accuracy = (inf_inf + not_not) / total
    expected = (
        (inf_inf + inf_not) * (inf_inf + not_inf) + (not_inf + not_not) * (inf_not + not_not)
    ) / total**2
    f1_not = safe_ratio(2 * not_not, 2 * not_not + not_inf + inf_not)
    f1_inf = safe_ratio(2 * inf_inf, 2 * inf_inf + inf_not + not_inf)
    return {
        "accuracy": accuracy,
        "cohen_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else float("nan"),
        "macro_f1": (f1_not + f1_inf) / 2,
        "f1_not_inferable": f1_not,
        "precision_not_inferable": safe_ratio(not_not, not_not + inf_not),
        "recall_not_inferable": safe_ratio(not_not, not_not + not_inf),
    }


def check_binary_metrics(fast: dict[str, float], reference: Record, where: str) -> None:
    pairs = {
        "accuracy": reference["accuracy"],
        "cohen_kappa": reference["cohen_kappa"],
        "macro_f1": reference["macro_f1"],
        "f1_not_inferable": reference["per_class"]["not_inferable"]["f1"],
    }
    for name, value in pairs.items():
        if not np.isnan(fast[name]) and abs(fast[name] - value) > 1e-4:
            raise SystemExit(f"{where}: {name} {fast[name]} != judge_metrics {value}")


def max_f1_not_inferable_optimum(scores: np.ndarray, labels: np.ndarray) -> Record | None:
    positives = int((~labels).sum())
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    ordered, ordered_labels = scores[order], labels[order]
    group_end = np.append(ordered[1:] != ordered[:-1], True)
    cut_scores = ordered[group_end]
    kept = np.concatenate([[0], np.flatnonzero(group_end) + 1])
    kept_inferable = np.concatenate([[0], np.cumsum(ordered_labels)[group_end]])
    kept_not = kept - kept_inferable
    true_not = positives - kept_not
    predicted_not = len(labels) - kept
    f1 = 2 * true_not / (predicted_not + positives)
    best = int(np.flatnonzero(f1 >= f1.max() - 1e-12)[-1])
    lowest_kept: float | None
    highest_excluded: float | None
    if best == 0:
        threshold = float(np.nextafter(cut_scores[0], np.inf))
        lowest_kept, highest_excluded = None, float(cut_scores[0])
    else:
        kept_score = float(cut_scores[best - 1])
        lowest_kept = kept_score
        highest_excluded = float(cut_scores[best]) if best < len(cut_scores) else None
        threshold = kept_score if highest_excluded is None else (kept_score + highest_excluded) / 2
    return {
        "threshold": threshold,
        "f1_not_inferable": float(f1[best]),
        "lowest_kept_score": lowest_kept,
        "highest_excluded_score": highest_excluded,
    }


def fit_rule(
    rule: str, scores: np.ndarray, labels: np.ndarray, config: VerifierConfig, probability: bool
) -> Record | None:
    if rule == "youden":
        return youden_optimum(scores, labels)
    if rule == "max_f1_not_inferable":
        return max_f1_not_inferable_optimum(scores, labels)
    return {"threshold": config.fixed_probability_threshold} if probability else None


def is_probability_system(system: str) -> bool:
    return not system.endswith(".cosine")


def hearing_draws(
    hearing_ids: np.ndarray, samples: int, rng: np.random.Generator
) -> list[np.ndarray]:
    groups = unit_groups(hearing_ids.tolist(), "hearing")
    return [resample_rows(rng, groups) for _ in range(samples)]


def finite_interval(values: np.ndarray, level: float) -> Record:
    return interval(values[~np.isnan(values)].tolist(), level)


def rounded_record(values: dict[str, float]) -> Record:
    return {name: None if np.isnan(value) else rounded(value) for name, value in values.items()}


def replicate_binary(
    labels: np.ndarray, predictions: np.ndarray, draws: list[np.ndarray]
) -> dict[str, np.ndarray]:
    values = [binary_metrics(labels[rows], predictions[rows]) for rows in draws]
    return {metric: np.array([v[metric] for v in values]) for metric in BINARY_METRICS}


def replicate_refit(
    scores: np.ndarray,
    labels: np.ndarray,
    draws: list[np.ndarray],
    refits: list[Record | None],
) -> dict[str, np.ndarray]:
    values = [
        binary_metrics(labels[rows], scores[rows] >= refit["threshold"])
        if refit is not None
        else {metric: float("nan") for metric in BINARY_METRICS}
        for rows, refit in zip(draws, refits, strict=True)
    ]
    return {metric: np.array([v[metric] for v in values]) for metric in BINARY_METRICS}


def evaluate_binary(
    predictions: np.ndarray, data: SplitData, draws: list[np.ndarray], level: float, where: str
) -> tuple[Record, dict[str, np.ndarray]]:
    metrics = judge_metrics(data.labels.tolist(), predictions.tolist())
    check_binary_metrics(binary_metrics(data.labels, predictions), metrics, where)
    replicates = replicate_binary(data.labels, predictions, draws)
    intervals = {m: finite_interval(replicates[m], level) for m in BINARY_METRICS}
    return {"metrics": metrics, "bootstrap": intervals}, replicates


def fit_record(rule: str, optimum: Record | None, scores: np.ndarray, data: SplitData) -> Record:
    if optimum is None:
        return {"applicable": False}
    predictions = scores >= optimum["threshold"]
    return {
        "applicable": True,
        "threshold": optimum["threshold"],
        "optimum": {k: v if v is None else rounded(v) for k, v in optimum.items()},
        "in_sample_metrics": judge_metrics(data.labels.tolist(), predictions.tolist()),
        "in_sample_note": (
            f"measured on the fit split that chose the {rule} threshold"
            if rule in FITTED_RULES
            else "fixed threshold, not fitted; measured on the fit split"
        ),
    }


def evaluate_system_split(
    system: str,
    data: SplitData,
    fitted: dict[str, Record | None],
    refits: dict[str, list[Record | None]],
    draws: list[np.ndarray],
    config: VerifierConfig,
) -> tuple[Record, dict[tuple[str, str], np.ndarray]]:
    scores, labels, level = data.scores[system], data.labels, config.confidence_level
    ranking_values = [ranking_metrics(labels[rows], scores[rows]) for rows in draws]
    replicates = {("ranking", m): np.array([v[m] for v in ranking_values]) for m in RANKING_METRICS}
    rules: Record = {}
    for rule, optimum in fitted.items():
        if optimum is None:
            continue
        where = f"{system} {rule} {data.split}"
        predictions = scores >= optimum["threshold"]
        result, fixed = evaluate_binary(predictions, data, draws, level, where)
        refit = replicate_refit(scores, labels, draws, refits[rule])
        replicates |= {(rule, m): fixed[m] for m in BINARY_METRICS}
        rules[rule] = {
            "threshold": rounded(optimum["threshold"]),
            "metrics": result["metrics"],
            "bootstrap_fixed_threshold": result["bootstrap"],
            "bootstrap_refit_threshold": {
                m: finite_interval(refit[m], level) for m in BINARY_METRICS
            },
        }
    return {
        "ranking": rounded_record(ranking_metrics(labels, scores)),
        "ranking_bootstrap": {
            m: finite_interval(replicates[("ranking", m)], level) for m in RANKING_METRICS
        },
        "score_by_label": {
            "inferable": describe(scores[labels]),
            "not_inferable": describe(scores[~labels]),
        },
        "rules": rules,
    }, replicates


def evaluate_system(
    system: str,
    fit: SplitData,
    evaluations: list[SplitData],
    fit_draws: list[np.ndarray],
    eval_draws: dict[str, list[np.ndarray]],
    config: VerifierConfig,
) -> tuple[Record, dict[str, dict[tuple[str, str], np.ndarray]]]:
    probability = is_probability_system(system)
    scores, labels = fit.scores[system], fit.labels
    fitted = {
        rule: fit_rule(rule, scores, labels, config, probability) for rule in config.threshold_rules
    }
    refits = {
        rule: [
            fit_rule(rule, scores[rows], labels[rows], config, probability) for rows in fit_draws
        ]
        if rule in FITTED_RULES
        else [optimum] * len(fit_draws)
        for rule, optimum in fitted.items()
    }
    result: Record = {
        "probability_score": probability,
        "fit": {rule: fit_record(rule, fitted[rule], scores, fit) for rule in fitted},
        "threshold_bootstrap_on_fit_split": {
            rule: interval(
                [r["threshold"] for r in refits[rule] if r is not None], config.confidence_level
            )
            for rule in FITTED_RULES
            if rule in fitted and fitted[rule] is not None
        },
        "evaluation": {},
    }
    replicates: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    for data in evaluations:
        split_result, split_replicates = evaluate_system_split(
            system, data, fitted, refits, eval_draws[data.split], config
        )
        result["evaluation"][data.split] = split_result
        replicates[data.split] = split_replicates
    return result, replicates


def judge_keys_of(rows: list[Record]) -> list[str]:
    keys = list(rows[0]["judge_inferable"])
    if any(list(row["judge_inferable"]) != keys for row in rows):
        raise SystemExit("benchmark rows do not share the same judge keys")
    return keys


def load_scores(path: Path) -> dict[str, Record]:
    return {row["id"]: row for row in load_jsonl(path)}


def split_data(
    split: str,
    rows: list[Record],
    systems_by_scorer: dict[str, list[str]],
    directories: dict[str, Path],
    judge_keys: list[str],
    own: set[str],
) -> SplitData:
    scored = {
        key: load_scores(score_file(directories[key], key, split)) for key in systems_by_scorer
    }
    id_sets = {key: set(values) for key, values in scored.items()}
    base = {key: ids for key, ids in id_sets.items() if key in own} or id_sets
    reference = next(iter(base.values()))
    if any(ids != reference for ids in base.values()):
        sizes = {key: len(ids) for key, ids in base.items()}
        raise SystemExit(f"{split}: scorers were run on different opinions {sizes}")
    short = [key for key, ids in id_sets.items() if key not in base and not reference <= ids]
    if short:
        raise SystemExit(f"{split}: imported score files {short} lack opinions of this run")
    members = [row for row in rows if row["split"] == split and row["id"] in reference]
    if len(members) != len(reference):
        raise SystemExit(f"{split}: score files name opinions that are not benchmark rows")
    ids = [row["id"] for row in members]
    for key, names in systems_by_scorer.items():
        stale = sorted({n for i in ids for n in names if n not in scored[key][i]["scores"]})
        if stale:
            raise SystemExit(
                f"{key} {split}: the score file lacks {stale}; it was written before the score "
                f"names changed, so rerun `score --scorers {key}` (the NLI logits are read from "
                "the cache, no model forward pass is repeated)"
            )
    scores = {
        f"{key}.{name}": np.array([scored[key][i]["scores"][name] for i in ids], dtype=np.float64)
        for key, names in systems_by_scorer.items()
        for name in names
    }
    return SplitData(
        split=split,
        ids=ids,
        labels=np.array([row["label_inferable"] for row in members], dtype=bool),
        hearing_ids=np.array([row["hearing_id"] for row in members], dtype=np.int64),
        judges={
            k: np.array([r["judge_inferable"][k] for r in members], dtype=bool) for k in judge_keys
        },
        scores=scores,
    )


def available_scorers(
    config: VerifierConfig,
    run_dir: Path,
    splits: tuple[str, ...],
    own: tuple[str, ...],
    imported: dict[str, str],
) -> tuple[dict[str, list[str]], Record, dict[str, Path]]:
    systems: dict[str, list[str]] = {}
    reports: Record = {}
    directories: dict[str, Path] = {}
    for key in (*own, *imported):
        directory = run_dir if key in own else run_dir.parent / imported[key]
        if not all(score_file(directory, key, split).exists() for split in splits):
            continue
        with open(score_report_file(directory, key)) as f:
            reports[key] = json.load(f)
        systems[key] = list(config.scorers[key].scores)
        directories[key] = directory
    if not systems:
        raise SystemExit(f"no scorer has score files for {list(splits)} in {run_dir}")
    subsets = {
        json.dumps(report["subset"], sort_keys=True)
        for key, report in reports.items()
        if key in own
    }
    if len(subsets) > 1:
        raise SystemExit(f"score files come from different subsets: {subsets}")
    return systems, reports, directories


def split_summary(data: SplitData) -> Record:
    inferable = int(data.labels.sum())
    return {
        "opinions": len(data.ids),
        "hearings": len(set(data.hearing_ids.tolist())),
        "inferable": inferable,
        "not_inferable": len(data.ids) - inferable,
        "not_inferable_share": rounded((len(data.ids) - inferable) / len(data.ids)),
    }


def evaluate_judges(
    fit: SplitData,
    evaluations: list[SplitData],
    eval_draws: dict[str, list[np.ndarray]],
    level: float,
) -> tuple[Record, dict[str, dict[str, dict[str, np.ndarray]]]]:
    fit_kappa = {
        key: judge_metrics(fit.labels.tolist(), fit.judges[key].tolist())["cohen_kappa"]
        for key in fit.judges
    }
    reference = max(fit.judges, key=lambda key: fit_kappa[key])
    result: Record = {
        "reference_judge": {"key": reference, "fit_cohen_kappa": fit_kappa[reference]},
        "fit_cohen_kappa": fit_kappa,
        "evaluation": {},
    }
    replicates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for data in evaluations:
        judged: Record = {}
        replicates[data.split] = {}
        for key, predictions in data.judges.items():
            summary, values = evaluate_binary(predictions, data, eval_draws[data.split], level, key)
            judged[key] = {**parse_judge_key(key), **summary}
            replicates[data.split][key] = values
        result["evaluation"][data.split] = judged
    return result, replicates


def evaluate_constant(data: SplitData, draws: list[np.ndarray], level: float) -> Record:
    summary, _ = evaluate_binary(np.ones(len(data.ids), dtype=bool), data, draws, level, "always")
    prevalence = float(data.labels.mean())
    return {
        **summary,
        "ranking": {
            "roc_auc": 0.5,
            "average_precision_inferable": rounded(prevalence),
            "average_precision_not_inferable": rounded(1 - prevalence),
        },
    }


def paired_delta(values: np.ndarray, reference: np.ndarray, point: float, level: float) -> Record:
    delta = values - reference
    valid = delta[~np.isnan(delta)]
    return {
        "delta": None if np.isnan(point) else rounded(point),
        "bootstrap": interval(valid.tolist(), level),
        "share_of_replicates_above_zero": rounded(float(np.mean(valid > 0)))
        if len(valid)
        else None,
    }


def point_value(results: Record, system: str, split: str, key: tuple[str, str]) -> float:
    evaluation = results[system]["evaluation"][split]
    if key[0] == "ranking":
        value = evaluation["ranking"][key[1]]
    else:
        metrics = evaluation["rules"][key[0]]["metrics"]
        value = (
            metrics["per_class"]["not_inferable"]["f1"]
            if key[1] == "f1_not_inferable"
            else metrics[key[1]]
        )
    return float("nan") if value is None else float(value)


def judge_point(judges: Record, split: str, key: str, metric: str) -> float:
    metrics = judges["evaluation"][split][key]["metrics"]
    return float(
        metrics["per_class"]["not_inferable"]["f1"]
        if metric == "f1_not_inferable"
        else metrics[metric]
    )


def compare_systems(
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    judges: Record,
    judge_replicates: dict[str, dict[str, dict[str, np.ndarray]]],
    splits: list[str],
    config: VerifierConfig,
) -> Record:
    level = config.confidence_level
    reference_system = config.reference_cosine_system
    reference_judge = judges["reference_judge"]["key"]
    comparisons: Record = {}
    for split in splits:
        against_cosine: Record = {}
        against_judge: Record = {}
        for system in results:
            if reference_system in results and system != reference_system:
                against_cosine[system] = {
                    metric: paired_delta(
                        replicates[system][split][("ranking", metric)],
                        replicates[reference_system][split][("ranking", metric)],
                        point_value(results, system, split, ("ranking", metric))
                        - point_value(results, reference_system, split, ("ranking", metric)),
                        level,
                    )
                    for metric in COMPARED_RANKING_METRICS
                }
            if (COMPARED_RULE, "cohen_kappa") in replicates[system][split]:
                against_judge[system] = {
                    metric: paired_delta(
                        replicates[system][split][(COMPARED_RULE, metric)],
                        judge_replicates[split][reference_judge][metric],
                        point_value(results, system, split, (COMPARED_RULE, metric))
                        - judge_point(judges, split, reference_judge, metric),
                        level,
                    )
                    for metric in COMPARED_BINARY_METRICS
                }
        comparisons[split] = {
            "against_reference_cosine": {"reference": reference_system, "deltas": against_cosine},
            "against_reference_judge": {
                "reference": reference_judge,
                "rule": COMPARED_RULE,
                "deltas": against_judge,
            },
        }
    return comparisons


def draw_bootstraps(
    fit: SplitData, evaluations: list[SplitData], config: VerifierConfig
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    fit_rng = np.random.default_rng([config.evaluation_seed, 0])
    fit_draws = hearing_draws(fit.hearing_ids, config.bootstrap_samples, fit_rng)
    eval_draws = {
        data.split: hearing_draws(
            data.hearing_ids,
            config.bootstrap_samples,
            np.random.default_rng([config.evaluation_seed, 1 + SPLIT_NAMES.index(data.split)]),
        )
        for data in evaluations
    }
    return fit_draws, eval_draws


def merge_fit_splits(parts: list[SplitData]) -> SplitData:
    if len(parts) == 1:
        return parts[0]
    return SplitData(
        split="+".join(part.split for part in parts),
        ids=[i for part in parts for i in part.ids],
        labels=np.concatenate([part.labels for part in parts]),
        hearing_ids=np.concatenate([part.hearing_ids for part in parts]),
        judges={k: np.concatenate([p.judges[k] for p in parts]) for k in parts[0].judges},
        scores={k: np.concatenate([p.scores[k] for p in parts]) for k in parts[0].scores},
    )


SUMMARY_OPTIONAL = ("language", "checks", "translation", "questions", "order_pairs_used")
TABLE_COLUMNS = (
    "split",
    "system",
    "scorer",
    "score",
    "kind",
    "language",
    "role",
    "roc_auc",
    "roc_auc_low",
    "roc_auc_high",
    "average_precision_not_inferable",
    "threshold_max_f1_not_inferable",
    "cohen_kappa",
    "cohen_kappa_low",
    "cohen_kappa_high",
    "f1_not_inferable",
    "comparison",
    "family",
    "reference",
    "metric",
    "delta",
    "delta_low",
    "delta_high",
    "p_value",
    "p_holm",
    "missing",
)


def scorer_summaries(reports: Record) -> Record:
    return {
        key: {
            "model": report["model"],
            "label_probes": report["label_probes"],
            "counts": report["counts"],
            "truncation": report["truncation"],
            "timing": report["timing"],
            "premise": report["premise"],
            "files": report["files"],
            "created_at": report["created_at"],
            "code": report["code"],
            **{name: report[name] for name in SUMMARY_OPTIONAL if name in report},
        }
        for key, report in reports.items()
    }


def missing_detail(
    keys: list[str], run_dir: Path, imported: dict[str, str], splits: tuple[str, ...]
) -> Record:
    detail: Record = {}
    for key in keys:
        directory = run_dir.parent / imported[key] if key in imported else run_dir
        detail[key] = {
            "directory": str(directory),
            "splits_without_score_file": [
                split for split in splits if not score_file(directory, key, split).exists()
            ],
        }
    return detail


def run_declaration(config: VerifierConfig, args: argparse.Namespace) -> RunDeclaration | None:
    name = args.declaration if args.declaration is not None else args.run_name
    declaration = config.declarations.get(name)
    if args.declaration is not None and declaration is None:
        raise SystemExit(f"no [declarations.{name}] table in the config")
    return declaration


def add_derived_systems(
    fit: SplitData,
    evaluations: list[SplitData],
    systems_by_scorer: dict[str, list[str]],
    config: VerifierConfig,
) -> tuple[list[str], Record, dict[str, dict[str, float]]]:
    names: list[str] = []
    details: Record = {}
    thresholds_by_scorer: dict[str, dict[str, float]] = {}
    for key in systems_by_scorer:
        spec = config.scorers[key]
        if spec.kind not in DECISION_KINDS:
            continue
        battery = require_battery(config)
        entry: Record = {}
        thresholds: dict[str, float] = {}
        for component in battery.panel_components:
            system = f"{key}.max.{component}"
            optimum = fit_rule(battery.consensus_rule, fit.scores[system], fit.labels, config, True)
            if optimum is not None:
                thresholds[system] = float(optimum["threshold"])
        if len(thresholds) == len(battery.panel_components):
            for data in (fit, *evaluations):
                data.scores[f"{key}.{CONSENSUS}"] = consensus_votes(data.scores, thresholds)
            names.append(f"{key}.{CONSENSUS}")
            thresholds_by_scorer[key] = thresholds
            entry["consensus"] = {
                "components": list(thresholds),
                "rule": battery.consensus_rule,
                "fitted_on": fit.split,
                "thresholds": {system: rounded(t) for system, t in thresholds.items()},
            }
        else:
            entry["consensus"] = {
                "applicable": False,
                "reason": "a component has no fitted threshold",
            }
        features = [f"{key}.max.{q}" for q in battery.stacked_features if q in spec.questions]
        if fit.labels.all() or not fit.labels.any():
            entry["stacked"] = {"applicable": False, "reason": "the fit split has one label only"}
        else:
            model = fit_stacked(
                fit.scores, fit.labels, features, battery.stacked_c, battery.stacked_seed
            )
            for data in (fit, *evaluations):
                data.scores[f"{key}.{STACKED}"] = model.predict(data.scores)
            names.append(f"{key}.{STACKED}")
            entry["stacked"] = {
                "fitted_on": fit.split,
                "opinions": len(fit.ids),
                **model.describe(),
            }
        details[key] = entry
    return names, details, thresholds_by_scorer


def robustness_report(
    evaluations: list[SplitData],
    systems_by_scorer: dict[str, list[str]],
    directories: dict[str, Path],
    config: VerifierConfig,
    thresholds_by_scorer: dict[str, dict[str, float]],
) -> Record:
    result: Record = {}
    for key in systems_by_scorer:
        spec = config.scorers[key]
        if spec.kind not in DECISION_KINDS:
            continue
        battery = require_battery(config)
        thresholds = thresholds_by_scorer.get(key, {})
        entry: Record = {}
        for data in evaluations:
            wanted = set(data.ids)
            rows = [
                row
                for row in load_jsonl(score_file(directories[key], key, data.split))
                if row["id"] in wanted
            ]
            predictions = {
                system: data.scores[system] >= threshold for system, threshold in thresholds.items()
            }
            entry[data.split] = {
                "order_changes": order_changes(battery, spec.questions, rows),
                "panel_agreement": component_agreement(predictions) if predictions else None,
            }
        result[key] = entry
    return result


def delta_entry(values: np.ndarray, reference: np.ndarray, point: float, level: float) -> Record:
    return {
        **paired_delta(values, reference, point, level),
        "p_value": bootstrap_p_value(values - reference),
    }


def system_delta(
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    system: str,
    reference: str,
    split: str,
    key: tuple[str, str],
    level: float,
) -> Record | None:
    if key not in replicates[system][split] or key not in replicates[reference][split]:
        return None
    point = point_value(results, system, split, key) - point_value(results, reference, split, key)
    return delta_entry(
        replicates[system][split][key], replicates[reference][split][key], point, level
    )


def comparison_entry(
    comparison: Record,
    family: str,
    split: str,
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    judges: Record,
    judge_replicates: dict[str, dict[str, dict[str, np.ndarray]]],
    level: float,
    hearings: int,
) -> Record:
    judge = judges["reference_judge"]["key"]
    system, reference = comparison["system"], comparison["reference"]
    entry: Record = {
        "name": comparison["name"],
        "family": family,
        **{key: comparison[key] for key in ("system", "reference", "metric")},
    }
    needed = [system] + ([] if reference == REFERENCE_JUDGE else [reference])
    missing = [name for name in needed if name not in results]
    if missing:
        return {**entry, "missing": missing, "p_value": None}
    if comparison["metric"] == "roc_auc":
        primary = system_delta(
            results, replicates, system, reference, split, ("ranking", "roc_auc"), level
        )
        entry |= primary or {"p_value": None}
        entry["secondary_uncorrected"] = {
            "average_precision_not_inferable": system_delta(
                results,
                replicates,
                system,
                reference,
                split,
                ("ranking", "average_precision_not_inferable"),
                level,
            ),
            "cohen_kappa": system_delta(
                results, replicates, system, reference, split, (COMPARED_RULE, "cohen_kappa"), level
            ),
        }
    else:
        key = (COMPARED_RULE, "cohen_kappa")
        if key not in replicates[system][split]:
            return {**entry, "missing": [f"{system} {COMPARED_RULE}"], "p_value": None}
        point = point_value(results, system, split, key) - judge_point(
            judges, split, judge, "cohen_kappa"
        )
        entry["reference_judge"] = judge
        entry |= delta_entry(
            replicates[system][split][key],
            judge_replicates[split][judge]["cohen_kappa"],
            point,
            level,
        )
    if hearings < MIN_HEARINGS_FOR_P_VALUE:
        entry["p_value"] = None
        entry["p_value_reason"] = (
            f"the {split} split has {hearings} hearing; a hearing bootstrap needs at least "
            f"{MIN_HEARINGS_FOR_P_VALUE}"
        )
    return entry


def declared_comparisons(
    declaration: RunDeclaration,
    results: Record,
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]],
    judges: Record,
    judge_replicates: dict[str, dict[str, dict[str, np.ndarray]]],
    splits: list[str],
    config: VerifierConfig,
    hearings: dict[str, int],
) -> Record:
    level = config.confidence_level
    by_name = {comparison["name"]: comparison for comparison in declaration.comparisons}
    output: Record = {}
    for split in splits:
        families: Record = {}
        ordered: list[Record] = []
        for family in declaration.families:
            entries = [
                comparison_entry(
                    by_name[name],
                    family.name,
                    split,
                    results,
                    replicates,
                    judges,
                    judge_replicates,
                    level,
                    hearings[split],
                )
                for name in family.comparisons
            ]
            apply_holm(entries)
            missing = [entry["name"] for entry in entries if entry.get("missing")]
            families[family.name] = {
                "question": family.question,
                "comparisons": list(family.comparisons),
                "family_size": len(entries),
                "missing": missing,
                "complete": not missing,
            }
            ordered += entries
        superseded = "families" in declaration.source
        output[split] = {
            "rule": declaration.source["comparison_rule"],
            "rule_superseded_by": "families_amendment" if superseded else None,
            "holm_scope": "each_family" if superseded else "whole_list",
            "families_rule": declaration.source.get("families_amendment"),
            "family_order": [family.name for family in declaration.families],
            "families": families,
            "comparisons": ordered,
        }
    return output


def system_role(system: str, declaration: RunDeclaration | None, config: VerifierConfig) -> str:
    primary = declaration.primary_system if declaration else config.primary_system
    if system == primary:
        return "primary"
    compared = set()
    if declaration is not None:
        compared = {c["system"] for c in declaration.comparisons} | {
            c["reference"] for c in declaration.comparisons
        }
    return "compared" if system in compared else "secondary"


def interval_bounds(bootstrap: Record) -> tuple[Any, Any]:
    return bootstrap.get("low"), bootstrap.get("high")


def comparison_table(
    report: Record, config: VerifierConfig, declaration: RunDeclaration | None
) -> list[Record]:
    rows: list[Record] = []
    for split in report["splits"]["evaluate"]:
        for system, result in report["systems"].items():
            key, score = system.split(".", 1)
            spec = config.scorers[key]
            evaluation = result["evaluation"][split]
            rule = evaluation["rules"].get(COMPARED_RULE)
            auc_low, auc_high = interval_bounds(evaluation["ranking_bootstrap"]["roc_auc"])
            kappa_low, kappa_high = (
                interval_bounds(rule["bootstrap_fixed_threshold"]["cohen_kappa"])
                if rule
                else (None, None)
            )
            rows.append(
                {
                    "split": split,
                    "system": system,
                    "scorer": key,
                    "score": score,
                    "kind": spec.kind,
                    "language": spec.language,
                    "role": system_role(system, declaration, config),
                    "roc_auc": evaluation["ranking"]["roc_auc"],
                    "roc_auc_low": auc_low,
                    "roc_auc_high": auc_high,
                    "average_precision_not_inferable": evaluation["ranking"][
                        "average_precision_not_inferable"
                    ],
                    "threshold_max_f1_not_inferable": rule["threshold"] if rule else None,
                    "cohen_kappa": rule["metrics"]["cohen_kappa"] if rule else None,
                    "cohen_kappa_low": kappa_low,
                    "cohen_kappa_high": kappa_high,
                    "f1_not_inferable": (
                        rule["metrics"]["per_class"]["not_inferable"]["f1"] if rule else None
                    ),
                }
            )
        for judge, judged in report["judges"]["evaluation"][split].items():
            low, high = interval_bounds(judged["bootstrap"]["cohen_kappa"])
            rows.append(
                {
                    "split": split,
                    "system": f"judge.{judge}",
                    "scorer": "llm_judge",
                    "score": judge,
                    "kind": "llm_judge",
                    "language": "",
                    "role": (
                        "reference_judge"
                        if judge == report["judges"]["reference_judge"]["key"]
                        else "judge"
                    ),
                    "roc_auc": None,
                    "roc_auc_low": None,
                    "roc_auc_high": None,
                    "average_precision_not_inferable": None,
                    "threshold_max_f1_not_inferable": None,
                    "cohen_kappa": judged["metrics"]["cohen_kappa"],
                    "cohen_kappa_low": low,
                    "cohen_kappa_high": high,
                    "f1_not_inferable": judged["metrics"]["per_class"]["not_inferable"]["f1"],
                }
            )
        rows += comparison_rows(report, config, split)
    return [{name: row.get(name) for name in TABLE_COLUMNS} for row in rows]


def comparison_rows(report: Record, config: VerifierConfig, split: str) -> list[Record]:
    declared = report.get("declared_comparisons", {}).get(split)
    if declared is None:
        return []
    rows = []
    for entry in declared["comparisons"]:
        key, score = entry["system"].split(".", 1)
        low, high = interval_bounds(entry.get("bootstrap") or {})
        rows.append(
            {
                "split": split,
                "system": entry["system"],
                "scorer": key,
                "score": score,
                "kind": "declared_comparison",
                "language": config.scorers[key].language,
                "role": "comparison",
                "comparison": entry["name"],
                "family": entry["family"],
                "reference": entry["reference"],
                "metric": entry["metric"],
                "delta": entry.get("delta"),
                "delta_low": low,
                "delta_high": high,
                "p_value": entry.get("p_value"),
                "p_holm": entry["p_holm"],
                "missing": ";".join(entry.get("missing", [])) or None,
            }
        )
    return rows


def write_table(rows: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: "" if row.get(name) is None else row[name] for name in TABLE_COLUMNS}
            )


def declared_block(config: VerifierConfig, declaration: RunDeclaration | None) -> Record:
    evaluation = config.source["evaluation"]
    block = {
        "primary_system": config.primary_system,
        "primary_metric": config.primary_metric,
        "primary_declaration": evaluation["primary_declaration"],
        "reference_cosine_system": config.reference_cosine_system,
        "reference_judge_rule": evaluation["reference_judge_rule"],
        "threshold_rules": evaluation["threshold_rules_definition"],
        "bootstrap_intervals": evaluation["bootstrap_intervals"],
    }
    if declaration is not None:
        block |= {
            "run_declaration": declaration.name,
            "primary_system": declaration.primary_system,
            "primary_metric": declaration.primary_metric,
            "primary_declaration": declaration.source["primary_declaration"],
            "declared": declaration.source["declared"],
            "v1_primary_system": config.primary_system,
        }
    return block


def command_evaluate(args: argparse.Namespace, config: VerifierConfig) -> None:
    started = time.perf_counter()
    declaration = run_declaration(config, args)
    own = declaration.scorers if declaration else config.evaluation_scorers
    imported = declaration.imported if declaration else {}
    eval_splits = evaluated_splits(config, args.final_test)
    splits = (*config.fit_splits, *eval_splits)
    run_dir = run_directory(config, args)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    benchmark = check_benchmark_file(config)
    rows = load_benchmark_rows(config, split_of, splits)
    judge_keys = judge_keys_of(rows)
    systems_by_scorer, reports, directories = available_scorers(
        config, run_dir, splits, own, imported
    )
    data = {
        split: split_data(split, rows, systems_by_scorer, directories, judge_keys, set(own))
        for split in splits
    }
    fit = merge_fit_splits([data[split] for split in config.fit_splits])
    evaluations = [data[split] for split in eval_splits]
    fit_draws, eval_draws = draw_bootstraps(fit, evaluations, config)
    systems = [f"{key}.{name}" for key, names in systems_by_scorer.items() for name in names]
    derived_names, derived, thresholds = add_derived_systems(
        fit, evaluations, systems_by_scorer, config
    )
    systems += derived_names
    results: Record = {}
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]] = {}
    for system in systems:
        results[system], replicates[system] = evaluate_system(
            system, fit, evaluations, fit_draws, eval_draws, config
        )
        print(f"evaluated {system}", flush=True)
    judges, judge_replicates = evaluate_judges(
        fit, evaluations, eval_draws, config.confidence_level
    )
    own_reports = [report for key, report in reports.items() if key in own]
    report = {
        "experiment": "nli_verifier",
        "run_name": args.run_name,
        "created_at": now(),
        "question": (
            declaration.source["question"]
            if declaration
            else (
                "can an open NLI model tell whether an opinion is inferable from the four "
                "retrieved chunks as the annotator judged it, compared with the 12 stored LLM "
                "judges and with cosine similarity under the production encoder"
            )
        ),
        "label_semantics": config.source["benchmark"]["label_semantics"],
        "splits": {
            "fit": list(config.fit_splits),
            "evaluate": list(eval_splits),
            "final_test_flag": args.final_test,
            "test_read": "test" in splits,
            "manifest": split_source,
            "hearing_ids": {
                split: sorted(set(d.hearing_ids.tolist())) for split, d in data.items()
            },
        },
        "opinions": {split: split_summary(d) for split, d in data.items()},
        "subset": (own_reports or list(reports.values()))[0]["subset"],
        "subset_warning": (
            None
            if (own_reports or list(reports.values()))[0]["subset"] is None
            else "a smoke run on a subset: its numbers check that the code runs and are not results"
        ),
        "declared": declared_block(config, declaration),
        "systems_evaluated": systems,
        "scorers_missing": [key for key in (*own, *imported) if key not in systems_by_scorer],
        "scorers_missing_detail": missing_detail(
            [key for key in (*own, *imported) if key not in systems_by_scorer],
            run_dir,
            imported,
            splits,
        ),
        "systems": results,
        "judges": judges,
        "always_inferable": {
            d.split: evaluate_constant(d, eval_draws[d.split], config.confidence_level)
            for d in evaluations
        },
        "comparisons": compare_systems(
            results, replicates, judges, judge_replicates, list(eval_splits), config
        ),
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "unit": "hearing",
            "confidence_level": config.confidence_level,
            "seed": config.evaluation_seed,
            "method": (
                "percentile; the same replicates are shared by every system and judge, "
                "so deltas are paired"
            ),
        },
        "scorers": scorer_summaries(reports),
        "sources": {
            "benchmark": benchmark,
            "splits": split_source,
            "score_files": {key: report["files"] for key, report in reports.items()},
        },
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    if declaration is not None:
        report["imported"] = {
            key: {"run": run, "available": key in systems_by_scorer}
            for key, run in imported.items()
        }
        report["derived_systems"] = derived
        report["declared_comparisons"] = declared_comparisons(
            declaration,
            results,
            replicates,
            judges,
            judge_replicates,
            list(eval_splits),
            config,
            {d.split: len(set(d.hearing_ids.tolist())) for d in evaluations},
        )
        report["robustness"] = robustness_report(
            evaluations, systems_by_scorer, directories, config, thresholds
        )
        table = comparison_table(report, config, declaration)
        report["table"] = {"columns": list(TABLE_COLUMNS), "rows": table}
        write_table(table, run_dir / "comparison_table.csv")
    report["timing"] = {"elapsed_seconds": round(time.perf_counter() - started, 1)}
    write_json(report, run_dir / "evaluation_report.json")
    print_evaluation(report)


def format_interval(value: Any, bootstrap: Record) -> str:
    if value is None:
        return "n/a"
    if not bootstrap.get("replicates"):
        return f"{value:.3f}"
    return f"{value:.3f} [{bootstrap['low']:.3f}, {bootstrap['high']:.3f}]"


def print_evaluation(report: Record) -> None:
    for split in report["splits"]["evaluate"]:
        print(f"== {split}: {json.dumps(report['opinions'][split])}")
        for system, result in report["systems"].items():
            evaluation = result["evaluation"][split]
            rule = evaluation["rules"].get(COMPARED_RULE)
            kappa = (
                "n/a"
                if rule is None
                else format_interval(
                    rule["metrics"]["cohen_kappa"], rule["bootstrap_fixed_threshold"]["cohen_kappa"]
                )
            )
            ranking, bootstrap = evaluation["ranking"], evaluation["ranking_bootstrap"]
            auc = format_interval(ranking["roc_auc"], bootstrap["roc_auc"])
            ap_not = format_interval(
                ranking["average_precision_not_inferable"],
                bootstrap["average_precision_not_inferable"],
            )
            print(f"  {system:44s} auc={auc} ap_not={ap_not} kappa@{COMPARED_RULE}={kappa}")
        for key, judged in report["judges"]["evaluation"][split].items():
            metrics = judged["metrics"]
            kappa = format_interval(metrics["cohen_kappa"], judged["bootstrap"]["cohen_kappa"])
            f1_not = metrics["per_class"]["not_inferable"]["f1"]
            print(f"  judge {key:38s} kappa={kappa} f1_not={f1_not:.3f}")
        always = report["always_inferable"][split]["metrics"]
        print(f"  always_inferable accuracy={always['accuracy']:.3f} kappa={always['cohen_kappa']}")
        declared = report.get("declared_comparisons", {}).get(split)
        if declared is None:
            continue
        for name in declared["family_order"]:
            family = declared["families"][name]
            print(
                f"  family {name}: Holm over {family['family_size']}, "
                f"{len(family['missing'])} with a missing system"
            )
            for entry in declared["comparisons"]:
                if entry["family"] != name:
                    continue
                if entry.get("missing"):
                    print(f"    {entry['name']:66s} missing {entry['missing']}")
                    continue
                delta = entry.get("delta")
                shown = "n/a" if delta is None else f"{delta:+.3f}"
                p_value = "n/a" if entry["p_value"] is None else f"{entry['p_value']:.4f}"
                print(
                    f"    {entry['name']:66s} {entry['metric']} delta={shown} "
                    f"p={p_value} p_holm={entry['p_holm']:.4f}"
                )
    if report.get("scorers_missing"):
        print(f"scorers missing: {report['scorers_missing']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLI verifier experiment on the NLI benchmark (train fit, validation "
        "evaluation) and entailment scoring of UDV-style (proposition, evidence) pairs."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/nli_verifier.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "fetch", help="download the pinned Laya checkpoints of the laya scorers (network)"
    )
    for name in ("score", "plan", "evaluate", "pairs"):
        command = commands.add_parser(name)
        command.add_argument("--run-name", required=True)
        command.add_argument("--output-dir", type=Path, default=None)
        command.add_argument(
            "--final-test",
            action="store_true",
            help="also read, score and evaluate the final_test splits; never used for choices",
        )
        if name in ("score", "pairs", "plan"):
            command.add_argument("--scorers", nargs="+", default=None)
        if name in ("score", "pairs"):
            command.add_argument("--device", choices=DEVICES, default=None)
            command.add_argument("--decision-mode", choices=DECISION_MODES, default="live")
        if name in ("score", "plan"):
            command.add_argument("--limit-per-split", type=int, default=None)
            command.add_argument("--subset-rule", choices=SUBSET_RULES, default="random")
            command.add_argument("--translation-cache-dir", type=Path, default=None)
        if name == "score":
            command.add_argument(
                "--splits",
                nargs="+",
                default=None,
                help="score only these of the fit and evaluate splits (a smoke run)",
            )
        if name == "plan":
            command.add_argument(
                "--timing-from",
                type=Path,
                default=None,
                help="a run directory whose score reports give the measured throughput",
            )
        if name == "evaluate":
            command.add_argument(
                "--declaration",
                default=None,
                help="evaluate with [declarations.<name>] (default: the run name, if declared)",
            )
        if name == "pairs":
            command.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def command_fetch(args: argparse.Namespace, config: VerifierConfig) -> None:
    targets = dict.fromkeys(
        (spec.name, spec.revision, spec.source["subfolder"])
        for spec in config.scorers.values()
        if spec.kind == "laya"
    )
    for name, revision, subfolder in targets:
        directory = fetch_laya(name, revision, subfolder)
        print(f"{name}@{revision} {subfolder or 'root'}: {directory}", flush=True)


def main() -> None:
    code_hashes()
    args = parse_args()
    config = load_config(args.config)
    if config.hf_hub_offline and args.command != "fetch":
        enforce_offline()
    seed_everything(config.seed)
    transformers.logging.set_verbosity_error()
    commands = {
        "fetch": command_fetch,
        "score": command_score,
        "plan": command_plan,
        "evaluate": command_evaluate,
        "pairs": command_pairs,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
