import argparse
import hashlib
import json
import math
import platform
import shlex
import time
import tomllib
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from utils import (
    build_udvs,
    calibrate_threshold,
    dataset_io,
    retrieval_data,
    retrieval_experiments,
    retrieval_models,
    retrieval_stats,
    retrieval_store,
    udv_pipeline,
)
from utils.build_udvs import load_config as load_udv_config
from utils.calibrate_threshold import load_split_lookup, youden_optimum
from utils.dataset_io import load_jsonl, sha256_of_file, write_json, write_jsonl
from utils.retrieval_data import BENCHES, SPLIT_NAMES, Query, Unit
from utils.retrieval_experiments import (
    ExperimentConfig,
    Workload,
    build_retriever,
    build_workload,
    evaluate_query,
    query_file,
    retriever_ids,
    unit_mean_chars,
)
from utils.retrieval_models import (
    DenseRetriever,
    Ranking,
    RerankRetriever,
    Retriever,
    RrfRetriever,
    Runtime,
    prefix_directory,
)
from utils.retrieval_stats import stream_rng
from utils.retrieval_store import VectorStore, slug

Record = dict[str, Any]

BASE_SIGNALS = ("top1_score", "margin", "zscore")
SUBSETS = ("multi_candidate", "nontrivial")
LOGISTIC_PREFIX = "logistic."
ENTAILMENT_PREFIX = "entailment."
DIFFERENCE_METRICS = ("roc_auc", "average_precision", "aurc", "e_aurc")
POLICY_METRICS = ("precision", "coverage", "recall")
ZERO_SPREAD_RELATIVE = 1.0e-12
DEGENERATE_YOUDEN = 1.0e-12
METRIC_CHECK_TOLERANCE = 1.0e-9
CODE_MODULES = (
    udv_pipeline,
    build_udvs,
    calibrate_threshold,
    dataset_io,
    retrieval_data,
    retrieval_experiments,
    retrieval_models,
    retrieval_stats,
    retrieval_store,
)
DEFINITIONS: Record = {
    "correct": "the harness rank of the first relevant unit is 1 (score descending, candidate "
    "index ascending on ties)",
    "correct_optimistic": "rank_optimistic == 1: correct under the most favourable tie order",
    "correct_pessimistic": "rank_pessimistic == 1: correct under the least favourable tie order",
    "multi_candidate": "n_units >= population.min_candidates",
    "nontrivial": "multi_candidate and n_relevant < n_units",
    "roc_auc": "probability that a correct query has a higher signal than an incorrect one, "
    "ties counted as one half (Mann-Whitney), checked against sklearn roc_auc_score",
    "average_precision": "average precision with correct as the positive class, one step per "
    "distinct signal value, checked against sklearn average_precision_score; the value of a "
    "random signal is base_rate",
    "risk": "share of incorrect top-1 among the accepted queries",
    "aurc": "mean risk over the coverages j/n, j = 1..n, queries sorted by signal descending, "
    "tied queries counted at their expected error share",
    "oracle_aurc": "aurc of a signal that puts every correct query first",
    "e_aurc": "aurc - oracle_aurc",
    "precision_at_coverage": "1 - risk at the first ceil(c * n) queries of the signal order; the "
    "cut is set on the evaluated split itself, so it describes ranking quality and is not a "
    "deployable policy",
    "precision": "share of correct top-1 among the accepted queries (null when none is accepted)",
    "coverage": "share of queries accepted",
    "recall": "share of the correct top-1 that are accepted",
    "fixed": "percentile interval over hearing-bootstrap replicates of the evaluated split, "
    "with every threshold and regression fitted once on the full fit split",
    "refit": "the same replicates, with thresholds and regressions refitted on a hearing-bootstrap "
    "replicate of the fit split drawn in the same iteration",
    "difference": "signal metric minus the metric of evaluation.reference_signal on the same "
    "replicate; for aurc and e_aurc a negative difference means the signal ranks better",
}


@dataclass(frozen=True)
class Target:
    name: str
    retriever: str
    unit: str


@dataclass(frozen=True)
class ConfidenceConfig:
    harness_config_path: Path
    harness_run_name: str
    manifest_path: Path
    fit_splits: tuple[str, ...]
    evaluate_splits: tuple[str, ...]
    final_test_splits: tuple[str, ...]
    targets: tuple[Target, ...]
    primary_target: str
    benches: tuple[str, ...]
    min_candidates: int
    base_signals: tuple[str, ...]
    zero_spread_value: float
    nli_config_path: Path
    entailment_scorers: tuple[str, ...]
    entailment_score_key: str
    entailment_units: tuple[str, ...]
    feature_sets: dict[str, tuple[str, ...]]
    logistic: Record
    conformal_signals: tuple[str, ...]
    conformal_alphas: tuple[float, ...]
    probability_threshold: float
    production_target: str
    production_signal: str
    udv_config_path: Path
    threshold_keys: tuple[str, ...]
    threshold_calibration: dict[str, Record]
    coverages: tuple[float, ...]
    grid_step: float
    reference_signal: str
    bootstrap_samples: int
    confidence_level: float
    seed: int
    score_tolerance: float
    run_name: str
    output_dir: Path
    source: Record = field(default_factory=dict)

    @property
    def entailment_signals(self) -> tuple[str, ...]:
        return tuple(f"{ENTAILMENT_PREFIX}{scorer}" for scorer in self.entailment_scorers)


@dataclass(frozen=True)
class LogisticFit:
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    model: LogisticRegression
    positive_column: int


@dataclass(frozen=True)
class Policy:
    name: str
    signal: str
    rule: str
    threshold: float | None
    details: Record


@dataclass
class Arrays:
    query_ids: list[str]
    hearing_ids: np.ndarray
    correct: np.ndarray
    nontrivial: np.ndarray
    signals: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.query_ids)


@dataclass
class Fitted:
    models: dict[str, LogisticFit | None]
    policies: list[Policy]
    fit_signals: dict[str, np.ndarray]


def parse_target(text: str) -> Target:
    if "." not in text:
        raise SystemExit(f"target {text!r} must be <retriever id>.<unit>")
    retriever, unit = text.rsplit(".", 1)
    return Target(text, retriever, unit)


def load_config(config_path: Path) -> ConfidenceConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    splits = raw["splits"]
    fit, evaluate = tuple(splits["fit"]), tuple(splits["evaluate"])
    final_test = tuple(splits["final_test"])
    if "test" in fit + evaluate:
        raise SystemExit("splits.fit and splits.evaluate must never include test")
    if not set(fit + evaluate + final_test) <= set(SPLIT_NAMES) or set(fit) & set(evaluate):
        raise SystemExit(f"splits must be disjoint names among {SPLIT_NAMES}")
    if len(fit) != 1:
        raise SystemExit("splits.fit must name exactly one split")
    base = tuple(raw["signals"]["base"])
    if not set(base) <= set(BASE_SIGNALS) or "top1_score" not in base:
        raise SystemExit(f"signals.base must be among {BASE_SIGNALS} and include top1_score")
    scorers = tuple(raw["entailment"]["scorers"])
    known = set(base) | {f"{ENTAILMENT_PREFIX}{scorer}" for scorer in scorers}
    feature_sets = {
        name: tuple(features) for name, features in raw["logistic"]["feature_sets"].items()
    }
    for name, features in feature_sets.items():
        if not features or not set(features) <= known:
            raise SystemExit(f"logistic.feature_sets.{name} must list signals among {known}")
    conformal_signals = tuple(raw["policies"]["conformal_signals"])
    alphas = tuple(float(alpha) for alpha in raw["policies"]["conformal_alphas"])
    if not set(conformal_signals) <= known or not all(0 < alpha < 1 for alpha in alphas):
        raise SystemExit("policies.conformal_signals must be known signals and alphas in (0, 1)")
    evaluation = raw["evaluation"]
    coverages = tuple(float(value) for value in evaluation["coverages"])
    if not all(0 < value <= 1 for value in coverages):
        raise SystemExit("evaluation.coverages must be in (0, 1]")
    if evaluation["reference_signal"] not in base:
        raise SystemExit("evaluation.reference_signal must be one of signals.base")
    if evaluation["bootstrap_unit"] != "hearing":
        raise SystemExit("evaluation.bootstrap_unit must be hearing")
    targets = tuple(parse_target(text) for text in raw["targets"]["list"])
    names = [target.name for target in targets]
    if len(set(names)) != len(names) or raw["targets"]["primary"] not in names:
        raise SystemExit("targets.list must be unique and include targets.primary")
    benches = tuple(raw["targets"]["benches"])
    if not set(benches) <= set(BENCHES):
        raise SystemExit(f"targets.benches must be among {BENCHES}")
    production = raw["production_rule"]
    if production["signal"] != "top1_score":
        raise SystemExit("production_rule.signal must be top1_score")
    calibration = production["calibration"]
    if set(calibration) != set(production["threshold_keys"]):
        raise SystemExit("production_rule.calibration must describe every threshold key")
    for key, spec in calibration.items():
        needed = {"hearings_source", "hearings_field", "threshold_source", "threshold_field"}
        if not needed <= set(spec):
            raise SystemExit(f"production_rule.calibration.{key} must give {sorted(needed)}")
    return ConfidenceConfig(
        harness_config_path=Path(raw["harness"]["config_path"]),
        harness_run_name=raw["harness"]["run_name"],
        manifest_path=Path(splits["manifest_path"]),
        fit_splits=fit,
        evaluate_splits=evaluate,
        final_test_splits=final_test,
        targets=targets,
        primary_target=raw["targets"]["primary"],
        benches=benches,
        min_candidates=int(raw["population"]["min_candidates"]),
        base_signals=base,
        zero_spread_value=float(raw["signals"]["zero_spread_value"]),
        nli_config_path=Path(raw["entailment"]["nli_config_path"]),
        entailment_scorers=scorers,
        entailment_score_key=raw["entailment"]["score_key"],
        entailment_units=tuple(raw["entailment"]["units"]),
        feature_sets=feature_sets,
        logistic=raw["logistic"],
        conformal_signals=conformal_signals,
        conformal_alphas=alphas,
        probability_threshold=float(raw["policies"]["probability_threshold"]),
        production_target=production["target"],
        production_signal=production["signal"],
        udv_config_path=Path(production["udv_config_path"]),
        threshold_keys=tuple(production["threshold_keys"]),
        threshold_calibration=calibration,
        coverages=coverages,
        grid_step=float(evaluation["risk_coverage_grid_step"]),
        reference_signal=evaluation["reference_signal"],
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        confidence_level=float(evaluation["confidence_level"]),
        seed=int(evaluation["seed"]),
        score_tolerance=float(raw["collect"]["score_tolerance"]),
        run_name=raw["run"]["run_name"],
        output_dir=Path(raw["run"]["output_dir"]),
        source=raw,
    )


def load_harness(config: ConfidenceConfig) -> ExperimentConfig:
    harness = retrieval_experiments.load_config(config.harness_config_path)
    if harness.manifest_path != config.manifest_path:
        raise SystemExit(f"the harness splits come from {harness.manifest_path}")
    return harness


def check_manifest_splits(
    loaded: dict[str, dict[str, dict[str, list[Record]]]], split_of: dict[int, str]
) -> None:
    for name, benches in loaded.items():
        for bench, by_split in benches.items():
            for split, rows in by_split.items():
                wrong = [row["query_id"] for row in rows if split_of[row["hearing_id"]] != split]
                if wrong or any(row["split"] != split for row in rows):
                    raise SystemExit(f"{name} {bench} {split}: rows of another split {wrong[:5]}")


def check_targets(targets: list[Target], harness: ExperimentConfig) -> None:
    ids = retriever_ids(harness.retrievers)
    for target in targets:
        if target.retriever not in ids or target.unit not in harness.unit_kinds:
            raise SystemExit(
                f"target {target.name}: retriever must be among {sorted(ids)} and unit among "
                f"{harness.unit_kinds}"
            )


def select_targets(config: ConfidenceConfig, requested: str | None) -> list[Target]:
    if not requested:
        return list(config.targets)
    return [parse_target(item.strip()) for item in requested.split(",") if item.strip()]


def evaluated_splits(config: ConfidenceConfig, final_test: bool) -> tuple[str, ...]:
    return config.evaluate_splits + (config.final_test_splits if final_test else ())


def resolve_splits(config: ConfidenceConfig, final_test: bool) -> tuple[str, ...]:
    return config.fit_splits + evaluated_splits(config, final_test)


def run_directory(config: ConfidenceConfig, run_name: str | None) -> Path:
    return config.output_dir / (run_name or config.run_name)


def features_file(run_dir: Path, bench: str, target: str, split: str) -> Path:
    return run_dir / "features" / bench / target / f"{split}.jsonl"


def predictions_file(run_dir: Path, bench: str, target: str, split: str) -> Path:
    return run_dir / "predictions" / bench / target / f"{split}.jsonl"


def pairs_input_file(run_dir: Path) -> Path:
    return run_dir / "nli" / "pairs_input.jsonl"


def entailment_file(run_dir: Path, run_name: str, scorer: str) -> Path:
    return run_dir / "nli" / "pairs" / run_name / f"{scorer}.jsonl"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def pair_id_of(bench: str, query_id: str, sha: str) -> str:
    return f"{bench}:{query_id}#{sha[:16]}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def code_hashes() -> Record:
    files = {
        f"utils/{Path(module.__file__).name}": Path(module.__file__)
        for module in CODE_MODULES
        if module.__file__
    }
    files["utils/confidence_policies.py"] = Path(__file__)
    return {name: sha256_of_file(path) for name, path in sorted(files.items())}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def store_directories(retriever: str, harness: ExperimentConfig) -> list[Path]:
    name, _ = retriever_ids(harness.retrievers)[retriever]
    spec = harness.retrievers[name]
    if spec["kind"] == "dense":
        base = Path(harness.cache["embeddings_dir"]) / slug(spec["model"]) / spec["revision"][:12]
        prefixes = dict.fromkeys((spec["query_prefix"], spec["passage_prefix"]))
        return [
            base / prefix_directory(prefix) / f"msl{spec['max_seq_length']}" for prefix in prefixes
        ]
    if spec["kind"] == "rrf":
        return [path for part in spec["components"] for path in store_directories(part, harness)]
    if spec["kind"] == "rerank":
        own = (
            Path(harness.cache["rerank_dir"])
            / slug(spec["model"])
            / spec["revision"][:12]
            / f"maxlen{spec['max_length']}"
        )
        return [own, *store_directories(spec["base"], harness)]
    return []


def model_revisions(retriever: str, harness: ExperimentConfig) -> list[Record]:
    name, _ = retriever_ids(harness.retrievers)[retriever]
    spec = harness.retrievers[name]
    if spec["kind"] in ("dense", "rerank"):
        own = [{"retriever": name, "model": spec["model"], "revision": spec["revision"]}]
        return own + (model_revisions(spec["base"], harness) if "base" in spec else [])
    if spec["kind"] == "rrf":
        return [item for part in spec["components"] for item in model_revisions(part, harness)]
    return []


def missing_inputs(
    target: Target,
    harness: ExperimentConfig,
    harness_dir: Path,
    benches: tuple[str, ...],
    splits: tuple[str, ...],
) -> list[str]:
    covered = harness_coverage(harness_dir, target)
    missing = [
        str(query_file(harness_dir, bench, target.unit, target.retriever, split))
        for bench in benches
        for split in splits
        if not query_file(harness_dir, bench, target.unit, target.retriever, split).exists()
        and (bench, split) not in covered
    ]
    missing += [
        str(path / "store.json")
        for path in store_directories(target.retriever, harness)
        if not (path / "store.json").exists()
    ]
    return missing


def refuse(what: str) -> Callable[..., Any]:
    def raise_error(*args: Any, **kwargs: Any) -> Any:
        raise SystemExit(
            f"{what} was requested but is not in the harness cache: run the harness step of this "
            "retriever first (collect never encodes, scores or writes cache files)"
        )

    return raise_error


def keep_in_memory(*args: Any, **kwargs: Any) -> None:
    return None


def replace_methods(target: object, replacements: dict[str, Callable[..., Any]]) -> None:
    for name, replacement in replacements.items():
        setattr(target, name, replacement)


def guard_store(store: VectorStore, what: str) -> None:
    if len(store) == 0:
        raise SystemExit(f"{store.directory}: the harness cache is empty")
    replace_methods(store, {"add": refuse(what), "flush": keep_in_memory})


def guard_retriever(retriever: Retriever) -> None:
    if isinstance(retriever, DenseRetriever):
        what = f"a {retriever.spec.model} vector"
        replace_methods(retriever.encoder, {"encode": refuse(what), "load": refuse(what)})
        for store in retriever.encoder.stores.values():
            guard_store(store, what)
    elif isinstance(retriever, RerankRetriever):
        what = f"a {retriever.spec.model} score"
        replace_methods(retriever, {"load": refuse(what)})
        guard_store(retriever.store, what)
        guard_retriever(retriever.base)
    elif isinstance(retriever, RrfRetriever):
        for component in retriever.components:
            guard_retriever(component)


def load_harness_rows(
    harness_dir: Path, target: Target, benches: tuple[str, ...], splits: tuple[str, ...]
) -> tuple[dict[tuple[str, str], Record], list[Record]]:
    rows: dict[tuple[str, str], Record] = {}
    files: list[Record] = []
    for bench in benches:
        for split in splits:
            path = query_file(harness_dir, bench, target.unit, target.retriever, split)
            if not path.exists():
                files.append({"path": str(path), "rows": 0, "sha256": None, "empty_run": True})
                continue
            members = load_jsonl(path)
            for row in members:
                expected = (bench, split, target.retriever, target.unit)
                if (row["bench"], row["split"], row["retriever"], row["unit"]) != expected:
                    raise SystemExit(f"{path}: row {row['query_id']} belongs to another file")
                rows[(bench, row["query_id"])] = row
            files.append({"path": str(path), "rows": len(members), "sha256": sha256_of_file(path)})
    return rows, files


def harness_coverage(harness_dir: Path, target: Target) -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for path in sorted((harness_dir / "runs").glob(f"{target.retriever}__*_report.json")):
        with open(path) as f:
            report = json.load(f)
        if report.get("retriever") == target.retriever and target.unit in report["units"]:
            covered.update(
                (bench, split) for bench in report["benches"] for split in report["splits_used"]
            )
    return covered


def harness_run_reports(harness_dir: Path, target: Target) -> list[Record]:
    reports = []
    for path in sorted((harness_dir / "runs").glob(f"{target.retriever}__*_report.json")):
        with open(path) as f:
            report = json.load(f)
        if report.get("retriever") != target.retriever or target.unit not in report["units"]:
            continue
        reports.append(
            {
                "path": str(path),
                "sha256": sha256_of_file(path),
                "created_at": report["created_at"],
                "splits_used": report["splits_used"],
                "units": report["units"],
                "device": report["device"],
                "code": report["code"],
            }
        )
    return reports


def ranking_signals(ranking: Ranking, zero_value: float) -> Record:
    display = np.asarray(ranking.display, dtype=np.float64)
    order = ranking.order
    top1 = float(display[order[0]])
    second = float(display[order[1]]) if len(order) > 1 else math.nan
    scored = display[~np.isnan(display)]
    mean, std = float(scored.mean()), float(scored.std())
    zero_spread = bool(std <= ZERO_SPREAD_RELATIVE * max(1.0, abs(mean)))
    defined = len(scored) >= 2 and not math.isnan(second)
    zscore = None
    if defined:
        zscore = zero_value if zero_spread else (top1 - mean) / std
    return {
        "top1_score": top1,
        "top2_score": None if math.isnan(second) else second,
        "margin": None if math.isnan(second) else top1 - second,
        "zscore": zscore,
        "score_mean": mean,
        "score_std": std,
        "scored_units": int(len(scored)),
        "zero_spread": zero_spread,
    }


def row_differences(built: Record, stored: Record, tolerance: float) -> list[str]:
    if set(built) != set(stored):
        return ["fields"]
    problems = [key for key in built if key != "top" and built[key] != stored[key]]
    built_ids = [item[0] for item in built["top"]]
    if built_ids != [item[0] for item in stored["top"]]:
        return [*problems, "top_ids"]
    for (_, mine), (_, theirs) in zip(built["top"], stored["top"], strict=True):
        if (mine is None) != (theirs is None) or (
            mine is not None and abs(mine - theirs) > tolerance
        ):
            return [*problems, "top_scores"]
    return problems


def feature_row(
    built: Record, target: Target, units: list[Unit], ranking: Ranking, zero_value: float
) -> Record:
    top1 = units[int(ranking.order[0])]
    return {
        "bench": built["bench"],
        "query_id": built["query_id"],
        "split": built["split"],
        "hearing_id": built["hearing_id"],
        "target": target.name,
        "retriever": target.retriever,
        "unit": target.unit,
        "n_units": built["n_units"],
        "n_relevant": built["n_relevant"],
        "rank": built["rank"],
        "rank_optimistic": built["rank_optimistic"],
        "rank_pessimistic": built["rank_pessimistic"],
        "correct": built["rank"] == 1,
        "correct_optimistic": built["rank_optimistic"] == 1,
        "correct_pessimistic": built["rank_pessimistic"] == 1,
        "top1_id": top1.unit_id,
        "top1_chars": len(top1.text),
        "top1_text_sha256": text_sha256(top1.text),
        **ranking_signals(ranking, zero_value),
    }


def collect_target(
    target: Target,
    harness: ExperimentConfig,
    harness_dir: Path,
    workload: Workload,
    runtime: Runtime,
    config: ConfidenceConfig,
    splits: tuple[str, ...],
    selected_hearings: set[int] | None,
) -> tuple[list[Record], Record]:
    stored_rows, row_files = load_harness_rows(harness_dir, target, config.benches, splits)
    retriever = build_retriever(target.retriever, harness, runtime)
    guard_retriever(retriever)
    started = time.perf_counter()
    features: list[Record] = []
    problems: Counter[str] = Counter()
    examples: list[Record] = []
    seen: set[tuple[str, str]] = set()
    for hearing in workload.hearings:
        queries = workload.queries[hearing.hearing_id]
        rankings = retriever.rank_hearing(hearing, target.unit, queries)
        means: dict[str, float] = {}
        for query in queries:
            units = hearing.contexts[query.context_key].units[target.unit]
            if query.context_key not in means:
                means[query.context_key] = unit_mean_chars(units)
            ranking = rankings[query.query_id]
            built = evaluate_query(
                query,
                target.unit,
                units,
                ranking,
                target.retriever,
                means[query.context_key],
                harness.evaluation,
            )
            key = (query.bench, query.query_id)
            stored = stored_rows.get(key)
            if stored is None:
                problems["query_without_harness_row"] += 1
                examples.append({"query_id": query.query_id, "problem": "no harness row"})
                continue
            seen.add(key)
            differences = row_differences(built, stored, config.score_tolerance)
            if differences:
                problems.update(differences)
                examples.append({"query_id": query.query_id, "fields": differences})
            features.append(feature_row(built, target, units, ranking, config.zero_spread_value))
    unvisited = [key for key in stored_rows if key not in seen]
    outside = [
        key
        for key in unvisited
        if selected_hearings is not None and stored_rows[key]["hearing_id"] not in selected_hearings
    ]
    if len(unvisited) > len(outside):
        problems["harness_row_not_rebuilt"] += len(unvisited) - len(outside)
    details = retriever.describe()
    retriever.close()
    if problems:
        raise SystemExit(
            f"{target.name}: rebuilt rankings disagree with the harness rows: {dict(problems)}; "
            f"first cases {examples[:5]}"
        )
    counts: Record = {}
    for row in features:
        split_counts = counts.setdefault(f"{row['bench']}.{row['split']}", Counter())
        split_counts["queries"] += 1
        split_counts["multi_candidate"] += int(row["n_units"] >= config.min_candidates)
        split_counts["correct"] += int(row["correct"])
        split_counts["label_depends_on_ties"] += int(
            row["correct_optimistic"] != row["correct_pessimistic"]
        )
        split_counts["zero_spread_multi_candidate"] += int(
            row["zero_spread"] and row["n_units"] >= config.min_candidates
        )
    report = {
        "target": target.name,
        "retriever": target.retriever,
        "unit": target.unit,
        "harness_rows": row_files,
        "harness_rows_checked": len(seen),
        "harness_rows_outside_selected_hearings": len(outside),
        "harness_run_reports": harness_run_reports(harness_dir, target),
        "model_revisions": model_revisions(target.retriever, harness),
        "cache_directories": [str(path) for path in store_directories(target.retriever, harness)],
        "retriever_details": details,
        "counts": {key: dict(value) for key, value in sorted(counts.items())},
        "seconds": round(time.perf_counter() - started, 1),
    }
    return features, report


def filter_workload(workload: Workload, hearing_ids: set[int] | None) -> Workload:
    if hearing_ids is None:
        return workload
    hearings = [hearing for hearing in workload.hearings if hearing.hearing_id in hearing_ids]
    missing = hearing_ids - {hearing.hearing_id for hearing in hearings}
    if missing:
        raise SystemExit(f"hearings {sorted(missing)} have no benchmark query in these splits")
    return Workload(
        hearings=hearings,
        queries={hearing.hearing_id: workload.queries[hearing.hearing_id] for hearing in hearings},
        checks=workload.checks,
        span_checks=workload.span_checks,
        sources=workload.sources,
        masked_texts_by_hearing=workload.masked_texts_by_hearing,
    )


def parse_hearing_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def split_record(config: ConfidenceConfig, final_test: bool) -> Record:
    return {
        "fit": list(config.fit_splits),
        "evaluate": list(config.evaluate_splits),
        "final_test": final_test,
        "final_test_splits": list(config.final_test_splits) if final_test else [],
    }


def command_collect(args: argparse.Namespace, config: ConfidenceConfig) -> None:
    harness = load_harness(config)
    targets = select_targets(config, args.targets)
    check_targets(targets, harness)
    splits = resolve_splits(config, args.final_test)
    harness_dir = Path(harness.run["output_dir"]) / config.harness_run_name
    hearing_ids = parse_hearing_ids(args.hearing_ids)
    default_name = f"{config.run_name}_smoke" if hearing_ids is not None else config.run_name
    run_dir = run_directory(config, args.run_name or default_name)
    available, skipped = [], {}
    for target in targets:
        missing = missing_inputs(target, harness, harness_dir, config.benches, splits)
        if missing:
            skipped[target.name] = missing
            print(f"[collect] skip {target.name}: missing {missing[:3]}", flush=True)
        else:
            available.append(target)
    if skipped:
        write_json(
            {"created_at": now(), "splits_used": list(splits), "skipped": skipped},
            run_dir / "collect" / "skipped_targets.json",
        )
    if not available:
        raise SystemExit("no target has harness rows and caches for every requested split")
    started = time.perf_counter()
    workload = filter_workload(
        build_workload(harness, splits, config.benches, None, None), hearing_ids
    )
    print(
        f"[collect] {len(workload.hearings)} hearings loaded in "
        f"{time.perf_counter() - started:.1f}s; targets {[t.name for t in available]}",
        flush=True,
    )
    runtime = Runtime(
        device="cpu",
        embeddings_dir=Path(harness.cache["embeddings_dir"]),
        rerank_dir=Path(harness.cache["rerank_dir"]),
        shard_size=harness.cache["shard_size"],
        udv_config=load_udv_config(Path(harness.cache["production_config"])),
        masked_texts_by_hearing=workload.masked_texts_by_hearing,
    )
    code = code_hashes()
    for target in available:
        features, details = collect_target(
            target, harness, harness_dir, workload, runtime, config, splits, hearing_ids
        )
        outputs = []
        for bench in config.benches:
            for split in splits:
                members = [r for r in features if r["bench"] == bench and r["split"] == split]
                path = features_file(run_dir, bench, target.name, split)
                write_jsonl(members, path)
                outputs.append(
                    {"path": str(path), "rows": len(members), "sha256": sha256_of_file(path)}
                )
        report = {
            "experiment": "confidence_policies",
            "step": "collect",
            "run_name": run_dir.name,
            "created_at": now(),
            "splits": split_record(config, args.final_test),
            "splits_used": list(splits),
            "hearing_filter": sorted(hearing_ids) if hearing_ids is not None else None,
            "device": "none: no model is loaded; dense vectors and cross-encoder scores are read "
            "from the harness caches and sparse retrievers are refitted on CPU",
            **details,
            "outputs": outputs,
            "sources": workload.sources,
            "code": code,
            "environment": environment(),
            "config": config.source,
            "harness_config": harness.source,
        }
        write_json(clean(report), run_dir / "collect" / f"{target.name}_report.json")
        print(f"[collect] {target.name}: {len(features)} rows, {details['counts']}", flush=True)


def load_features(
    run_dir: Path, target: Target, benches: tuple[str, ...], splits: tuple[str, ...]
) -> tuple[dict[str, dict[str, list[Record]]], list[Record]] | None:
    by_bench: dict[str, dict[str, list[Record]]] = {}
    files = []
    for bench in benches:
        for split in splits:
            path = features_file(run_dir, bench, target.name, split)
            if not path.exists():
                return None
            rows = load_jsonl(path)
            by_bench.setdefault(bench, {})[split] = rows
            files.append({"path": str(path), "rows": len(rows), "sha256": sha256_of_file(path)})
    return by_bench, files


def collect_reports(run_dir: Path, names: list[str]) -> Record:
    reports: Record = {}
    for name in names:
        path = run_dir / "collect" / f"{name}_report.json"
        if not path.exists():
            reports[name] = {"path": str(path), "available": False}
            continue
        with open(path) as f:
            report = json.load(f)
        reports[name] = {
            "path": str(path),
            "sha256": sha256_of_file(path),
            "created_at": report["created_at"],
            "hearing_filter": report["hearing_filter"],
            "harness_rows_checked": report["harness_rows_checked"],
            "harness_run_reports": [item["path"] for item in report["harness_run_reports"]],
            "model_revisions": report["model_revisions"],
            "code_sha256": report["code"].get("utils/confidence_policies.py"),
        }
        if report["hearing_filter"] is not None:
            print(f"WARNING {name}: collected on hearings {report['hearing_filter']} only")
    return reports


def command_pairs(args: argparse.Namespace, config: ConfidenceConfig) -> None:
    harness = load_harness(config)
    splits = resolve_splits(config, args.final_test)
    run_dir = run_directory(config, args.run_name)
    targets = [t for t in select_targets(config, args.targets) if t.unit in config.entailment_units]
    loaded: dict[str, dict[str, dict[str, list[Record]]]] = {}
    inputs: list[Record] = []
    for target in targets:
        result = load_features(run_dir, target, config.benches, splits)
        if result is not None:
            loaded[target.name], files = result
            inputs += files
    if not loaded:
        raise SystemExit(f"no collected features for targets with units {config.entailment_units}")
    workload = build_workload(harness, splits, config.benches, None, None)
    lookup: dict[tuple[str, str, str], tuple[Query, list[Unit]]] = {}
    unit_of = {target.name: target.unit for target in targets}
    for hearing in workload.hearings:
        for query in workload.queries[hearing.hearing_id]:
            for unit_name in {unit_of[name] for name in loaded}:
                lookup[(query.bench, query.query_id, unit_name)] = (
                    query,
                    hearing.contexts[query.context_key].units[unit_name],
                )
    pairs: dict[str, Record] = {}
    for name, benches in loaded.items():
        for bench, by_split in benches.items():
            for rows in by_split.values():
                for row in rows:
                    query, units = lookup[(bench, row["query_id"], unit_of[name])]
                    unit = next(u for u in units if u.unit_id == row["top1_id"])
                    if text_sha256(unit.text) != row["top1_text_sha256"]:
                        raise SystemExit(f"{name} {row['query_id']}: top-1 text changed")
                    pair_id = pair_id_of(bench, row["query_id"], row["top1_text_sha256"])
                    entry = pairs.setdefault(
                        pair_id,
                        {
                            "pair_id": pair_id,
                            "query_id": row["query_id"],
                            "hearing_id": row["hearing_id"],
                            "proposition": query.text,
                            "evidence": unit.text,
                            "meta": {
                                "bench": bench,
                                "split": row["split"],
                                "top1_text_sha256": row["top1_text_sha256"],
                                "targets": [],
                            },
                        },
                    )
                    entry["meta"]["targets"].append(name)
    ordered = sorted(
        pairs.values(), key=lambda p: (p["meta"]["bench"], p["hearing_id"], p["pair_id"])
    )
    path = pairs_input_file(run_dir)
    write_jsonl(ordered, path)
    counts = Counter(f"{p['meta']['bench']}.{p['meta']['split']}" for p in ordered)
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "utils.nli_verifier_experiments",
        "--config",
        str(config.nli_config_path),
        "pairs",
        "--input",
        str(path),
        "--run-name",
        run_dir.name,
        "--output-dir",
        str(run_dir / "nli"),
        "--scorers",
        *config.entailment_scorers,
        *(["--final-test"] if args.final_test else []),
    ]
    report = {
        "experiment": "confidence_policies",
        "step": "pairs",
        "run_name": run_dir.name,
        "created_at": now(),
        "splits": split_record(config, args.final_test),
        "splits_used": list(splits),
        "targets": sorted(loaded),
        "pairs": len(ordered),
        "pairs_by_bench_split": dict(sorted(counts.items())),
        "collect_reports": collect_reports(run_dir, sorted(loaded)),
        "feature_rows": sum(
            len(rows)
            for benches in loaded.values()
            for s in benches.values()
            for rows in s.values()
        ),
        "pair_rule": config.source["entailment"]["pair"],
        "output": {"path": str(path), "sha256": sha256_of_file(path)},
        "scoring_command": shlex.join(command),
        "inputs": inputs,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    write_json(clean(report), run_dir / "nli" / "pairs_input_report.json")
    print(f"[pairs] {len(ordered)} pairs {dict(counts)} -> {path}", flush=True)
    print(f"[pairs] score them with:\n{shlex.join(command)}", flush=True)


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(scores)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    n_positive = int(positive.sum())
    if n_positive == 0:
        return math.nan
    order = np.argsort(-scores, kind="stable")
    ordered, hits = scores[order], positive[order]
    last = np.append(np.flatnonzero(ordered[1:] != ordered[:-1]), len(ordered) - 1)
    true_positives = np.cumsum(hits)[last]
    precision = true_positives / (last + 1)
    recall = true_positives / n_positive
    return float(np.sum(np.diff(np.append(0.0, recall)) * precision))


def expected_cumulative_errors(scores: np.ndarray, errors: np.ndarray) -> np.ndarray:
    size = len(scores)
    if size == 0:
        return np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    ordered, wrong = scores[order], errors[order].astype(np.float64)
    starts = np.flatnonzero(np.append(True, ordered[1:] != ordered[:-1]))
    sizes = np.diff(np.append(starts, size))
    group_errors = np.add.reduceat(wrong, starts)
    before = np.append(0.0, np.cumsum(group_errors)[:-1])
    group = np.repeat(np.arange(len(starts)), sizes)
    position = np.arange(size) - starts[group] + 1
    return before[group] + position * group_errors[group] / sizes[group]


def coverage_count(coverage: float, size: int) -> int:
    return max(1, min(size, math.ceil(coverage * size - 1e-9)))


def signal_metrics(scores: np.ndarray, correct: np.ndarray, coverages: tuple[float, ...]) -> Record:
    size = len(scores)
    if size == 0:
        return {"queries": 0}
    cumulative = expected_cumulative_errors(scores, ~correct)
    counts = np.arange(1, size + 1)
    n_correct = int(correct.sum())
    oracle = np.maximum(0, counts - n_correct) / counts
    aurc = float((cumulative / counts).mean())
    metrics: Record = {
        "queries": size,
        "base_rate": n_correct / size,
        "roc_auc": roc_auc(scores, correct),
        "average_precision": average_precision(scores, correct),
        "aurc": aurc,
        "oracle_aurc": float(oracle.mean()),
        "e_aurc": aurc - float(oracle.mean()),
    }
    for coverage in coverages:
        count = coverage_count(coverage, size)
        metrics[f"precision_at_{coverage:g}"] = float(1 - cumulative[count - 1] / count)
    return metrics


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray, step: float) -> list[list[float]]:
    size = len(scores)
    if size == 0:
        return []
    cumulative = expected_cumulative_errors(scores, ~correct)
    grid = np.round(np.arange(step, 1 + 1e-9, step), 10)
    points = []
    for coverage in grid:
        count = coverage_count(float(coverage), size)
        points.append([float(coverage), round(float(cumulative[count - 1] / count), 6)])
    return points


def check_fast_metrics(scores: np.ndarray, correct: np.ndarray, where: str) -> None:
    if correct.all() or not correct.any():
        return
    pairs = (
        (roc_auc(scores, correct), roc_auc_score(correct, scores), "roc_auc"),
        (
            average_precision(scores, correct),
            average_precision_score(correct, scores),
            "average_precision",
        ),
    )
    for fast, reference, name in pairs:
        if abs(fast - float(reference)) > METRIC_CHECK_TOLERANCE:
            raise SystemExit(f"{where}: {name} {fast} differs from sklearn {reference}")


def policy_metrics(accepted: np.ndarray, correct: np.ndarray) -> Record:
    size = len(accepted)
    kept = int(accepted.sum())
    n_correct = int(correct.sum())
    kept_correct = int((accepted & correct).sum())
    return {
        "queries": size,
        "accepted": kept,
        "accepted_correct": kept_correct,
        "coverage": kept / size if size else math.nan,
        "precision": kept_correct / kept if kept else math.nan,
        "recall": kept_correct / n_correct if n_correct else math.nan,
    }


def fit_logistic(
    signals: dict[str, np.ndarray],
    correct: np.ndarray,
    features: tuple[str, ...],
    params: Record,
    seed: int,
) -> LogisticFit | None:
    if len(correct) == 0 or correct.all() or not correct.any():
        return None
    matrix = np.column_stack([signals[name] for name in features])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    scale = np.where(std > 0, std, 1.0)
    model = LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=None if params["class_weight"] == "none" else params["class_weight"],
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit((matrix - mean) / scale, correct)
    return LogisticFit(features, mean, scale, model, list(model.classes_).index(True))


def predict_logistic(fit: LogisticFit | None, signals: dict[str, np.ndarray]) -> np.ndarray:
    size = len(next(iter(signals.values())))
    values = np.full(size, math.nan)
    if fit is None:
        return values
    matrix = np.column_stack([signals[name] for name in fit.features])
    finite = np.isfinite(matrix).all(axis=1)
    if finite.any():
        standardized = (matrix[finite] - fit.mean) / fit.scale
        values[finite] = fit.model.predict_proba(standardized)[:, fit.positive_column]
    return values


def logistic_record(fit: LogisticFit | None, correct: np.ndarray) -> Record:
    if fit is None:
        return {"fitted": False, "reason": "the fit rows hold one class only or no row"}
    return {
        "fitted": True,
        "features": list(fit.features),
        "coefficients_standardized": dict(
            zip(fit.features, fit.model.coef_[0].tolist(), strict=True)
        ),
        "intercept": float(fit.model.intercept_[0]),
        "feature_mean": dict(zip(fit.features, fit.mean.tolist(), strict=True)),
        "feature_scale": dict(zip(fit.features, fit.scale.tolist(), strict=True)),
        "iterations": int(fit.model.n_iter_[0]),
        "fit_rows": len(correct),
        "fit_positive_rate": float(correct.mean()),
    }


def youden_policy(signal: str, values: np.ndarray, correct: np.ndarray) -> Policy:
    optimum = youden_optimum(values, correct)
    name = f"youden:{signal}"
    if optimum is None:
        return Policy(name, signal, "youden", None, {"reason": "one class only in the fit rows"})
    details = {key: rounded(value) for key, value in optimum.items()}
    details["degenerate"] = optimum["youden_j"] <= DEGENERATE_YOUDEN
    return Policy(name, signal, "youden", float(optimum["threshold"]), details)


def conformal_policy(signal: str, values: np.ndarray, correct: np.ndarray, alpha: float) -> Policy:
    calibration = np.sort(values[correct])
    size = len(calibration)
    rank = math.floor(alpha * (size + 1))
    name = f"conformal:{signal}:alpha={alpha:g}"
    details = {"alpha": alpha, "target_recall": 1 - alpha, "calibration_queries": size, "k": rank}
    if size == 0:
        return Policy(name, signal, "conformal", None, {**details, "reason": "no correct fit row"})
    threshold = -math.inf if rank == 0 else float(calibration[rank - 1])
    return Policy(name, signal, "conformal", threshold, {**details, "accepts_all": rank == 0})


def fit_policies(
    signals: dict[str, np.ndarray], correct: np.ndarray, names: list[str], config: ConfidenceConfig
) -> list[Policy]:
    policies = [youden_policy(name, signals[name], correct) for name in names]
    for name in config.conformal_signals:
        if name in names:
            policies += [
                conformal_policy(name, signals[name], correct, alpha)
                for alpha in config.conformal_alphas
            ]
    policies += [
        Policy(
            f"probability:{name}",
            name,
            "probability",
            config.probability_threshold,
            {"threshold": config.probability_threshold},
        )
        for name in names
        if name.startswith(LOGISTIC_PREFIX)
    ]
    return policies


def accepted_by(policy: Policy, values: np.ndarray) -> np.ndarray | None:
    if policy.threshold is None:
        return None
    return np.asarray(values >= policy.threshold)


def fit_all(
    fit: Arrays,
    raw_names: list[str],
    feature_sets: dict[str, tuple[str, ...]],
    config: ConfidenceConfig,
) -> Fitted:
    raw = {name: fit.signals[name] for name in raw_names}
    models = {
        f"{LOGISTIC_PREFIX}{name}": fit_logistic(
            raw, fit.correct, features, config.logistic, config.seed
        )
        for name, features in feature_sets.items()
    }
    signals = dict(raw)
    for name, model in models.items():
        signals[name] = predict_logistic(model, raw)
    names = [
        name
        for name in signals
        if models.get(name, True) is not None and len(fit) and np.isfinite(signals[name]).all()
    ]
    return Fitted(models, fit_policies(signals, fit.correct, names, config), signals)


def apply_models(fitted: Fitted, arrays: Arrays, raw_names: list[str]) -> dict[str, np.ndarray]:
    raw = {name: arrays.signals[name] for name in raw_names}
    signals = dict(raw)
    for name, model in fitted.models.items():
        signals[name] = predict_logistic(model, raw)
    return signals


def arrays_of(rows: list[Record], names: list[str]) -> Arrays:
    return Arrays(
        query_ids=[row["query_id"] for row in rows],
        hearing_ids=np.array([row["hearing_id"] for row in rows], dtype=np.int64),
        correct=np.array([row["correct"] for row in rows], dtype=bool),
        nontrivial=np.array([row["n_relevant"] < row["n_units"] for row in rows], dtype=bool),
        signals={
            name: np.array(
                [math.nan if row.get(name) is None else row[name] for row in rows],
                dtype=np.float64,
            )
            for name in names
        },
    )


def take(arrays: Arrays, index: np.ndarray) -> Arrays:
    return Arrays(
        query_ids=[arrays.query_ids[i] for i in index],
        hearing_ids=arrays.hearing_ids[index],
        correct=arrays.correct[index],
        nontrivial=arrays.nontrivial[index],
        signals={name: values[index] for name, values in arrays.signals.items()},
    )


def hearing_members(hearing_ids: np.ndarray, universe: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(hearing_ids == hearing) for hearing in universe]


def replicate_index(members: list[np.ndarray], draw: np.ndarray) -> np.ndarray:
    parts = [members[position] for position in draw]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def finite_interval(values: list[float], level: float) -> Record:
    array = np.array(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"low": None, "high": None, "replicates": 0}
    tail = (1 - level) / 2
    return {
        "low": rounded(float(np.quantile(finite, tail))),
        "high": rounded(float(np.quantile(finite, 1 - tail))),
        "replicates": int(len(finite)),
    }


def excludes_zero(interval: Record) -> bool | None:
    if interval["low"] is None or interval["high"] is None:
        return None
    return bool(interval["low"] > 0 or interval["high"] < 0)


def subset_index(arrays: Arrays, index: np.ndarray, subset: str) -> np.ndarray:
    if subset == "nontrivial":
        return index[arrays.nontrivial[index]]
    return index


class Replicates:
    def __init__(self) -> None:
        self.values: dict[tuple[str, ...], list[float]] = {}

    def add(self, key: tuple[str, ...], value: float) -> None:
        self.values.setdefault(key, []).append(value)

    def interval(self, key: tuple[str, ...], level: float) -> Record | None:
        if key not in self.values:
            return None
        return finite_interval(self.values[key], level)


def record_signal_replicates(
    store: Replicates,
    mode: str,
    signals: dict[str, np.ndarray],
    names: list[str],
    arrays: Arrays,
    index: np.ndarray,
    config: ConfidenceConfig,
) -> None:
    for subset in SUBSETS:
        chosen = subset_index(arrays, index, subset)
        if len(chosen) == 0:
            continue
        correct = arrays.correct[chosen]
        reference = signal_metrics(signals[config.reference_signal][chosen], correct, ())
        for name in names:
            values = signals[name][chosen]
            if not np.isfinite(values).all():
                continue
            metrics = signal_metrics(values, correct, config.coverages)
            for metric, value in metrics.items():
                if metric != "queries":
                    store.add((mode, subset, name, metric), value)
            if name == config.reference_signal:
                continue
            for metric in DIFFERENCE_METRICS:
                store.add(
                    (mode, subset, name, f"difference_{metric}"),
                    metrics[metric] - reference[metric],
                )


def record_policy_replicates(
    store: Replicates,
    mode: str,
    policies: list[Policy],
    signals: dict[str, np.ndarray],
    correct: np.ndarray,
    index: np.ndarray,
) -> None:
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal][index])
        if accepted is None:
            continue
        metrics = policy_metrics(accepted, correct[index])
        for metric in POLICY_METRICS:
            store.add((mode, policy.name, metric), metrics[metric])


def policy_threshold_record(policy: Policy) -> float | str | None:
    if policy.threshold is None:
        return None
    if math.isinf(policy.threshold):
        return "accept_all"
    return rounded(policy.threshold, 8)


def evaluate_split(
    fitted: Fitted,
    fit: Arrays,
    evaluation: Arrays,
    raw_names: list[str],
    feature_sets: dict[str, tuple[str, ...]],
    config: ConfidenceConfig,
    fit_draws: tuple[np.ndarray, np.ndarray],
    eval_draws: tuple[np.ndarray, np.ndarray],
    production: list[Policy],
    where: str,
) -> Record:
    signals = apply_models(fitted, evaluation, raw_names)
    names = [name for name in signals if np.isfinite(signals[name]).all()]
    logistic_names = [name for name in names if name.startswith(LOGISTIC_PREFIX)]
    point: Record = {}
    curves: Record = {}
    for subset in SUBSETS:
        chosen = subset_index(evaluation, np.arange(len(evaluation)), subset)
        correct = evaluation.correct[chosen]
        point[subset], curves[subset] = {}, {}
        for name in names:
            values = signals[name][chosen]
            check_fast_metrics(values, correct, f"{where}|{subset}|{name}")
            point[subset][name] = signal_metrics(values, correct, config.coverages)
            curves[subset][name] = risk_coverage_curve(values, correct, config.grid_step)
    policies = fitted.policies + production
    policy_point = {}
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal])
        if accepted is not None:
            policy_point[policy.name] = policy_metrics(accepted, evaluation.correct)
    store = Replicates()
    fit_universe, fit_matrix = fit_draws
    eval_universe, eval_matrix = eval_draws
    fit_members = hearing_members(fit.hearing_ids, fit_universe)
    eval_members = hearing_members(evaluation.hearing_ids, eval_universe)
    for replicate in range(config.bootstrap_samples):
        eval_index = replicate_index(eval_members, eval_matrix[replicate])
        record_signal_replicates(store, "fixed", signals, names, evaluation, eval_index, config)
        record_policy_replicates(store, "fixed", policies, signals, evaluation.correct, eval_index)
        fit_index = replicate_index(fit_members, fit_matrix[replicate])
        refit = fit_all(take(fit, fit_index), raw_names, feature_sets, config)
        refit_signals = apply_models(refit, evaluation, raw_names)
        refit_names = [name for name in logistic_names if np.isfinite(refit_signals[name]).all()]
        record_signal_replicates(
            store, "refit", refit_signals, refit_names, evaluation, eval_index, config
        )
        record_policy_replicates(
            store, "refit", refit.policies, refit_signals, evaluation.correct, eval_index
        )
    level = config.confidence_level
    signal_report: Record = {}
    for subset in SUBSETS:
        signal_report[subset] = {}
        for name in names:
            metrics = point[subset][name]
            entry: Record = {"queries": metrics.get("queries", 0)}
            for metric, value in metrics.items():
                if metric == "queries":
                    continue
                entry[metric] = {"point": rounded(value)}
                for mode in ("fixed", "refit") if name in logistic_names else ("fixed",):
                    interval = store.interval((mode, subset, name, metric), level)
                    if interval is not None:
                        entry[metric][mode] = interval
            reference = point[subset][config.reference_signal]
            if name != config.reference_signal and "aurc" in reference and "aurc" in metrics:
                entry["difference_vs_reference"] = {}
                for metric in DIFFERENCE_METRICS:
                    difference: Record = {"point": rounded(metrics[metric] - reference[metric])}
                    for mode in ("fixed", "refit") if name in logistic_names else ("fixed",):
                        interval = store.interval(
                            (mode, subset, name, f"difference_{metric}"), level
                        )
                        if interval is not None:
                            difference[mode] = {
                                **interval,
                                "excludes_zero": excludes_zero(interval),
                            }
                    entry["difference_vs_reference"][metric] = difference
            signal_report[subset][name] = entry
    policy_report: Record = {}
    for policy in policies:
        fitted_policy = policy.rule != "production"
        entry = {
            "signal": policy.signal,
            "rule": policy.rule,
            "threshold": policy_threshold_record(policy),
            "fit_details": policy.details,
            "evaluate": {
                key: rounded(value) if isinstance(value, float) else value
                for key, value in policy_point.get(policy.name, {}).items()
            },
        }
        for mode in ("fixed", "refit") if fitted_policy else ("fixed",):
            entry[mode] = {
                metric: store.interval((mode, policy.name, metric), level)
                for metric in POLICY_METRICS
            }
        policy_report[policy.name] = entry
    return {
        "queries": len(evaluation),
        "hearings": int(len(np.unique(evaluation.hearing_ids))),
        "correct": int(evaluation.correct.sum()),
        "nontrivial": int(evaluation.nontrivial.sum()),
        "signals_evaluated": names,
        "signal_metrics": signal_report,
        "risk_coverage": curves,
        "policies": policy_report,
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "unit": "hearing",
            "evaluate_hearings_in_universe": int(len(eval_universe)),
            "fit_hearings_in_universe": int(len(fit_universe)),
        },
    }


def fit_split_description(fitted: Fitted, fit: Arrays, config: ConfidenceConfig) -> Record:
    description: Record = {}
    for name, values in fitted.fit_signals.items():
        if not np.isfinite(values).all():
            continue
        metrics = signal_metrics(values, fit.correct, config.coverages)
        description[name] = {
            key: rounded(value) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
    return description


def dotted_value(payload: Any, field_path: str) -> Any:
    value = payload
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"field {field_path!r} is missing")
        value = value[part]
    return value


def calibration_hearings(spec: Record) -> list[int]:
    path = Path(spec["hearings_source"])
    if path.suffix == ".jsonl":
        values = [row[spec["hearings_field"]] for row in load_jsonl(path)]
    else:
        with open(path) as f:
            values = list(dotted_value(json.load(f), spec["hearings_field"]))
    if not values or not all(isinstance(value, int) for value in values):
        raise SystemExit(f"{path}: {spec['hearings_field']} gives no hearing ids")
    return sorted(set(values))


def calibration_record(key: str, value: float, spec: Record, udv_evidence: Record) -> Record:
    threshold_path = Path(spec["threshold_source"])
    with open(threshold_path) as f:
        recorded = float(dotted_value(json.load(f), spec["threshold_field"]))
    if abs(recorded - value) > 1e-9:
        raise SystemExit(
            f"production_rule.calibration.{key}: {threshold_path} {spec['threshold_field']} is "
            f"{recorded}, udv config {key} is {value}"
        )
    if key == "embedding_threshold" and udv_evidence.get("calibration_source") != str(
        spec["hearings_source"]
    ):
        raise SystemExit(
            f"production_rule.calibration.{key}.hearings_source differs from the udv config "
            f"calibration_source {udv_evidence.get('calibration_source')!r}"
        )
    hearings = calibration_hearings(spec)
    return {
        "hearings_source": spec["hearings_source"],
        "hearings_source_sha256": sha256_of_file(Path(spec["hearings_source"])),
        "hearings_field": spec["hearings_field"],
        "threshold_source": spec["threshold_source"],
        "threshold_field": spec["threshold_field"],
        "threshold_recorded_there": recorded,
        "description": spec.get("description"),
        "role": spec.get("role"),
        "calibration_hearing_ids": hearings,
    }


def production_policies(config: ConfidenceConfig, target: Target) -> tuple[list[Policy], Record]:
    if target.name != config.production_target:
        return [], {}
    udv = load_udv_config(config.udv_config_path)
    evidence = udv.source["evidence"]
    thresholds = {key: float(evidence[key]) for key in config.threshold_keys}
    calibration = {
        key: calibration_record(key, value, config.threshold_calibration[key], evidence)
        for key, value in thresholds.items()
    }
    policies = [
        Policy(
            f"production:{key}",
            config.production_signal,
            "production",
            value,
            {
                "udv_config_key": key,
                "threshold": value,
                "calibration_source": calibration[key]["hearings_source"],
                "calibration_role": calibration[key]["role"],
                "calibration_hearings": len(calibration[key]["calibration_hearing_ids"]),
            },
        )
        for key, value in thresholds.items()
    ]
    source = {
        "udv_config": str(config.udv_config_path),
        "sha256": sha256_of_file(config.udv_config_path),
        "thresholds": thresholds,
        "calibration": calibration,
    }
    return policies, source


def subset_draws(
    rows: list[Record], label: str, config: ConfidenceConfig
) -> tuple[np.ndarray, np.ndarray]:
    hearings = np.array(sorted({row["hearing_id"] for row in rows}), dtype=np.int64)
    rng = stream_rng(config.seed, label)
    matrix = (
        rng.integers(0, len(hearings), size=(config.bootstrap_samples, len(hearings)))
        if len(hearings)
        else np.zeros((config.bootstrap_samples, 0), dtype=np.int64)
    )
    return hearings, matrix


def calibration_overlap(
    calibration: Record, key: str, rows: list[Record], population: list[Record]
) -> Record:
    hearing_ids = set(calibration[key]["calibration_hearing_ids"])
    split_hearings = sorted({row["hearing_id"] for row in rows})
    overlap = [hearing for hearing in split_hearings if hearing in hearing_ids]
    return {
        "calibration_hearings_in_split": overlap,
        "split_hearings": len(split_hearings),
        "queries_in_those_hearings": sum(1 for row in rows if row["hearing_id"] in hearing_ids),
        "multi_candidate_queries_in_those_hearings": sum(
            1 for row in population if row["hearing_id"] in hearing_ids
        ),
        "in_sample": bool(overlap),
    }


def production_outside_calibration(
    production: list[Policy],
    calibration: Record,
    rows: list[Record],
    population: list[Record],
    label: str,
    config: ConfidenceConfig,
) -> Record:
    excluded = set().union(
        *(set(entry["calibration_hearing_ids"]) for entry in calibration.values())
    )
    kept_rows = [row for row in rows if row["hearing_id"] not in excluded]
    kept_population = [row for row in population if row["hearing_id"] not in excluded]
    return {
        "rule": (
            "the queries of this split whose hearing is in no calibration set of any production "
            "threshold, so every cut is evaluated out of sample on the same queries; intervals "
            "resample these hearings only"
        ),
        "hearings_excluded": sorted(excluded & {row["hearing_id"] for row in rows}),
        "hearings_kept": len({row["hearing_id"] for row in kept_rows}),
        "multi_candidate": {
            "queries": len(kept_population),
            "policies": production_on_all(
                production,
                kept_population,
                subset_draws(kept_population, f"{label}|outside_calibration|multi", config),
                config,
            ),
        },
        "all_queries": {
            "queries": len(kept_rows),
            "policies": production_on_all(
                production,
                kept_rows,
                subset_draws(kept_rows, f"{label}|outside_calibration|all", config),
                config,
            ),
        },
    }


def universes(
    rows_by_split: dict[str, list[Record]],
    splits: tuple[str, ...],
    bench: str,
    seed: int,
    samples: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    draws = {}
    for split in splits:
        hearings = np.array(sorted({row["hearing_id"] for row in rows_by_split[split]}))
        rng = stream_rng(seed, f"{bench}|{split}|hearings")
        matrix = (
            rng.integers(0, len(hearings), size=(samples, len(hearings)))
            if len(hearings)
            else np.zeros((samples, 0), dtype=np.int64)
        )
        draws[split] = (hearings, matrix)
    return draws


def attach_entailment(
    rows: list[Record],
    target: Target,
    config: ConfidenceConfig,
    scores: dict[str, dict[str, Record]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for scorer in config.entailment_scorers:
            name = f"{ENTAILMENT_PREFIX}{scorer}"
            row[name] = None
            if target.unit not in config.entailment_units:
                continue
            entry = scores.get(scorer, {}).get(
                pair_id_of(row["bench"], row["query_id"], row["top1_text_sha256"])
            )
            if entry is None:
                counts[f"{scorer}.missing"] += 1
                continue
            if entry["split"] != row["split"] or entry["hearing_id"] != row["hearing_id"]:
                raise SystemExit(f"{entry['pair_id']}: split or hearing differs from the features")
            row[name] = float(entry["scores"][config.entailment_score_key])
            counts[f"{scorer}.scored"] += 1
            counts[f"{scorer}.truncated"] += int(any(i and i["truncated"] for i in entry["items"]))
    return counts


def load_entailment(
    run_dir: Path, config: ConfidenceConfig
) -> tuple[dict[str, dict[str, Record]], Record]:
    scores: dict[str, dict[str, Record]] = {}
    sources: Record = {}
    pairs_path = pairs_input_file(run_dir)
    pairs_sha = sha256_of_file(pairs_path) if pairs_path.exists() else None
    for scorer in config.entailment_scorers:
        path = entailment_file(run_dir, run_dir.name, scorer)
        if not path.exists():
            sources[scorer] = {"path": str(path), "available": False}
            continue
        rows = load_jsonl(path)
        scores[scorer] = {row["pair_id"]: row for row in rows}
        report_path = path.with_name(f"{scorer}_report.json")
        report: Record = {}
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
        scored_input = report.get("sources", {}).get("input", {}).get("sha256")
        sources[scorer] = {
            "available": True,
            "path": str(path),
            "sha256": sha256_of_file(path),
            "rows": len(rows),
            "model": rows[0]["model"] if rows else None,
            "revision": rows[0]["revision"] if rows else None,
            "report": str(report_path) if report else None,
            "report_created_at": report.get("created_at"),
            "scored_input_sha256": scored_input,
            "scored_input_matches_current_pairs_file": None
            if scored_input is None or pairs_sha is None
            else scored_input == pairs_sha,
            "label_probes": report.get("label_probes"),
        }
        if scored_input is not None and pairs_sha is not None and scored_input != pairs_sha:
            print(f"WARNING {scorer}: scores were computed on another pairs file", flush=True)
    return scores, sources


def prediction_rows(
    rows: list[Record],
    fitted: Fitted,
    raw_names: list[str],
    policies: list[Policy],
    config: ConfidenceConfig,
) -> list[Record]:
    arrays = arrays_of(rows, raw_names)
    signals = apply_models(fitted, arrays, raw_names)
    multi = np.array([row["n_units"] >= config.min_candidates for row in rows], dtype=bool)
    output = []
    for position, row in enumerate(rows):
        values = {name: clean(float(signals[name][position])) for name in signals}
        accepted: Record = {}
        for policy in policies:
            value = values.get(policy.signal)
            eligible = multi[position] or policy.rule == "production"
            if policy.threshold is None or value is None or not eligible:
                accepted[policy.name] = None
            else:
                accepted[policy.name] = bool(value >= policy.threshold)
        output.append(
            {
                "query_id": row["query_id"],
                "hearing_id": row["hearing_id"],
                "split": row["split"],
                "n_units": row["n_units"],
                "n_relevant": row["n_relevant"],
                "multi_candidate": bool(multi[position]),
                "nontrivial": bool(multi[position] and row["n_relevant"] < row["n_units"]),
                "correct": row["correct"],
                "correct_optimistic": row["correct_optimistic"],
                "correct_pessimistic": row["correct_pessimistic"],
                "zero_spread": row["zero_spread"],
                "top1_id": row["top1_id"],
                "signals": values,
                "accepted": accepted,
            }
        )
    return output


def split_counts(rows: list[Record], config: ConfidenceConfig) -> Record:
    multi = [row for row in rows if row["n_units"] >= config.min_candidates]
    return {
        "queries": len(rows),
        "hearings": len({row["hearing_id"] for row in rows}),
        "multi_candidate": len(multi),
        "single_candidate": len(rows) - len(multi),
        "nontrivial": sum(1 for row in multi if row["n_relevant"] < row["n_units"]),
        "correct_all": sum(1 for row in rows if row["correct"]),
        "correct_multi_candidate": sum(1 for row in multi if row["correct"]),
        "label_depends_on_ties_multi_candidate": sum(
            1 for row in multi if row["correct_optimistic"] != row["correct_pessimistic"]
        ),
        "zero_spread_multi_candidate": sum(1 for row in multi if row["zero_spread"]),
    }


def production_on_all(
    policies: list[Policy],
    rows: list[Record],
    draws: tuple[np.ndarray, np.ndarray],
    config: ConfidenceConfig,
) -> Record:
    arrays = arrays_of(rows, [config.production_signal])
    universe, matrix = draws
    members = hearing_members(arrays.hearing_ids, universe)
    store = Replicates()
    signals = arrays.signals
    for replicate in range(config.bootstrap_samples):
        index = replicate_index(members, matrix[replicate])
        record_policy_replicates(store, "fixed", policies, signals, arrays.correct, index)
    report = {}
    for policy in policies:
        accepted = accepted_by(policy, signals[policy.signal])
        if accepted is None:
            continue
        point = policy_metrics(accepted, arrays.correct)
        report[policy.name] = {
            "threshold": policy.threshold,
            "evaluate": {k: rounded(v) if isinstance(v, float) else v for k, v in point.items()},
            "fixed": {
                metric: store.interval(("fixed", policy.name, metric), config.confidence_level)
                for metric in POLICY_METRICS
            },
        }
    return report


def evaluate_target(
    bench: str,
    target: Target,
    rows_by_split: dict[str, list[Record]],
    eval_splits: tuple[str, ...],
    config: ConfidenceConfig,
    draws: dict[str, tuple[np.ndarray, np.ndarray]],
    run_dir: Path,
) -> tuple[Record, list[Record]]:
    population = {
        split: [row for row in rows if row["n_units"] >= config.min_candidates]
        for split, rows in rows_by_split.items()
    }
    fit_rows = [row for split in config.fit_splits for row in population[split]]
    decision_splits = (*config.fit_splits, *config.evaluate_splits)
    final_splits = [split for split in eval_splits if split not in decision_splits]
    entailment_names = [
        name
        for name in config.entailment_signals
        if target.unit in config.entailment_units
        and all(row.get(name) is not None for split in decision_splits for row in population[split])
    ]
    for split in final_splits:
        for name in entailment_names:
            unscored = sum(1 for row in population[split] if row.get(name) is None)
            if unscored:
                raise SystemExit(
                    f"{bench} {target.name}: {name} is scored on every "
                    f"{'+'.join(decision_splits)} query but missing for {unscored} {split} "
                    f"queries; the feature sets are fixed on {'+'.join(decision_splits)}, so the "
                    f"{split} evaluation is refused until its pairs are built and scored "
                    "(confidence_policies pairs --final-test, then nli_verifier_experiments "
                    "pairs --final-test)"
                )
    raw_names = [*config.base_signals, *entailment_names]
    missing_entailment = {
        name: sum(
            1 for split in decision_splits for row in population[split] if row.get(name) is None
        )
        for name in config.entailment_signals
        if name not in entailment_names
    }
    feature_sets = {
        name: features
        for name, features in config.feature_sets.items()
        if set(features) <= set(raw_names)
    }
    fit = arrays_of(fit_rows, raw_names)
    for name in raw_names:
        if not np.isfinite(fit.signals[name]).all():
            raise SystemExit(f"{bench} {target.name}: {name} is not finite on every fit row")
    fitted = fit_all(fit, raw_names, feature_sets, config)
    production, production_source = production_policies(config, target)
    fit_draws = draws[config.fit_splits[0]]
    report: Record = {
        "target": target.name,
        "retriever": target.retriever,
        "unit": target.unit,
        "counts": {split: split_counts(rows, config) for split, rows in rows_by_split.items()},
        "raw_signals": raw_names,
        "entailment_decision_splits": list(decision_splits),
        "entailment_missing_rows": missing_entailment,
        "feature_sets_used": {name: list(features) for name, features in feature_sets.items()},
        "feature_sets_skipped": sorted(set(config.feature_sets) - set(feature_sets)),
        "logistic": {
            name: logistic_record(model, fit.correct) for name, model in fitted.models.items()
        },
        "fit_split_description": fit_split_description(fitted, fit, config),
        "evaluate": {},
    }
    if production_source:
        report["production_rule"] = {
            "source": production_source,
            "all_queries": {},
            "calibration_overlap": {},
            "outside_calibration": {},
            "comparison_rule": config.source["production_rule"]["comparison_rule"],
        }
    for split in eval_splits:
        evaluation_rows = population[split]
        if not evaluation_rows:
            report["evaluate"][split] = {"queries": 0}
            continue
        evaluation = arrays_of(evaluation_rows, raw_names)
        report["evaluate"][split] = evaluate_split(
            fitted,
            fit,
            evaluation,
            raw_names,
            feature_sets,
            config,
            fit_draws,
            draws[split],
            production,
            f"{bench}|{target.name}|{split}",
        )
        if production:
            calibration = production_source["calibration"]
            overlap = {
                policy.name: calibration_overlap(
                    calibration,
                    policy.details["udv_config_key"],
                    rows_by_split[split],
                    evaluation_rows,
                )
                for policy in production
            }
            for name, entry in overlap.items():
                report["evaluate"][split]["policies"][name]["calibration_overlap"] = entry
            report["production_rule"]["calibration_overlap"][split] = overlap
            report["production_rule"]["all_queries"][split] = production_on_all(
                production, rows_by_split[split], draws[split], config
            )
            report["production_rule"]["outside_calibration"][split] = (
                production_outside_calibration(
                    production,
                    calibration,
                    rows_by_split[split],
                    evaluation_rows,
                    f"{bench}|{split}",
                    config,
                )
            )
    predictions: list[Record] = []
    for split, rows in rows_by_split.items():
        output = prediction_rows(rows, fitted, raw_names, fitted.policies + production, config)
        path = predictions_file(run_dir, bench, target.name, split)
        write_jsonl(output, path)
        predictions.append({"path": str(path), "rows": len(output), "sha256": sha256_of_file(path)})
    return report, predictions


def check_query_sets(loaded: dict[str, dict[str, dict[str, list[Record]]]], bench: str) -> None:
    reference: dict[str, set[str]] | None = None
    for name, benches in loaded.items():
        sets = {split: {row["query_id"] for row in rows} for split, rows in benches[bench].items()}
        if reference is None:
            reference = sets
        elif sets != reference:
            raise SystemExit(f"{bench}: {name} has another query set than the first target")


def primary_summary(results: Record, config: ConfidenceConfig) -> Record:
    summary: Record = {}
    for bench, targets in results.items():
        target = targets.get(config.primary_target)
        if target is None:
            summary[bench] = {"available": False}
            continue
        summary[bench] = {}
        for split, evaluation in target["evaluate"].items():
            metrics = evaluation.get("signal_metrics", {}).get("multi_candidate", {})
            summary[bench][split] = {}
            for name, entry in metrics.items():
                item: Record = {"aurc": entry["aurc"]["point"]}
                if "difference_vs_reference" in entry:
                    difference = entry["difference_vs_reference"]["aurc"]
                    mode = "refit" if name.startswith(LOGISTIC_PREFIX) else "fixed"
                    interval = difference.get(mode)
                    item["difference_aurc"] = difference["point"]
                    item["interval_mode"] = mode
                    item["interval"] = interval
                    item["better_than_reference"] = bool(
                        interval is not None
                        and interval["high"] is not None
                        and interval["high"] < 0
                    )
                summary[bench][split][name] = item
    return summary


def command_evaluate(args: argparse.Namespace, config: ConfidenceConfig) -> None:
    run_dir = run_directory(config, args.run_name)
    eval_splits = evaluated_splits(config, args.final_test)
    splits = resolve_splits(config, args.final_test)
    targets = select_targets(config, args.targets)
    loaded: dict[str, dict[str, dict[str, list[Record]]]] = {}
    inputs: list[Record] = []
    missing: dict[str, str] = {}
    for target in targets:
        result = load_features(run_dir, target, config.benches, splits)
        if result is None:
            missing[target.name] = "features not collected for every bench and split"
            continue
        loaded[target.name], files = result
        inputs += files
    if not loaded:
        raise SystemExit(f"no collected features under {run_dir}")
    harness = load_harness(config)
    split_of, split_source = load_split_lookup(config.manifest_path, harness.lds_sha256)
    check_manifest_splits(loaded, split_of)
    scores, entailment_sources = load_entailment(run_dir, config)
    entailment_counts: Record = {}
    by_name = {target.name: target for target in targets}
    for name, benches in loaded.items():
        counts: Counter[str] = Counter()
        for by_split in benches.values():
            for rows in by_split.values():
                counts.update(attach_entailment(rows, by_name[name], config, scores))
        entailment_counts[name] = dict(counts)
    results: Record = {}
    outputs: list[Record] = []
    started = time.perf_counter()
    for bench in config.benches:
        check_query_sets(loaded, bench)
        first = next(iter(loaded.values()))[bench]
        draws = universes(first, splits, bench, config.seed, config.bootstrap_samples)
        results[bench] = {}
        for name, benches in loaded.items():
            report, predictions = evaluate_target(
                bench, by_name[name], benches[bench], eval_splits, config, draws, run_dir
            )
            results[bench][name] = report
            outputs += predictions
            print(
                f"[evaluate] {bench} {name} done ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    report = {
        "experiment": "confidence_policies",
        "step": "evaluate",
        "run_name": run_dir.name,
        "created_at": now(),
        "question": "which signal best predicts that the top-1 unit of a retriever is relevant, "
        "so that a confidence tier built on it means something",
        "splits": split_record(config, args.final_test),
        "splits_used": list(splits),
        "primary": {
            "declaration": config.source["evaluation"]["primary_declaration"],
            "target": config.primary_target,
            "metric": config.source["evaluation"]["primary_metric"],
            "results": primary_summary(results, config),
        },
        "definitions": DEFINITIONS,
        "targets": {"evaluated": sorted(loaded), "missing": missing},
        "collect_reports": collect_reports(run_dir, sorted(loaded)),
        "entailment": {"sources": entailment_sources, "counts": entailment_counts},
        "results": results,
        "inputs": inputs,
        "sources": {"splits": split_source},
        "outputs": outputs,
        "seconds": round(time.perf_counter() - started, 1),
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    suffix = "final_test_report" if args.final_test else "report"
    path = run_dir / f"{run_dir.name}_{suffix}.json"
    write_json(clean(report), path)
    print_summary(clean(report))
    print(f"[evaluate] report -> {path}", flush=True)


def format_interval(entry: Record | None) -> str:
    if not entry or entry.get("low") is None:
        return "[  n/a  ]"
    return f"[{entry['low']:.3f},{entry['high']:.3f}]"


def print_summary(report: Record) -> None:
    for bench, targets in report["results"].items():
        for name, target in targets.items():
            for split, evaluation in target["evaluate"].items():
                if not evaluation.get("queries"):
                    continue
                print(
                    f"{bench} {name} {split}: n={evaluation['queries']} "
                    f"correct={evaluation['correct']} hearings={evaluation['hearings']}"
                )
                metrics = evaluation["signal_metrics"]["multi_candidate"]
                for signal, entry in metrics.items():
                    mode = "refit" if signal.startswith(LOGISTIC_PREFIX) else "fixed"
                    auc, aurc = entry["roc_auc"], entry["aurc"]
                    auc_text = "  n/a" if auc["point"] is None else f"{auc['point']:.3f}"
                    print(
                        f"  {signal:34s} auc={auc_text} {format_interval(auc.get(mode))} "
                        f"aurc={aurc['point']:.3f} {format_interval(aurc.get(mode))}"
                    )
                for policy, entry in evaluation["policies"].items():
                    point = entry["evaluate"]
                    precision = point.get("precision")
                    coverage = point.get("coverage")
                    mode = "fixed" if entry["rule"] == "production" else "refit"
                    print(
                        f"  {policy:48s} precision="
                        f"{'n/a' if precision is None else f'{precision:.3f}'} "
                        f"{format_interval(entry.get(mode, {}).get('precision'))} coverage="
                        f"{'n/a' if coverage is None else f'{coverage:.3f}'} "
                        f"{format_interval(entry.get(mode, {}).get('coverage'))}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confidence policies for retrieved top-1 evidence: collect per-query signals "
        "from the retrieval harness, build entailment pairs for the NLI verifier, fit policies on "
        "train and evaluate them on validation."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/confidence_policies.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "pairs", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-name", default=None, help="default from run.run_name")
        command.add_argument("--targets", default=None, help="comma list of <retriever>.<unit>")
        command.add_argument(
            "--final-test",
            action="store_true",
            help="also read and evaluate the final_test splits; nothing is fitted on them",
        )
        if name == "collect":
            command.add_argument(
                "--hearing-ids", default=None, help="comma list of hearings (smoke tests only)"
            )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = {"collect": command_collect, "pairs": command_pairs, "evaluate": command_evaluate}
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
