"""Confidence experiment configuration, targets, run paths and the metric definitions."""

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.common.splits import SPLIT_NAMES
from experiments.retrieval import experiments as retrieval_experiments
from experiments.retrieval.data import BENCHES
from experiments.retrieval.experiments import (
    ExperimentConfig,
    retriever_ids,
)

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
