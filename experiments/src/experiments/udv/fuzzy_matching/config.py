"""Settings of the fuzzy matching experiment, read and checked from its TOML file."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Record = dict[str, Any]

CONFIG_SPLITS_ALLOWED = ("train", "validation")
QUOTE_METHODS = ("char", "token")
NULL_KINDS = ("same_hearing_other_person", "other_hearing_same_split")
NAME_METRICS = ("token_set_ratio", "jaro_winkler")
COMBINED_NAME_RULE = "both"
SEMANTIC_TIERS = ("semantic_match_high", "semantic_match_weak")
REVIEW_KINDS = ("quotes", "names")
NULL_DRAW_STREAMS = (0, 1)
MAX_SCORE = 100
MIN_REVIEW_LABELS = 3


@dataclass(frozen=True)
class FuzzyConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    splits: tuple[str, ...]
    reference_runs: tuple[Path, ...]
    udv_config_path: Path
    cache_device_label: str
    prefix_levels: tuple[int, ...]
    min_prefix_words: int
    quote_thresholds: tuple[float, ...]
    null_draws: int
    quote_review_min_score: float
    score_bin_width: int
    name_metrics: tuple[str, ...]
    name_thresholds: tuple[float, ...]
    name_review_min_score: float
    speech_excerpt_chars: int
    impostor_examples: int
    bootstrap_samples: int
    confidence_level: float
    seed: int
    version: str
    output_dir: Path
    review_context_chars: int
    review_item_prefixes: dict[str, str]
    review_order_streams: dict[str, tuple[int, ...]]
    review_labels: dict[str, tuple[str, ...]]
    review_positive: dict[str, str]
    review_unsure: dict[str, str]
    source: Record = field(default_factory=dict)


def validate_thresholds(values: list[float], key: str) -> tuple[float, ...]:
    if not values or any(not 0 <= value <= MAX_SCORE for value in values):
        raise SystemExit(f"{key} must be a non-empty list of scores in [0, 100]")
    return tuple(sorted(float(value) for value in values))


def check_splits(splits: tuple[str, ...]) -> None:
    if "test" in splits:
        raise SystemExit("splits.use must never list test; pass --final-test instead")
    if not splits or any(split not in CONFIG_SPLITS_ALLOWED for split in splits):
        raise SystemExit(f"splits.use must be a non-empty subset of {CONFIG_SPLITS_ALLOWED}")
    if len(set(splits)) != len(splits):
        raise SystemExit("splits.use must list distinct split names")


def check_quotes(quotes: Record) -> tuple[int, ...]:
    levels = tuple(quotes["prefix_word_levels"])
    if not levels or list(levels) != sorted(set(levels), reverse=True):
        raise SystemExit("quotes.prefix_word_levels must be distinct and in decreasing order")
    if quotes["min_prefix_words"] < 1 or quotes["null_draws_per_opinion"] < 1:
        raise SystemExit("quotes.min_prefix_words and quotes.null_draws_per_opinion must be >= 1")
    return levels


def check_names_and_bootstrap(names: Record, bootstrap: Record) -> None:
    if any(metric not in NAME_METRICS for metric in names["metrics"]):
        raise SystemExit(f"names.metrics must be among {NAME_METRICS}")
    if bootstrap["unit"] != "hearing":
        raise SystemExit(
            "bootstrap.unit must be hearing: opinions of one hearing are not independent"
        )
    if bootstrap["samples"] < 1 or not 0 < bootstrap["confidence_level"] < 1:
        raise SystemExit("bootstrap.samples must be >= 1 and confidence_level in (0, 1)")


def review_labels(review: Record) -> dict[str, tuple[str, ...]]:
    labels = {kind: tuple(review[kind]["labels"]) for kind in REVIEW_KINDS}
    for kind in REVIEW_KINDS:
        chosen = {review[kind]["positive_label"], review[kind]["unsure_label"]}
        if len(set(labels[kind])) != len(labels[kind]) or not chosen <= set(labels[kind]):
            raise SystemExit(f"review.{kind}: labels must be distinct and hold both chosen labels")
        if len(chosen) != 2 or len(labels[kind]) < MIN_REVIEW_LABELS:
            raise SystemExit(f"review.{kind}: needs a positive, a negative and an unsure label")
    return labels


def review_item_prefixes(review: Record) -> dict[str, str]:
    prefixes = {kind: review["item_prefixes"][kind] for kind in REVIEW_KINDS}
    if len(set(prefixes.values())) != len(prefixes):
        raise SystemExit("review.item_prefixes must differ between sheets")
    return prefixes


def review_order_streams(review: Record) -> dict[str, tuple[int, ...]]:
    streams = {kind: tuple(review["order_streams"][kind]) for kind in REVIEW_KINDS}
    if len(set(streams.values())) != len(streams) or any(
        stream[0] in NULL_DRAW_STREAMS for stream in streams.values()
    ):
        raise SystemExit("review.order_streams must differ and not reuse the null-draw streams")
    return streams


def load_config(config_path: Path) -> FuzzyConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    splits = tuple(raw["splits"]["use"])
    check_splits(splits)
    quotes, names, bootstrap, review = raw["quotes"], raw["names"], raw["bootstrap"], raw["review"]
    levels = check_quotes(quotes)
    check_names_and_bootstrap(names, bootstrap)
    labels = review_labels(review)
    prefixes = review_item_prefixes(review)
    streams = review_order_streams(review)
    return FuzzyConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        splits=splits,
        reference_runs=tuple(Path(run) for run in raw["reference"]["udv_runs"]),
        udv_config_path=Path(raw["encoder_baseline"]["udv_config_path"]),
        cache_device_label=raw["encoder_baseline"]["cache_device_label"],
        prefix_levels=levels,
        min_prefix_words=quotes["min_prefix_words"],
        quote_thresholds=validate_thresholds(quotes["thresholds"], "quotes.thresholds"),
        null_draws=quotes["null_draws_per_opinion"],
        quote_review_min_score=float(quotes["review_min_score"]),
        score_bin_width=quotes["score_bin_width"],
        name_metrics=tuple(names["metrics"]),
        name_thresholds=validate_thresholds(names["thresholds"], "names.thresholds"),
        name_review_min_score=float(names["review_min_score"]),
        speech_excerpt_chars=names["speech_excerpt_chars"],
        impostor_examples=names["impostor_examples"],
        bootstrap_samples=bootstrap["samples"],
        confidence_level=bootstrap["confidence_level"],
        seed=raw["run"]["seed"],
        version=raw["run"]["version"],
        output_dir=Path(raw["run"]["output_dir"]),
        review_context_chars=review["context_chars"],
        review_item_prefixes=prefixes,
        review_order_streams=streams,
        review_labels=labels,
        review_positive={kind: review[kind]["positive_label"] for kind in REVIEW_KINDS},
        review_unsure={kind: review[kind]["unsure_label"] for kind in REVIEW_KINDS},
        source=raw,
    )


def run_name_for(config: FuzzyConfig, final_test: bool) -> str:
    return f"{config.version}_final_test" if final_test else config.version
