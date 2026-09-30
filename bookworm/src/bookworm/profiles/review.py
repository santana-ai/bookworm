"""Blind human review of profile pairs: stratified sample and Wilson intervals."""

import csv
import io
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, get_args

import numpy as np

from bookworm.data.io import JsonObject, read_json_object, sha256_of_file, write_json
from bookworm.errors import ConfigError
from bookworm.profiles.config import (
    ProfileValidationConfig,
    ReviewSettings,
    load_profile_validation_config,
)
from bookworm.profiles.schemas import GROUPS, Group, ProfilePair, read_pairs
from bookworm.profiles.validate import check_new_outputs
from bookworm.profiles.validation_metrics import rounded

Judgment = Literal["sustentada", "compativel", "contradita", "sem_relacao"]

JUDGMENTS: tuple[Judgment, ...] = get_args(Judgment)
JUDGMENT_BY_LABEL: dict[str, Judgment] = {judgment: judgment for judgment in JUDGMENTS}
SUPPORT_OUTCOMES: dict[str, tuple[Judgment, ...]] = {
    "strict_support": ("sustentada",),
    "tolerant_support": ("sustentada", "compativel"),
}
FILL_COLUMNS = ("julgamento", "observacao")
DISPLAY_COLUMNS = (
    "item_id",
    "udv_id",
    "actor",
    "proposition",
    "udv_evidence",
    "profile_evidence",
    "profiles_file",
)
CSV_COLUMNS = (*DISPLAY_COLUMNS, *FILL_COLUMNS)
CHECKED_COLUMNS = ("udv_id", "actor")
CSV_DELIMITER = ";"
CSV_DELIMITERS = (";", ",", "\t")
CSV_ENCODING = "utf-8-sig"
ITEM_ID_PREFIX = "P"
ITEM_ID_MIN_DIGITS = 3
DRAW_STREAM = 0
ORDER_STREAM = 1
ALL_SCORES_BAND = "all"
LABEL_NORMALIZATION = (
    "strip, lower case, accents removed, spaces and hyphens turned into underscores"
)
STRATIFIED_INTERVAL = (
    "normal approximation, variance sum_h W_h^2 (1 - n_h / N_h) p_h (1 - p_h) / (n_h - 1), "
    "W_h = N_h / N"
)


def band_labels(edges: Sequence[float]) -> list[str]:
    if not edges:
        return [ALL_SCORES_BAND]
    bounds = ["-inf", *(f"{edge:.2f}" for edge in edges), "inf"]
    return [f"[{low}, {high})" for low, high in pairwise(bounds)]


def score_band(score: float, edges: Sequence[float]) -> str:
    labels = band_labels(edges)
    return labels[sum(1 for edge in edges if score >= edge)]


@dataclass(frozen=True)
class Stratum:
    group: Group
    band: str
    population: list[ProfilePair]


def build_strata(pairs: Sequence[ProfilePair], edges: Sequence[float]) -> list[Stratum]:
    ordered = sorted(pairs, key=lambda pair: pair.udv_id)
    return [
        Stratum(
            group,
            band,
            [
                pair
                for pair in ordered
                if pair.group == group and score_band(pair.score, edges) == band
            ],
        )
        for group in GROUPS
        for band in band_labels(edges)
    ]


def csv_row(item_id: str, pair: ProfilePair, profiles_file: Path) -> dict[str, str]:
    return {
        "item_id": item_id,
        "udv_id": pair.udv_id,
        "actor": pair.actor,
        "proposition": pair.proposition,
        "udv_evidence": pair.udv_evidence or "",
        "profile_evidence": pair.profile_sentence,
        "profiles_file": str(profiles_file),
        "julgamento": "",
        "observacao": "",
    }


def item_key(pair: ProfilePair, band: str) -> JsonObject:
    return {
        "udv_id": pair.udv_id,
        "actor": pair.actor,
        "group": pair.group,
        "split": pair.split,
        "hearing_id": pair.hearing_id,
        "score": pair.score,
        "score_band": band,
    }


def write_review_csv(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_COLUMNS, delimiter=CSV_DELIMITER, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_strata(
    pairs: Sequence[ProfilePair], review: ReviewSettings
) -> tuple[list[tuple[ProfilePair, str]], list[JsonObject]]:
    """Draw up to the configured size from every stratum, without replacement."""
    draw_rng = np.random.default_rng([review.seed, DRAW_STREAM])
    drawn: list[tuple[ProfilePair, str]] = []
    strata_report: list[JsonObject] = []
    for stratum in build_strata(pairs, review.score_bands):
        target = review.sizes.get(stratum.group, 0)
        size = min(target, len(stratum.population))
        chosen = sorted(draw_rng.choice(len(stratum.population), size=size, replace=False))
        drawn.extend((stratum.population[int(index)], stratum.band) for index in chosen)
        strata_report.append(
            {
                "group": stratum.group,
                "band": stratum.band,
                "population": len(stratum.population),
                "target": target,
                "drawn": size,
                "shortfall": target - size,
            }
        )
    return drawn, strata_report


def review_item_ids(count: int) -> list[str]:
    width = max(ITEM_ID_MIN_DIGITS, len(str(count)))
    return [f"{ITEM_ID_PREFIX}{number:0{width}d}" for number in range(1, count + 1)]


def sample_profile_review(
    config: ProfileValidationConfig, *, overwrite: bool = False
) -> JsonObject:
    """Write a seeded, stratified review sample as a CSV with empty judgments."""
    review = config.review
    check_new_outputs((review.output, config.review_sample_path), overwrite)
    pairs = read_pairs(config.pairs_path)
    drawn, strata_report = draw_strata(pairs, review)
    order = np.random.default_rng([review.seed, ORDER_STREAM]).permutation(len(drawn))
    item_ids = review_item_ids(len(drawn))
    ordered = [drawn[int(index)] for index in order]
    rows = [
        csv_row(item_id, pair, config.profiles_path)
        for item_id, (pair, _) in zip(item_ids, ordered, strict=True)
    ]
    write_review_csv(rows, review.output)
    sample = {
        "name": config.name,
        "pairs": {"path": str(config.pairs_path), "sha256": sha256_of_file(config.pairs_path)},
        "seed": review.seed,
        "seed_streams": {"stratum_draw": DRAW_STREAM, "row_order": ORDER_STREAM},
        "score_bands": list(review.score_bands),
        "sizes": dict(review.sizes),
        "judgments": list(JUDGMENTS),
        "csv": {"path": str(review.output), "skeleton_sha256": sha256_of_file(review.output)},
        "strata": strata_report,
        "items": {
            item_id: item_key(pair, band)
            for item_id, (pair, band) in zip(item_ids, ordered, strict=True)
        },
    }
    write_json(sample, config.review_sample_path)
    return {
        "items": len(rows),
        "strata": strata_report,
        "csv": str(review.output),
        "sample": str(config.review_sample_path),
    }


def normalize_label(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.strip().lower())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return "_".join(plain.replace("-", " ").split())


def read_review_csv(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding=CSV_ENCODING)
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: annotation file not found") from error
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path}: cannot read annotation file: {error}") from error
    for delimiter in CSV_DELIMITERS:
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if reader.fieldnames is not None and set(CSV_COLUMNS) <= set(reader.fieldnames):
            return [dict(row) for row in reader]
    raise ConfigError(f"{path}: header does not contain the columns {list(CSV_COLUMNS)}")


@dataclass(frozen=True)
class JudgedItem:
    item_id: str
    group: Group
    band: str
    judgment: Judgment


def check_sample_items(
    rows: Sequence[Mapping[str, str]], expected: Mapping[str, object], path: Path
) -> None:
    seen = [row["item_id"] for row in rows]
    if sorted(seen) != sorted(expected):
        missing = sorted(set(expected) - set(seen))
        extra = sorted(set(seen) - set(expected))
        raise ConfigError(
            f"{path}: items differ from the sample (missing {missing}, unexpected {extra})"
        )


def sampled_pair(
    row: Mapping[str, str],
    key: Mapping[str, Any],
    pairs: Mapping[str, ProfilePair],
    edges: Sequence[float],
    path: Path,
) -> ProfilePair:
    item_id = row["item_id"]
    pair = pairs.get(str(key.get("udv_id")))
    if pair is None or dict(key) != item_key(pair, score_band(pair.score, edges)):
        raise ConfigError(f"review sample key: {item_id} does not match the pairs file")
    changed = [column for column in CHECKED_COLUMNS if row[column] != key[column]]
    if changed:
        raise ConfigError(f"{path}: {item_id} has edited columns {changed}")
    return pair


def judged_items(
    rows: Sequence[Mapping[str, str]],
    sample: Mapping[str, Any],
    pairs: Mapping[str, ProfilePair],
    edges: Sequence[float],
    path: Path,
) -> list[JudgedItem]:
    """The judged review rows, checked against the sample; every row needs a valid judgment."""
    expected: dict[str, Mapping[str, Any]] = dict(sample["items"])
    check_sample_items(rows, expected, path)
    invalid: list[str] = []
    empty: list[str] = []
    items: list[JudgedItem] = []
    for row in rows:
        item_id = row["item_id"]
        key = expected[item_id]
        pair = sampled_pair(row, key, pairs, edges, path)
        label = normalize_label(row["julgamento"] or "")
        judgment = JUDGMENT_BY_LABEL.get(label)
        if not label:
            empty.append(item_id)
        elif judgment is None:
            invalid.append(f"{item_id}={row['julgamento']!r}")
        else:
            items.append(JudgedItem(item_id, pair.group, str(key["score_band"]), judgment))
    if invalid:
        raise ConfigError(f"{path}: invalid judgments {invalid}; valid values are {JUDGMENTS}")
    if empty:
        raise ConfigError(f"{path}: {len(empty)} items have no judgment: {empty}")
    return items


def wilson_interval(successes: int, trials: int, confidence_level: float) -> JsonObject:
    if trials == 0:
        return {"successes": 0, "trials": 0, "estimate": None, "low": None, "high": None}
    z = NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2))
    half /= denominator
    return {
        "successes": successes,
        "trials": trials,
        "estimate": rounded(proportion),
        "low": rounded(max(0.0, center - half)),
        "high": rounded(min(1.0, center + half)),
    }


def proportions(items: Sequence[JudgedItem], confidence_level: float) -> JsonObject:
    counts = Counter(item.judgment for item in items)
    labels = {
        label: wilson_interval(counts[label], len(items), confidence_level) for label in JUDGMENTS
    }
    outcomes = {
        name: wilson_interval(
            sum(counts[label] for label in accepted), len(items), confidence_level
        )
        for name, accepted in SUPPORT_OUTCOMES.items()
    }
    return {"items": len(items), "labels": labels, "outcomes": outcomes}


def stratified_estimate(
    items: Sequence[JudgedItem],
    strata: Sequence[Mapping[str, Any]],
    accepted: Sequence[Judgment],
    confidence_level: float,
) -> JsonObject:
    population = sum(int(stratum["population"]) for stratum in strata)
    uncovered = [
        stratum["band"]
        for stratum in strata
        if stratum["population"] > 0 and not any(item.band == stratum["band"] for item in items)
    ]
    if population == 0 or uncovered:
        return {"estimate": None, "low": None, "high": None, "uncovered_bands": uncovered}
    estimate = 0.0
    variance: float | None = 0.0
    for stratum in strata:
        size = int(stratum["population"])
        if size == 0:
            continue
        judged = [item for item in items if item.band == stratum["band"]]
        weight = size / population
        share = sum(item.judgment in accepted for item in judged) / len(judged)
        estimate += weight * share
        correction = 1 - len(judged) / size
        if correction == 0 or variance is None:
            continue
        if len(judged) < 2:
            variance = None
            continue
        variance += weight**2 * correction * share * (1 - share) / (len(judged) - 1)
    if variance is None:
        return {"estimate": rounded(estimate), "low": None, "high": None, "uncovered_bands": []}
    z = NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)
    half = z * math.sqrt(variance)
    return {
        "estimate": rounded(estimate),
        "low": rounded(max(0.0, estimate - half)),
        "high": rounded(min(1.0, estimate + half)),
        "uncovered_bands": [],
    }


def group_review(
    group: Group,
    items: Sequence[JudgedItem],
    strata: Sequence[Mapping[str, Any]],
    confidence_level: float,
) -> JsonObject:
    group_items = [item for item in items if item.group == group]
    group_strata = [stratum for stratum in strata if stratum["group"] == group]
    return {
        "sample": proportions(group_items, confidence_level),
        "by_band": {
            stratum["band"]: {
                "population": stratum["population"],
                **proportions(
                    [item for item in group_items if item.band == stratum["band"]],
                    confidence_level,
                ),
            }
            for stratum in group_strata
        },
        "population_weighted": {
            name: stratified_estimate(group_items, group_strata, accepted, confidence_level)
            for name, accepted in SUPPORT_OUTCOMES.items()
        },
    }


def score_profile_review(config: ProfileValidationConfig, annotations: Path) -> JsonObject:
    """Score a filled review CSV: support proportions with Wilson intervals."""
    sample = read_json_object(config.review_sample_path, "review sample file")
    pairs = {pair.udv_id: pair for pair in read_pairs(config.pairs_path)}
    if sha256_of_file(config.pairs_path) != sample["pairs"]["sha256"]:
        raise ConfigError(
            f"{config.pairs_path}: pairs file changed after the review sample was drawn"
        )
    rows = read_review_csv(annotations)
    items = judged_items(rows, sample, pairs, sample["score_bands"], annotations)
    report = {
        "name": config.name,
        "annotations": {"path": str(annotations), "sha256": sha256_of_file(annotations)},
        "sample": {
            "path": str(config.review_sample_path),
            "sha256": sha256_of_file(config.review_sample_path),
        },
        "confidence_level": config.confidence_level,
        "proportion_interval": "wilson",
        "stratified_interval": STRATIFIED_INTERVAL,
        "label_normalization": LABEL_NORMALIZATION,
        "outcomes": {name: list(accepted) for name, accepted in SUPPORT_OUTCOMES.items()},
        "groups": {
            group: group_review(group, items, sample["strata"], config.confidence_level)
            for group in GROUPS
        },
    }
    write_json(report, config.review_report_path)
    return report


def run_sample_profile_review(config_path: Path, *, overwrite: bool = False) -> JsonObject:
    return sample_profile_review(load_profile_validation_config(config_path), overwrite=overwrite)


def run_score_profile_review(config_path: Path, annotations: Path) -> JsonObject:
    config = load_profile_validation_config(config_path)
    report = score_profile_review(config, annotations)
    return {
        "groups": {group: report["groups"][group]["sample"]["outcomes"] for group in GROUPS},
        "report": str(config.review_report_path),
    }
