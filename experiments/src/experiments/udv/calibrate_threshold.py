import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import sentence_transformers
import torch
from bookworm import load_jsonl, sha256_of_file, write_json, write_jsonl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.common import transcript, udv_run
from experiments.common.provenance import module_path, source_hashes, source_label
from experiments.common.transcript import (
    find_opinion_turn_quote_match,
    is_trusted_quote,
    sentences_agree,
)
from experiments.common.udv_run import (
    UdvConfig,
    encode_with_cache,
    load_config,
    load_encoder,
    load_lds_records,
    resolve_hearing_people,
    seed_everything,
    select_device,
    sentence_slices_by_person,
)

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
CALIBRATION_SPLITS_ALLOWED = ("train", "validation")
EMBEDDING_KINDS = ("sentences", "opinions", "masked")
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
SCORE_DECIMALS = 4
QUERY_SCORE_DECIMALS = 6
RULE_DEFINITIONS: Record = {
    "legacy_random": (
        "positives: unmasked opinion vs the sentence holding its trusted quote; negatives: the "
        "same opinion vs a sentence drawn uniformly from all sentences of the other calibration "
        "hearings; threshold: midpoint of the configured positive and negative quantiles"
    ),
    "hard_negative": (
        "positives as in legacy_random; negatives: the same opinion vs a sentence drawn uniformly "
        "from the speaker's own sentences that do not agree with the target; threshold as in "
        "legacy_random; opinions whose speaker has no such sentence are left out"
    ),
    "masked_hard_negative": (
        "positives: masked opinion (quoted spans removed, masked_quotes benchmark) vs its "
        "target; negatives: the masked opinion vs a non-target sentence of the same speaker; "
        "threshold as in legacy_random"
    ),
    "masked_top1_youden": (
        "for each masked opinion the top-1 sentence by cosine among the speaker's sentences "
        "(first index on ties, as in the UDV builder) is correct when it agrees with the "
        "target; the threshold maximizes Youden's J = TPR - FPR of 'top-1 score >= t predicts "
        "a correct top-1', the lowest threshold wins ties, and the reported value is the "
        "midpoint between the lowest top-1 score kept and the highest one excluded"
    ),
}
LABEL_SEMANTICS: Record = {
    "target": (
        "silver label: the sentence of the speaker's own turn that encloses the matched quote "
        "prefix of 6+ words (udv_pipeline.find_opinion_turn_quote_match), found by string "
        "matching and not validated by a human"
    ),
    "positive_score": (
        "highest cosine between the query and the candidate sentences that agree with the "
        "target (sentences_agree); a target spanning two candidate sentences has two"
    ),
    "masked_queries": (
        "masked opinions stand in for opinions without a quote, the ones the threshold is "
        "applied to; their wording may differ from paraphrases written without any quote"
    ),
    "in_sample": (
        "every number here is measured on the calibration splits only; precision and coverage "
        "at a threshold chosen on the same queries are optimistic, and an unbiased estimate "
        "needs held-out queries"
    ),
}


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


@dataclass
class EncodingLedger:
    allowed_hearing_ids: frozenset[int]
    encoded_hearing_ids: set[int] = field(default_factory=set)
    by_kind: dict[str, Record] = field(default_factory=dict)


@dataclass(frozen=True)
class SentencePool:
    hearing_ids: np.ndarray
    sentence_indices: np.ndarray
    scores: np.ndarray


class NegativeDraw(NamedTuple):
    scores: np.ndarray
    hearing_ids: np.ndarray
    sentence_indices: np.ndarray


NegativeSampler = Callable[[np.random.Generator, np.ndarray], NegativeDraw]


def load_calibration_config(udv_config: UdvConfig) -> CalibrationConfig:
    raw = udv_config.source.get("calibration")
    if raw is None:
        raise SystemExit("the [calibration] section is missing from the UDV config")
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
    if not all(0 <= raw[key] <= 1 for key in ("positive_quantile", "negative_quantile")):
        raise SystemExit("calibration quantiles must be in [0, 1]")
    return CalibrationConfig(
        version=raw["version"],
        splits=splits,
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


def load_split_lookup(manifest_path: Path, lds_sha256: str) -> tuple[dict[int, str], Record]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != lds_sha256:
        raise SystemExit(f"{manifest_path} was built from another LDS file")
    lookup: dict[int, str] = {}
    for name in SPLIT_NAMES:
        for hearing_id in manifest[name]:
            if hearing_id in lookup:
                raise SystemExit(f"hearing {hearing_id} is listed in two splits")
            lookup[hearing_id] = name
    source = {
        "path": str(manifest_path),
        "sha256": sha256_of_file(manifest_path),
        "split_version": manifest["split_version"],
    }
    return lookup, source


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


def encode_guarded(
    encoder: SentenceTransformer,
    texts: list[str],
    udv_config: UdvConfig,
    device: str,
    kind: str,
    hearing_id: int,
    ledger: EncodingLedger,
) -> np.ndarray:
    if hearing_id not in ledger.allowed_hearing_ids:
        raise SystemExit(f"refusing hearing {hearing_id}: it is outside the calibration splits")
    label = f"{kind}_{hearing_id}"
    before = set(udv_config.cache_dir.glob(f"{label}_*.npy"))
    embeddings = encode_with_cache(encoder, texts, udv_config, device, label)
    written = set(udv_config.cache_dir.glob(f"{label}_*.npy")) - before
    counts = ledger.by_kind.setdefault(
        kind, {"calls": 0, "texts": 0, "cache_hits": 0, "encoded": 0, "empty": 0}
    )
    counts["calls"] += 1
    counts["texts"] += len(texts)
    counts["empty" if not texts else "encoded" if written else "cache_hits"] += 1
    ledger.encoded_hearing_ids.add(hearing_id)
    return embeddings


def trusted_quote_queries(hearing: Record, people: list[Record]) -> tuple[list[Record], Record]:
    slices = sentence_slices_by_person(people)
    opinions = [
        (person, opinion_index, opinion_text)
        for person in people
        for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"])
    ]
    counts = {
        "opinions": len(opinions),
        "resolved_person": 0,
        "trusted_quote": 0,
        "target_not_in_candidates": 0,
    }
    queries = []
    for position, (person, opinion_index, opinion_text) in enumerate(opinions):
        if not person["matched_turns"]:
            continue
        counts["resolved_person"] += 1
        quote_match = find_opinion_turn_quote_match(opinion_text, person["matched_turns"])
        if quote_match is None or not is_trusted_quote(quote_match):
            continue
        counts["trusted_quote"] += 1
        target = quote_match["sentence"]
        target_indices = [
            index
            for index, sentence in enumerate(person["sentences"])
            if sentences_agree(sentence, target)
        ]
        if not target_indices:
            counts["target_not_in_candidates"] += 1
            continue
        queries.append(
            {
                "id": f"udv-{hearing['id']}-{person['index']}-{opinion_index}",
                "hearing_id": hearing["id"],
                "person_index": person["index"],
                "opinion": opinion_text,
                "opinion_position": position,
                "target_text": target,
                "target_indices": target_indices,
                "candidates": person["sentences"],
                "sentence_offset": slices[person["index"]].start,
            }
        )
    return queries, counts


def masked_row_problems(row: Record, base_queries: dict[str, Record]) -> list[str]:
    query = base_queries.get(row["id"])
    if query is None:
        return ["not_a_trusted_quote_query"]
    checks = {
        "opinion_mismatch": row["opinion"] == query["opinion"],
        "person_mismatch": row["person"]["index"] == query["person_index"],
        "candidates_mismatch": row["candidates"] == query["candidates"],
        "target_text_mismatch": row["target"]["text"] == query["target_text"],
        "target_indices_mismatch": row["target_indices"] == query["target_indices"],
    }
    return [problem for problem, passed in checks.items() if not passed]


def score_query(
    query: Record, kind: str, query_embedding: np.ndarray, candidate_embeddings: np.ndarray
) -> Record:
    scores = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
    targets = query["target_indices"]
    top1 = int(scores.argmax())
    return {
        **query,
        "query": kind,
        "candidate_scores": scores,
        "non_target_indices": np.setdiff1d(np.arange(len(scores)), targets),
        "positive_score": float(scores[targets].max()),
        "top1_index": top1,
        "top1_score": float(scores[top1]),
        "top1_correct": top1 in targets,
    }


def collect_hearing(
    hearing: Record,
    rows: list[Record],
    encoder: SentenceTransformer,
    udv_config: UdvConfig,
    device: str,
    ledger: EncodingLedger,
) -> Record:
    hearing_id = hearing["id"]
    people = resolve_hearing_people(hearing)
    sentences = [sentence for person in people for sentence in person["sentences"]]
    sentence_embeddings = encode_guarded(
        encoder, sentences, udv_config, device, "sentences", hearing_id, ledger
    )
    slices = sentence_slices_by_person(people)
    base_queries, funnel = trusted_quote_queries(hearing, people)
    unmasked: list[Record] = []
    query_embeddings = np.empty((0, sentence_embeddings.shape[1]), dtype=np.float32)
    if base_queries:
        opinion_texts = [text for person in people for text in person["participant"]["opinioes"]]
        opinion_embeddings = encode_guarded(
            encoder, opinion_texts, udv_config, device, "opinions", hearing_id, ledger
        )
        query_embeddings = opinion_embeddings[[q["opinion_position"] for q in base_queries]]
        unmasked = [
            score_query(
                query,
                "opinion",
                opinion_embeddings[query["opinion_position"]],
                sentence_embeddings[slices[query["person_index"]]],
            )
            for query in base_queries
        ]
    by_id = {query["id"]: query for query in base_queries}
    problems = {row["id"]: masked_row_problems(row, by_id) for row in rows}
    problems = {row_id: found for row_id, found in problems.items() if found}
    masked: list[Record] = []
    if rows and not problems:
        masked_embeddings = encode_guarded(
            encoder,
            [row["masked_opinion"] for row in rows],
            udv_config,
            device,
            "masked",
            hearing_id,
            ledger,
        )
        masked = [
            score_query(
                {**by_id[row["id"]], "masked_opinion": row["masked_opinion"]},
                "masked_opinion",
                masked_embeddings[position],
                sentence_embeddings[slices[row["person"]["index"]]],
            )
            for position, row in enumerate(rows)
        ]
    return {
        "unmasked": unmasked,
        "query_embeddings": query_embeddings,
        "masked": masked,
        "sentence_embeddings": sentence_embeddings,
        "funnel": funnel,
        "problems": problems,
    }


def build_pool(
    query_embeddings: np.ndarray, sentence_embeddings: dict[int, np.ndarray]
) -> SentencePool:
    hearing_ids = sorted(h for h, embeddings in sentence_embeddings.items() if len(embeddings))
    return SentencePool(
        hearing_ids=np.concatenate(
            [np.full(len(sentence_embeddings[h]), h, dtype=np.int64) for h in hearing_ids]
        ),
        sentence_indices=np.concatenate(
            [np.arange(len(sentence_embeddings[h]), dtype=np.int64) for h in hearing_ids]
        ),
        scores=np.hstack(
            [cosine_similarity(query_embeddings, sentence_embeddings[h]) for h in hearing_ids]
        ),
    )


def draw_pool_negatives(
    rng: np.random.Generator,
    rows: np.ndarray,
    negatives_per_query: int,
    query_hearing_ids: np.ndarray,
    pool: SentencePool,
) -> NegativeDraw:
    repeated = np.repeat(rows, negatives_per_query)
    hearings = query_hearing_ids[repeated]
    picks = rng.integers(0, len(pool.hearing_ids), size=len(repeated))
    clash = pool.hearing_ids[picks] == hearings
    while clash.any():
        picks[clash] = rng.integers(0, len(pool.hearing_ids), size=int(clash.sum()))
        clash = pool.hearing_ids[picks] == hearings
    return NegativeDraw(
        pool.scores[repeated, picks], pool.hearing_ids[picks], pool.sentence_indices[picks]
    )


def draw_hard_negatives(
    rng: np.random.Generator, rows: np.ndarray, negatives_per_query: int, queries: list[Record]
) -> NegativeDraw:
    repeated = np.repeat(rows, negatives_per_query)
    sizes = np.array([len(queries[row]["non_target_indices"]) for row in repeated], dtype=np.int64)
    picks = rng.integers(0, sizes) if len(repeated) else np.empty(0, dtype=np.int64)
    candidates = [
        int(queries[row]["non_target_indices"][pick])
        for row, pick in zip(repeated, picks, strict=True)
    ]
    return NegativeDraw(
        np.array(
            [
                queries[row]["candidate_scores"][candidate]
                for row, candidate in zip(repeated, candidates, strict=True)
            ],
            dtype=np.float64,
        ),
        np.array([queries[row]["hearing_id"] for row in repeated], dtype=np.int64),
        np.array(
            [
                queries[row]["sentence_offset"] + candidate
                for row, candidate in zip(repeated, candidates, strict=True)
            ],
            dtype=np.int64,
        ),
    )


def rounded(value: float) -> float:
    return round(float(value), SCORE_DECIMALS)


def describe(values: np.ndarray) -> Record:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": rounded(values.min()),
        "q25": rounded(np.quantile(values, 0.25)),
        "median": rounded(np.median(values)),
        "q75": rounded(np.quantile(values, 0.75)),
        "max": rounded(values.max()),
        "mean": rounded(values.mean()),
    }


def interval(values: list[float], confidence_level: float) -> Record:
    if not values:
        return {"replicates": 0}
    array = np.array(values, dtype=np.float64)
    tail = (1 - confidence_level) / 2
    return {
        "replicates": len(values),
        "low": rounded(np.quantile(array, tail)),
        "high": rounded(np.quantile(array, 1 - tail)),
        "median": rounded(np.median(array)),
        "std": rounded(array.std(ddof=1)) if len(values) > 1 else None,
    }


def unit_groups(hearing_ids: list[int], unit: str) -> list[np.ndarray]:
    if unit == "query":
        return [np.array([row], dtype=np.int64) for row in range(len(hearing_ids))]
    groups: dict[int, list[int]] = {}
    for row, hearing_id in enumerate(hearing_ids):
        groups.setdefault(hearing_id, []).append(row)
    return [np.array(rows, dtype=np.int64) for _, rows in sorted(groups.items())]


def resample_rows(rng: np.random.Generator, groups: list[np.ndarray]) -> np.ndarray:
    picks = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[pick] for pick in picks])


def rule_rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def pair_threshold(
    positives: np.ndarray, negatives: np.ndarray, config: CalibrationConfig
) -> Record:
    positive_value = float(np.quantile(positives, config.positive_quantile))
    negative_value = float(np.quantile(negatives, config.negative_quantile))
    return {
        "threshold": (positive_value + negative_value) / 2,
        "positive_quantile_value": positive_value,
        "negative_quantile_value": negative_value,
    }


def evaluate_pair_rule(
    queries: list[Record],
    sampler: NegativeSampler,
    config: CalibrationConfig,
    rng: np.random.Generator,
) -> tuple[Record, NegativeDraw]:
    positives = np.array([query["positive_score"] for query in queries], dtype=np.float64)
    point_draw = sampler(rng, np.arange(len(queries), dtype=np.int64))
    point = pair_threshold(positives, point_draw.scores, config)
    groups = unit_groups([query["hearing_id"] for query in queries], config.bootstrap_unit)
    replicates = []
    for _ in range(config.bootstrap_samples):
        rows = resample_rows(rng, groups)
        replicates.append(pair_threshold(positives[rows], sampler(rng, rows).scores, config))
    threshold = point["threshold"]
    result = {
        "pairs": {
            "queries": len(queries),
            "hearings": len({query["hearing_id"] for query in queries}),
            "positives": len(positives),
            "negatives": len(point_draw.scores),
        },
        "threshold": rounded(threshold),
        "threshold_exact": threshold,
        "threshold_rounded": round(threshold, config.threshold_decimals),
        "positive_quantile_value": rounded(point["positive_quantile_value"]),
        "negative_quantile_value": rounded(point["negative_quantile_value"]),
        "positives_below_threshold": int((positives < threshold).sum()),
        "negatives_at_or_above_threshold": int((point_draw.scores >= threshold).sum()),
        "positive_scores": describe(positives),
        "negative_scores": describe(point_draw.scores),
        "bootstrap": {
            key: interval([replicate[key] for replicate in replicates], config.confidence_level)
            for key in ("threshold", "positive_quantile_value", "negative_quantile_value")
        },
    }
    return result, point_draw


def youden_optimum(scores: np.ndarray, correct: np.ndarray) -> Record | None:
    positives = int(correct.sum())
    negatives = len(correct) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered, labels = scores[order], correct[order]
    group_end = np.append(ordered[1:] != ordered[:-1], True)
    cut_scores = ordered[group_end]
    tpr = np.cumsum(labels)[group_end] / positives
    fpr = np.cumsum(~labels)[group_end] / negatives
    youden = tpr - fpr
    best = int(np.flatnonzero(youden >= youden.max() - 1e-12)[-1])
    lowest_kept = float(cut_scores[best])
    highest_excluded = float(cut_scores[best + 1]) if best + 1 < len(cut_scores) else None
    threshold = lowest_kept if highest_excluded is None else (lowest_kept + highest_excluded) / 2
    return {
        "threshold": threshold,
        "youden_j": float(youden[best]),
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
        "lowest_kept_score": lowest_kept,
        "highest_excluded_score": highest_excluded,
    }


def operating_point(scores: np.ndarray, correct: np.ndarray, threshold: float) -> Record:
    kept = scores >= threshold
    correct_total = int(correct.sum())
    incorrect_total = len(correct) - correct_total
    return {
        "threshold": rounded(threshold),
        "kept": int(kept.sum()),
        "kept_correct": int((kept & correct).sum()),
        "coverage": rounded(kept.mean()) if len(scores) else None,
        "precision": rounded(correct[kept].mean()) if kept.any() else None,
        "tpr": rounded((kept & correct).sum() / correct_total) if correct_total else None,
        "fpr": rounded((kept & ~correct).sum() / incorrect_total) if incorrect_total else None,
    }


def top1_arrays(queries: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.array([query["top1_score"] for query in queries], dtype=np.float64)
    correct = np.array([query["top1_correct"] for query in queries], dtype=bool)
    return scores, correct


def evaluate_primary_rule(
    queries: list[Record],
    unmasked_by_id: dict[str, Record],
    config: CalibrationConfig,
    rng: np.random.Generator,
) -> Record:
    scores, correct = top1_arrays(queries)
    optimum = youden_optimum(scores, correct)
    groups = unit_groups([query["hearing_id"] for query in queries], config.bootstrap_unit)
    accuracy, thresholds, youden, precision, coverage = [], [], [], [], []
    without_optimum = 0
    for _ in range(config.bootstrap_samples):
        rows = resample_rows(rng, groups)
        sample_scores, sample_correct = scores[rows], correct[rows]
        accuracy.append(float(sample_correct.mean()))
        replicate = youden_optimum(sample_scores, sample_correct)
        if replicate is None:
            without_optimum += 1
        else:
            thresholds.append(replicate["threshold"])
            youden.append(replicate["youden_j"])
        if optimum is not None:
            kept = sample_scores >= optimum["threshold"]
            coverage.append(float(kept.mean()))
            if kept.any():
                precision.append(float(sample_correct[kept].mean()))
    n_candidates = np.array([len(query["candidates"]) for query in queries], dtype=np.float64)
    level = config.confidence_level
    return {
        "pairs": {
            "queries": len(queries),
            "hearings": len({query["hearing_id"] for query in queries}),
            "top1_correct": int(correct.sum()),
            "top1_incorrect": int((~correct).sum()),
        },
        "acc_at_1": rounded(correct.mean()),
        "random_acc_at_1": rounded(
            np.mean([len(q["target_indices"]) / len(q["candidates"]) for q in queries])
        ),
        "unmasked_acc_at_1_same_queries": rounded(
            np.mean([unmasked_by_id[query["id"]]["top1_correct"] for query in queries])
        ),
        "candidates": describe(n_candidates),
        "top1_score_correct": describe(scores[correct]),
        "top1_score_incorrect": describe(scores[~correct]),
        "threshold": None if optimum is None else rounded(optimum["threshold"]),
        "threshold_exact": None if optimum is None else optimum["threshold"],
        "threshold_rounded": (
            None if optimum is None else round(optimum["threshold"], config.threshold_decimals)
        ),
        "optimum": (
            None
            if optimum is None
            else {key: None if value is None else rounded(value) for key, value in optimum.items()}
        ),
        "at_threshold": (
            None if optimum is None else operating_point(scores, correct, optimum["threshold"])
        ),
        "bootstrap": {
            "acc_at_1": interval(accuracy, level),
            "threshold": interval(thresholds, level),
            "youden_j": interval(youden, level),
            "precision_at_point_threshold": interval(precision, level),
            "coverage_at_point_threshold": interval(coverage, level),
            "replicates_without_optimum": without_optimum,
        },
    }


def negatives_by_query(
    queries: list[Record], draw: NegativeDraw, negatives_per_query: int
) -> dict[str, list[Record]]:
    return {
        query["id"]: [
            {
                "hearing_id": int(draw.hearing_ids[index]),
                "sentence_index": int(draw.sentence_indices[index]),
                "score": round(float(draw.scores[index]), QUERY_SCORE_DECIMALS),
            }
            for index in range(position * negatives_per_query, (position + 1) * negatives_per_query)
        ]
        for position, query in enumerate(queries)
    }


def run_rules(
    unmasked: list[Record],
    masked: list[Record],
    pool: SentencePool,
    config: CalibrationConfig,
) -> tuple[Record, dict[str, dict[str, list[Record]]]]:
    k = config.negatives_per_query
    with_hard = [query for query in unmasked if len(query["non_target_indices"])]
    masked_with_hard = [query for query in masked if len(query["non_target_indices"])]
    masked_ids = {query["id"] for query in masked}
    inputs: dict[str, tuple[list[Record], NegativeSampler]] = {
        "legacy_random": (
            unmasked,
            partial(
                draw_pool_negatives,
                negatives_per_query=k,
                query_hearing_ids=np.array([q["hearing_id"] for q in unmasked], dtype=np.int64),
                pool=pool,
            ),
        ),
        "hard_negative": (
            with_hard,
            partial(draw_hard_negatives, negatives_per_query=k, queries=with_hard),
        ),
        "masked_hard_negative": (
            masked_with_hard,
            partial(draw_hard_negatives, negatives_per_query=k, queries=masked_with_hard),
        ),
    }
    results: Record = {}
    draws: dict[str, dict[str, list[Record]]] = {}
    for stream, name in enumerate(PAIR_RULES):
        queries, sampler = inputs[name]
        result, draw = evaluate_pair_rule(queries, sampler, config, rule_rng(config.seed, stream))
        results[name] = {"definition": RULE_DEFINITIONS[name], **result}
        draws[name] = negatives_by_query(queries, draw, k)
    results["hard_negative"]["queries_without_hard_negative"] = len(unmasked) - len(with_hard)
    results["masked_hard_negative"]["queries_without_hard_negative"] = len(masked) - len(
        masked_with_hard
    )
    same_ids = [query for query in with_hard if query["id"] in masked_ids]
    same_ids_result, _ = evaluate_pair_rule(
        same_ids,
        partial(draw_hard_negatives, negatives_per_query=k, queries=same_ids),
        config,
        rule_rng(config.seed, RULES.index("hard_negative"), 1),
    )
    results["hard_negative"]["on_masked_query_ids"] = {
        "definition": (
            "hard_negative restricted to the opinions that are also masked queries, so that "
            "masked_hard_negative differs from it only by the masking"
        ),
        **same_ids_result,
    }
    unmasked_by_id = {query["id"]: query for query in unmasked}
    results[PRIMARY_RULE] = {
        "definition": RULE_DEFINITIONS[PRIMARY_RULE],
        **evaluate_primary_rule(
            masked, unmasked_by_id, config, rule_rng(config.seed, RULES.index(PRIMARY_RULE))
        ),
    }
    return results, draws


def operating_points(results: Record, masked: list[Record], configured_threshold: float) -> Record:
    scores, correct = top1_arrays(masked)
    thresholds = {name: results[name]["threshold_exact"] for name in RULES}
    thresholds["configured_embedding_threshold"] = configured_threshold
    return {
        name: None if threshold is None else operating_point(scores, correct, threshold)
        for name, threshold in thresholds.items()
    }


def leak_check(
    split_of: dict[int, str],
    config: CalibrationConfig,
    ledger: EncodingLedger,
    query_hearings: dict[str, set[int]],
    pool: SentencePool,
) -> Record:
    pool_hearings = {int(hearing_id) for hearing_id in np.unique(pool.hearing_ids)}
    used = set(ledger.encoded_hearing_ids) | pool_hearings
    for hearing_ids in query_hearings.values():
        used |= hearing_ids
    outside = [name for name in SPLIT_NAMES if name not in config.splits]
    intersections = {
        name: sorted(hearing_id for hearing_id in used if split_of[hearing_id] == name)
        for name in outside
    }
    passed = used <= ledger.allowed_hearing_ids and not any(intersections.values())
    if not passed:
        raise SystemExit(f"hearing leak outside the calibration splits: {intersections}")
    return {
        "calibration_splits": list(config.splits),
        "allowed_hearings": len(ledger.allowed_hearing_ids),
        "hearings_used": len(used),
        "hearing_ids_used": sorted(used),
        "encoded_hearing_ids": sorted(ledger.encoded_hearing_ids),
        "query_hearing_ids": {name: sorted(ids) for name, ids in query_hearings.items()},
        "negative_pool_hearing_ids": sorted(pool_hearings),
        "intersection_with": intersections,
        "passed": passed,
    }


def query_rows(queries: list[Record], draws: dict[str, dict[str, list[Record]]]) -> list[Record]:
    return [
        {
            "id": query["id"],
            "hearing_id": query["hearing_id"],
            "query": query["query"],
            "n_candidates": len(query["candidates"]),
            "target_indices": query["target_indices"],
            "positive_score": round(query["positive_score"], QUERY_SCORE_DECIMALS),
            "top1_index": query["top1_index"],
            "top1_score": round(query["top1_score"], QUERY_SCORE_DECIMALS),
            "top1_correct": query["top1_correct"],
            "negatives": {
                name: by_id[query["id"]]
                for name, by_id in draws.items()
                if RULE_QUERY_KIND[name] == query["query"] and query["id"] in by_id
            },
        }
        for query in queries
    ]


def benchmark_source(path: Path) -> Record:
    report_path = path.with_name(f"{path.stem}_report.json")
    recorded = None
    if report_path.exists():
        with open(report_path) as f:
            recorded = json.load(f).get("code", {}).get(source_label(transcript))
    current = sha256_of_file(module_path(transcript))
    return {
        "path": str(path),
        "sha256": sha256_of_file(path),
        "report_path": str(report_path) if report_path.exists() else None,
        "report_udv_pipeline_sha256": recorded,
        "current_udv_pipeline_sha256": current,
        "udv_pipeline_sha256_matches": recorded == current,
    }


def sum_counts(counts: list[Record]) -> Record:
    return {key: sum(count[key] for count in counts) for key in (counts[0] if counts else {})}


def build_report(
    results: Record,
    points: Record,
    leak: Record,
    ledger: EncodingLedger,
    funnel: Record,
    masked_counts: Record,
    sources: Record,
    artifacts: Record,
    encoder: SentenceTransformer,
    device: str,
    elapsed_seconds: float,
    udv_config: UdvConfig,
    config: CalibrationConfig,
) -> Record:
    primary = results[config.primary_rule]
    return {
        "calibration_version": config.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "primary_rule": config.primary_rule,
        "primary_rule_declaration": config.source["primary_rule_declaration"],
        "primary_threshold": primary["threshold"],
        "primary_threshold_rounded": primary["threshold_rounded"],
        "configured_embedding_threshold": udv_config.embedding_threshold,
        "configured_embedding_threshold_modified": False,
        "label_semantics": LABEL_SEMANTICS,
        "rules": results,
        "pair_counts": {name: results[name]["pairs"] for name in RULES},
        "operating_points_on_primary_queries": points,
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "unit": config.bootstrap_unit,
            "confidence_level": config.confidence_level,
            "method": "percentile; pair rules redraw their random negatives in every replicate",
            "seed": config.seed,
        },
        "queries": {"unmasked": funnel, "masked": masked_counts},
        "leak_check": leak,
        "embedding_cache": {
            "cache_dir": str(udv_config.cache_dir),
            "labels": {kind: f"{kind}_{{hearing_id}}" for kind in EMBEDDING_KINDS},
            "by_kind": ledger.by_kind,
        },
        "encoder": {
            "name": udv_config.model_name,
            "revision": udv_config.model_revision,
            "device": device,
            "embedding_dimension": encoder.get_embedding_dimension(),
            "max_seq_length": encoder.max_seq_length,
        },
        "sources": sources,
        "artifacts": artifacts,
        "code": {
            **source_hashes(transcript, *transcript.SOURCES),
            **source_hashes(udv_run),
            **source_hashes(Path(__file__)),
        },
        "timing": {"elapsed_seconds": round(elapsed_seconds, 1)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "platform": platform.platform(),
        },
        "config": udv_config.source,
    }


def print_summary(report: Record) -> None:
    for name in PAIR_RULES:
        result = report["rules"][name]
        ci = result["bootstrap"]["threshold"]
        print(
            f"{name:22s} threshold={result['threshold']:.4f} "
            f"[{ci['low']:.4f}, {ci['high']:.4f}] pairs={result['pairs']['positives']}"
            f"+{result['pairs']['negatives']}"
        )
    primary = report["rules"][PRIMARY_RULE]
    ci = primary["bootstrap"]["threshold"]
    interval_text = f"[{ci['low']:.4f}, {ci['high']:.4f}]" if ci["replicates"] else "[n/a]"
    print(
        f"{PRIMARY_RULE:22s} threshold={primary['threshold']} {interval_text} "
        f"queries={primary['pairs']['queries']} acc@1={primary['acc_at_1']}"
    )
    print(
        f"primary threshold ({report['primary_rule']}): {report['primary_threshold']} "
        f"(rounded {report['primary_threshold_rounded']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the embedding threshold on the calibration splits only, with "
        "bootstrap intervals, from the embeddings cached by the UDV builder."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    udv_config = load_config(args.config)
    config = load_calibration_config(udv_config)
    seed_everything(udv_config.seed)
    lds = load_lds_records(udv_config)
    split_of, split_source = load_split_lookup(config.manifest_path, udv_config.expected_sha256)
    hearings = select_calibration_hearings(lds, split_of, config.splits)
    ledger = EncodingLedger(allowed_hearing_ids=frozenset(hearing["id"] for hearing in hearings))
    masked_rows, masked_counts = load_masked_rows(
        config.masked_benchmark_path, split_of, config.splits
    )
    device = select_device(udv_config.device)
    print(
        f"{len(hearings)} calibration hearings ({', '.join(config.splits)}) | "
        f"{udv_config.model_name}@{udv_config.model_revision[:7]} on {device}",
        flush=True,
    )
    encoder = load_encoder(udv_config, device)

    unmasked: list[Record] = []
    masked: list[Record] = []
    query_embeddings: list[np.ndarray] = []
    sentence_embeddings: dict[int, np.ndarray] = {}
    funnels: list[Record] = []
    problems: dict[str, list[str]] = {}
    for number, hearing in enumerate(hearings, start=1):
        collected = collect_hearing(
            hearing, masked_rows.get(hearing["id"], []), encoder, udv_config, device, ledger
        )
        unmasked.extend(collected["unmasked"])
        masked.extend(collected["masked"])
        query_embeddings.append(collected["query_embeddings"])
        sentence_embeddings[hearing["id"]] = collected["sentence_embeddings"]
        funnels.append(collected["funnel"])
        problems.update(collected["problems"])
        print(
            f"[{number}/{len(hearings)}] hearing {hearing['id']}: "
            f"{len(collected['unmasked'])} quote queries, {len(collected['masked'])} masked",
            flush=True,
        )
    unused_rows = sorted(set(masked_rows) - ledger.allowed_hearing_ids)
    if problems or unused_rows:
        raise SystemExit(
            "the masked benchmark does not match the current pipeline; rebuild it with "
            f"experiments.data.quote_benchmark: {dict(list(problems.items())[:10])} {unused_rows}"
        )
    if not unmasked or not masked:
        raise SystemExit("no calibration queries: check the splits and the masked benchmark")

    pool = build_pool(np.vstack(query_embeddings), sentence_embeddings)
    pool_hearings = {int(hearing_id) for hearing_id in np.unique(pool.hearing_ids)}
    if any(not pool_hearings - {query["hearing_id"]} for query in unmasked):
        raise SystemExit("the random-negative pool needs sentences from another hearing")
    del sentence_embeddings
    results, draws = run_rules(unmasked, masked, pool, config)
    query_hearings = {
        "legacy_random": {query["hearing_id"] for query in unmasked},
        "hard_negative": {
            query["hearing_id"] for query in unmasked if len(query["non_target_indices"])
        },
        "masked_hard_negative": {
            query["hearing_id"] for query in masked if len(query["non_target_indices"])
        },
        PRIMARY_RULE: {query["hearing_id"] for query in masked},
    }
    leak = leak_check(split_of, config, ledger, query_hearings, pool)
    points = operating_points(results, masked, udv_config.embedding_threshold)

    rows_path = config.output_dir / f"{config.version}_queries.jsonl"
    rows = query_rows(unmasked, draws) + query_rows(masked, draws)
    write_jsonl(rows, rows_path)
    artifacts = {
        "queries": {
            "path": str(rows_path),
            "rows": len(rows),
            "sha256": sha256_of_file(rows_path),
        }
    }
    sources = {
        "lds": {"path": str(udv_config.lds_path), "sha256": udv_config.expected_sha256},
        "splits": split_source,
        "masked_benchmark": benchmark_source(config.masked_benchmark_path),
    }
    funnel = {
        **sum_counts(funnels),
        "used_by_legacy_random": len(unmasked),
        "used_by_hard_negative": results["hard_negative"]["pairs"]["queries"],
    }
    masked_summary = {**masked_counts, "used": len(masked)}
    report = build_report(
        results,
        points,
        leak,
        ledger,
        funnel,
        masked_summary,
        sources,
        artifacts,
        encoder,
        device,
        time.perf_counter() - started,
        udv_config,
        config,
    )
    write_json(report, config.output_dir / f"{config.version}.json")
    print_summary(report)
    if report["primary_threshold"] is None:
        raise SystemExit("the primary rule has no threshold: top-1 is correct for all or none")


if __name__ == "__main__":
    main()
