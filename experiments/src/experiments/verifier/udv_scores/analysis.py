"""Descriptive analysis of the verifier scores over the UDVs."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file
from scipy.stats import spearmanr

from experiments.common.transcript import normalize_whitespace, split_sentences
from experiments.udv.calibrate_threshold import rounded
from experiments.verifier.exploration.candidates import (
    feature_candidate,
)
from experiments.verifier.exploration.config import (
    MATCH_TOLERANCE,
    Candidate,
    ScorerData,
)
from experiments.verifier.exploration.scores import candidate_scores
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.scoring import (
    score_report_file,
)
from experiments.verifier.udv_scores.config import (
    EVIDENCE_COSINE_FEATURES,
    SENTENCE_CHECK_RULE,
    SENTENCE_UNIT,
    WINDOW_CHECK_RULE,
    UdvVerifierConfig,
)
from experiments.verifier.udv_scores.units import support_type

Record = dict[str, Any]


def pool_agreement(candidate: Candidate, data: dict[str, ScorerData]) -> Record:
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for feature in candidate.features:
        scorer, signal, pool = feature.split(":")
        groups.setdefault((scorer, signal), {})[pool] = feature
    gaps: dict[str, float] = {}
    for (scorer, signal), pools in groups.items():
        if len(pools) < 2:
            continue
        columns = [candidate_scores(feature_candidate(f), data) for f in pools.values()]
        gap = float(np.max(np.abs(np.ptp(np.column_stack(columns), axis=1))))
        if gap > MATCH_TOLERANCE:
            raise SystemExit(f"{scorer}:{signal}: pools {sorted(pools)} differ by {gap}")
        gaps[scorer] = max(gaps.get(scorer, 0.0), gap)
    return {
        "rule": "with one premise item the max, mean and concatenated pools of a signal coincide",
        "max_abs_gap_by_scorer": gaps,
    }


def quantile_summary(values: np.ndarray, quantiles: tuple[float, ...]) -> Record:
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": rounded(float(np.mean(values))),
        "quantiles": {str(q): rounded(float(np.quantile(values, q))) for q in quantiles},
    }


def grouped_summaries(
    values: np.ndarray, groups: list[str], quantiles: tuple[float, ...]
) -> Record:
    keys = list(dict.fromkeys(groups))
    return {
        key: quantile_summary(values[[g == key for g in groups]], quantiles) for key in keys
    } | {"all": quantile_summary(values, quantiles)}


def supported_shares(decisions: np.ndarray, groups: list[str]) -> Record:
    result: Record = {}
    for key in [*dict.fromkeys(groups), "all"]:
        mask = np.ones(len(groups), bool) if key == "all" else np.array([g == key for g in groups])
        count = int(mask.sum())
        supported = int(decisions[mask].sum())
        result[key] = {
            "n": count,
            "supported": supported,
            "share": rounded(supported / count) if count else None,
        }
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> Record:
    if len(first) < 3:
        return {"n": int(len(first)), "rho": None, "p_value": None}
    result = spearmanr(first, second)
    return {
        "n": int(len(first)),
        "rho": rounded(float(result.statistic)),
        "p_value": float(result.pvalue),
    }


def cosine_relation(
    scores: dict[str, np.ndarray], cosine: np.ndarray, tiers: list[str], semantic: tuple[str, ...]
) -> Record:
    result: Record = {}
    for label, members in [*((tier, (tier,)) for tier in semantic), ("semantic_all", semantic)]:
        mask = np.array([tier in members for tier in tiers])
        result[label] = {
            name: spearman(values[mask], cosine[mask]) for name, values in scores.items()
        }
    return result


def udv_semantic_unit(udv_path: Path) -> str:
    coverage_path = udv_path.with_name(f"{udv_path.stem}_coverage.json")
    with open(coverage_path) as f:
        pipeline = json.load(f).get("pipeline") or {}
    unit = pipeline.get("semantic_unit", SENTENCE_UNIT)
    if not isinstance(unit, str):
        raise SystemExit(f"{coverage_path}: pipeline.semantic_unit is not text")
    return unit


def evidence_cosine_feature(semantic_unit: str) -> str:
    return EVIDENCE_COSINE_FEATURES[SENTENCE_UNIT if semantic_unit == SENTENCE_UNIT else "window"]


def encoded_window_text(evidence_text: str) -> str:
    return " ".join(split_sentences(normalize_whitespace(evidence_text)))


def gap_summary(gaps: np.ndarray) -> Record:
    return {
        "n": int(len(gaps)),
        "max_abs_gap": rounded(float(gaps.max())) if len(gaps) else None,
        "within_1e-4": int((gaps <= 1e-4).sum()),
    }


def evidence_score_check(
    recomputed: np.ndarray,
    recorded: np.ndarray,
    records: list[Record],
    semantic: tuple[str, ...],
    semantic_unit: str = SENTENCE_UNIT,
) -> Record:
    mask = np.array([record["tier"] in semantic for record in records])
    gaps = np.abs(recomputed[mask] - recorded[mask])
    if semantic_unit == SENTENCE_UNIT:
        return {"rule": SENTENCE_CHECK_RULE, **gap_summary(gaps)}
    semantic_records = [record for record in records if record["tier"] in semantic]
    equal = np.array(
        [
            encoded_window_text(record["evidence"]["text"])
            == normalize_whitespace(record["evidence"]["text"])
            for record in semantic_records
        ],
        dtype=bool,
    )
    return {
        "rule": WINDOW_CHECK_RULE,
        "semantic_unit": semantic_unit,
        "semantic_udvs": int(len(gaps)),
        **gap_summary(gaps[equal]),
        "encoded_text_differs": {
            **gap_summary(gaps[~equal]),
            "udv_ids": [
                record["id"]
                for record, same in zip(semantic_records, equal, strict=True)
                if not same
            ],
        },
    }


def lowest_udvs(
    records: list[Record],
    primary: np.ndarray,
    secondary: np.ndarray,
    tiers: tuple[str, ...],
    count: int,
) -> Record:
    result: Record = {}
    for tier in tiers:
        rows = [
            (float(primary[index]), record["id"], index)
            for index, record in enumerate(records)
            if record["tier"] == tier
        ]
        rows.sort()
        result[tier] = [
            {
                "udv_id": records[index]["id"],
                "hearing_id": records[index]["hearing_id"],
                "support_type": support_type(records[index]),
                "primary_probability": rounded(score),
                "p4_supports": rounded(float(secondary[index])),
                "evidence_score": records[index]["evidence"]["score"],
                "proposition": records[index]["proposition"],
                "evidence_text": records[index]["evidence"]["text"],
            }
            for score, _, index in rows[:count]
        ]
    return result


def benchmark_overlap(verifier: Record, units: list[PremiseUnit]) -> Record:
    splits_of: dict[str, set[str]] = {}
    for row in load_jsonl(Path(verifier["benchmark"]["path"])):
        splits_of.setdefault(normalize_whitespace(row["opinion"]), set()).add(row["split"])
    matched: Counter[str] = Counter()
    for unit in units:
        found = splits_of.get(unit.hypothesis)
        matched["+".join(sorted(found)) if found else "not_in_benchmark"] += 1
    return {
        "rule": "UDVs whose normalized proposition equals a normalized NLI benchmark opinion, by "
        "the benchmark split of that opinion; train opinions were hypotheses of the verifier fit, "
        "paired there with retrieved chunks instead of the evidence sentence",
        "counts": dict(sorted(matched.items())),
    }


def score_reports(config: UdvVerifierConfig) -> Record:
    reports: Record = {}
    for key in config.scorers:
        path = score_report_file(config.run_dir, key)
        with open(path) as f:
            report = json.load(f)
        reports[key] = {
            "path": str(path),
            "sha256": sha256_of_file(path),
            "created_at": report["created_at"],
            "device": (report.get("model") or {}).get("device"),
            "timing": report["timing"],
            "truncation": report["truncation"],
            "label_probes_agreeing": None
            if report.get("label_probes") is None
            else f"{report['label_probes']['agreeing']}/{report['label_probes']['total']}",
            "files": report["files"],
        }
    return reports


def analysis(
    config: UdvVerifierConfig,
    scored_records: list[Record],
    primary: np.ndarray,
    secondary: np.ndarray,
    decisions: np.ndarray,
    recomputed_cosine: np.ndarray,
    splits: list[str],
    semantic_unit: str = SENTENCE_UNIT,
) -> Record:
    tiers = [record["tier"] for record in scored_records]
    types = [support_type(record) for record in scored_records]
    evidence_scores = np.array(
        [
            np.nan if record["evidence"]["score"] is None else record["evidence"]["score"]
            for record in scored_records
        ],
        dtype=float,
    )
    scores = {"primary_probability": primary, "p4_supports": secondary}
    distributions = {
        name: {
            "by_tier": grouped_summaries(values, tiers, config.quantiles),
            "by_support_type": grouped_summaries(values, types, config.quantiles),
            "by_hearing_split": grouped_summaries(values, splits, config.quantiles),
        }
        for name, values in scores.items()
    }
    return {
        "distributions": distributions,
        "supported_at_train_threshold": {
            "by_tier": supported_shares(decisions, tiers),
            "by_support_type": supported_shares(decisions, types),
            "by_hearing_split": supported_shares(decisions, splits),
        },
        "cosine_relation": {
            "rule": config.raw["report"]["cosine_rule"],
            "spearman": cosine_relation(scores, evidence_scores, tiers, config.semantic_tiers),
        },
        "evidence_score_check": evidence_score_check(
            recomputed_cosine,
            evidence_scores,
            scored_records,
            config.semantic_tiers,
            semantic_unit,
        ),
        "lowest": {
            "rule": config.raw["report"]["lowest_rule"],
            "udvs": lowest_udvs(
                scored_records,
                primary,
                secondary,
                config.lowest_tiers,
                config.lowest_count,
            ),
        },
    }
