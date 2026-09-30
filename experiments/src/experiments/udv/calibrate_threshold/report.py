"""The calibration report: leak check, per-query rows, sources and the printed summary."""

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
from bookworm import sha256_of_file, write_jsonl
from sentence_transformers import SentenceTransformer

from experiments.common import transcript, udv_run
from experiments.common.provenance import code_section, module_path, source_label
from experiments.common.reporting import file_record, utc_timestamp
from experiments.common.splits import SPLIT_NAMES
from experiments.common.udv_run import UdvConfig
from experiments.udv.calibrate_threshold.config import (
    PAIR_RULES,
    PRIMARY_RULE,
    RULE_QUERY_KIND,
    RULES,
    CalibrationConfig,
)
from experiments.udv.calibrate_threshold.negatives import QUERY_SCORE_DECIMALS, SentencePool
from experiments.udv.calibrate_threshold.queries import EMBEDDING_KINDS, EncodingLedger

Record = dict[str, Any]

PACKAGE_DIR = Path(__file__).parent
LABEL_SEMANTICS: Record = {
    "target": (
        "silver label: the sentence of the speaker's own turn that encloses the matched quote "
        "prefix of 6+ words (find_opinion_turn_quote_match of bookworm.udv.quotes), found by "
        "string matching and not validated by a human"
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


def write_query_rows(
    unmasked: list[Record],
    masked: list[Record],
    draws: dict[str, dict[str, list[Record]]],
    config: CalibrationConfig,
) -> Record:
    rows_path = config.output_dir / f"{config.version}_queries.jsonl"
    rows = query_rows(unmasked, draws) + query_rows(masked, draws)
    write_jsonl(rows, rows_path)
    return file_record(rows_path, len(rows))


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


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "platform": platform.platform(),
    }


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
        "created_at": utc_timestamp(),
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
        "code": code_section(*transcript.CODE_SOURCES, udv_run, PACKAGE_DIR),
        "timing": {"elapsed_seconds": round(elapsed_seconds, 1)},
        "environment": environment(),
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
