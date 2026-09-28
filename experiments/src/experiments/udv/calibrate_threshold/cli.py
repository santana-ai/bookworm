"""Calibrate the embedding threshold of a UDV run on the calibration splits only."""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import write_json
from sentence_transformers import SentenceTransformer

from experiments.common.splits import load_split_lookup
from experiments.common.udv_run import (
    UdvConfig,
    load_config,
    load_encoder,
    load_lds_records,
    seed_everything,
    select_device,
)
from experiments.udv.calibrate_threshold.config import (
    load_calibration_config,
    load_masked_rows,
    select_calibration_hearings,
)
from experiments.udv.calibrate_threshold.negatives import SentencePool, build_pool
from experiments.udv.calibrate_threshold.queries import EncodingLedger, collect_hearing
from experiments.udv.calibrate_threshold.report import (
    benchmark_source,
    build_report,
    leak_check,
    print_summary,
    sum_counts,
    write_query_rows,
)
from experiments.udv.calibrate_threshold.rules import (
    operating_points,
    rule_query_hearings,
    run_rules,
)

Record = dict[str, Any]


@dataclass
class CollectedQueries:
    unmasked: list[Record] = field(default_factory=list)
    masked: list[Record] = field(default_factory=list)
    query_embeddings: list[np.ndarray] = field(default_factory=list)
    sentence_embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    funnels: list[Record] = field(default_factory=list)
    problems: dict[str, list[str]] = field(default_factory=dict)


def collect_queries(
    hearings: list[Record],
    masked_rows: dict[int, list[Record]],
    encoder: SentenceTransformer,
    udv_config: UdvConfig,
    device: str,
    ledger: EncodingLedger,
) -> CollectedQueries:
    collected = CollectedQueries()
    for number, hearing in enumerate(hearings, start=1):
        found = collect_hearing(
            hearing, masked_rows.get(hearing["id"], []), encoder, udv_config, device, ledger
        )
        collected.unmasked.extend(found["unmasked"])
        collected.masked.extend(found["masked"])
        collected.query_embeddings.append(found["query_embeddings"])
        collected.sentence_embeddings[hearing["id"]] = found["sentence_embeddings"]
        collected.funnels.append(found["funnel"])
        collected.problems.update(found["problems"])
        print(
            f"[{number}/{len(hearings)}] hearing {hearing['id']}: "
            f"{len(found['unmasked'])} quote queries, {len(found['masked'])} masked",
            flush=True,
        )
    return collected


def check_collected(
    collected: CollectedQueries, masked_rows: dict[int, list[Record]], ledger: EncodingLedger
) -> None:
    unused_rows = sorted(set(masked_rows) - ledger.allowed_hearing_ids)
    if collected.problems or unused_rows:
        raise SystemExit(
            "the masked benchmark does not match the current pipeline; rebuild it with "
            f"experiments.data.quote_benchmark: {dict(list(collected.problems.items())[:10])} "
            f"{unused_rows}"
        )
    if not collected.unmasked or not collected.masked:
        raise SystemExit("no calibration queries: check the splits and the masked benchmark")


def negative_pool(collected: CollectedQueries) -> SentencePool:
    pool = build_pool(np.vstack(collected.query_embeddings), collected.sentence_embeddings)
    pool_hearings = {int(hearing_id) for hearing_id in np.unique(pool.hearing_ids)}
    if any(not pool_hearings - {query["hearing_id"]} for query in collected.unmasked):
        raise SystemExit("the random-negative pool needs sentences from another hearing")
    return pool


def calibrate(config_path: Path) -> None:
    started = time.perf_counter()
    udv_config = load_config(config_path)
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
    collected = collect_queries(hearings, masked_rows, encoder, udv_config, device, ledger)
    check_collected(collected, masked_rows, ledger)
    unmasked, masked = collected.unmasked, collected.masked
    pool = negative_pool(collected)
    collected.sentence_embeddings.clear()
    results, draws = run_rules(unmasked, masked, pool, config)
    leak = leak_check(split_of, config, ledger, rule_query_hearings(unmasked, masked), pool)
    points = operating_points(results, masked, udv_config.embedding_threshold)
    artifacts = {"queries": write_query_rows(unmasked, masked, draws, config)}
    sources = {
        "lds": {"path": str(udv_config.lds_path), "sha256": udv_config.expected_sha256},
        "splits": split_source,
        "masked_benchmark": benchmark_source(config.masked_benchmark_path),
    }
    funnel = {
        **sum_counts(collected.funnels),
        "used_by_legacy_random": len(unmasked),
        "used_by_hard_negative": results["hard_negative"]["pairs"]["queries"],
    }
    report = build_report(
        results,
        points,
        leak,
        ledger,
        funnel,
        {**masked_counts, "used": len(masked)},
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the embedding threshold on the calibration splits only, with "
        "bootstrap intervals, from the embeddings cached by the UDV builder."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    return parser.parse_args()


def main() -> None:
    calibrate(parse_args().config)
