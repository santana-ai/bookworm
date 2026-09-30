"""The queries of a harness run: benchmark rows checked against the manifest and the pipeline."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm import load_gated_jsonl, load_jsonl, sha256_of_file

from experiments.common.splits import load_split_lookup
from experiments.retrieval.config import ExperimentConfig
from experiments.retrieval.data import HearingData, Query, build_queries, get_context, load_hearing

Record = dict[str, Any]

MASKED_BENCH = "masked_quotes"
MISMATCH_MARKER = "mismatch"


@dataclass
class Workload:
    hearings: list[HearingData]
    queries: dict[int, list[Query]]
    checks: dict[str, Counter[str]]
    span_checks: Counter[str]
    sources: Record
    masked_texts_by_hearing: dict[int, list[str]]


def load_bench_rows(
    path: Path, split_of: dict[int, str], splits: tuple[str, ...]
) -> tuple[dict[int, list[Record]], Record]:
    by_hearing: dict[int, list[Record]] = {}
    counts: Counter[str] = Counter()
    for row in load_jsonl(path):
        counts["rows_read"] += 1
        if split_of.get(row["hearing_id"]) != row["split"]:
            raise SystemExit(f"{row['id']}: split {row['split']!r} differs from the manifest")
        if row["split"] not in splits:
            counts["rows_outside_requested_splits"] += 1
            continue
        counts[f"rows_{row['split']}"] += 1
        by_hearing.setdefault(row["hearing_id"], []).append(row)
    return by_hearing, {"path": str(path), "sha256": sha256_of_file(path), **counts}


def selected_records(
    lds: list[Record],
    split_of: dict[int, str],
    splits: tuple[str, ...],
    rows: dict[str, dict[int, list[Record]]],
    limit_hearings: int | None,
) -> list[Record]:
    records = [
        record
        for record in lds
        if split_of[record["id"]] in splits
        and any(record["id"] in by_id for by_id in rows.values())
    ]
    return records if limit_hearings is None else records[:limit_hearings]


def check_queries(checks: dict[str, Counter[str]]) -> None:
    problems = {
        bench: {key: count for key, count in counter.items() if MISMATCH_MARKER in key}
        for bench, counter in checks.items()
    }
    if any(problems.values()):
        raise SystemExit(f"benchmark rows disagree with the current pipeline: {problems}")


def masked_texts(rows: dict[str, dict[int, list[Record]]]) -> dict[int, list[str]]:
    return {
        hearing_id: [row["masked_opinion"] for row in hearing_rows]
        for hearing_id, hearing_rows in rows.get(MASKED_BENCH, {}).items()
    }


def build_workload(
    config: ExperimentConfig,
    splits: tuple[str, ...],
    benches: tuple[str, ...],
    limit_hearings: int | None,
    limit_queries: int | None,
) -> Workload:
    """Load the hearings of the requested splits and build their queries, bench by bench."""
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    rows: dict[str, dict[int, list[Record]]] = {}
    bench_sources: Record = {}
    for bench in benches:
        rows[bench], bench_sources[bench] = load_bench_rows(
            Path(config.benches[bench]["path"]), split_of, splits
        )
    checks: dict[str, Counter[str]] = {bench: Counter() for bench in benches}
    span_checks: Counter[str] = Counter()
    hearings: list[HearingData] = []
    queries: dict[int, list[Query]] = {}
    remaining = {bench: limit_queries for bench in benches}
    for record in selected_records(lds, split_of, splits, rows, limit_hearings):
        if all(remaining[bench] == 0 for bench in benches):
            break
        hearing = load_hearing(record, split_of[record["id"]])
        span_checks.update(hearing.span_checks)
        get_context(hearing, None, config.window_sizes)
        hearing_queries: list[Query] = []
        for bench in benches:
            built = build_queries(
                bench,
                rows[bench].get(record["id"], []),
                hearing,
                config.window_sizes,
                checks[bench],
            )
            limit = remaining[bench]
            if limit is not None:
                built = built[:limit]
                remaining[bench] = limit - len(built)
            hearing_queries.extend(built)
        if hearing_queries:
            hearings.append(hearing)
            queries[hearing.hearing_id] = hearing_queries
    check_queries(checks)
    return Workload(
        hearings=hearings,
        queries=queries,
        checks=checks,
        span_checks=span_checks,
        sources={
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "splits": split_source,
            "benchmarks": bench_sources,
        },
        masked_texts_by_hearing=masked_texts(rows),
    )


def workload_counts(workload: Workload) -> Record:
    counts: Record = {}
    for hearing in workload.hearings:
        for query in workload.queries[hearing.hearing_id]:
            key = f"{query.bench}.{query.split}"
            counts.setdefault(key, {"queries": 0, "hearings": set()})
            counts[key]["queries"] += 1
            counts[key]["hearings"].add(query.hearing_id)
    return {
        key: {"queries": value["queries"], "hearings": len(value["hearings"])}
        for key, value in sorted(counts.items())
    }
