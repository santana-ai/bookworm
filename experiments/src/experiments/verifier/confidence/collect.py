"""The collect command: ranking signals of every query from the retrieval harness."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, sha256_of_file, write_json, write_jsonl

from experiments.common.provenance import source_label
from experiments.common.reporting import utc_timestamp
from experiments.common.udv_run import load_config as load_udv_config
from experiments.retrieval.experiments import (
    ExperimentConfig,
    Workload,
    build_retriever,
    build_workload,
    evaluate_query,
    unit_mean_chars,
)
from experiments.retrieval.models import (
    Runtime,
)
from experiments.verifier.confidence.config import (
    ConfidenceConfig,
    Target,
    check_targets,
    features_file,
    load_harness,
    resolve_splits,
    run_directory,
    select_targets,
)
from experiments.verifier.confidence.harness import (
    feature_row,
    guard_retriever,
    harness_run_reports,
    load_harness_rows,
    missing_inputs,
    model_revisions,
    row_differences,
    store_directories,
)
from experiments.verifier.confidence.provenance import ENTRY_POINT, code_hashes, environment
from experiments.verifier.confidence.records import clean

Record = dict[str, Any]


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
            {"created_at": utc_timestamp(), "splits_used": list(splits), "skipped": skipped},
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
            "created_at": utc_timestamp(),
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
            "code_sha256": report["code"].get(source_label(ENTRY_POINT)),
        }
        if report["hearing_filter"] is not None:
            print(f"WARNING {name}: collected on hearings {report['hearing_filter']} only")
    return reports
