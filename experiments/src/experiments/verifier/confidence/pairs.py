"""The pairs command: NLI input pairs for the entailment signals."""

import argparse
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from bookworm import write_json, write_jsonl

from experiments.common.reporting import file_record, utc_timestamp
from experiments.retrieval.data import Query, Unit
from experiments.retrieval.workload import Workload, build_workload
from experiments.verifier.confidence.collect import collect_reports, load_features, split_record
from experiments.verifier.confidence.config import (
    ConfidenceConfig,
    Target,
    load_harness,
    pair_id_of,
    pairs_input_file,
    resolve_splits,
    run_directory,
    select_targets,
    text_sha256,
)
from experiments.verifier.confidence.provenance import code_hashes, environment
from experiments.verifier.confidence.records import clean

Record = dict[str, Any]


Features = dict[str, dict[str, dict[str, list[Record]]]]
Lookup = dict[tuple[str, str, str], tuple[Query, list[Unit]]]


def load_target_features(
    run_dir: Path, targets: list[Target], config: ConfidenceConfig, splits: tuple[str, ...]
) -> tuple[Features, list[Record]]:
    loaded: Features = {}
    inputs: list[Record] = []
    for target in targets:
        result = load_features(run_dir, target, config.benches, splits)
        if result is not None:
            loaded[target.name], files = result
            inputs += files
    if not loaded:
        raise SystemExit(f"no collected features for targets with units {config.entailment_units}")
    return loaded, inputs


def unit_lookup(workload: Workload, unit_names: set[str]) -> Lookup:
    """The query and the candidate units of each (bench, query id, unit)."""
    lookup: Lookup = {}
    for hearing in workload.hearings:
        for query in workload.queries[hearing.hearing_id]:
            for unit_name in unit_names:
                lookup[(query.bench, query.query_id, unit_name)] = (
                    query,
                    hearing.contexts[query.context_key].units[unit_name],
                )
    return lookup


def new_pair(pair_id: str, bench: str, row: Record, query: Query, unit: Unit) -> Record:
    return {
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
    }


def build_pairs(loaded: Features, unit_of: dict[str, str], lookup: Lookup) -> list[Record]:
    """One pair per distinct top-1 text of a query, listing the targets that retrieved it."""
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
                    entry = pairs.setdefault(pair_id, new_pair(pair_id, bench, row, query, unit))
                    entry["meta"]["targets"].append(name)
    return sorted(pairs.values(), key=lambda p: (p["meta"]["bench"], p["hearing_id"], p["pair_id"]))


def scoring_command(
    config: ConfidenceConfig, path: Path, run_dir: Path, final_test: bool
) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "experiments.verifier.nli_experiments",
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
        *(["--final-test"] if final_test else []),
    ]


def feature_row_count(loaded: Features) -> int:
    return sum(
        len(rows) for benches in loaded.values() for s in benches.values() for rows in s.values()
    )


def command_pairs(args: argparse.Namespace, config: ConfidenceConfig) -> None:
    harness = load_harness(config)
    splits = resolve_splits(config, args.final_test)
    run_dir = run_directory(config, args.run_name)
    targets = [t for t in select_targets(config, args.targets) if t.unit in config.entailment_units]
    loaded, inputs = load_target_features(run_dir, targets, config, splits)
    workload = build_workload(harness, splits, config.benches, None, None)
    unit_of = {target.name: target.unit for target in targets}
    lookup = unit_lookup(workload, {unit_of[name] for name in loaded})
    ordered = build_pairs(loaded, unit_of, lookup)
    path = pairs_input_file(run_dir)
    write_jsonl(ordered, path)
    counts = Counter(f"{p['meta']['bench']}.{p['meta']['split']}" for p in ordered)
    command = scoring_command(config, path, run_dir, args.final_test)
    report = {
        "experiment": "confidence_policies",
        "step": "pairs",
        "run_name": run_dir.name,
        "created_at": utc_timestamp(),
        "splits": split_record(config, args.final_test),
        "splits_used": list(splits),
        "targets": sorted(loaded),
        "pairs": len(ordered),
        "pairs_by_bench_split": dict(sorted(counts.items())),
        "collect_reports": collect_reports(run_dir, sorted(loaded)),
        "feature_rows": feature_row_count(loaded),
        "pair_rule": config.source["entailment"]["pair"],
        "output": file_record(path),
        "scoring_command": shlex.join(command),
        "inputs": inputs,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    write_json(clean(report), run_dir / "nli" / "pairs_input_report.json")
    print(f"[pairs] {len(ordered)} pairs {dict(counts)} -> {path}", flush=True)
    print(f"[pairs] score them with:\n{shlex.join(command)}", flush=True)
