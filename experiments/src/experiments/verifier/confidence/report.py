"""The evaluate command and its console summary."""

import argparse
import time
from collections import Counter
from typing import Any

from bookworm import write_json

from experiments.common.reporting import utc_timestamp
from experiments.common.splits import load_split_lookup
from experiments.verifier.confidence.bootstrap import FIXED, REFIT
from experiments.verifier.confidence.collect import collect_reports, load_features, split_record
from experiments.verifier.confidence.config import (
    DEFINITIONS,
    LOGISTIC_PREFIX,
    ConfidenceConfig,
    check_manifest_splits,
    evaluated_splits,
    load_harness,
    resolve_splits,
    run_directory,
    select_targets,
)
from experiments.verifier.confidence.production import universes
from experiments.verifier.confidence.provenance import code_hashes, environment
from experiments.verifier.confidence.records import clean
from experiments.verifier.confidence.target import (
    attach_entailment,
    evaluate_target,
    load_entailment,
)

Record = dict[str, Any]


def check_query_sets(loaded: dict[str, dict[str, dict[str, list[Record]]]], bench: str) -> None:
    reference: dict[str, set[str]] | None = None
    for name, benches in loaded.items():
        sets = {split: {row["query_id"] for row in rows} for split, rows in benches[bench].items()}
        if reference is None:
            reference = sets
        elif sets != reference:
            raise SystemExit(f"{bench}: {name} has another query set than the first target")


def primary_summary(results: Record, config: ConfidenceConfig) -> Record:
    summary: Record = {}
    for bench, targets in results.items():
        target = targets.get(config.primary_target)
        if target is None:
            summary[bench] = {"available": False}
            continue
        summary[bench] = {}
        for split, evaluation in target["evaluate"].items():
            metrics = evaluation.get("signal_metrics", {}).get("multi_candidate", {})
            summary[bench][split] = {}
            for name, entry in metrics.items():
                item: Record = {"aurc": entry["aurc"]["point"]}
                if "difference_vs_reference" in entry:
                    difference = entry["difference_vs_reference"]["aurc"]
                    mode = REFIT if name.startswith(LOGISTIC_PREFIX) else FIXED
                    interval = difference.get(mode)
                    item["difference_aurc"] = difference["point"]
                    item["interval_mode"] = mode
                    item["interval"] = interval
                    item["better_than_reference"] = bool(
                        interval is not None
                        and interval["high"] is not None
                        and interval["high"] < 0
                    )
                summary[bench][split][name] = item
    return summary


def command_evaluate(args: argparse.Namespace, config: ConfidenceConfig) -> None:
    run_dir = run_directory(config, args.run_name)
    eval_splits = evaluated_splits(config, args.final_test)
    splits = resolve_splits(config, args.final_test)
    targets = select_targets(config, args.targets)
    loaded: dict[str, dict[str, dict[str, list[Record]]]] = {}
    inputs: list[Record] = []
    missing: dict[str, str] = {}
    for target in targets:
        result = load_features(run_dir, target, config.benches, splits)
        if result is None:
            missing[target.name] = "features not collected for every bench and split"
            continue
        loaded[target.name], files = result
        inputs += files
    if not loaded:
        raise SystemExit(f"no collected features under {run_dir}")
    harness = load_harness(config)
    split_of, split_source = load_split_lookup(config.manifest_path, harness.lds_sha256)
    check_manifest_splits(loaded, split_of)
    scores, entailment_sources = load_entailment(run_dir, config)
    entailment_counts: Record = {}
    by_name = {target.name: target for target in targets}
    for name, benches in loaded.items():
        counts: Counter[str] = Counter()
        for by_split in benches.values():
            for rows in by_split.values():
                counts.update(attach_entailment(rows, by_name[name], config, scores))
        entailment_counts[name] = dict(counts)
    results: Record = {}
    outputs: list[Record] = []
    started = time.perf_counter()
    for bench in config.benches:
        check_query_sets(loaded, bench)
        first = next(iter(loaded.values()))[bench]
        draws = universes(first, splits, bench, config.seed, config.bootstrap_samples)
        results[bench] = {}
        for name, benches in loaded.items():
            report, predictions = evaluate_target(
                bench, by_name[name], benches[bench], eval_splits, config, draws, run_dir
            )
            results[bench][name] = report
            outputs += predictions
            print(
                f"[evaluate] {bench} {name} done ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    report = {
        "experiment": "confidence_policies",
        "step": "evaluate",
        "run_name": run_dir.name,
        "created_at": utc_timestamp(),
        "question": "which signal best predicts that the top-1 unit of a retriever is relevant, "
        "so that a confidence tier built on it means something",
        "splits": split_record(config, args.final_test),
        "splits_used": list(splits),
        "primary": {
            "declaration": config.source["evaluation"]["primary_declaration"],
            "target": config.primary_target,
            "metric": config.source["evaluation"]["primary_metric"],
            "results": primary_summary(results, config),
        },
        "definitions": DEFINITIONS,
        "targets": {"evaluated": sorted(loaded), "missing": missing},
        "collect_reports": collect_reports(run_dir, sorted(loaded)),
        "entailment": {"sources": entailment_sources, "counts": entailment_counts},
        "results": results,
        "inputs": inputs,
        "sources": {"splits": split_source},
        "outputs": outputs,
        "seconds": round(time.perf_counter() - started, 1),
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    suffix = "final_test_report" if args.final_test else "report"
    path = run_dir / f"{run_dir.name}_{suffix}.json"
    write_json(clean(report), path)
    print_summary(clean(report))
    print(f"[evaluate] report -> {path}", flush=True)


def format_interval(entry: Record | None) -> str:
    if not entry or entry.get("low") is None:
        return "[  n/a  ]"
    return f"[{entry['low']:.3f},{entry['high']:.3f}]"


def print_summary(report: Record) -> None:
    for bench, targets in report["results"].items():
        for name, target in targets.items():
            for split, evaluation in target["evaluate"].items():
                if not evaluation.get("queries"):
                    continue
                print(
                    f"{bench} {name} {split}: n={evaluation['queries']} "
                    f"correct={evaluation['correct']} hearings={evaluation['hearings']}"
                )
                metrics = evaluation["signal_metrics"]["multi_candidate"]
                for signal, entry in metrics.items():
                    mode = REFIT if signal.startswith(LOGISTIC_PREFIX) else FIXED
                    auc, aurc = entry["roc_auc"], entry["aurc"]
                    auc_text = "  n/a" if auc["point"] is None else f"{auc['point']:.3f}"
                    print(
                        f"  {signal:34s} auc={auc_text} {format_interval(auc.get(mode))} "
                        f"aurc={aurc['point']:.3f} {format_interval(aurc.get(mode))}"
                    )
                for policy, entry in evaluation["policies"].items():
                    point = entry["evaluate"]
                    precision = point.get("precision")
                    coverage = point.get("coverage")
                    mode = FIXED if entry["rule"] == "production" else REFIT
                    print(
                        f"  {policy:48s} precision="
                        f"{'n/a' if precision is None else f'{precision:.3f}'} "
                        f"{format_interval(entry.get(mode, {}).get('precision'))} coverage="
                        f"{'n/a' if coverage is None else f'{coverage:.3f}'} "
                        f"{format_interval(entry.get(mode, {}).get('coverage'))}"
                    )
