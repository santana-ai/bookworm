"""The evaluate command and its console summary."""

import argparse
import json
import time
from typing import Any

import numpy as np
from bookworm import write_json

from experiments.udv.calibrate_threshold import load_split_lookup
from experiments.verifier.nli.benchmark import check_benchmark_file, load_benchmark_rows
from experiments.verifier.nli.comparisons import (
    add_derived_systems,
    declared_comparisons,
    missing_detail,
    robustness_report,
    run_declaration,
    scorer_summaries,
)
from experiments.verifier.nli.config import COMPARED_RULE, VerifierConfig, evaluated_splits
from experiments.verifier.nli.evaluation import (
    available_scorers,
    compare_systems,
    draw_bootstraps,
    evaluate_constant,
    evaluate_judges,
    evaluate_system,
    judge_keys_of,
    merge_fit_splits,
    split_data,
    split_summary,
)
from experiments.verifier.nli.provenance import code_hashes, environment
from experiments.verifier.nli.scoring import run_directory
from experiments.verifier.nli.table import (
    TABLE_COLUMNS,
    comparison_table,
    declared_block,
    write_table,
)
from experiments.verifier.runtime import now

Record = dict[str, Any]


def command_evaluate(args: argparse.Namespace, config: VerifierConfig) -> None:
    started = time.perf_counter()
    declaration = run_declaration(config, args)
    own = declaration.scorers if declaration else config.evaluation_scorers
    imported = declaration.imported if declaration else {}
    eval_splits = evaluated_splits(config, args.final_test)
    splits = (*config.fit_splits, *eval_splits)
    run_dir = run_directory(config, args)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    benchmark = check_benchmark_file(config)
    rows = load_benchmark_rows(config, split_of, splits)
    judge_keys = judge_keys_of(rows)
    systems_by_scorer, reports, directories = available_scorers(
        config, run_dir, splits, own, imported
    )
    data = {
        split: split_data(split, rows, systems_by_scorer, directories, judge_keys, set(own))
        for split in splits
    }
    fit = merge_fit_splits([data[split] for split in config.fit_splits])
    evaluations = [data[split] for split in eval_splits]
    fit_draws, eval_draws = draw_bootstraps(fit, evaluations, config)
    systems = [f"{key}.{name}" for key, names in systems_by_scorer.items() for name in names]
    derived_names, derived, thresholds = add_derived_systems(
        fit, evaluations, systems_by_scorer, config
    )
    systems += derived_names
    results: Record = {}
    replicates: dict[str, dict[str, dict[tuple[str, str], np.ndarray]]] = {}
    for system in systems:
        results[system], replicates[system] = evaluate_system(
            system, fit, evaluations, fit_draws, eval_draws, config
        )
        print(f"evaluated {system}", flush=True)
    judges, judge_replicates = evaluate_judges(
        fit, evaluations, eval_draws, config.confidence_level
    )
    own_reports = [report for key, report in reports.items() if key in own]
    report = {
        "experiment": "nli_verifier",
        "run_name": args.run_name,
        "created_at": now(),
        "question": (
            declaration.source["question"]
            if declaration
            else (
                "can an open NLI model tell whether an opinion is inferable from the four "
                "retrieved chunks as the annotator judged it, compared with the 12 stored LLM "
                "judges and with cosine similarity under the production encoder"
            )
        ),
        "label_semantics": config.source["benchmark"]["label_semantics"],
        "splits": {
            "fit": list(config.fit_splits),
            "evaluate": list(eval_splits),
            "final_test_flag": args.final_test,
            "test_read": "test" in splits,
            "manifest": split_source,
            "hearing_ids": {
                split: sorted(set(d.hearing_ids.tolist())) for split, d in data.items()
            },
        },
        "opinions": {split: split_summary(d) for split, d in data.items()},
        "subset": (own_reports or list(reports.values()))[0]["subset"],
        "subset_warning": (
            None
            if (own_reports or list(reports.values()))[0]["subset"] is None
            else "a smoke run on a subset: its numbers check that the code runs and are not results"
        ),
        "declared": declared_block(config, declaration),
        "systems_evaluated": systems,
        "scorers_missing": [key for key in (*own, *imported) if key not in systems_by_scorer],
        "scorers_missing_detail": missing_detail(
            [key for key in (*own, *imported) if key not in systems_by_scorer],
            run_dir,
            imported,
            splits,
        ),
        "systems": results,
        "judges": judges,
        "always_inferable": {
            d.split: evaluate_constant(d, eval_draws[d.split], config.confidence_level)
            for d in evaluations
        },
        "comparisons": compare_systems(
            results, replicates, judges, judge_replicates, list(eval_splits), config
        ),
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "unit": "hearing",
            "confidence_level": config.confidence_level,
            "seed": config.evaluation_seed,
            "method": (
                "percentile; the same replicates are shared by every system and judge, "
                "so deltas are paired"
            ),
        },
        "scorers": scorer_summaries(reports),
        "sources": {
            "benchmark": benchmark,
            "splits": split_source,
            "score_files": {key: report["files"] for key, report in reports.items()},
        },
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    if declaration is not None:
        report["imported"] = {
            key: {"run": run, "available": key in systems_by_scorer}
            for key, run in imported.items()
        }
        report["derived_systems"] = derived
        report["declared_comparisons"] = declared_comparisons(
            declaration,
            results,
            replicates,
            judges,
            judge_replicates,
            list(eval_splits),
            config,
            {d.split: len(set(d.hearing_ids.tolist())) for d in evaluations},
        )
        report["robustness"] = robustness_report(
            evaluations, systems_by_scorer, directories, config, thresholds
        )
        table = comparison_table(report, config, declaration)
        report["table"] = {"columns": list(TABLE_COLUMNS), "rows": table}
        write_table(table, run_dir / "comparison_table.csv")
    report["timing"] = {"elapsed_seconds": round(time.perf_counter() - started, 1)}
    write_json(report, run_dir / "evaluation_report.json")
    print_evaluation(report)


def format_interval(value: Any, bootstrap: Record) -> str:
    if value is None:
        return "n/a"
    if not bootstrap.get("replicates"):
        return f"{value:.3f}"
    return f"{value:.3f} [{bootstrap['low']:.3f}, {bootstrap['high']:.3f}]"


def print_evaluation(report: Record) -> None:
    for split in report["splits"]["evaluate"]:
        print(f"== {split}: {json.dumps(report['opinions'][split])}")
        for system, result in report["systems"].items():
            evaluation = result["evaluation"][split]
            rule = evaluation["rules"].get(COMPARED_RULE)
            kappa = (
                "n/a"
                if rule is None
                else format_interval(
                    rule["metrics"]["cohen_kappa"], rule["bootstrap_fixed_threshold"]["cohen_kappa"]
                )
            )
            ranking, bootstrap = evaluation["ranking"], evaluation["ranking_bootstrap"]
            auc = format_interval(ranking["roc_auc"], bootstrap["roc_auc"])
            ap_not = format_interval(
                ranking["average_precision_not_inferable"],
                bootstrap["average_precision_not_inferable"],
            )
            print(f"  {system:44s} auc={auc} ap_not={ap_not} kappa@{COMPARED_RULE}={kappa}")
        for key, judged in report["judges"]["evaluation"][split].items():
            metrics = judged["metrics"]
            kappa = format_interval(metrics["cohen_kappa"], judged["bootstrap"]["cohen_kappa"])
            f1_not = metrics["per_class"]["not_inferable"]["f1"]
            print(f"  judge {key:38s} kappa={kappa} f1_not={f1_not:.3f}")
        always = report["always_inferable"][split]["metrics"]
        print(f"  always_inferable accuracy={always['accuracy']:.3f} kappa={always['cohen_kappa']}")
        declared = report.get("declared_comparisons", {}).get(split)
        if declared is None:
            continue
        for name in declared["family_order"]:
            family = declared["families"][name]
            print(
                f"  family {name}: Holm over {family['family_size']}, "
                f"{len(family['missing'])} with a missing system"
            )
            for entry in declared["comparisons"]:
                if entry["family"] != name:
                    continue
                if entry.get("missing"):
                    print(f"    {entry['name']:66s} missing {entry['missing']}")
                    continue
                delta = entry.get("delta")
                shown = "n/a" if delta is None else f"{delta:+.3f}"
                p_value = "n/a" if entry["p_value"] is None else f"{entry['p_value']:.4f}"
                print(
                    f"    {entry['name']:66s} {entry['metric']} delta={shown} "
                    f"p={p_value} p_holm={entry['p_holm']:.4f}"
                )
    if report.get("scorers_missing"):
        print(f"scorers missing: {report['scorers_missing']}")
