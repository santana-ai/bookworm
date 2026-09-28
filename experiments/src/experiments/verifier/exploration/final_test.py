"""The final-test command: evaluate the declared final candidates once on test."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.reporting import file_record, utc_timestamp
from experiments.verifier.exploration.candidates import (
    check_reading,
    fixed_candidates,
    learned_candidates,
    resolve_reference,
)
from experiments.verifier.exploration.config import MATCH_TOLERANCE, ExplorationConfig, ScorerData
from experiments.verifier.exploration.confirm import compare_systems, evaluate_split, system_scores
from experiments.verifier.exploration.provenance import code_hashes
from experiments.verifier.exploration.scores import load_all, load_labels, load_scorer, output_dir
from experiments.verifier.nli.provenance import environment

Record = dict[str, Any]


def top_confirmed(out: Path, count: int) -> list[str]:
    with open(out / "confirmation.csv") as f:
        rows = [row for row in csv.DictReader(f) if row["role"] != "reference"]
    rows.sort(key=lambda row: -float(row["roc_auc"]))
    return [row["system"] for row in rows[:count]]


def check_train_copy(fit_data: dict[str, ScorerData], copy: dict[str, ScorerData]) -> Record:
    checked: Record = {}
    for key, scorer in copy.items():
        gaps = [
            float(np.max(np.abs(values - fit_data[key].stored[name])))
            for name, values in scorer.stored.items()
        ]
        worst = max(gaps)
        if worst > MATCH_TOLERANCE:
            raise SystemExit(f"{key}: train scores of the test run differ from explore by {worst}")
        checked[key] = {"scores_compared": len(gaps), "max_abs_gap": worst}
    return checked


def command_final_test(args: argparse.Namespace, config: ExplorationConfig) -> None:
    final = config.raw["final_test"]
    out = output_dir(config, args.output_dir)
    target = out / "final_test.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the final test runs once")
    keys = list(final["candidates"])
    if keys != top_confirmed(out, len(keys)):
        raise SystemExit("final_test.candidates differs from the top rows of confirmation.csv")
    references = tuple(final["references"])
    fit_split = config.raw["fit_split"]
    _, fit_labels, fit_hearings, fit_data, _ = load_all(config, fit_split)
    fit_ids, _, _ = load_labels(config, fit_split)
    ids, labels, hearings = load_labels(config, "test")
    run = final["score_run"]
    test_data: dict[str, ScorerData] = {}
    train_copy: dict[str, ScorerData] = {}
    for key, (kind, _) in config.scorers.items():
        scorer = load_scorer(key, kind, run, "test", ids, config)
        if scorer is None:
            continue
        test_data[key] = scorer
        copy = load_scorer(key, kind, run, fit_split, fit_ids, config)
        if copy is None or key not in fit_data:
            raise SystemExit(f"{key}: the test run lacks its {fit_split} scores")
        train_copy[key] = copy
    reading = check_reading(config, test_data)
    same_train = check_train_copy(fit_data, train_copy)
    built = {
        c.key: c
        for c in [*fixed_candidates(config, fit_data), *learned_candidates(config, fit_data)]
    }
    systems: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    learned: Record = {}
    for key in keys:
        fit_scores, scores, model = system_scores(
            built[key], fit_data, test_data, fit_labels, fit_hearings, config
        )
        systems[key] = (fit_scores, scores)
        if model:
            learned[key] = model
    for reference in references:
        systems[reference] = (
            resolve_reference(reference, config, fit_data),
            resolve_reference(reference, config, test_data),
        )
    results, replicates = evaluate_split(systems, fit_labels, labels, hearings, config)
    for name, result in results.items():
        result["role"] = (
            "primary" if name == final["primary"] else "reference" if name in references else "top5"
        )
        result["threshold_fitted_on"] = fit_split
    comparisons = compare_systems(keys, references, results, replicates, config)
    report = {
        "experiment": "nli_verifier_exploration",
        "name": config.name,
        "stage": "final_test",
        "created_at": utc_timestamp(),
        "splits_read": [fit_split, "test"],
        "declared": final["declared"],
        "primary": final["primary"],
        "score_run": run,
        "files_read": {key: file_record(s.path) for key, s in test_data.items()},
        "reading_check": reading,
        "train_copy_check": same_train,
        "opinions": {
            "count": len(ids),
            "hearings": int(len(set(hearings.tolist()))),
            "not_inferable": int((~labels).sum()),
        },
        "learned": learned,
        "results": results,
        "comparisons": comparisons,
        "comparison_rule": final["comparison_rule"],
        "label_semantics": config.verifier["benchmark"]["label_semantics"],
        "config": file_record(config.path),
        "code": code_hashes(),
        "environment": environment(),
    }
    with open(target, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"results": results, "comparisons": comparisons}, indent=2))
