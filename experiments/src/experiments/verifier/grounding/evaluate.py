"""The evaluate-ea command: existing and literature systems on train and validation."""

import argparse
import json
import time
from typing import Any

import numpy as np
from bookworm import write_json

from experiments.common.reporting import file_record, rounded, utc_timestamp
from experiments.verifier.exploration.candidates import feature_candidate
from experiments.verifier.exploration.config import (
    ExplorationConfig,
    ScorerData,
    load_exploration_config,
)
from experiments.verifier.exploration.scores import candidate_scores, load_labels, load_scorer
from experiments.verifier.grounding.config import (
    COMBINATIONS,
    EA_SPLITS,
    EXISTING,
    EXISTING_FEATURES,
    PRIMARY,
    REFERENCE,
    Config,
)
from experiments.verifier.grounding.laya import laya_systems, spread_flag_table
from experiments.verifier.grounding.metrics import (
    evaluate_set,
    new_scores,
    paired_comparisons,
    point_metrics,
    with_combinations,
)
from experiments.verifier.grounding.provenance import run_record
from experiments.verifier.udv_scores.config import (
    FittedPrimary,
    candidate_scorers,
    primary_candidate,
)
from experiments.verifier.udv_scores.config import load_config as load_udv_verifier_config
from experiments.verifier.udv_scores.primary import fit_primary, load_fit_data, refit_check

Record = dict[str, Any]


def subset_table(table: Record, names: list[str]) -> Record:
    return {name: table[name] for name in names if name in table}


def exploration_data(
    expl: ExplorationConfig, keys: tuple[str, ...], split: str, ids: list[str]
) -> dict[str, ScorerData]:
    data = {}
    for key in keys:
        kind, run = expl.scorers[key]
        scorer = load_scorer(key, kind, run, split, ids, expl)
        if scorer is None:
            raise SystemExit(f"{key}: no {split} score file in run {run}")
        data[key] = scorer
    return data


def fitted_primary(
    config: Config,
) -> tuple[FittedPrimary, ExplorationConfig, Record]:
    udv_config = load_udv_verifier_config(config.udv_verifier_config)
    expl = load_exploration_config(udv_config.exploration_config)
    candidate = primary_candidate(expl, udv_config.primary)
    with open(udv_config.selection_path) as f:
        selection = json.load(f)
    with open(udv_config.final_test_path) as f:
        final_test = json.load(f)
    scorers = candidate_scorers(candidate)
    labels, hearings, data, files = load_fit_data(expl, scorers, selection)
    fitted = fit_primary(candidate, labels, hearings, data, expl)
    check = refit_check(fitted, final_test, udv_config.primary)
    return fitted, expl, {"refit_check": check, "fit_files": files, "key": udv_config.primary}


def ea_existing(
    split: str,
    ids: list[str],
    fitted: FittedPrimary,
    expl: ExplorationConfig,
) -> tuple[dict[str, np.ndarray], Record]:
    scorer_keys = tuple(
        dict.fromkeys(
            [
                *candidate_scorers(fitted.candidate),
                *(feature.split(":")[0] for feature in EXISTING_FEATURES.values()),
            ]
        )
    )
    data = exploration_data(expl, scorer_keys, split, ids)
    systems = {
        name: candidate_scores(feature_candidate(feature), data)
        for name, feature in EXISTING_FEATURES.items()
    }
    systems[PRIMARY] = fitted.probabilities(data)
    sources = {key: file_record(d.path) for key, d in data.items()}
    return {name: systems[name] for name in EXISTING}, sources


def available_candidates(config: Config, splits: tuple[str, ...]) -> tuple[list[str], Record]:
    present, absent = [], {}
    for key in config.order:
        paths = [config.scores_dir / f"{key}_{split}.jsonl" for split in splits]
        if all(path.exists() for path in paths):
            present.append(key)
        else:
            absent[key] = "no score file (dropped at the smoke or not run)"
    return present, absent


def literature_scores(
    config: Config, candidates: list[str], split: str, ids: list[str], pool: str
) -> tuple[dict[str, np.ndarray], Record]:
    systems: dict[str, np.ndarray] = {}
    sources: Record = {}
    for key in candidates:
        if pool == "concatenated" and not config.candidates[key].concatenated:
            continue
        systems[key], sources[key] = new_scores(config, key, split, ids, pool)
    return systems, sources


def split_extra(
    extra: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spreads = {name: values for name, values in extra.items() if "_spread_" in name}
    others = {name: values for name, values in extra.items() if "_spread_" not in name}
    return others, spreads


def added_value_pairs(extra: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    return [
        (name, name.removesuffix("_q11") + "_q7")
        for name in extra
        if name.endswith("_q11") and "_spread_" not in name
    ]


def spread_thresholds(config: Config) -> tuple[dict[str, float], Record]:
    udv_config = load_udv_verifier_config(config.udv_verifier_config)
    expl = load_exploration_config(udv_config.exploration_config)
    ids, _, _ = load_labels(expl, "train")
    _, extra, sources = laya_systems(config, "train", ids, "max")
    _, spreads = split_extra(extra)
    return {name: float(np.median(values)) for name, values in spreads.items()}, sources


def spread_section(
    spreads: dict[str, np.ndarray],
    thresholds: dict[str, float],
    positive: np.ndarray,
    hearings: np.ndarray,
    table: Record,
    config: Config,
) -> Record:
    return {
        name: {
            "as_signal": table[f"neg_{name}"],
            "flag": spread_flag_table(
                values,
                thresholds[name],
                positive,
                hearings,
                config.bootstrap_samples,
                config.seed,
                config.level,
            ),
            "median": rounded(float(np.median(values))),
        }
        for name, values in spreads.items()
        if name in thresholds
    }


def evaluate_all(
    config: Config,
    base: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
    positive: np.ndarray,
    hearings: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Record]:
    systems = with_combinations(base)
    others, spreads = split_extra(extra)
    signals = {**systems, **others, **{f"neg_{name}": -values for name, values in spreads.items()}}
    result = evaluate_set(
        signals, positive, hearings, config.bootstrap_samples, config.seed, config.level
    )
    return systems, spreads, result


def command_evaluate_ea(args: argparse.Namespace, config: Config) -> None:
    path = config.output_dir / "ea_report.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    started = time.perf_counter()
    fitted, expl, primary_record = fitted_primary(config)
    candidates, absent = available_candidates(config, EA_SPLITS)
    thresholds, _ = spread_thresholds(config)
    results: Record = {}
    inputs: Record = {}
    for split in EA_SPLITS:
        ids, labels, hearings = load_labels(expl, split)
        existing, sources = ea_existing(split, ids, fitted, expl)
        laya_main, laya_extra, laya_sources = laya_systems(config, split, ids, "max")
        literature, literature_sources = literature_scores(config, candidates, split, ids, "max")
        base = {**existing, **laya_main, **literature}
        systems, spreads, result = evaluate_all(config, base, laya_extra, labels, hearings)
        table, replicates = result["systems"], result["_replicates"]
        family = [(name, REFERENCE) for name in base if name != REFERENCE]
        combos = [(name, REFERENCE) for name in COMBINATIONS]
        extra_names = [name for name in laya_extra if "_spread_" not in name]
        joined_laya, joined_extra, _ = laya_systems(config, split, ids, "concatenated")
        joined_literature, _ = literature_scores(config, candidates, split, ids, "concatenated")
        joined = {**joined_literature, **joined_laya, **split_extra(joined_extra)[0]}
        signals = {**systems, **{name: laya_extra[name] for name in extra_names}}
        results[split] = {
            "items": result["items"],
            "hearings": result["hearings"],
            "positives": result["positives"],
            "negatives": result["negatives"],
            "systems": subset_table(table, list(systems)),
            "comparisons": paired_comparisons(signals, labels, replicates, family, config.level),
            "combination_comparisons": paired_comparisons(
                signals, labels, replicates, combos, config.level
            ),
            "added_questions": {
                "systems": subset_table(table, extra_names),
                "comparisons": paired_comparisons(
                    signals, labels, replicates, added_value_pairs(laya_extra), config.level
                ),
            },
            "spread": spread_section(spreads, thresholds, labels, hearings, table, config),
            "concatenated": {name: point_metrics(v, labels) for name, v in joined.items()},
            "in_sample": [PRIMARY, *COMBINATIONS] if split == "train" else [],
            "bootstrap": result["bootstrap"],
        }
        if split == "train":
            results[split]["note"] = (
                "descriptive: the Holm families are the validation ones; the E3x primary, and "
                "the rank combinations that use it, are in-sample on train"
            )
        inputs[split] = {**sources, **laya_sources, **literature_sources}
    report = {
        "experiment": config.name,
        "part": "E-A",
        "created_at": utc_timestamp(),
        "splits_read": list(EA_SPLITS),
        "label": config.raw["evaluation"]["positive_ea"],
        "label_semantics": config.raw["experiment"]["label_semantics"],
        "premise": {
            "max": config.raw["premise"]["ea_max"],
            "concatenated": config.raw["premise"]["ea_concatenated"],
            "domain_shift": config.raw["premise"]["domain_shift"],
        },
        "family": config.raw["evaluation"]["ea_family"],
        "laya": {
            key: config.raw["laya"][key]
            for key in ("support_rule", "aggregate_rule", "spread_rule")
        },
        "spread_thresholds_train": {name: rounded(value) for name, value in thresholds.items()},
        "candidates": {"literature_scored": candidates, "absent": absent},
        "primary": primary_record,
        "results": results,
        "inputs": inputs,
        "seconds": round(time.perf_counter() - started, 3),
        **run_record(config),
    }
    write_json(report, path)
    print_table(results["validation"], "E-A validation")


def print_table(result: Record, title: str) -> None:
    print(f"== {title}: {result['items']} items, {result['negatives']} negatives", flush=True)
    for name, metrics in result["systems"].items():
        cells = []
        for metric in ("roc_auc", "aurc", "precision_at_0.8", "precision_at_0.9"):
            entry = metrics[metric]
            span = entry["interval"]
            cells.append(f"{metric} {entry['point']:.3f} [{span.get('low')}, {span.get('high')}]")
        print(f"{name:20s} " + " | ".join(cells), flush=True)
