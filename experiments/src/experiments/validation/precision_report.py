import argparse
import math
import platform
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from utils.dataset_io import sha256_of_file, write_json
from utils.generate_validation_sample import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    REANNOTATION_CSV,
    REANNOTATION_KEY,
    ValidationConfig,
    canonical_sha256,
    code_hashes,
    existing_precision_reports,
    load_key,
    load_validation_config,
    min_successes_to_pass,
    now_iso,
    read_annotation_csv,
    require_final_test,
    rounded,
    validate_annotation,
    wilson_interval,
)

Record = dict[str, Any]

METRICS = ("strict_precision", "tolerant_precision")
SUPPORT_QUESTION = "trecho_sustenta"
SPEAKER_QUESTION = "pessoa_falou"
LABEL_SEMANTICS: Record = {
    "trecho_sustenta": (
        "one annotator's reading of whether the passage, spoken by the participant, supports "
        "the statement the LDS attributes to them; it says nothing about whether the statement "
        "is true in the world"
    ),
    "strict_precision": "share of judged UDVs of a stratum marked correta",
    "tolerant_precision": "share of judged UDVs of a stratum marked correta or parcial",
    "pessoa_falou": (
        "whether the transcript records words spoken by the participant; a no_evidence or "
        "person_not_resolved UDV is counted correct when the answer is nao_falou, and nao_sei "
        "is left out of every denominator"
    ),
    "false_absence_rate": (
        "share of decided speaker_check answers that are falou, i.e. the participant spoke but "
        "the pipeline recorded no usable speech"
    ),
    "population": (
        "N_h is the number of UDVs of stratum h in the sampled splits, frozen in the key when the "
        "sample was drawn"
    ),
}


@dataclass(frozen=True)
class ReportConfig:
    confidence_level: float
    strict_success: tuple[str, ...]
    tolerant_success: tuple[str, ...]
    speaker_correct: tuple[str, ...]
    speaker_spoke: tuple[str, ...]
    speaker_undecided: tuple[str, ...]
    output_name: str
    interim_output_prefix: str


def load_report_config(config: ValidationConfig) -> ReportConfig:
    raw = config.source["report"]
    if raw["proportion_interval"] != "wilson":
        raise SystemExit("report.proportion_interval must be wilson")
    report = ReportConfig(
        confidence_level=raw["confidence_level"],
        strict_success=tuple(raw["strict_success"]),
        tolerant_success=tuple(raw["tolerant_success"]),
        speaker_correct=tuple(raw["speaker_correct"]),
        speaker_spoke=tuple(raw["speaker_spoke"]),
        speaker_undecided=tuple(raw["speaker_undecided"]),
        output_name=raw["output_name"],
        interim_output_prefix=raw["interim_output_prefix"],
    )
    if not report.interim_output_prefix.startswith(report.output_name.removesuffix(".json")):
        raise SystemExit("report.interim_output_prefix must start with the output_name stem")
    support = set(config.questions[SUPPORT_QUESTION].judgments)
    if not set(report.strict_success) <= set(report.tolerant_success) <= support:
        raise SystemExit("report.strict_success must be within tolerant_success, within labels")
    speaker_groups = (report.speaker_correct, report.speaker_spoke, report.speaker_undecided)
    speaker_labels = [label for group in speaker_groups for label in group]
    if sorted(speaker_labels) != sorted(config.questions[SPEAKER_QUESTION].judgments):
        raise SystemExit("report speaker groups must partition the pessoa_falou labels")
    return report


def check_rules(rules: list[Record], strata: dict[str, str]) -> None:
    for rule in rules:
        if strata.get(rule["stratum"]) != SUPPORT_QUESTION:
            raise SystemExit(
                f"criterion {rule['name']}: {rule['stratum']} is not an evidence stratum"
            )
        if rule["metric"] not in METRICS:
            raise SystemExit(f"criterion {rule['name']}: metric must be one of {METRICS}")
        if not 0 < rule["min_wilson_lower"] < 1:
            raise SystemExit(f"criterion {rule['name']}: min_wilson_lower must be in (0, 1)")


def judged_units(key: Record, judged: dict[str, Record]) -> list[Record]:
    return [
        {
            "udv_id": udv_id,
            "item_id": item_id,
            "stratum": item["stratum"],
            "question": item["question"],
            "tier": item["tier"],
            "quote_cue_in_trecho": item.get("quote_cue_in_trecho"),
            **judged[item_id],
        }
        for item_id, item in key["items"].items()
        for udv_id in item["udv_ids"]
    ]


def label_counts(labels: list[str], order: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: counts[label] for label in order}


def quote_cue_breakdown(units: list[Record], rc: ReportConfig) -> Record | None:
    flags = [unit["quote_cue_in_trecho"] for unit in units]
    if any(flag is None for flag in flags):
        return None
    breakdown: Record = {
        "note": (
            "rows whose afirmacao quotes words that reappear in trecho are recognizable on the "
            "sheet (see blinding.known_leaks in sample_report.json), so the judgment of those "
            "rows was not blind to how the passage was found"
        )
    }
    for name, value in (("with_cue", True), ("without_cue", False)):
        members = [unit for unit in units if unit["quote_cue_in_trecho"] is value]
        strict = sum(1 for unit in members if unit["judgment"] in rc.strict_success)
        tolerant = sum(1 for unit in members if unit["judgment"] in rc.tolerant_success)
        breakdown[name] = {
            "udvs": len(members),
            "strict_precision": wilson_interval(strict, len(members), rc.confidence_level),
            "tolerant_precision": wilson_interval(tolerant, len(members), rc.confidence_level),
        }
    return breakdown


def evidence_stratum(
    units: list[Record], population: int, config: ValidationConfig, rc: ReportConfig
) -> Record:
    order = config.questions[SUPPORT_QUESTION].judgments
    judgments = [unit["judgment"] for unit in units]
    strict = sum(1 for label in judgments if label in rc.strict_success)
    tolerant = sum(1 for label in judgments if label in rc.tolerant_success)
    better = Counter(unit["better_passage"] for unit in units)
    return {
        "question": SUPPORT_QUESTION,
        "population": population,
        "judged_udvs": len(units),
        "judgments": label_counts(judgments, order),
        "strict_precision": wilson_interval(strict, len(units), rc.confidence_level),
        "tolerant_precision": wilson_interval(tolerant, len(units), rc.confidence_level),
        "better_passage": {value: better[value] for value in config.better_passage_values},
        "by_quote_cue": quote_cue_breakdown(units, rc),
        "udv_ids_by_judgment": {
            label: sorted(unit["udv_id"] for unit in units if unit["judgment"] == label)
            for label in order
        },
        "estimator_inputs": {
            "strict_precision": {"successes": strict, "trials": len(units)},
            "tolerant_precision": {"successes": tolerant, "trials": len(units)},
        },
    }


def speaker_rates(labels: list[str], rc: ReportConfig) -> Record:
    decided = [label for label in labels if label not in rc.speaker_undecided]
    correct = sum(1 for label in decided if label in rc.speaker_correct)
    spoke = sum(1 for label in decided if label in rc.speaker_spoke)
    return {
        "answers": len(labels),
        "undecided": len(labels) - len(decided),
        "correct_absence_rate": wilson_interval(correct, len(decided), rc.confidence_level),
        "false_absence_rate": wilson_interval(spoke, len(decided), rc.confidence_level),
    }


def speaker_stratum(
    units: list[Record],
    items: list[tuple[str, Record, Record]],
    population: int,
    config: ValidationConfig,
    rc: ReportConfig,
) -> Record:
    order = config.questions[SPEAKER_QUESTION].judgments
    unit_labels = [unit["judgment"] for unit in units]
    person_labels = [judged["judgment"] for _, _, judged in items]
    decided = [label for label in unit_labels if label not in rc.speaker_undecided]
    correct = sum(1 for label in decided if label in rc.speaker_correct)
    tiers = sorted({unit["tier"] for unit in units})
    return {
        "question": SPEAKER_QUESTION,
        "population": population,
        "judged_udvs": len(units),
        "judged_persons": len(items),
        "judgments_udvs": label_counts(unit_labels, order),
        "judgments_persons": label_counts(person_labels, order),
        "udvs": speaker_rates(unit_labels, rc),
        "persons": speaker_rates(person_labels, rc),
        "by_tier": {
            tier: {
                "udvs": speaker_rates([u["judgment"] for u in units if u["tier"] == tier], rc),
                "persons": speaker_rates(
                    [judged["judgment"] for _, item, judged in items if item["tier"] == tier], rc
                ),
            }
            for tier in tiers
        },
        "udv_ids_by_judgment": {
            label: sorted(unit["udv_id"] for unit in units if unit["judgment"] == label)
            for label in order
        },
        "clustering_note": (
            "one answer per participant is copied to each of that participant's drawn UDVs, so "
            "UDV-level intervals treat correlated answers as independent; the person-level "
            "intervals use the independent unit"
        ),
        "estimator_inputs": {
            metric: {"successes": correct, "trials": len(decided)} for metric in METRICS
        },
    }


def stratum_results(
    key: Record,
    units: list[Record],
    judged: dict[str, Record],
    config: ValidationConfig,
    rc: ReportConfig,
) -> dict[str, Record]:
    population = key["population"]["sampled_splits"]
    results: dict[str, Record] = {}
    for stratum in key["strata"]:
        name = stratum["name"]
        members = [unit for unit in units if unit["stratum"] == name]
        if stratum["question"] == SUPPORT_QUESTION:
            results[name] = evidence_stratum(members, population[name], config, rc)
        else:
            items = [
                (item_id, item, judged[item_id])
                for item_id, item in key["items"].items()
                if item["stratum"] == name
            ]
            results[name] = speaker_stratum(members, items, population[name], config, rc)
        results[name]["sample"] = key["sample"][name]
    return results


def stratified_estimate(inputs: list[Record], level: float, fpc: bool) -> Record:
    active = [entry for entry in inputs if entry["population"] > 0]
    population = sum(entry["population"] for entry in active)
    empty = [entry["stratum"] for entry in active if entry["trials"] == 0]
    if empty:
        return {
            "population": population,
            "estimated_correct": None,
            "reason": f"strata with UDVs in the population but no decided judgment: {empty}",
        }
    total = 0.0
    variance = 0.0
    undefined: list[str] = []
    zero_not_census: list[str] = []
    strata = []
    for entry in active:
        size, trials, successes = entry["population"], entry["trials"], entry["successes"]
        if trials > size:
            raise SystemExit(f"{entry['stratum']}: {trials} judged UDVs exceed population {size}")
        share = successes / trials
        correction = 1 - trials / size if fpc else 1.0
        if trials == 1 and correction > 0:
            undefined.append(entry["stratum"])
            contribution = None
        else:
            contribution = (
                0.0 if trials == 1 else size**2 * correction * share * (1 - share) / (trials - 1)
            )
            variance += contribution
            if contribution == 0 and correction > 0:
                zero_not_census.append(entry["stratum"])
        total += size * share
        strata.append(
            {
                **entry,
                "estimate": share,
                "estimated_correct": size * share,
                "finite_population_correction": correction,
                "variance_contribution": contribution,
            }
        )
    z = float(norm.ppf(0.5 + level / 2))
    result: Record = {
        "population": population,
        "estimated_correct": total,
        "proportion": total / population,
        "finite_population_correction": fpc,
        "z": z,
        "strata": strata,
        "undefined_variance_strata": undefined,
        "zero_variance_strata_not_census": zero_not_census,
    }
    if undefined:
        result.update({"standard_error_total": None, "interval_total": None})
        result["interval_proportion"] = None
        return result
    error = math.sqrt(variance)
    low, high = max(0.0, total - z * error), min(float(population), total + z * error)
    result.update(
        {
            "standard_error_total": error,
            "interval_total": [low, high],
            "interval_proportion": [low / population, high / population],
        }
    )
    return result


def estimator_inputs(
    results: dict[str, Record], populations: dict[str, int], metric: str, names: list[str]
) -> list[Record]:
    return [
        {
            "stratum": name,
            "population": populations[name],
            **results[name]["estimator_inputs"][metric],
        }
        for name in names
    ]


def stratified_section(key: Record, results: dict[str, Record], level: float) -> Record:
    evidence = [s["name"] for s in key["strata"] if s["question"] == SUPPORT_QUESTION]
    scopes = {"all_udvs": [s["name"] for s in key["strata"]], "udvs_with_evidence": evidence}
    sampled = key["population"]["sampled_splits"]
    section: Record = {
        "sampled_splits": {
            scope: {
                metric: stratified_estimate(
                    estimator_inputs(results, sampled, metric, names), level, True
                )
                for metric in METRICS
            }
            for scope, names in scopes.items()
        },
        "interval": (
            "normal approximation of the stratified total, variance sum_h N_h^2 (1 - n_h / N_h) "
            "p_h (1 - p_h) / (n_h - 1); a stratum with p_h of 0 or 1 contributes no variance, "
            "which understates the uncertainty unless the stratum is a census"
        ),
    }
    all_splits = key["population"]["all_splits"]
    if all_splits is None:
        section["projection_all_splits"] = None
        section["projection_note"] = key["population"]["all_splits_note"]
        return section
    section["projection_all_splits"] = {
        scope: {
            metric: stratified_estimate(
                estimator_inputs(results, all_splits, metric, names), level, False
            )
            for metric in METRICS
        }
        for scope, names in scopes.items()
    }
    section["projection_note"] = (
        "per-stratum precision measured on the sampled splits applied to the stratum sizes of the "
        "whole run, without finite population correction; this assumes the other splits have the "
        "same per-stratum precision, which the sample does not test"
    )
    return section


def evaluate_criteria(
    rules: list[Record], results: dict[str, Record], level: float, dry_run: bool
) -> list[Record]:
    evaluated = []
    for rule in rules:
        interval = results[rule["stratum"]][rule["metric"]]
        trials = interval["trials"]
        needed = min_successes_to_pass(trials, rule["min_wilson_lower"], level)
        if trials == 0:
            status = "NOT_EVALUABLE"
        else:
            status = "PASS" if interval["low"] >= rule["min_wilson_lower"] else "FAIL"
        evaluated.append(
            {
                **rule,
                "status": status,
                "valid_for_decision": not dry_run,
                "observed": interval,
                "min_successes_to_pass_at_this_n": needed,
            }
        )
    return evaluated


def kappa(first: list[Any], second: list[Any], labels: list[Any]) -> float | None:
    if not first:
        return None
    count_first, count_second = Counter(first), Counter(second)
    expected = sum(count_first[label] * count_second[label] for label in labels) / len(first) ** 2
    if expected >= 1 - 1e-12:
        return None
    return float(cohen_kappa_score(first, second, labels=labels))


def binary_agreement(first: list[bool], second: list[bool], level: float) -> Record:
    same = sum(1 for a, b in zip(first, second, strict=True) if a == b)
    return {
        "raw_agreement": wilson_interval(same, len(first), level),
        "cohen_kappa": kappa(first, second, [True, False]),
    }


def question_agreement(
    pairs: list[Record], question: str, config: ValidationConfig, rc: ReportConfig
) -> Record:
    labels = list(config.questions[question].judgments)
    first = [pair["first"] for pair in pairs]
    second = [pair["second"] for pair in pairs]
    same = sum(1 for a, b in zip(first, second, strict=True) if a == b)
    section: Record = {
        "pairs": len(pairs),
        "raw_agreement": wilson_interval(same, len(pairs), rc.confidence_level),
        "cohen_kappa": kappa(first, second, labels),
        "confusion": {
            "labels": labels,
            "rows_first_pass_columns_second_pass": confusion_matrix(
                first, second, labels=labels
            ).tolist(),
        },
        "disagreements": [
            {key: pair[key] for key in ("item_id", "original_item_id", "first", "second")}
            for pair in pairs
            if pair["first"] != pair["second"]
        ],
    }
    if question == SUPPORT_QUESTION:
        for name, successes in (("strict", rc.strict_success), ("tolerant", rc.tolerant_success)):
            section[f"{name}_binary"] = binary_agreement(
                [label in successes for label in first],
                [label in successes for label in second],
                rc.confidence_level,
            )
        both = [p for p in pairs if p["first_better"] and p["second_better"]]
        section["better_passage_raw_agreement"] = wilson_interval(
            sum(1 for p in both if p["first_better"] == p["second_better"]),
            len(both),
            rc.confidence_level,
        )
    return section


def agreement_section(
    key: Record,
    first: dict[str, Record],
    repeat_key: Record,
    second: dict[str, Record],
    annotation_sha256: str,
    config: ValidationConfig,
    rc: ReportConfig,
) -> Record:
    pairs = []
    for item_id, item in repeat_key["items"].items():
        original = item["original_item_id"]
        pairs.append(
            {
                "item_id": item_id,
                "original_item_id": original,
                "question": item["question"],
                "stratum": key["items"][original]["stratum"],
                "first": first[original]["judgment"],
                "second": second[item_id]["judgment"],
                "first_better": first[original]["better_passage"],
                "second_better": second[item_id]["better_passage"],
            }
        )
    questions = sorted({pair["question"] for pair in pairs})
    return {
        "status": "computed",
        "pairs": len(pairs),
        "by_stratum": dict(Counter(pair["stratum"] for pair in pairs)),
        "by_question": {
            question: question_agreement(
                [pair for pair in pairs if pair["question"] == question], question, config, rc
            )
            for question in questions
        },
        "first_pass_sha256_when_repeat_created": repeat_key["first_pass"]["sha256"],
        "first_pass_sha256_now": annotation_sha256,
        "first_pass_changed_after_repeat_created": (
            repeat_key["first_pass"]["sha256"] != annotation_sha256
        ),
        "hours_between_first_pass_and_repeat_sheet": repeat_key["first_pass"][
            "hours_since_modification"
        ],
        "repeat_created_at": repeat_key["created_at"],
        "kappa_note": (
            "Cohen's kappa is None when expected agreement is 1 (both passes used a single label); "
            "with about 20 pairs its sampling error is large"
        ),
    }


def speaker_absence_summary(results: dict[str, Record], key: Record) -> Record:
    summary: Record = {}
    for stratum in key["strata"]:
        if stratum["question"] != SPEAKER_QUESTION:
            continue
        for tier, rates in results[stratum["name"]]["by_tier"].items():
            summary[tier] = {
                "persons": rates["persons"]["false_absence_rate"],
                "udvs": rates["udvs"]["false_absence_rate"],
                "undecided_persons": rates["persons"]["undecided"],
            }
    return summary


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return rounded(value)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def interim_section(
    sample_dir: Path, rc: ReportConfig, repeat_key: Record | None, writing: str
) -> Record:
    interim = [
        entry
        for entry in existing_precision_reports(sample_dir)
        if entry["file"].startswith(rc.interim_output_prefix) and entry["file"] != writing
    ]
    repeat_created = repeat_key["created_at"] if repeat_key is not None else None
    for entry in interim:
        entry["before_repeat_sheet"] = (
            None
            if repeat_created is None or entry["created_at"] is None
            else entry["created_at"] < repeat_created
        )
    recorded = (
        repeat_key.get("precision_reports_before_repeat_sheet") if repeat_key is not None else None
    )
    seen_before = bool(recorded) or any(entry["before_repeat_sheet"] for entry in interim)
    return {
        "interim_reports_found": interim,
        "reports_recorded_when_repeat_sheet_was_created": recorded,
        "precision_seen_before_repeat_sheet": seen_before if repeat_key is not None else None,
        "rule": (
            "--without-reannotation writes an interim report under its own timestamped name, "
            "never overwritten; the repeat stage records every precision report present when the "
            "repeat sheet is created, and this section lists both, so an early look at the numbers "
            "stays visible in the final report"
        ),
    }


def integrity_section(key: Record, sample_dir: Path, config: ValidationConfig) -> Record:
    run_path = Path(key["run"]["path"])
    run_now = sha256_of_file(run_path) if run_path.exists() else None
    frozen = key["criteria_sha256"]
    current = canonical_sha256(config.source["criteria"])
    return {
        "annotation_key_sha256": sha256_of_file(sample_dir / ANNOTATION_KEY),
        "annotation_csv_sha256": sha256_of_file(sample_dir / ANNOTATION_CSV),
        "annotation_csv_sha256_at_creation": key["csv"]["sha256_at_creation"],
        "run_path": str(run_path),
        "run_sha256_at_sampling": key["run"]["sha256"],
        "run_sha256_now": run_now,
        "run_changed_since_sampling": run_now != key["run"]["sha256"],
        "criteria_sha256_frozen_in_key": frozen,
        "criteria_sha256_in_config": current,
        "criteria_changed_since_sampling": frozen != current,
    }


def print_summary(report: Record) -> None:
    print(
        f"{report['sample_name']} ({', '.join(report['splits_used'])}, dry_run={report['dry_run']})"
    )
    for name, result in report["strata"].items():
        if result["question"] == SUPPORT_QUESTION:
            strict, tolerant = result["strict_precision"], result["tolerant_precision"]
            print(
                f"  {name:26s} n={result['judged_udvs']:3d}/{result['population']:4d} "
                f"strict={strict['estimate']} [{strict['low']}, {strict['high']}] "
                f"tolerant={tolerant['estimate']} [{tolerant['low']}, {tolerant['high']}]"
            )
        else:
            rate = result["persons"]["false_absence_rate"]
            print(
                f"  {name:26s} persons={result['judged_persons']:3d} "
                f"udvs={result['judged_udvs']:3d} "
                f"false_absence(persons)={rate['estimate']} [{rate['low']}, {rate['high']}]"
            )
    for metric in METRICS:
        estimate = report["stratified_estimate"]["sampled_splits"]["all_udvs"][metric]
        print(
            f"  stratified {metric}: {estimate.get('estimated_correct')} of "
            f"{estimate['population']} UDVs, interval {estimate.get('interval_total')}"
        )
    for rule in report["criteria"]["rules"]:
        print(
            f"  criterion {rule['name']}: {rule['status']} (lower {rule['observed']['low']} vs "
            f"{rule['min_wilson_lower']}, needs {rule['min_successes_to_pass_at_this_n']} of "
            f"{rule['observed']['trials']})"
        )
    agreement = report["intra_annotator_agreement"]
    if agreement["status"] == "computed":
        for question, section in agreement["by_question"].items():
            print(
                f"  agreement {question}: raw={section['raw_agreement']['estimate']} "
                f"kappa={section['cohen_kappa']} pairs={section['pairs']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute precision per stratum, the stratified estimate of correct UDVs, the "
        "pre-declared criteria and intra-annotator agreement from the filled validation sheets."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/validation_sample.toml"))
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument(
        "--final-test", action="store_true", help="required when the sample holds test hearings"
    )
    parser.add_argument(
        "--without-reannotation",
        action="store_true",
        help="report before the repeat sheet exists (its agreement section is then not computed)",
    )
    return parser.parse_args()


def load_judgments(
    key: Record, sample_dir: Path, config: ValidationConfig, without_reannotation: bool
) -> tuple[dict[str, Record], tuple[Record, dict[str, Record]] | None]:
    first, first_problems = validate_annotation(
        read_annotation_csv(sample_dir / ANNOTATION_CSV, config), key, config
    )
    problems = [f"{ANNOTATION_CSV}: {problem}" for problem in first_problems]
    repeat: tuple[Record, dict[str, Record]] | None = None
    if (sample_dir / REANNOTATION_KEY).exists():
        repeat_key = load_key(sample_dir / REANNOTATION_KEY, "reannotation")
        if repeat_key["sample_name"] != key["sample_name"]:
            problems.append(f"{REANNOTATION_KEY} belongs to another sample")
        second, second_problems = validate_annotation(
            read_annotation_csv(sample_dir / REANNOTATION_CSV, config), repeat_key, config
        )
        problems += [f"{REANNOTATION_CSV}: {problem}" for problem in second_problems]
        repeat = (repeat_key, second)
    elif not without_reannotation:
        problems.append(
            f"{REANNOTATION_KEY} does not exist: create and annotate the repeat sheet "
            "(generate_validation_sample --stage repeat) before any number is computed, or pass "
            "--without-reannotation"
        )
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} validation problems; no report written")
    return first, repeat


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_validation_config(args.config)
    rc = load_report_config(config)
    key = load_key(args.sample_dir / ANNOTATION_KEY, "annotation")
    require_final_test(key, args.final_test)
    rules = key["criteria"]["rules"]
    check_rules(rules, {stratum["name"]: stratum["question"] for stratum in key["strata"]})
    first, repeat = load_judgments(key, args.sample_dir, config, args.without_reannotation)
    units = judged_units(key, first)
    results = stratum_results(key, units, first, config, rc)
    integrity = integrity_section(key, args.sample_dir, config)
    interim = repeat is None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_name = f"{rc.interim_output_prefix}_{stamp}.json" if interim else rc.output_name
    output_path = args.sample_dir / output_name
    if interim and output_path.exists():
        raise SystemExit(f"{output_path} exists; interim reports are never overwritten")
    integrity["interim_reports"] = interim_section(
        args.sample_dir, rc, repeat[0] if repeat is not None else None, output_name
    )
    if repeat is None:
        agreement: Record = {"status": "not_computed", "reason": "--without-reannotation"}
    else:
        agreement = agreement_section(
            key, first, repeat[0], repeat[1], integrity["annotation_csv_sha256"], config, rc
        )
        integrity["reannotation_csv_sha256"] = sha256_of_file(args.sample_dir / REANNOTATION_CSV)
        integrity["reannotation_key_sha256"] = sha256_of_file(args.sample_dir / REANNOTATION_KEY)
    report = {
        "sample_name": key["sample_name"],
        "created_at": now_iso(),
        "interim": interim,
        "interim_note": (
            "computed with --without-reannotation before the repeat sheet was annotated; written "
            "under its own name and never overwritten"
            if interim
            else None
        ),
        "dry_run": key["dry_run"],
        "final_test": args.final_test,
        "splits_used": key["splits_used"],
        "splits_declared": key["splits_declared"],
        "sample_matches_declared_design": (
            key["splits_used"] == key["splits_declared"] and not key["dry_run"]
        ),
        "run": key["run"],
        "label_semantics": LABEL_SEMANTICS,
        "counts": {
            "rows": len(first),
            "udvs": len(units),
            "rows_by_stratum": dict(Counter(item["stratum"] for item in key["items"].values())),
        },
        "strata": results,
        "false_absence_by_tier": speaker_absence_summary(results, key),
        "stratified_estimate": stratified_section(key, results, rc.confidence_level),
        "criteria": {
            "declared_on": key["criteria"]["declared_on"],
            "declaration": key["criteria"]["declaration"],
            "source": "frozen copy in annotation_key.json",
            "confidence_level": rc.confidence_level,
            "rules": evaluate_criteria(rules, results, rc.confidence_level, key["dry_run"]),
        },
        "intra_annotator_agreement": agreement,
        "integrity": integrity,
        "code": {
            **code_hashes(),
            "utils/precision_report.py": sha256_of_file(Path(__file__)),
        },
        "timing": {"elapsed_seconds": round(time.perf_counter() - started, 2)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "config": config.source,
    }
    report = round_floats(report)
    write_json(report, output_path)
    print_summary(report)
    print(f"report -> {output_path}")


if __name__ == "__main__":
    main()
