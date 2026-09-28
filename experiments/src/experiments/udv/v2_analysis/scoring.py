"""Subcommand ``score-annotation``: udv_v1 and udv_v2 precision from the filled sheets."""

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookworm import sha256_of_file, write_json

from experiments.common.provenance import code_section
from experiments.common.reporting import file_record, utc_timestamp
from experiments.udv.v2_analysis.paths import PACKAGE_DIR, PRECISION_OUTPUT_PATTERN
from experiments.udv.v2_analysis.plan import inherited_items, load_plan, plan_items
from experiments.udv.v2_analysis.printing import print_score_summary
from experiments.udv.v2_analysis.relations import RELATIONS, load_inheritance
from experiments.udv.v2_analysis.supplement import load_supplement
from experiments.validation import precision_report as precision
from experiments.validation.generate_sample import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    ValidationConfig,
    load_key,
    load_validation_config,
    read_annotation_csv,
    require_final_test,
    validate_annotation,
    wilson_interval,
)
from experiments.validation.precision_report import ReportConfig

Record = dict[str, Any]
ItemSources = dict[str, tuple[str, str]]

NO_JUDGMENTS = "no judgments yet"
V1_SHEET = "udv_v1 sheet"
SUPPLEMENT_SHEET = "supplementary sheet"
COMBINED_RULE = (
    "udv_v2 precision per stratum from every item of the udv_v1 sample: an item whose udv_v2 "
    "evidence is same, superset or no_evidence (inheritance section of the supplementary key) "
    "keeps its judgment from the udv_v1 sheet, and a moved item counts only with its judgment "
    "from the supplementary sheet; the stratum of an item is the stratum of its udv_v2 records, "
    "and the population is the udv_v2 count of the sampled splits"
)
COMBINED_CAVEATS = [
    "the items were drawn from the udv_v1 strata; an item that changed stratum in udv_v2 keeps "
    "the inclusion probability of its udv_v1 stratum, so the udv_v2 strata are not simple random "
    "samples of the udv_v2 populations",
    "udv_v2 UDVs of a stratum whose udv_v1 stratum was sampled at another rate, or not drawn, are "
    "represented only through the items that were drawn",
    "the supplementary rows show opinions the annotator already judged with their udv_v1 passage, "
    "so their judgments are not independent of the first sheet",
    "a superset item keeps a udv_v1 label given to a shorter passage; the inheritance section "
    "states the assumption this rests on",
]
COMBINED_INHERITANCE_FIELDS = (
    "inherit_relations",
    "relation_rules",
    "inheritance_rule",
    "justification",
    "consequence",
    "assumption",
)
INTERIM_NOTE = (
    "INTERIM: computed on the judged rows only; the criteria statuses below are not "
    "a decision and change as the sheet is filled"
)
INVALID_LABELS_RULE = (
    "a filled julgamento outside the labels of its question is not counted and is listed "
    "here; the report is interim while any row is empty or invalid"
)
UNITS_RULE = "precision per UDV; the judgment of an item counts for each of its UDVs"
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def tolerated_problems(item_id: str, raw_label: str) -> set[str]:
    return {
        f"{item_id}: julgamento is empty",
        f"{item_id}: existe_trecho_melhor is empty",
        f"{item_id}: julgamento {raw_label!r} is not one of",
    }


def is_tolerated(problem: str, tolerated: dict[str, set[str]]) -> bool:
    item_id = problem.split(":", 1)[0]
    return any(problem.startswith(prefix) for prefix in tolerated.get(item_id, set()))


def partial_judgments(rows: list[Record], key: Record, config: ValidationConfig) -> Record:
    """The valid judgments of a partly filled sheet; any problem but a missing label is fatal."""
    judged, problems = validate_annotation(rows, key, config)
    raw = {row.get("item_id", "").strip(): row.get("julgamento", "") for row in rows}
    tolerated = {item_id: tolerated_problems(item_id, raw.get(item_id, "")) for item_id in judged}
    fatal = [problem for problem in problems if not is_tolerated(problem, tolerated)]
    if fatal:
        raise SystemExit("\n".join(fatal))
    valid: dict[str, Record] = {}
    invalid: list[Record] = []
    for item_id, entry in sorted(judged.items()):
        if not entry["judgment"]:
            continue
        allowed = config.questions[key["items"][item_id]["question"]].judgments
        if entry["judgment"] in allowed:
            valid[item_id] = entry
        else:
            invalid.append({"item_id": item_id, "julgamento": raw[item_id].strip()})
    return {
        "valid": valid,
        "invalid_labels": invalid,
        "better_passage_missing": sorted(
            item_id for item_id, entry in valid.items() if entry["better_passage"] is None
        ),
    }


def subset_key(key: Record, items: dict[str, Record], population: dict[str, int]) -> Record:
    return {
        **key,
        "items": items,
        "population": {**key["population"], "sampled_splits": population},
    }


def run_results(
    key: Record,
    items: dict[str, Record],
    judged: dict[str, Record],
    population: dict[str, int],
    sample_items: dict[str, int],
    config: ValidationConfig,
    rc: ReportConfig,
    decision_valid: bool,
) -> Record:
    """Precision per stratum and the criteria for one run, from the judged items given."""
    run_key = subset_key(key, items, population)
    run_judged = {item_id: judged[item_id] for item_id in items}
    units = precision.judged_units(run_key, run_judged)
    results = precision.stratum_results(run_key, units, run_judged, config, rc)
    for name, result in results.items():
        result["judged_items"] = sum(1 for item in items.values() if item["stratum"] == name)
        result["sample_items"] = sample_items.get(name, 0)
        result["judged_of_sample"] = f"{result['judged_items']} of {result['sample_items']}"
    rules = precision.evaluate_criteria(
        key["criteria"]["rules"], results, rc.confidence_level, not decision_valid
    )
    if not decision_valid:
        rules = [{**rule, "status": f"INTERIM_{rule['status']}"} for rule in rules]
    return {"strata": precision.round_floats(results), "criteria": precision.round_floats(rules)}


def support_precision(
    items: dict[str, Record], judged: dict[str, Record], field: str, rc: ReportConfig
) -> Record:
    groups: dict[str, list[str]] = {}
    for item_id, item in items.items():
        if item["question"] != precision.SUPPORT_QUESTION:
            continue
        for _ in item["udv_ids"]:
            groups.setdefault(str(item[field]), []).append(judged[item_id]["judgment"])
    result: Record = {}
    for name, labels in sorted(groups.items()):
        strict = sum(1 for label in labels if label in rc.strict_success)
        tolerant = sum(1 for label in labels if label in rc.tolerant_success)
        result[name] = {
            "judged_udvs": len(labels),
            "strict_precision": wilson_interval(strict, len(labels), rc.confidence_level),
            "tolerant_precision": wilson_interval(tolerant, len(labels), rc.confidence_level),
        }
    return precision.round_floats(result)


def item_sources(inherited: dict[str, Record], supplement: Record) -> ItemSources:
    """(relation, sheet the label comes from) of every item of the combined udv_v2 sample."""
    sources = {
        item_id: (supplement["inherited_items"][item_id]["relation"], V1_SHEET)
        for item_id in inherited
    }
    for item_id, item in supplement["items"].items():
        sources[item_id] = (item["v1_relation"], SUPPLEMENT_SHEET)
    return sources


def source_counts(
    items: dict[str, Record],
    judged_items: dict[str, Record],
    sources: ItemSources,
    names: list[str],
) -> tuple[Record, Record]:
    totals: Record = {}
    for relation, sheet in sorted(set(sources.values()), key=lambda pair: RELATIONS.index(pair[0])):
        members = [item_id for item_id in items if sources[item_id] == (relation, sheet)]
        totals[relation] = {
            "label_from": sheet,
            "items": len(members),
            "judged": sum(1 for item_id in members if item_id in judged_items),
        }
    by_stratum = {
        name: {
            relation: sum(
                1
                for item_id, item in items.items()
                if item["stratum"] == name and sources[item_id][0] == relation
            )
            for relation in totals
        }
        for name in names
    }
    return totals, by_stratum


def precision_by_stratum_and_tier(
    items: dict[str, Record], judged: dict[str, Record], rc: ReportConfig
) -> Record:
    return {
        "by_stratum": support_precision(items, judged, "stratum", rc),
        "by_tier": support_precision(items, judged, "tier", rc),
    }


def sensitivity_section(
    judged_items: dict[str, Record],
    judged: dict[str, Record],
    sources: ItemSources,
    inheritance: Record,
    rc: ReportConfig,
) -> Record:
    """udv_v2 precision again with the superset items whose udv_v1 label is not correta left out."""
    superset = ("superset", V1_SHEET)
    left_out = sorted(
        item_id
        for item_id in judged_items
        if sources[item_id] == superset and judged[item_id]["judgment"] not in rc.strict_success
    )
    kept = {item_id: item for item_id, item in judged_items.items() if item_id not in left_out}
    return {
        "rule": inheritance["sensitivity_rule"],
        "superset_judged": sum(1 for item_id in judged_items if sources[item_id] == superset),
        "superset_left_out": len(left_out),
        "superset_left_out_items": left_out,
        "superset_inherited": precision_by_stratum_and_tier(judged_items, judged, rc),
        "superset_only_if_v1_correta": precision_by_stratum_and_tier(kept, judged, rc),
    }


def combined_results(
    key: Record,
    plan: Record,
    v1_valid: dict[str, Record],
    supplement: Record,
    supplement_valid: dict[str, Record],
    config: ValidationConfig,
    rc: ReportConfig,
) -> Record:
    """udv_v2 precision from the inherited udv_v1 labels and the supplementary sheet together."""
    inherited = plan_items(key, plan, set(supplement["inherited_items"]))
    items = {**inherited, **supplement["items"]}
    judged = {
        **{item_id: v1_valid[item_id] for item_id in inherited if item_id in v1_valid},
        **supplement_valid,
    }
    judged_items = {item_id: item for item_id, item in items.items() if item_id in judged}
    names = [stratum["name"] for stratum in key["strata"]]
    population = {name: plan["v2_population_sampled_splits"].get(name, 0) for name in names}
    sample_items = dict(Counter(str(item["stratum"]) for item in items.values()))
    sources = item_sources(inherited, supplement)
    totals, by_stratum = source_counts(items, judged_items, sources, names)
    complete = len(judged_items) == len(items)
    decision_valid = complete and not key["dry_run"] and not supplement["dry_run"]
    inheritance = supplement["inheritance"]
    return {
        "status": "final" if complete else "interim",
        "rule": COMBINED_RULE,
        "inheritance": {name: inheritance[name] for name in COMBINED_INHERITANCE_FIELDS},
        "caveats": COMBINED_CAVEATS,
        "items_in_sample": len(items),
        "judged": f"{len(judged_items)} of {len(items)}",
        "judged_inherited": sum(1 for item_id in judged_items if item_id in inherited),
        "judged_reannotated": sum(1 for item_id in judged_items if item_id in supplement["items"]),
        "sources": totals,
        "sources_by_stratum": by_stratum,
        "by_tier": support_precision(judged_items, judged, "tier", rc),
        "sensitivity": sensitivity_section(judged_items, judged, sources, inheritance, rc),
        **run_results(
            {**key, "sample": by_stratum},
            judged_items,
            judged,
            population,
            sample_items,
            config,
            rc,
            decision_valid,
        ),
    }


def supplement_results(
    args: argparse.Namespace,
    key: Record,
    plan: Record,
    valid: dict[str, Record],
    config: ValidationConfig,
    rc: ReportConfig,
) -> tuple[Record, Record]:
    sample_dir = Path(args.sample_dir)
    supplement_dir = Path(args.supplement_dir)
    supplement = load_supplement(
        supplement_dir,
        sample_dir / ANNOTATION_KEY,
        plan,
        Path(args.plan),
        load_inheritance(config),
    )
    supplement_path = (
        Path(args.supplement_annotation)
        if args.supplement_annotation
        else supplement_dir / ANNOTATION_CSV
    )
    judgments = partial_judgments(read_annotation_csv(supplement_path, config), supplement, config)
    combined = combined_results(key, plan, valid, supplement, judgments["valid"], config, rc)
    combined["invalid_labels"] = judgments["invalid_labels"]
    source = {
        "annotation_csv": file_record(supplement_path),
        "annotation_key": file_record(supplement_dir / ANNOTATION_KEY),
    }
    return combined, source


def unchanged_v2_results(
    key: Record,
    plan: Record,
    valid: dict[str, Record],
    config: ValidationConfig,
    rc: ReportConfig,
    decision_valid: bool,
) -> Record:
    """udv_v2 precision from the items whose evidence did not change, labels inherited as is."""
    names = [stratum["name"] for stratum in key["strata"]]
    v2_all, v2_sample = inherited_items(key, plan)
    v2_items = {item_id: item for item_id, item in v2_all.items() if item_id in valid}
    v2_population = {name: plan["v2_population_sampled_splits"].get(name, 0) for name in names}
    return {
        "rule": plan["rule"],
        "caveat": plan["caveat"],
        "items_in_sample": len(v2_all),
        **run_results(key, v2_items, valid, v2_population, v2_sample, config, rc, decision_valid),
    }


def load_sheet(
    args: argparse.Namespace, config: ValidationConfig
) -> tuple[Path, Record, Record, Record]:
    """The sheet path, key, plan and judgments of the udv_v1 sample, all checked."""
    sample_dir = Path(args.sample_dir)
    annotation_path = Path(args.annotation) if args.annotation else sample_dir / ANNOTATION_CSV
    key = load_key(sample_dir / ANNOTATION_KEY, "annotation")
    require_final_test(key, args.final_test)
    precision.check_rules(
        key["criteria"]["rules"],
        {stratum["name"]: stratum["question"] for stratum in key["strata"]},
    )
    plan = load_plan(Path(args.plan), sample_dir / ANNOTATION_KEY)
    judgments = partial_judgments(read_annotation_csv(annotation_path, config), key, config)
    if not judgments["valid"]:
        raise SystemExit(
            f"{NO_JUDGMENTS}: {annotation_path} has no valid julgamento in any of its "
            f"{len(key['items'])} rows; fill the sheet and run this command again"
        )
    return annotation_path, key, plan, judgments


def output_path(args: argparse.Namespace, status: str) -> Path:
    if args.output:
        return Path(args.output)
    stamp = datetime.now(UTC).strftime(STAMP_FORMAT)
    return Path(PRECISION_OUTPUT_PATTERN.format(status=status, stamp=stamp))


def command_score(args: argparse.Namespace) -> None:
    config = load_validation_config(args.config)
    rc = precision.load_report_config(config)
    annotation_path, key, plan, judgments = load_sheet(args, config)
    valid = judgments["valid"]
    v1_complete = len(valid) == len(key["items"])
    decision_valid = v1_complete and not key["dry_run"]
    combined: tuple[Record, Record] | None = None
    if args.supplement_dir:
        combined = supplement_results(args, key, plan, valid, config, rc)
    complete = v1_complete and (combined is None or combined[0]["status"] == "final")
    v1_items = {item_id: key["items"][item_id] for item_id in valid}
    v1_sample = dict(Counter(item["stratum"] for item in key["items"].values()))
    status = "final" if complete else "interim"
    report = {
        "created_at": utc_timestamp(),
        "status": status,
        "status_note": "every row of the sheet has a valid julgamento"
        if complete
        else INTERIM_NOTE,
        "sample_name": key["sample_name"],
        "splits_used": key["splits_used"],
        "judged": f"{len(valid)} of {len(key['items'])}",
        "judged_items": len(valid),
        "items": len(key["items"]),
        "invalid_labels": judgments["invalid_labels"],
        "invalid_labels_rule": INVALID_LABELS_RULE,
        "better_passage_missing": len(judgments["better_passage_missing"]),
        "annotation_csv": file_record(annotation_path),
        "annotation_key": file_record(Path(args.sample_dir) / ANNOTATION_KEY),
        "plan": {"path": str(args.plan), "sha256": sha256_of_file(Path(args.plan))},
        "confidence_level": rc.confidence_level,
        "units": UNITS_RULE,
        "label_semantics": precision.LABEL_SEMANTICS,
        "criteria_declaration": key["criteria"]["declaration"],
        "criteria_declared_on": key["criteria"]["declared_on"],
        "udv_v1": run_results(
            key,
            v1_items,
            valid,
            key["population"]["sampled_splits"],
            v1_sample,
            config,
            rc,
            decision_valid,
        ),
        "udv_v2_unchanged": unchanged_v2_results(key, plan, valid, config, rc, decision_valid),
        "code": code_section(PACKAGE_DIR, precision),
    }
    if combined is not None:
        report["udv_v2"], report["supplement"] = combined
    output = output_path(args, status)
    if output.exists():
        raise SystemExit(f"{output} exists; precision reports are never overwritten")
    write_json(report, output)
    print_score_summary(report)
    print(f"report -> {output}")
