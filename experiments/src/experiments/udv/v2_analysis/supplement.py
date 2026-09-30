"""Subcommand ``supplement-sheet``: the blind udv_v2 sheet of the sampled items that moved."""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_gated_jsonl, load_jsonl, sha256_of_file, write_json

from experiments.common.provenance import code_section
from experiments.common.reporting import file_record, utc_timestamp
from experiments.udv.v2_analysis.paths import PACKAGE_DIR
from experiments.udv.v2_analysis.plan import key_strata, load_plan
from experiments.udv.v2_analysis.relations import (
    IDENTICAL_RELATIONS,
    RELATIONS,
    is_inherited,
    item_relations,
    load_inheritance,
)
from experiments.validation.generate_sample import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    Stratum,
    ValidationConfig,
    assign_item_ids,
    blinding_section,
    canonical_sha256,
    ensure_new_dir,
    evidence_item,
    hearing_view,
    load_key,
    load_validation_config,
    require_final_test,
    sheet_rows,
    write_annotation_csv,
    write_transcripts,
)
from experiments.validation.generate_sample.sheets import csv_section
from experiments.validation.precision_report import SUPPORT_QUESTION

Record = dict[str, Any]
Reannotated = tuple[str, Record, str, str]

SUPPLEMENT_NAME = "human_validation_v1_udv_v2_supplement"
SUPPLEMENT_KIND = "udv_v2_supplement"
SUPPLEMENT_ITEM_PREFIX = "C"
SUPPLEMENT_ORDER_STREAM = 4
SUPPLEMENT_RULE = (
    "one row per item of the udv_v1 sample whose udv_v2 evidence is moved under the relation "
    "rules of the inheritance section: the same sampled opinion, shown with its udv_v2 evidence, "
    "context and link; the rows are put in a new random order and get new item ids, and the "
    "udv_v1 item id and relation are kept in the key only"
)
GENERATOR = "numpy PCG64"


def reannotated_records(
    plan: Record, relations: dict[str, Record], v2: dict[str, Record]
) -> list[Reannotated]:
    """(udv_v1 item id, udv_v2 record, udv_v2 stratum, relation) of each item to judge again."""
    entries: list[Reannotated] = []
    for item_id, entry in sorted(plan["item_plan"].items()):
        if relations[item_id]["inherited"]:
            continue
        if entry["question"] != SUPPORT_QUESTION or len(entry["udv_ids"]) != 1:
            raise SystemExit(f"{item_id}: only single-UDV evidence items can be reannotated")
        record = v2[entry["udv_ids"][0]]
        if record.get("evidence") is None or entry["v2_stratum"] is None:
            raise SystemExit(f"{item_id}: the udv_v2 record has no evidence stratum")
        entries.append((item_id, record, str(entry["v2_stratum"]), relations[item_id]["relation"]))
    return entries


def supplement_items(
    entries: list[Reannotated],
    views: dict[int, Record],
    strata: dict[str, Stratum],
    chars: int,
    rng: np.random.Generator,
) -> dict[str, Record]:
    items = [
        {
            **evidence_item(record, views[record["hearing_id"]], strata[name], chars),
            "v1_item_id": item_id,
            "v1_relation": relation,
        }
        for item_id, record, name, relation in entries
    ]
    return assign_item_ids(items, rng, SUPPLEMENT_ITEM_PREFIX)


def inherited_section(plan: Record, relations: dict[str, Record]) -> dict[str, Record]:
    return {
        item_id: {
            "relation": relation["relation"],
            "question": relation["question"],
            "v2_stratum": plan["item_plan"][item_id]["v2_stratum"],
            "v2_tiers": plan["item_plan"][item_id]["v2_tiers"],
        }
        for item_id, relation in sorted(relations.items())
        if relation["inherited"]
    }


def relation_counts(relations: dict[str, Record]) -> Record:
    by_question: dict[str, Counter[str]] = {}
    for relation in relations.values():
        by_question.setdefault(relation["question"], Counter())[relation["relation"]] += 1
    return {
        question: {name: counts[name] for name in RELATIONS if counts[name]}
        for question, counts in sorted(by_question.items())
    }


def supplement_rng(config: ValidationConfig) -> np.random.Generator:
    if SUPPLEMENT_ORDER_STREAM in config.seed_streams.values():
        raise SystemExit(f"seed stream {SUPPLEMENT_ORDER_STREAM} is already used by the sample")
    return np.random.default_rng([config.seed, SUPPLEMENT_ORDER_STREAM])


def supplement_key(
    key: Record,
    key_path: Path,
    plan: Record,
    plan_path: Path,
    runs: dict[str, Path],
    inheritance: Record,
    final_test: bool,
) -> Record:
    return {
        "role": "annotation",
        "kind": SUPPLEMENT_KIND,
        "sample_name": SUPPLEMENT_NAME,
        "created_at": utc_timestamp(),
        "dry_run": False,
        "final_test": final_test,
        "splits_used": key["splits_used"],
        "splits_declared": key["splits_declared"],
        "rule": SUPPLEMENT_RULE,
        "inheritance": inheritance,
        "inheritance_sha256": canonical_sha256(inheritance),
        "base_sample": {"sample_name": key["sample_name"], "annotation_key": file_record(key_path)},
        "plan": file_record(plan_path),
        "runs": {name: file_record(path) for name, path in runs.items()},
        "population": {"unit": "UDV", "sampled_splits": plan["v2_population_sampled_splits"]},
        "strata": key["strata"],
        "criteria": key["criteria"],
        "criteria_sha256": key["criteria_sha256"],
        "code": code_section(PACKAGE_DIR),
    }


def check_plan_runs(plan: Record, plan_path: Path, runs: dict[str, Path]) -> None:
    for name, path in runs.items():
        if plan["inputs"][name]["sha256"] != sha256_of_file(path):
            raise SystemExit(f"{plan_path} was built from another {path}")


def hearing_views(config: ValidationConfig, hearing_ids: set[int]) -> dict[int, Record]:
    return {
        hearing["id"]: hearing_view(hearing)
        for hearing in load_gated_jsonl(config.lds_path, config.lds_sha256)
        if hearing["id"] in hearing_ids
    }


def print_supplement_summary(
    supplement: Record, rows: int, csv_path: Path, transcripts_dir: Path
) -> None:
    print(f"{SUPPLEMENT_NAME}: {rows} rows to judge -> {csv_path}")
    for question, counts in supplement["relation_counts"].items():
        print(f"  {question:26s} {counts}")
    for name, entry in supplement["sample"].items():
        print(f"  {name:26s} rows={entry['rows']}")
    print(f"  transcripts in {transcripts_dir}")


def command_supplement(args: argparse.Namespace) -> None:
    config = load_validation_config(args.config)
    inheritance = load_inheritance(config)
    key_path = Path(args.sample_dir) / ANNOTATION_KEY
    key = load_key(key_path, "annotation")
    require_final_test(key, args.final_test)
    plan_path = Path(args.plan)
    plan = load_plan(plan_path, key_path)
    runs = {"udv_v1": Path(args.udv_v1), "udv_v2": Path(args.udv_v2)}
    check_plan_runs(plan, plan_path, runs)
    output_dir = Path(args.output_dir)
    transcripts_dir = Path(args.transcripts_dir)
    ensure_new_dir(output_dir)
    ensure_new_dir(transcripts_dir)
    v1 = {record["id"]: record for record in load_jsonl(runs["udv_v1"])}
    v2 = {record["id"]: record for record in load_jsonl(runs["udv_v2"])}
    relations = item_relations(key, plan, v1, v2, inheritance)
    entries = reannotated_records(plan, relations, v2)
    views = hearing_views(config, {record["hearing_id"] for _, record, _, _ in entries})
    items = supplement_items(
        entries, views, key_strata(key), config.context_chars, supplement_rng(config)
    )
    rows = sheet_rows(items)
    blinding = blinding_section(rows, items, config)
    csv_path = output_dir / ANNOTATION_CSV
    write_annotation_csv(rows, csv_path, config)
    rows_by_stratum = dict(sorted(Counter(item["stratum"] for item in items.values()).items()))
    supplement = {
        **supplement_key(key, key_path, plan, plan_path, runs, inheritance, args.final_test),
        "relation_counts": relation_counts(relations),
        "sample": {name: {"rows": count} for name, count in rows_by_stratum.items()},
        "seeds": {
            "seed": config.seed,
            "row_order_stream": SUPPLEMENT_ORDER_STREAM,
            "generator": GENERATOR,
        },
        "blinding": blinding,
        "csv": csv_section(ANNOTATION_CSV, csv_path, config),
        "transcripts": write_transcripts(views, transcripts_dir),
        "transcripts_dir": str(transcripts_dir),
        "inherited_items": inherited_section(plan, relations),
        "items": items,
    }
    write_json(supplement, output_dir / ANNOTATION_KEY)
    print_supplement_summary(supplement, len(rows), csv_path, transcripts_dir)


def check_supplement_sources(
    supplement: Record, path: Path, key_path: Path, plan_path: Path, inheritance: Record
) -> None:
    if supplement.get("kind") != SUPPLEMENT_KIND:
        raise SystemExit(f"{path / ANNOTATION_KEY} is not a {SUPPLEMENT_KIND} key")
    if supplement["base_sample"]["annotation_key"]["sha256"] != sha256_of_file(key_path):
        raise SystemExit(f"{path} was drawn from another annotation key")
    if supplement["plan"]["sha256"] != sha256_of_file(plan_path):
        raise SystemExit(f"{path} was drawn from another annotation plan")
    if supplement["criteria_sha256"] != canonical_sha256(supplement["criteria"]):
        raise SystemExit(f"{path}: the criteria differ from the ones declared with the sample")
    if supplement["inheritance_sha256"] != canonical_sha256(supplement["inheritance"]):
        raise SystemExit(f"{path}: the inheritance section was edited after the sheet was drawn")
    if supplement["inheritance_sha256"] != canonical_sha256(inheritance):
        raise SystemExit(f"{path}: the inheritance section differs from the config")


def check_supplement_coverage(
    supplement: Record, path: Path, plan: Record, inheritance: Record
) -> None:
    """Every udv_v1 item is either inherited or has one row, as the relation rules say."""
    inherited = supplement["inherited_items"]
    covered = [item["v1_item_id"] for item in supplement["items"].values()]
    if len(set(covered)) != len(covered) or set(covered) & set(inherited):
        raise SystemExit(f"{path}: an item of the udv_v1 sample is covered more than once")
    if set(covered) | set(inherited) != set(plan["item_plan"]):
        raise SystemExit(f"{path}: the rows and the inherited items do not cover the plan")
    identical = {
        item_id for item_id, entry in plan["item_plan"].items() if entry["action"] == "inherit"
    }
    for item_id, entry in inherited.items():
        if not is_inherited(entry["relation"], inheritance):
            raise SystemExit(f"{path}: {item_id} is inherited with relation {entry['relation']}")
        if (item_id in identical) != (entry["relation"] in IDENTICAL_RELATIONS):
            raise SystemExit(f"{path}: {item_id} relation disagrees with the annotation plan")
    for item in supplement["items"].values():
        if is_inherited(item["v1_relation"], inheritance) or item["v1_item_id"] in identical:
            raise SystemExit(f"{path}: {item['v1_item_id']} has a row but would be inherited")


def load_supplement(
    path: Path, key_path: Path, plan: Record, plan_path: Path, inheritance: Record
) -> Record:
    supplement = load_key(path / ANNOTATION_KEY, "annotation")
    check_supplement_sources(supplement, path, key_path, plan_path, inheritance)
    check_supplement_coverage(supplement, path, plan, inheritance)
    return supplement
