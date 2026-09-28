"""Stage ``repeat``: after the first pass is complete, draw the blind repeat sheet of a sample."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookworm import sha256_of_file, write_json

from experiments.common.reporting import utc_timestamp
from experiments.validation.generate_sample.config import ValidationConfig, stream_rng
from experiments.validation.generate_sample.items import assign_item_ids
from experiments.validation.generate_sample.report import (
    code_hashes,
    environment,
    file_entry,
    sample_name_of,
    seeds_section,
)
from experiments.validation.generate_sample.sheets import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    REANNOTATION_CSV,
    REANNOTATION_KEY,
    REANNOTATION_REPORT,
    csv_section,
    existing_precision_reports,
    load_key,
    read_annotation_csv,
    require_final_test,
    sheet_rows,
    validate_annotation,
    write_annotation_csv,
)
from experiments.validation.generate_sample.sources import resolve_locations

Record = dict[str, Any]

SECONDS_PER_HOUR = 3600
PROBLEMS_SHOWN = 50


def repeat_same_relative_order(chosen: list[str], new_items: dict[str, Record]) -> bool:
    return [item["original_item_id"] for item in new_items.values()] == sorted(chosen)


def warn_about_earlier_reports(earlier_reports: list[Record]) -> None:
    if earlier_reports:
        print(
            f"WARNING precision numbers were computed before the repeat sheet exists "
            f"({[entry['file'] for entry in earlier_reports]}); the repeat sheet is created and "
            "this is recorded in reannotation_key.json and in the final precision report"
        )


def first_pass_section(first_path: Path, key: Record, config: ValidationConfig) -> Record:
    """Refuse an incomplete or too recent first pass, and record when it was last modified."""
    _, problems = validate_annotation(read_annotation_csv(first_path, config), key, config)
    if problems:
        print("\n".join(problems[:PROBLEMS_SHOWN]))
        raise SystemExit(
            f"{len(problems)} problems in {first_path}; the repeat sheet is created only after "
            "the first pass is complete"
        )
    modified = datetime.fromtimestamp(first_path.stat().st_mtime, UTC)
    hours = (datetime.now(UTC) - modified).total_seconds() / SECONDS_PER_HOUR
    if hours < config.repeat_min_hours:
        raise SystemExit(
            f"{first_path} was last modified {hours:.1f} h ago; the repeat sheet needs at least "
            f"{config.repeat_min_hours:g} h"
        )
    return {
        "file": ANNOTATION_CSV,
        "sha256": sha256_of_file(first_path),
        "modified_at": modified.isoformat(timespec="seconds"),
        "hours_since_modification": round(hours, 2),
    }


def draw_repeat_items(
    key: Record, config: ValidationConfig
) -> tuple[list[str], list[str], dict[str, Record]]:
    """The eligible item ids, the ones drawn again, and the drawn items under new ids."""
    eligible = sorted(
        item_id
        for item_id, item in key["items"].items()
        if item["question"] in config.repeat_questions
    )
    size = min(config.repeat_items, len(eligible))
    picks = stream_rng(config, "repeat_draw").choice(len(eligible), size=size, replace=False)
    chosen = sorted(eligible[int(index)] for index in picks)
    originals = [
        {
            "original_item_id": item_id,
            "question": key["items"][item_id]["question"],
            "display": key["items"][item_id]["display"],
        }
        for item_id in chosen
    ]
    new_items = assign_item_ids(
        originals, stream_rng(config, "repeat_order"), config.repeat_item_id_prefix
    )
    return eligible, chosen, new_items


def run_repeat_stage(args: argparse.Namespace, config: ValidationConfig) -> None:
    started = time.perf_counter()
    if args.splits is not None:
        raise SystemExit("--splits is not used by the repeat stage")
    sample_name = sample_name_of(config, args.run_name)
    locations = resolve_locations(config, sample_name, args.dry_run, args.output_dir, False)
    key_path = locations.sample_dir / ANNOTATION_KEY
    key = load_key(key_path, "annotation")
    if key["sample_name"] != sample_name or key["dry_run"] != args.dry_run:
        raise SystemExit(f"{key_path} belongs to another sample or mode")
    require_final_test(key, args.final_test)
    csv_path = locations.sample_dir / REANNOTATION_CSV
    repeat_key_path = locations.sample_dir / REANNOTATION_KEY
    if csv_path.exists() or repeat_key_path.exists():
        raise SystemExit(f"the repeat sheet of {sample_name} already exists; it is never redrawn")
    earlier_reports = existing_precision_reports(locations.sample_dir)
    warn_about_earlier_reports(earlier_reports)
    first_pass = first_pass_section(locations.sample_dir / ANNOTATION_CSV, key, config)
    eligible, chosen, new_items = draw_repeat_items(key, config)
    write_annotation_csv(sheet_rows(new_items), csv_path, config)
    repeat_key = {
        "role": "reannotation",
        "sample_name": sample_name,
        "created_at": utc_timestamp(),
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": key["splits_used"],
        "annotation_key_sha256": sha256_of_file(key_path),
        "first_pass": first_pass,
        "precision_reports_before_repeat_sheet": earlier_reports,
        "eligible_questions": list(config.repeat_questions),
        "eligible_items": len(eligible),
        "requested_items": config.repeat_items,
        "drawn_items": len(chosen),
        "same_relative_order_as_first_pass": repeat_same_relative_order(chosen, new_items),
        "csv": csv_section(REANNOTATION_CSV, csv_path, config),
        "items": new_items,
    }
    write_json(repeat_key, repeat_key_path)
    report = {
        "stage": "repeat",
        "sample_name": sample_name,
        "created_at": repeat_key["created_at"],
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": key["splits_used"],
        "first_pass": first_pass,
        "precision_reports_before_repeat_sheet": earlier_reports,
        "eligible_questions": list(config.repeat_questions),
        "eligible_reason": config.source["repeat"]["eligible_reason"],
        "eligible_items": len(eligible),
        "drawn_items": len(chosen),
        "same_relative_order_as_first_pass": repeat_key["same_relative_order_as_first_pass"],
        "seeds": seeds_section(config),
        "outputs": {
            "reannotation_csv": file_entry(csv_path),
            "reannotation_key": file_entry(repeat_key_path),
        },
        "code": code_hashes(),
        "timing": {"elapsed_seconds": round(time.perf_counter() - started, 1)},
        "environment": environment(),
        "config": config.source,
    }
    write_json(report, locations.sample_dir / REANNOTATION_REPORT)
    print(f"repeat sheet with {len(chosen)} of {len(eligible)} eligible items -> {csv_path}")
