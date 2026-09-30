"""Stage ``sample``: draw the strata of a UDV run and write the blind sheet, its key and report."""

import argparse
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from bookworm import load_gated_jsonl, write_json

from experiments.common.reporting import utc_timestamp
from experiments.common.splits import SPLIT_NAMES, load_split_lookup
from experiments.validation.generate_sample.blinding import blinding_section
from experiments.validation.generate_sample.config import ValidationConfig, stream_rng
from experiments.validation.generate_sample.draw import (
    draw_strata,
    population_counts,
    sum_population,
)
from experiments.validation.generate_sample.items import (
    assign_item_ids,
    build_items,
    hearing_view,
    write_transcripts,
)
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
    SAMPLE_REPORT,
    canonical_sha256,
    csv_section,
    ensure_new_dir,
    sheet_rows,
    write_annotation_csv,
)
from experiments.validation.generate_sample.sources import (
    Locations,
    load_coverage,
    load_run,
    readable_splits,
    resolve_locations,
    resolve_splits,
    threshold_calibration_check,
)
from experiments.validation.generate_sample.statistics import criteria_feasibility

Record = dict[str, Any]


def checked_calibration(
    args: argparse.Namespace,
    config: ValidationConfig,
    splits: tuple[str, ...],
    split_of: dict[int, str],
    manifest_sha256: str,
) -> Record:
    """The calibration check of the run; a failed check is refused unless this is a dry run."""
    calibration = threshold_calibration_check(
        load_coverage(config, args.run_name),
        splits,
        split_of,
        manifest_sha256,
        config.source["source"]["threshold_calibration_rule"],
    )
    if not calibration["passed"]:
        message = (
            f"the tier boundary of {args.run_name} (embedding_threshold "
            f"{calibration['embedding_threshold']}) has no calibration record that excludes test "
            f"and the sampled splits: {calibration['problems']}"
        )
        if not args.dry_run:
            raise SystemExit(f"{message}; the sample is refused")
        print(f"WARNING {message}; accepted only because this is a dry run, and recorded")
    return calibration


def population_section(
    by_split: dict[str, dict[str, int]], splits: tuple[str, ...], readable: set[str]
) -> Record:
    all_available = set(SPLIT_NAMES) <= readable
    return {
        "unit": "UDV",
        "sampled_splits": sum_population(by_split, splits),
        "by_split": by_split,
        "all_splits": sum_population(by_split, SPLIT_NAMES) if all_available else None,
        "all_splits_note": (
            "all three splits were readable"
            if all_available
            else "test was not readable "
            "(no --final-test), so the population of the whole run is not recorded"
        ),
    }


def trials_by_stratum(items: dict[str, Record]) -> dict[str, int]:
    trials: dict[str, int] = {}
    for item in items.values():
        trials[item["stratum"]] = trials.get(item["stratum"], 0) + len(item["udv_ids"])
    return trials


def count_rows(draw_summary: dict[str, Record], items: dict[str, Record]) -> None:
    rows_by_stratum = Counter(item["stratum"] for item in items.values())
    for name, summary in draw_summary.items():
        summary["rows"] = rows_by_stratum.get(name, 0)


@dataclass(frozen=True)
class DrawnSample:
    run_source: Record
    population: Record
    draw_summary: dict[str, Record]
    items: dict[str, Record]
    rows: list[Record]
    views: dict[int, Record]
    hearing_ids: set[int]


def draw_sample(
    args: argparse.Namespace,
    config: ValidationConfig,
    splits: tuple[str, ...],
    readable: set[str],
    split_of: dict[int, str],
) -> DrawnSample:
    records, run_source = load_run(config, args.run_name, split_of, readable)
    by_split = population_counts(records, split_of, config.strata, readable)
    sampled_records = [r for r in records if split_of[r["hearing_id"]] in splits]
    drawn, draw_summary = draw_strata(
        sampled_records, config.strata, stream_rng(config, "stratum_draw")
    )
    hearing_ids = {record["hearing_id"] for group in drawn.values() for record in group}
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    views = {h["id"]: hearing_view(h) for h in lds if h["id"] in hearing_ids}
    del lds
    items = assign_item_ids(
        build_items(drawn, config.strata, views, config.context_chars),
        stream_rng(config, "row_order"),
        config.item_id_prefix,
    )
    return DrawnSample(
        run_source=run_source,
        population=population_section(by_split, splits, readable),
        draw_summary=draw_summary,
        items=items,
        rows=sheet_rows(items),
        views=views,
        hearing_ids=hearing_ids,
    )


def sample_key(
    args: argparse.Namespace,
    config: ValidationConfig,
    sample_name: str,
    splits: tuple[str, ...],
    calibration: Record,
    sample: DrawnSample,
    locations: Locations,
) -> Record:
    criteria = config.source["criteria"]
    csv_path = locations.sample_dir / ANNOTATION_CSV
    return {
        "role": "annotation",
        "sample_name": sample_name,
        "sample_version": config.version,
        "created_at": utc_timestamp(),
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": list(splits),
        "splits_declared": list(config.sample_splits),
        "run": sample.run_source,
        "threshold_calibration": calibration,
        "population": sample.population,
        "strata": [asdict(stratum) for stratum in config.strata],
        "sample": sample.draw_summary,
        "criteria": criteria,
        "criteria_sha256": canonical_sha256(criteria),
        "csv": csv_section(ANNOTATION_CSV, csv_path, config),
        "transcripts_dir": str(locations.transcripts_dir),
        "items": sample.items,
    }


def sample_section(sample: DrawnSample) -> Record:
    items, summary = sample.items, sample.draw_summary
    return {
        "by_stratum": summary,
        "rows": len(sample.rows),
        "udvs": sum(len(item["udv_ids"]) for item in items.values()),
        "hearings": len(sample.hearing_ids),
        "evidence_rows_with_offsets": sum(
            1 for item in items.values() if item.get("start_char") is not None
        ),
        "strata_with_shortfall": [n for n, s in summary.items() if s["shortfall"] > 0],
    }


def sample_report(
    args: argparse.Namespace,
    config: ValidationConfig,
    key: Record,
    readable: set[str],
    split_source: Record,
    sample: DrawnSample,
    blinding: Record,
    outputs: Record,
    elapsed: float,
) -> Record:
    return {
        "stage": "sample",
        "sample_name": key["sample_name"],
        "created_at": key["created_at"],
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": key["splits_used"],
        "splits_declared": key["splits_declared"],
        "splits_reason": config.source["splits"]["sample_splits_reason"],
        "splits_readable": sorted(readable),
        "split_manifest": split_source,
        "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
        "run": sample.run_source,
        "threshold_calibration": key["threshold_calibration"],
        "population": sample.population,
        "sample": sample_section(sample),
        "criteria": key["criteria"],
        "criteria_sha256": key["criteria_sha256"],
        "criteria_feasibility_at_drawn_size": criteria_feasibility(
            config, trials_by_stratum(sample.items)
        ),
        "blinding": blinding,
        "seeds": seeds_section(config),
        "outputs": outputs,
        "code": code_hashes(),
        "timing": {"elapsed_seconds": round(elapsed, 1)},
        "environment": environment(),
        "config": config.source,
    }


def print_sample_summary(
    sample_name: str, splits: tuple[str, ...], sample: DrawnSample, locations: Locations
) -> None:
    print(f"sample {sample_name} from {list(splits)} -> {locations.sample_dir}")
    for name, summary in sample.draw_summary.items():
        print(
            f"  {name:26s} population={summary['population']:5d} target={summary['target']:3d} "
            f"drawn={summary['drawn_udvs']:3d} rows={summary['rows']:3d}"
            + (" (all taken)" if summary["all_taken"] else "")
        )
    print(f"  {len(sample.rows)} rows, transcripts in {locations.transcripts_dir}")


def run_sample_stage(args: argparse.Namespace, config: ValidationConfig) -> None:
    started = time.perf_counter()
    splits = resolve_splits(config, args.splits, args.dry_run, args.final_test)
    readable = readable_splits(splits, args.final_test)
    sample_name = sample_name_of(config, args.run_name)
    locations = resolve_locations(config, sample_name, args.dry_run, args.output_dir, True)
    ensure_new_dir(locations.sample_dir)
    ensure_new_dir(locations.transcripts_dir)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    calibration = checked_calibration(args, config, splits, split_of, split_source["sha256"])
    sample = draw_sample(args, config, splits, readable, split_of)
    blinding = blinding_section(sample.rows, sample.items, config)
    count_rows(sample.draw_summary, sample.items)
    csv_path = locations.sample_dir / ANNOTATION_CSV
    write_annotation_csv(sample.rows, csv_path, config)
    transcripts = write_transcripts(sample.views, locations.transcripts_dir)
    key = sample_key(args, config, sample_name, splits, calibration, sample, locations)
    key_path = locations.sample_dir / ANNOTATION_KEY
    write_json(key, key_path)
    outputs = {
        "annotation_csv": file_entry(csv_path),
        "annotation_key": file_entry(key_path),
        "transcripts": transcripts,
    }
    report = sample_report(
        args,
        config,
        key,
        readable,
        split_source,
        sample,
        blinding,
        outputs,
        time.perf_counter() - started,
    )
    write_json(report, locations.sample_dir / SAMPLE_REPORT)
    print_sample_summary(sample_name, splits, sample, locations)
