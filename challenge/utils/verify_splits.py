import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from utils.build_splits import (
    SPLIT_NAMES,
    choose_boundary,
    cut_candidates,
    dates_by_hearing,
    load_config,
    summarize_date_extraction,
)
from utils.dataset_io import load_gated_jsonl, load_jsonl

Record = dict[str, Any]

MANIFEST_KEYS = ("split_version", "grouping_method", "seed", "train", "validation", "test")


def check_manifest_shape(manifest: Record) -> list[str]:
    problems = [f"missing_key:{key}" for key in MANIFEST_KEYS if key not in manifest]
    for name in SPLIT_NAMES:
        ids = manifest.get(name)
        if ids is not None and not all(isinstance(i, int) for i in ids):
            problems.append(f"non_integer_ids:{name}")
    return problems


def check_partition(manifest: Record, dates: dict[int, date]) -> list[str]:
    assigned = [i for name in SPLIT_NAMES for i in manifest[name]]
    problems: list[str] = []
    for hearing_id, count in Counter(assigned).items():
        if count > 1:
            problems.append(f"duplicate_hearing:{hearing_id}")
    for hearing_id in sorted(set(dates) - set(assigned)):
        problems.append(f"unassigned_hearing:{hearing_id}")
    for hearing_id in sorted(set(assigned) - set(dates)):
        problems.append(f"unknown_hearing:{hearing_id}")
    return problems


def check_dates(manifest: Record, dates: dict[int, date]) -> list[str]:
    recorded = manifest.get("article_dates", {})
    problems: list[str] = []
    if len(recorded) != len(dates):
        problems.append(f"article_dates_size:{len(recorded)} != {len(dates)}")
    for hearing_id, day in sorted(dates.items()):
        if recorded.get(str(hearing_id)) != day.isoformat():
            problems.append(f"article_date_mismatch:{hearing_id}")
    return problems


def check_chronology(manifest: Record, dates: dict[int, date], min_gap_days: int) -> list[str]:
    ranges = {
        name: (
            min(dates[i] for i in manifest[name]),
            max(dates[i] for i in manifest[name]),
        )
        for name in SPLIT_NAMES
        if manifest[name]
    }
    problems: list[str] = []
    for earlier, later in (("train", "validation"), ("validation", "test")):
        if earlier not in ranges or later not in ranges:
            problems.append(f"empty_split:{earlier if earlier not in ranges else later}")
            continue
        gap = (ranges[later][0] - ranges[earlier][1]).days
        if gap <= 0:
            problems.append(f"not_chronological:{earlier}->{later}")
        elif gap < min_gap_days:
            problems.append(f"boundary_gap_below_minimum:{earlier}->{later}:{gap}")
    days_by_split = defaultdict(set)
    for name in SPLIT_NAMES:
        for hearing_id in manifest[name]:
            days_by_split[dates[hearing_id]].add(name)
    for day, names in sorted(days_by_split.items()):
        if len(names) > 1:
            problems.append(f"date_straddles_splits:{day.isoformat()}")
    return problems


def check_boundaries(manifest: Record, dates: dict[int, date], config: Record) -> list[str]:
    candidates = cut_candidates(dates, config.min_boundary_gap_days)
    train_cut = choose_boundary(candidates, config.train_fraction, None)
    validation_cut = choose_boundary(
        candidates, config.train_fraction + config.validation_fraction, train_cut["date"]
    )
    recomputed = {
        "train_end": train_cut["date"].isoformat(),
        "validation_start": train_cut["next_date"].isoformat(),
        "validation_end": validation_cut["date"].isoformat(),
        "test_start": validation_cut["next_date"].isoformat(),
        "train_gap_days": train_cut["gap_days"],
        "test_gap_days": validation_cut["gap_days"],
    }
    recorded = manifest["boundaries"]
    return [
        f"boundary_mismatch:{key}: manifest {recorded.get(key)}, recomputed {value}"
        for key, value in recomputed.items()
        if recorded.get(key) != value
    ]


def check_gap_against_date_uncertainty(manifest: Record, extraction: Record) -> list[str]:
    lags = [int(lag) for lag in extraction["lag_days_when_agreeing"]]
    uncertainty = max(lags) if lags else 0
    return [
        f"boundary_gap_within_date_uncertainty:{key}:{manifest['boundaries'][key]}"
        for key in ("train_gap_days", "test_gap_days")
        if manifest["boundaries"][key] <= uncertainty
    ]


def check_report(
    report: Record, manifest: Record, records: list[Record], udvs: list[Record]
) -> list[str]:
    records_by_id = {record["id"]: record for record in records}
    udvs_by_hearing: dict[int, int] = Counter(udv["hearing_id"] for udv in udvs)
    problems: list[str] = []
    for name in SPLIT_NAMES:
        ids = manifest[name]
        summary = report["splits"][name]
        recount = {
            "hearings": len(ids),
            "people": sum(len(records_by_id[i]["metadados"]["envolvidos"]) for i in ids),
            "opinions": sum(
                len(person["opinioes"])
                for i in ids
                for person in records_by_id[i]["metadados"]["envolvidos"]
            ),
            "udvs": sum(udvs_by_hearing[i] for i in ids),
        }
        for key, value in recount.items():
            if summary[key] != value:
                problems.append(f"report.{name}.{key}: reported {summary[key]}, recomputed {value}")
    if (
        udvs
        and sum(len(manifest[name]) for name in SPLIT_NAMES)
        and len(udvs) != sum(report["splits"][name]["udvs"] for name in SPLIT_NAMES)
    ):
        problems.append("report.udvs_total_mismatch")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and cross-check a split manifest against the LDS file."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/splits.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = config.output_dir / f"{config.split_version}.json"
    report_path = config.output_dir / f"{config.split_version}_report.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(report_path) as f:
        report = json.load(f)

    records = load_gated_jsonl(config.lds_path, config.expected_sha256)
    dates = dates_by_hearing(records)
    udvs = load_jsonl(config.udv_path) if config.udv_path.exists() else []
    extraction = summarize_date_extraction(records)

    problems: list[str] = []
    problems.extend(check_manifest_shape(manifest))
    problems.extend(check_partition(manifest, dates))
    problems.extend(check_dates(manifest, dates))
    problems.extend(check_chronology(manifest, dates, config.min_boundary_gap_days))
    problems.extend(check_boundaries(manifest, dates, config))
    problems.extend(check_gap_against_date_uncertainty(manifest, extraction))
    problems.extend(check_report(report, manifest, records, udvs))
    if manifest["seed"] != config.seed:
        problems.append("seed_mismatch")
    if manifest["dataset"]["sha256"] != config.expected_sha256:
        problems.append("dataset_sha256_mismatch")
    if (
        extraction["confirmed_by_no_mention"]
        != report["date_extraction"]["confirmed_by_no_mention"]
    ):
        problems.append("date_extraction_mismatch")

    print(
        json.dumps(
            {
                "split_version": manifest["split_version"],
                "hearings": {name: len(manifest[name]) for name in SPLIT_NAMES},
                "udvs": {name: report["splits"][name]["udvs"] for name in SPLIT_NAMES},
                "boundaries": manifest["boundaries"],
                "date_extraction": {
                    "with_publication_timestamp": extraction["with_publication_timestamp"],
                    "confirmed_by_at_least_one_mention": extraction[
                        "confirmed_by_at_least_one_mention"
                    ],
                    "max_lag_days": max(
                        (int(lag) for lag in extraction["lag_days_when_agreeing"]), default=0
                    ),
                },
                "problems": problems,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
