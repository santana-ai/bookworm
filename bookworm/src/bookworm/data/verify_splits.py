"""Independent verification of a temporal split manifest and report."""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from bookworm.config import SplitConfig
from bookworm.data.dates import DateExtractionSummary, summarize_date_extraction
from bookworm.data.io import JsonObject, is_json_integer, is_json_integer_list, json_path
from bookworm.data.schemas import HearingRecord
from bookworm.data.splits import (
    RECOMPUTED_BOUNDARY_KEYS,
    SPLIT_NAMES,
    SplitBoundaries,
    assign_splits,
    choose_cuts,
    cut_candidates,
    dates_by_hearing,
    opinion_count,
    people_count,
)
from bookworm.udv.schemas import UdvRecord

MANIFEST_KEYS = (
    "split_version",
    "grouping_method",
    "seed",
    "dataset",
    "boundaries",
    "train",
    "validation",
    "test",
)
MANIFEST_OBJECT_KEYS = ("dataset", "boundaries", "article_dates")
REPORT_SPLIT_KEYS = ("hearings", "people", "opinions", "udvs")
GAP_KEYS = ("train_gap_days", "test_gap_days")
CHRONOLOGY_PAIRS = (("train", "validation"), ("validation", "test"))


@dataclass(frozen=True)
class SplitVerification:
    split_version: Any
    hearings: dict[str, int | None]
    udvs: dict[str, Any]
    boundaries: Any
    extraction: DateExtractionSummary
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_report(self) -> JsonObject:
        return {
            "split_version": self.split_version,
            "hearings": dict(self.hearings),
            "udvs": dict(self.udvs),
            "boundaries": self.boundaries,
            "date_extraction": {
                "with_publication_timestamp": self.extraction.with_publication_timestamp,
                "confirmed_by_at_least_one_mention": (
                    self.extraction.confirmed_by_at_least_one_mention
                ),
                "max_lag_days": self.extraction.max_lag_days,
            },
            "problems": list(self.problems),
        }


def check_manifest_shape(manifest: Mapping[str, Any]) -> list[str]:
    problems = [f"missing_key:{key}" for key in MANIFEST_KEYS if key not in manifest]
    problems.extend(
        f"non_integer_ids:{name}"
        for name in SPLIT_NAMES
        if name in manifest and not is_json_integer_list(manifest[name])
    )
    problems.extend(
        f"non_object:{key}"
        for key in MANIFEST_OBJECT_KEYS
        if key in manifest and not isinstance(manifest[key], dict)
    )
    return problems


def check_partition(manifest: Mapping[str, Any], dates: Mapping[int, date]) -> list[str]:
    assigned = [i for name in SPLIT_NAMES for i in manifest[name]]
    problems = [
        f"duplicate_hearing:{hearing_id}"
        for hearing_id, count in Counter(assigned).items()
        if count > 1
    ]
    problems.extend(
        f"unassigned_hearing:{hearing_id}" for hearing_id in sorted(set(dates) - set(assigned))
    )
    problems.extend(
        f"unknown_hearing:{hearing_id}" for hearing_id in sorted(set(assigned) - set(dates))
    )
    return problems


def check_dates(manifest: Mapping[str, Any], dates: Mapping[int, date]) -> list[str]:
    recorded = manifest.get("article_dates", {})
    problems: list[str] = []
    if len(recorded) != len(dates):
        problems.append(f"article_dates_size:{len(recorded)} != {len(dates)}")
    for hearing_id, day in sorted(dates.items()):
        if recorded.get(str(hearing_id)) != day.isoformat():
            problems.append(f"article_date_mismatch:{hearing_id}")
    return problems


def known_ids(manifest: Mapping[str, Any], name: str, dates: Mapping[int, date]) -> list[int]:
    return [i for i in manifest[name] if i in dates]


def check_chronology(
    manifest: Mapping[str, Any], dates: Mapping[int, date], min_gap_days: int
) -> list[str]:
    ranges = {
        name: (
            min(dates[i] for i in known_ids(manifest, name, dates)),
            max(dates[i] for i in known_ids(manifest, name, dates)),
        )
        for name in SPLIT_NAMES
        if known_ids(manifest, name, dates)
    }
    problems: list[str] = []
    for earlier, later in CHRONOLOGY_PAIRS:
        if earlier not in ranges or later not in ranges:
            problems.append(f"empty_split:{earlier if earlier not in ranges else later}")
            continue
        gap = (ranges[later][0] - ranges[earlier][1]).days
        if gap <= 0:
            problems.append(f"not_chronological:{earlier}->{later}")
        elif gap < min_gap_days:
            problems.append(f"boundary_gap_below_minimum:{earlier}->{later}:{gap}")
    days_by_split: dict[date, set[str]] = defaultdict(set)
    for name in SPLIT_NAMES:
        for hearing_id in known_ids(manifest, name, dates):
            days_by_split[dates[hearing_id]].add(name)
    for day, names in sorted(days_by_split.items()):
        if len(names) > 1:
            problems.append(f"date_straddles_splits:{day.isoformat()}")
    return problems


def recompute_boundaries(dates: Mapping[int, date], config: SplitConfig) -> SplitBoundaries:
    candidates = cut_candidates(dates, config.min_boundary_gap_days)
    train_cut, validation_cut = choose_cuts(
        candidates, config.train_fraction, config.validation_fraction
    )
    return SplitBoundaries.from_cuts(train_cut, validation_cut)


def check_boundaries(
    manifest: Mapping[str, Any], dates: Mapping[int, date], config: SplitConfig
) -> list[str]:
    recomputed = recompute_boundaries(dates, config).to_dict()
    recorded = manifest["boundaries"]
    return [
        f"boundary_mismatch:{key}: manifest {recorded.get(key)}, recomputed {recomputed[key]}"
        for key in RECOMPUTED_BOUNDARY_KEYS
        if recorded.get(key) != recomputed[key]
    ]


def check_assignment(
    manifest: Mapping[str, Any], dates: Mapping[int, date], config: SplitConfig
) -> list[str]:
    boundaries = recompute_boundaries(dates, config)
    expected = {
        hearing_id: name
        for name, ids in assign_splits(
            dates, boundaries.train_end, boundaries.validation_end
        ).items()
        for hearing_id in ids
    }
    recorded: dict[int, set[str]] = defaultdict(set)
    for name in SPLIT_NAMES:
        for hearing_id in known_ids(manifest, name, dates):
            recorded[hearing_id].add(name)
    return [
        f"assignment_mismatch:{hearing_id}"
        for hearing_id in sorted(recorded)
        if recorded[hearing_id] != {expected[hearing_id]}
    ]


def check_gap_against_date_uncertainty(
    manifest: Mapping[str, Any], extraction: DateExtractionSummary
) -> list[str]:
    uncertainty = extraction.max_lag_days
    recorded = manifest["boundaries"]
    return [
        f"boundary_gap_within_date_uncertainty:{key}:{recorded[key]}"
        for key in GAP_KEYS
        if isinstance(recorded.get(key), int) and recorded[key] <= uncertainty
    ]


def check_report_counter(report: Mapping[str, Any], path: tuple[str, ...]) -> list[str]:
    found, value = json_path(report, path)
    if not found:
        return [f"report_missing_key:{'.'.join(path)}"]
    if not is_json_integer(value):
        return [f"report_non_integer:{'.'.join(path)}"]
    return []


def check_report_shape(report: Mapping[str, Any]) -> list[str]:
    paths: list[tuple[str, ...]] = [
        ("splits", name, key) for name in SPLIT_NAMES for key in REPORT_SPLIT_KEYS
    ]
    paths.append(("date_extraction", "confirmed_by_no_mention"))
    return [problem for path in paths for problem in check_report_counter(report, path)]


def check_report(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    hearings: Sequence[HearingRecord],
    udvs: Sequence[UdvRecord],
) -> list[str]:
    hearings_by_id = {hearing.id: hearing for hearing in hearings}
    udvs_by_hearing = Counter(udv.hearing_id for udv in udvs)
    problems: list[str] = []
    for name in SPLIT_NAMES:
        ids = manifest[name]
        known = [i for i in ids if i in hearings_by_id]
        summary = report["splits"][name]
        recount = {
            "hearings": len(ids),
            "people": people_count(known, hearings_by_id),
            "opinions": opinion_count(known, hearings_by_id),
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


def check_run_metadata(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    extraction: DateExtractionSummary,
    config: SplitConfig,
    report_readable: bool,
) -> list[str]:
    problems: list[str] = []
    if manifest["seed"] != config.seed:
        problems.append("seed_mismatch")
    if manifest["dataset"].get("sha256") != config.expected_sha256:
        problems.append("dataset_sha256_mismatch")
    if (
        report_readable
        and extraction.confirmed_by_no_mention
        != report["date_extraction"]["confirmed_by_no_mention"]
    ):
        problems.append("date_extraction_mismatch")
    return problems


def report_udvs(report: Mapping[str, Any]) -> dict[str, Any]:
    splits = report.get("splits")
    return {
        name: splits[name].get("udvs")
        if isinstance(splits, dict) and isinstance(splits.get(name), dict)
        else None
        for name in SPLIT_NAMES
    }


def verify_split_run(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    hearings: Sequence[HearingRecord],
    udvs: Sequence[UdvRecord],
    config: SplitConfig,
) -> SplitVerification:
    """Recompute a temporal split from the LDS and check a manifest and report."""
    dates = dates_by_hearing(hearings)
    extraction = summarize_date_extraction(hearings)
    problems = check_manifest_shape(manifest)
    report_problems = check_report_shape(report)
    if not problems:
        problems.extend(check_partition(manifest, dates))
        problems.extend(check_dates(manifest, dates))
        problems.extend(check_chronology(manifest, dates, config.min_boundary_gap_days))
        problems.extend(check_boundaries(manifest, dates, config))
        problems.extend(check_assignment(manifest, dates, config))
        problems.extend(check_gap_against_date_uncertainty(manifest, extraction))
        problems.extend(report_problems)
        if not report_problems:
            problems.extend(check_report(report, manifest, hearings, udvs))
        problems.extend(
            check_run_metadata(manifest, report, extraction, config, not report_problems)
        )
    else:
        problems.extend(report_problems)
    ids = {name: manifest.get(name) for name in SPLIT_NAMES}
    return SplitVerification(
        split_version=manifest.get("split_version"),
        hearings={name: len(v) if isinstance(v, list) else None for name, v in ids.items()},
        udvs=report_udvs(report),
        boundaries=manifest.get("boundaries"),
        extraction=extraction,
        problems=problems,
    )
