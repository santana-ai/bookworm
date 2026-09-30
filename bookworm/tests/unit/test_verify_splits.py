import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest

from bookworm import (
    HearingRecord,
    SplitConfig,
    SplitVerification,
    UdvRecord,
    build_temporal_split,
    summarize_date_extraction,
    verify_split_run,
)
from bookworm.data.io import JsonObject
from bookworm.data.verify_splits import (
    check_assignment,
    check_chronology,
    check_gap_against_date_uncertainty,
    check_manifest_shape,
    check_partition,
)

Mutation = Callable[[JsonObject, JsonObject], None]
DATES = {
    1: date(2023, 3, 1),
    2: date(2023, 3, 1),
    3: date(2023, 3, 2),
    4: date(2023, 3, 9),
    5: date(2023, 3, 14),
    6: date(2023, 3, 16),
    7: date(2023, 3, 23),
    8: date(2023, 4, 5),
    9: date(2023, 4, 12),
    10: date(2023, 5, 3),
}
CLEAN_GROUPS: JsonObject = {"train": [1, 2, 3, 4, 5, 6, 7], "validation": [8], "test": [9, 10]}


@dataclass(frozen=True)
class CleanRun:
    manifest: JsonObject
    report: JsonObject
    hearings: list[HearingRecord]
    udvs: list[UdvRecord]
    config: SplitConfig

    def verify(self, mutation: Mutation | None = None) -> SplitVerification:
        manifest = copy.deepcopy(self.manifest)
        report = copy.deepcopy(self.report)
        if mutation is not None:
            mutation(manifest, report)
        return verify_split_run(manifest, report, self.hearings, self.udvs, self.config)


@pytest.fixture
def clean(
    split_hearings: list[HearingRecord], split_config: SplitConfig, split_udvs: list[UdvRecord]
) -> CleanRun:
    artifacts = build_temporal_split(
        split_hearings,
        split_config,
        split_udvs,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        environment={},
    )
    return CleanRun(
        manifest=json.loads(json.dumps(artifacts.manifest)),
        report=json.loads(json.dumps(artifacts.report)),
        hearings=split_hearings,
        udvs=split_udvs,
        config=split_config,
    )


def swap_hearing_between_splits(manifest: JsonObject, report: JsonObject) -> None:
    manifest["train"].remove(4)
    manifest["test"].insert(0, 4)


def remove_hearing(manifest: JsonObject, report: JsonObject) -> None:
    manifest["test"].remove(10)


def alter_date(manifest: JsonObject, report: JsonObject) -> None:
    manifest["article_dates"]["5"] = "2023-03-15"


def alter_boundary(manifest: JsonObject, report: JsonObject) -> None:
    manifest["boundaries"]["train_end"] = "2023-03-16"


def alter_report_counter(manifest: JsonObject, report: JsonObject) -> None:
    report["splits"]["validation"]["udvs"] = 2


INJECTED_DEFECTS: dict[str, tuple[Mutation, list[str]]] = {
    "hearing_swapped_between_splits": (
        swap_hearing_between_splits,
        [
            "not_chronological:validation->test",
            "assignment_mismatch:4",
            "report.train.hearings: reported 7, recomputed 6",
            "report.train.people: reported 9, recomputed 8",
            "report.train.opinions: reported 10, recomputed 9",
            "report.test.hearings: reported 2, recomputed 3",
            "report.test.people: reported 3, recomputed 4",
            "report.test.opinions: reported 3, recomputed 4",
        ],
    ),
    "hearing_removed": (
        remove_hearing,
        [
            "unassigned_hearing:10",
            "report.test.hearings: reported 2, recomputed 1",
            "report.test.people: reported 3, recomputed 2",
            "report.test.opinions: reported 3, recomputed 2",
        ],
    ),
    "date_altered": (alter_date, ["article_date_mismatch:5"]),
    "boundary_altered": (
        alter_boundary,
        ["boundary_mismatch:train_end: manifest 2023-03-16, recomputed 2023-03-23"],
    ),
    "report_counter_altered": (
        alter_report_counter,
        [
            "report.validation.udvs: reported 2, recomputed 1",
            "report.udvs_total_mismatch",
        ],
    ),
}


def test_clean_run_has_no_problems(clean: CleanRun) -> None:
    verification = clean.verify()
    assert verification.ok
    assert verification.to_report() == {
        "split_version": "mini_temporal",
        "hearings": {"train": 7, "validation": 1, "test": 2},
        "udvs": {"train": 2, "validation": 1, "test": 1},
        "boundaries": clean.manifest["boundaries"],
        "date_extraction": {
            "with_publication_timestamp": 10,
            "confirmed_by_at_least_one_mention": 6,
            "max_lag_days": 1,
        },
        "problems": [],
    }


@pytest.mark.parametrize("defect", sorted(INJECTED_DEFECTS))
def test_injected_defect_is_reported(clean: CleanRun, defect: str) -> None:
    mutation, expected = INJECTED_DEFECTS[defect]
    verification = clean.verify(mutation)
    assert not verification.ok
    assert verification.problems == expected


def test_missing_article_date_entry(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        del manifest["article_dates"]["3"]

    assert clean.verify(mutate).problems == [
        "article_dates_size:9 != 10",
        "article_date_mismatch:3",
    ]


def test_seed_dataset_and_extraction_mismatches(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        manifest["seed"] = 8
        manifest["dataset"]["sha256"] = "0" * 64
        report["date_extraction"]["confirmed_by_no_mention"] = 0

    assert clean.verify(mutate).problems == [
        "seed_mismatch",
        "dataset_sha256_mismatch",
        "date_extraction_mismatch",
    ]


def test_missing_keys_stop_the_verification(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        del manifest["validation"]
        del manifest["boundaries"]
        manifest["test"] = [9, "10"]

    verification = clean.verify(mutate)
    assert verification.problems == [
        "missing_key:boundaries",
        "missing_key:validation",
        "non_integer_ids:test",
    ]
    assert verification.to_report()["hearings"] == {"train": 7, "validation": None, "test": 2}
    assert verification.to_report()["boundaries"] is None


def test_manifest_shape_reports_non_objects() -> None:
    manifest: JsonObject = {
        "split_version": "v",
        "grouping_method": "g",
        "seed": 1,
        "dataset": [],
        "boundaries": "none",
        "article_dates": [],
        "train": 3,
        "validation": [],
        "test": [],
    }
    assert check_manifest_shape(manifest) == [
        "non_integer_ids:train",
        "non_object:dataset",
        "non_object:boundaries",
        "non_object:article_dates",
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("train", None),
        ("validation", [True]),
        ("validation", [8.0]),
        ("test", "9,10"),
        ("test", {"9": 10}),
    ],
)
def test_split_lists_must_hold_integers(clean: CleanRun, name: str, value: object) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        manifest[name] = value

    verification = clean.verify(mutate)
    assert verification.problems == [f"non_integer_ids:{name}"]
    expected_count = len(value) if isinstance(value, list) else None
    assert verification.to_report()["hearings"][name] == expected_count


@pytest.mark.parametrize("value", ["1", None, True, 1.0, [1]])
def test_report_split_counters_must_be_integers(clean: CleanRun, value: object) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        report["splits"]["validation"]["udvs"] = value

    assert clean.verify(mutate).problems == ["report_non_integer:splits.validation.udvs"]


def test_report_extraction_counter_must_be_an_integer(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        report["date_extraction"]["confirmed_by_no_mention"] = "4"
        report["splits"]["train"]["people"] = None

    assert clean.verify(mutate).problems == [
        "report_non_integer:splits.train.people",
        "report_non_integer:date_extraction.confirmed_by_no_mention",
    ]


def test_report_shape_problems_skip_the_counter_checks(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        del report["splits"]["test"]["people"]
        report["splits"]["validation"] = []
        del report["date_extraction"]

    verification = clean.verify(mutate)
    assert verification.problems == [
        "report_missing_key:splits.validation.hearings",
        "report_missing_key:splits.validation.people",
        "report_missing_key:splits.validation.opinions",
        "report_missing_key:splits.validation.udvs",
        "report_missing_key:splits.test.people",
        "report_missing_key:date_extraction.confirmed_by_no_mention",
    ]
    assert verification.udvs == {"train": 2, "validation": None, "test": 1}


def test_report_without_splits(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        del report["splits"]
        del manifest["seed"]

    verification = clean.verify(mutate)
    assert verification.problems[0] == "missing_key:seed"
    assert "report_missing_key:splits.train.hearings" in verification.problems
    assert verification.udvs == {"train": None, "validation": None, "test": None}


def test_partition_problems() -> None:
    manifest: JsonObject = {"train": [1, 2, 3, 3], "validation": [99], "test": [9, 10]}
    assert check_partition(manifest, DATES) == [
        "duplicate_hearing:3",
        "unassigned_hearing:4",
        "unassigned_hearing:5",
        "unassigned_hearing:6",
        "unassigned_hearing:7",
        "unassigned_hearing:8",
        "unknown_hearing:99",
    ]


def test_unknown_hearing_does_not_break_the_other_checks(clean: CleanRun) -> None:
    def mutate(manifest: JsonObject, report: JsonObject) -> None:
        manifest["test"].append(99)

    assert clean.verify(mutate).problems == [
        "unknown_hearing:99",
        "report.test.hearings: reported 2, recomputed 3",
    ]


def test_date_straddling_two_splits() -> None:
    manifest: JsonObject = {"train": [1, 3, 4], "validation": [2, 5], "test": [9, 10]}
    assert check_chronology(manifest, DATES, 2) == [
        "not_chronological:train->validation",
        "date_straddles_splits:2023-03-01",
    ]


def test_splits_that_share_their_boundary_date_are_not_chronological() -> None:
    manifest: JsonObject = {"train": [1], "validation": [2], "test": [9, 10]}
    assert check_chronology(manifest, DATES, 2) == [
        "not_chronological:train->validation",
        "date_straddles_splits:2023-03-01",
    ]


def test_gap_below_the_minimum() -> None:
    manifest: JsonObject = {"train": [1, 2], "validation": [3, 4], "test": [9, 10]}
    assert check_chronology(manifest, DATES, 2) == [
        "boundary_gap_below_minimum:train->validation:1",
    ]
    assert check_chronology(manifest, DATES, 1) == []


def test_empty_split() -> None:
    manifest: JsonObject = {"train": [1, 2, 3], "validation": [], "test": [9, 10]}
    assert check_chronology(manifest, DATES, 2) == [
        "empty_split:validation",
        "empty_split:validation",
    ]
    assert check_chronology({"train": [], "validation": [], "test": []}, DATES, 2) == [
        "empty_split:train",
        "empty_split:validation",
    ]


def test_assignment_mismatch_on_a_boundary_hearing(split_config: SplitConfig) -> None:
    manifest: JsonObject = {**CLEAN_GROUPS, "train": [1, 2, 3, 4, 5, 6], "validation": [7, 8]}
    assert check_chronology(manifest, DATES, 2) == []
    assert check_assignment(manifest, DATES, split_config) == ["assignment_mismatch:7"]
    assert check_assignment(CLEAN_GROUPS, DATES, split_config) == []


def test_duplicate_hearing_is_also_an_assignment_mismatch(split_config: SplitConfig) -> None:
    manifest: JsonObject = {**CLEAN_GROUPS, "validation": [7, 8]}
    assert check_assignment(manifest, DATES, split_config) == ["assignment_mismatch:7"]


def test_gap_within_the_date_uncertainty(split_hearings: list[HearingRecord]) -> None:
    extraction = summarize_date_extraction(split_hearings)
    assert extraction.max_lag_days == 1
    manifest: JsonObject = {"boundaries": {"train_gap_days": 1, "test_gap_days": 2}}
    assert check_gap_against_date_uncertainty(manifest, extraction) == [
        "boundary_gap_within_date_uncertainty:train_gap_days:1",
    ]
    assert check_gap_against_date_uncertainty({"boundaries": {}}, extraction) == []


def test_verify_accepts_a_run_without_udvs(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    artifacts = build_temporal_split(split_hearings, split_config, None)
    verification = verify_split_run(
        artifacts.manifest, artifacts.report, split_hearings, [], split_config
    )
    assert verification.ok
    assert verification.udvs == {"train": 0, "validation": 0, "test": 0}
