import copy
import json
from datetime import UTC, date, datetime
from typing import Any

import pytest

from bookworm import (
    CutCandidate,
    HearingRecord,
    SplitBoundaries,
    SplitConfig,
    SplitError,
    UdvRecord,
    build_manifest,
    build_report,
    build_temporal_split,
    compute_temporal_split,
)
from bookworm.data.splits import (
    GROUPING_METHOD,
    assign_splits,
    choose_boundary,
    cut_candidates,
    dates_by_hearing,
)

CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
ENVIRONMENT = {"python": "3.12.0", "platform": "test"}
FIXTURE_DATES = {
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


def candidate(day: date, gap_days: int, fraction: float) -> CutCandidate:
    return CutCandidate(
        day=day,
        next_day=date.fromordinal(day.toordinal() + gap_days),
        gap_days=gap_days,
        cumulative=0,
        fraction=fraction,
    )


def with_changes(config: SplitConfig, **changes: Any) -> SplitConfig:
    return SplitConfig.model_validate({**config.model_dump(), **changes})


def test_dates_by_hearing_reads_every_fixture_date(split_hearings: list[HearingRecord]) -> None:
    assert dates_by_hearing(split_hearings) == FIXTURE_DATES


def test_dates_by_hearing_rejects_an_undated_article(split_hearings: list[HearingRecord]) -> None:
    undated = split_hearings[3].model_copy(update={"materia": "Sem carimbo de publicação."})
    with pytest.raises(SplitError, match="hearing 4: no publication timestamp"):
        dates_by_hearing([*split_hearings[:3], undated])


def test_cut_candidates_respect_the_minimum_gap() -> None:
    candidates = cut_candidates(FIXTURE_DATES, 2)
    assert [(c.day.isoformat(), c.gap_days, c.cumulative) for c in candidates] == [
        ("2023-03-02", 7, 3),
        ("2023-03-09", 5, 4),
        ("2023-03-14", 2, 5),
        ("2023-03-16", 7, 6),
        ("2023-03-23", 13, 7),
        ("2023-04-05", 7, 8),
        ("2023-04-12", 21, 9),
    ]
    assert [c.fraction for c in candidates] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert candidates[0].next_day == date(2023, 3, 9)


def test_a_one_day_gap_is_a_candidate_only_with_minimum_one() -> None:
    assert cut_candidates(FIXTURE_DATES, 1)[0] == CutCandidate(
        day=date(2023, 3, 1), next_day=date(2023, 3, 2), gap_days=1, cumulative=2, fraction=0.2
    )
    assert [c.day for c in cut_candidates(FIXTURE_DATES, 8)] == [
        date(2023, 3, 23),
        date(2023, 4, 12),
    ]


def test_the_last_date_is_never_a_candidate() -> None:
    assert cut_candidates({1: date(2023, 1, 1)}, 1) == []
    assert all(c.day < date(2023, 5, 3) for c in cut_candidates(FIXTURE_DATES, 1))


def test_choose_boundary_prefers_the_closest_fraction() -> None:
    near = candidate(date(2023, 1, 10), 2, 0.5)
    far = candidate(date(2023, 1, 5), 30, 0.25)
    assert choose_boundary([far, near], 0.5, None) is near


def test_choose_boundary_breaks_distance_ties_by_the_larger_gap() -> None:
    short_gap = candidate(date(2023, 1, 5), 3, 0.25)
    long_gap = candidate(date(2023, 1, 20), 9, 0.75)
    assert choose_boundary([short_gap, long_gap], 0.5, None) is long_gap


def test_choose_boundary_breaks_gap_ties_by_the_earlier_date() -> None:
    later = candidate(date(2023, 1, 20), 9, 0.75)
    earlier = candidate(date(2023, 1, 5), 9, 0.25)
    assert choose_boundary([later, earlier], 0.5, None) is earlier


def test_choose_boundary_only_considers_dates_after_the_previous_cut() -> None:
    first = candidate(date(2023, 1, 5), 9, 0.5)
    second = candidate(date(2023, 1, 20), 3, 0.9)
    assert choose_boundary([first, second], 0.5, date(2023, 1, 5)) is second
    with pytest.raises(SplitError, match=r"no cut date available for target fraction 0\.5"):
        choose_boundary([first, second], 0.5, date(2023, 1, 20))


def test_choose_boundary_with_no_candidates() -> None:
    with pytest.raises(SplitError):
        choose_boundary([], 0.7, None)


def test_assign_splits_uses_inclusive_ends_and_date_then_id_order() -> None:
    groups = assign_splits(FIXTURE_DATES, date(2023, 3, 1), date(2023, 3, 14))
    assert groups == {
        "train": (1, 2),
        "validation": (3, 4, 5),
        "test": (6, 7, 8, 9, 10),
    }
    reordered = {2: date(2023, 3, 1), 9: date(2023, 1, 1), 1: date(2023, 3, 1)}
    assert assign_splits(reordered, date(2023, 3, 1), date(2023, 3, 1))["train"] == (9, 1, 2)


def test_compute_temporal_split_on_the_fixture(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    split = compute_temporal_split(split_hearings, split_config)
    assert split.groups == {
        "train": (1, 2, 3, 4, 5, 6, 7),
        "validation": (8,),
        "test": (9, 10),
    }
    assert split.boundaries == SplitBoundaries(
        train_end=date(2023, 3, 23),
        validation_start=date(2023, 4, 5),
        validation_end=date(2023, 4, 5),
        test_start=date(2023, 4, 12),
        train_gap_days=13,
        test_gap_days=7,
        train_fraction_reached=0.7,
        validation_end_fraction_reached=0.8,
    )
    assert len(split.candidates) == 7
    assert dict(split.dates) == FIXTURE_DATES


def test_compute_temporal_split_without_room_for_the_validation_cut(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    config = with_changes(split_config, train_fraction=0.9, validation_fraction=0.05)
    with pytest.raises(SplitError, match="no cut date available"):
        compute_temporal_split(split_hearings, config)


def test_boundaries_serialize_in_the_manifest_key_order() -> None:
    boundaries = SplitBoundaries.from_cuts(
        CutCandidate(date(2023, 3, 23), date(2023, 4, 5), 13, 7, 0.699999),
        CutCandidate(date(2023, 4, 5), date(2023, 4, 12), 7, 8, 0.85444),
    )
    assert json.dumps(boundaries.to_dict()) == json.dumps(
        {
            "train_end": "2023-03-23",
            "validation_start": "2023-04-05",
            "validation_end": "2023-04-05",
            "test_start": "2023-04-12",
            "train_gap_days": 13,
            "test_gap_days": 7,
            "train_fraction_reached": 0.7,
            "validation_end_fraction_reached": 0.8544,
        }
    )


def test_build_manifest(split_hearings: list[HearingRecord], split_config: SplitConfig) -> None:
    split = compute_temporal_split(split_hearings, split_config)
    manifest = build_manifest(split, split_config, created_at=CREATED_AT)
    assert list(manifest) == [
        "split_version",
        "grouping_method",
        "seed",
        "created_at",
        "dataset",
        "date_field",
        "boundaries",
        "train",
        "validation",
        "test",
        "article_dates",
    ]
    assert manifest["split_version"] == "mini_temporal"
    assert manifest["grouping_method"] == GROUPING_METHOD
    assert manifest["seed"] == 7
    assert manifest["created_at"] == "2026-01-02T03:04:05+00:00"
    assert manifest["dataset"] == {
        "path": "splits_mini.jsonl",
        "sha256": split_config.expected_sha256,
    }
    assert manifest["train"] == [1, 2, 3, 4, 5, 6, 7]
    assert manifest["validation"] == [8]
    assert manifest["test"] == [9, 10]
    assert list(manifest["article_dates"]) == [str(i) for i in range(1, 11)]
    assert manifest["article_dates"]["2"] == "2023-03-01"


def test_build_report(
    split_hearings: list[HearingRecord], split_config: SplitConfig, split_udvs: list[UdvRecord]
) -> None:
    split = compute_temporal_split(split_hearings, split_config)
    report = build_report(
        split_hearings,
        split,
        split_config,
        split_udvs,
        created_at=CREATED_AT,
        environment=ENVIRONMENT,
    )
    assert list(report) == [
        "split_version",
        "created_at",
        "date_extraction",
        "calendar",
        "boundaries",
        "cut_candidates_considered",
        "splits",
        "leakage",
        "udv_source",
        "environment",
        "config",
    ]
    assert report["calendar"] == {
        "first_date": "2023-03-01",
        "last_date": "2023-05-03",
        "distinct_dates": 9,
        "hearings_per_date_max": 2,
        "by_year": {2023: 10},
    }
    assert report["cut_candidates_considered"] == 7
    assert report["splits"] == {
        "train": {
            "hearings": 7,
            "share_of_hearings": 0.7,
            "first_date": "2023-03-01",
            "last_date": "2023-03-23",
            "distinct_dates": 6,
            "people": 9,
            "opinions": 10,
            "udvs": 2,
            "udvs_by_tier": {"quote_found": 1, "semantic_match_high": 1},
        },
        "validation": {
            "hearings": 1,
            "share_of_hearings": 0.1,
            "first_date": "2023-04-05",
            "last_date": "2023-04-05",
            "distinct_dates": 1,
            "people": 1,
            "opinions": 1,
            "udvs": 1,
            "udvs_by_tier": {"no_evidence": 1},
        },
        "test": {
            "hearings": 2,
            "share_of_hearings": 0.2,
            "first_date": "2023-04-12",
            "last_date": "2023-05-03",
            "distinct_dates": 2,
            "people": 3,
            "opinions": 3,
            "udvs": 1,
            "udvs_by_tier": {"person_not_resolved": 1},
        },
    }
    assert report["leakage"]["actors"] == {
        "distinct_actors": 9,
        "in_more_than_one_split": 2,
        "in_train_and_test": 2,
    }
    assert report["udv_source"] == "udv/mini.jsonl"
    assert report["environment"] == ENVIRONMENT
    assert report["config"] == split_config.source
    assert report["date_extraction"]["hearings_without_any_agreeing_mention"] == [6]


def test_near_duplicates_find_the_repeated_article(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    split = compute_temporal_split(split_hearings, split_config)
    near = build_report(split_hearings, split, split_config)["leakage"]["near_duplicates"]
    assert near["method"] == "tfidf cosine over materia, min_df=2, sublinear_tf"
    assert near["threshold"] == 0.5
    assert near["cross_split_pairs"] == 23
    assert len(near["top_pairs"]) == 3
    top = near["top_pairs"][0]
    assert top["hearings"] == [4, 9]
    assert top["splits"] == ["train", "test"]
    assert top["similarity"] == near["max"]
    assert top["similarity"] > near["top_pairs"][1]["similarity"]
    similarities = [pair["similarity"] for pair in near["top_pairs"]]
    assert similarities == sorted(similarities, reverse=True)
    listed_above = sum(1 for score in similarities if score >= near["threshold"])
    assert near["above_threshold"] >= listed_above >= 1
    assert near["median"] <= near["p99"] <= near["max"]


def test_integer_near_duplicate_threshold_is_written_as_in_the_toml(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    raw = copy.deepcopy(split_config.source)
    raw["leakage"]["near_duplicate_threshold"] = 1
    config = SplitConfig.from_mapping(raw)
    report = build_temporal_split(
        split_hearings, config, created_at=CREATED_AT, environment=ENVIRONMENT
    ).report
    near = report["leakage"]["near_duplicates"]
    assert type(near["threshold"]) is int
    assert '"threshold": 1,' in json.dumps(near)
    assert near["above_threshold"] == 0
    assert report["config"]["leakage"]["near_duplicate_threshold"] == near["threshold"]


def test_report_without_udvs(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    split = compute_temporal_split(split_hearings, split_config)
    report = build_report(split_hearings, split, split_config, None, environment=ENVIRONMENT)
    assert report["udv_source"] is None
    assert {name: summary["udvs"] for name, summary in report["splits"].items()} == {
        "train": 0,
        "validation": 0,
        "test": 0,
    }
    assert report["splits"]["train"]["udvs_by_tier"] == {}


def test_report_environment_defaults_to_the_running_interpreter(
    split_hearings: list[HearingRecord], split_config: SplitConfig
) -> None:
    split = compute_temporal_split(split_hearings, split_config)
    report = build_report(split_hearings, split, split_config)
    assert list(report["environment"]) == ["python", "platform"]
    assert report["created_at"].endswith("+00:00")


def test_build_temporal_split_shares_one_timestamp(
    split_hearings: list[HearingRecord], split_config: SplitConfig, split_udvs: list[UdvRecord]
) -> None:
    artifacts = build_temporal_split(split_hearings, split_config, split_udvs)
    assert artifacts.manifest["created_at"] == artifacts.report["created_at"]
    assert artifacts.manifest["boundaries"] == artifacts.report["boundaries"]
    assert artifacts.split.groups["validation"] == (8,)
    fixed = build_temporal_split(
        split_hearings, split_config, split_udvs, created_at=CREATED_AT, environment=ENVIRONMENT
    )
    assert fixed.report["created_at"] == "2026-01-02T03:04:05+00:00"
    assert fixed.report["environment"] == ENVIRONMENT
