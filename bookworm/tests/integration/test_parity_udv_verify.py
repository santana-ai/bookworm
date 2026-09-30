import json
from pathlib import Path
from typing import Any

import pytest
from conftest import udv_artifact_path

from bookworm import (
    HearingRecord,
    UdvRecord,
    UdvVerification,
    compare_with_baseline,
    load_udv_jsonl,
    verify_udv_run,
)

pytestmark = pytest.mark.dataset

PEOPLE = (1065, 1020)
RECORDS = 2203
SUPPORT_TYPES = {
    "direct_quote": 277,
    "semantic_with_short_quote": 111,
    "semantic_similarity": 1717,
}
EXPECTED: dict[str, dict[str, Any]] = {
    "udv_v1": {
        "threshold": 0.45,
        "by_tier": {
            "quote_found": 277,
            "semantic_match_high": 1785,
            "semantic_match_weak": 43,
            "no_evidence": 8,
            "person_not_resolved": 90,
        },
    },
    "udv_v1_pre": {
        "threshold": 0.47,
        "by_tier": {
            "quote_found": 277,
            "semantic_match_high": 1776,
            "semantic_match_weak": 52,
            "no_evidence": 8,
            "person_not_resolved": 90,
        },
    },
}
NEW_QUOTE_FOUND = [
    "udv-1-6-0",
    "udv-6-0-0",
    "udv-39-3-0",
    "udv-63-0-1",
    "udv-63-1-1",
    "udv-73-4-1",
    "udv-119-0-3",
    "udv-128-0-0",
    "udv-128-1-0",
    "udv-128-3-2",
    "udv-154-0-0",
    "udv-154-1-0",
    "udv-154-1-1",
    "udv-154-2-1",
    "udv-154-3-0",
    "udv-183-1-1",
    "udv-197-1-0",
]
SHORT_QUOTE_CHANGES = [
    "udv-1-6-0",
    "udv-6-0-0",
    "udv-35-0-2",
    "udv-54-2-1",
    "udv-63-2-2",
    "udv-70-3-0",
    "udv-71-4-0",
    "udv-73-0-2",
    "udv-79-4-0",
    "udv-99-0-2",
    "udv-103-3-0",
    "udv-119-0-1",
    "udv-119-0-2",
    "udv-171-2-0",
    "udv-200-0-0",
]
UDV_V0_PROBLEMS = {
    "evidence_not_in_single_actor_turn": ["udv-1-1-2"],
    "evidence_not_person_sentence": ["udv-1-1-2", "udv-45-0-2", "udv-169-2-0", "udv-170-4-0"],
    "evidence_offsets_missing": ["udv-1-1-2"],
    "semantic_but_trusted_quote_findable": NEW_QUOTE_FOUND,
    "short_quote_support_mismatch": SHORT_QUOTE_CHANGES,
    "short_quote_prefix_mismatch": SHORT_QUOTE_CHANGES,
    "quote_prefix_mismatch": ["udv-24-1-0"],
    "quote_text_mismatch": ["udv-48-2-0", "udv-65-1-0", "udv-82-2-1"],
    "quote_turn_mismatch": ["udv-65-1-0"],
}


def read_artifact_lines(udv_artifacts_dir: Path, run_name: str) -> list[str]:
    text = udv_artifact_path(udv_artifacts_dir, f"{run_name}.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    return text.removesuffix("\n").split("\n")


def read_coverage(udv_artifacts_dir: Path, run_name: str) -> dict[str, Any]:
    path = udv_artifact_path(udv_artifacts_dir, f"{run_name}_coverage.json")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def run_name(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


@pytest.fixture(scope="module")
def artifact_lines(run_name: str, udv_artifacts_dir: Path) -> list[str]:
    return read_artifact_lines(udv_artifacts_dir, run_name)


@pytest.fixture(scope="module")
def coverage(run_name: str, udv_artifacts_dir: Path) -> dict[str, Any]:
    return read_coverage(udv_artifacts_dir, run_name)


@pytest.fixture(scope="module")
def records(artifact_lines: list[str]) -> list[UdvRecord]:
    return [UdvRecord.from_json_line(line) for line in artifact_lines]


@pytest.fixture(scope="module")
def verification(
    run_name: str,
    records: list[UdvRecord],
    coverage: dict[str, Any],
    lds_hearings: list[HearingRecord],
) -> UdvVerification:
    return verify_udv_run(run_name, records, coverage, lds_hearings)


def test_every_line_round_trips_byte_for_byte(
    records: list[UdvRecord], artifact_lines: list[str]
) -> None:
    assert len(records) == len(artifact_lines)
    mismatched = [
        record.id
        for record, line in zip(records, artifact_lines, strict=True)
        if record.to_json_line() != line
    ]
    assert mismatched == []


def test_verification_finds_no_problem(verification: UdvVerification) -> None:
    assert verification.problems == {}
    assert verification.ok


def test_counters_equal_the_coverage_file(
    verification: UdvVerification, coverage: dict[str, Any]
) -> None:
    assert verification.records == coverage["opinions"]["total"]
    assert verification.expected_records == coverage["opinions"]["total"]
    assert verification.people_total == coverage["people"]["total"]
    assert verification.people_resolved == coverage["people"]["resolved"]
    assert verification.by_tier == coverage["opinions"]["by_tier"]
    assert verification.by_support_type == coverage["evidence_support_types"]


def test_counters_equal_the_published_numbers(
    verification: UdvVerification, coverage: dict[str, Any], run_name: str
) -> None:
    expected = EXPECTED[run_name]
    assert coverage["config"]["evidence"]["embedding_threshold"] == expected["threshold"]
    assert (verification.people_total, verification.people_resolved) == PEOPLE
    assert verification.records == RECORDS
    assert verification.by_tier == expected["by_tier"]
    assert verification.by_support_type == SUPPORT_TYPES


@pytest.fixture(scope="module")
def udv_v0_records(udv_artifacts_dir: Path) -> list[UdvRecord]:
    return load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, "udv_v0.jsonl"))


def test_udv_v0_fails_exactly_on_the_records_the_current_rules_change(
    udv_v0_records: list[UdvRecord],
    udv_artifacts_dir: Path,
    lds_hearings: list[HearingRecord],
) -> None:
    coverage = read_coverage(udv_artifacts_dir, "udv_v0")
    verification = verify_udv_run("udv_v0", udv_v0_records, coverage, lds_hearings)
    assert verification.problems == UDV_V0_PROBLEMS


def test_udv_v1_differs_from_udv_v0_in_42_evidences(
    udv_v0_records: list[UdvRecord], udv_artifacts_dir: Path
) -> None:
    current = load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, "udv_v1.jsonl"))
    previous = load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, "udv_v1_pre.jsonl"))
    assert compare_with_baseline(previous, udv_v0_records) == {
        "baseline_records": RECORDS,
        "evidence_changed": 42,
        "tier_moves": {"semantic_match_high->quote_found": len(NEW_QUOTE_FOUND)},
        "extra_in_run": 0,
    }
    assert compare_with_baseline(current, previous) == {
        "baseline_records": RECORDS,
        "evidence_changed": 0,
        "tier_moves": {"semantic_match_weak->semantic_match_high": 9},
        "extra_in_run": 0,
    }
    old_tiers = {record.id: record.tier for record in udv_v0_records}
    moved = [
        record.id
        for record in current
        if record.tier == "quote_found" and old_tiers[record.id] != "quote_found"
    ]
    assert moved == NEW_QUOTE_FOUND
