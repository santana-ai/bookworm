import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from conftest import udv_artifact_path

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    TfidfEncoder,
    UdvRecord,
    UdvRun,
    build_udvs,
    load_udv_jsonl,
    summarize_run,
    verify_udv_run,
)
from bookworm.udv.build import udv_corpus
from bookworm.udv.verify import coverage_hearing_ids

pytestmark = pytest.mark.dataset

MAX_FEATURES = 4096
HEARINGS = 20
RUN_NAME = "tfidf_first20"
TFIDF_CHARACTERIZATION = {
    "by_tier": {
        "quote_found": 40,
        "semantic_match_high": 85,
        "semantic_match_weak": 130,
        "no_evidence": 0,
        "person_not_resolved": 13,
    },
    "evidence_support_types": {
        "direct_quote": 40,
        "semantic_with_short_quote": 12,
        "semantic_similarity": 203,
    },
    "evidence_offsets": {"total": 255, "located": 255},
}


@pytest.fixture(scope="module")
def udv_v1_coverage(udv_artifacts_dir: Path) -> dict[str, Any]:
    path = udv_artifact_path(udv_artifacts_dir, "udv_v1_coverage.json")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module")
def first_ids(udv_v1_coverage: dict[str, Any]) -> list[int]:
    return coverage_hearing_ids(udv_v1_coverage)[:HEARINGS]


@pytest.fixture(scope="module")
def udv_v1_subset(udv_artifacts_dir: Path, first_ids: list[int]) -> list[UdvRecord]:
    wanted = set(first_ids)
    records = load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, "udv_v1.jsonl"))
    return [record for record in records if record.hearing_id in wanted]


@pytest.fixture(scope="module")
def tfidf_run(
    udv_v1_coverage: dict[str, Any], first_ids: list[int], lds_hearings: list[HearingRecord]
) -> UdvRun:
    wanted = set(first_ids)
    hearings = [hearing for hearing in lds_hearings if hearing.id in wanted]
    encoder = TfidfEncoder.fit(udv_corpus(hearings), max_features=MAX_FEATURES)
    threshold = udv_v1_coverage["config"]["evidence"]["embedding_threshold"]
    return build_udvs(hearings, CachedEncoder(encoder), EvidenceSettings(threshold))


@pytest.fixture(scope="module")
def tfidf_summary(tfidf_run: UdvRun, udv_v1_coverage: dict[str, Any]) -> dict[str, Any]:
    return summarize_run(
        RUN_NAME,
        tfidf_run.records,
        tfidf_run.people,
        tfidf_run.hearings,
        encoder_runtime={},
        hearing_seconds=tfidf_run.hearing_seconds,
        config_source=udv_v1_coverage["config"],
    )


def test_tfidf_run_passes_every_verification_check(
    tfidf_run: UdvRun, tfidf_summary: dict[str, Any], lds_hearings: list[HearingRecord]
) -> None:
    verification = verify_udv_run(RUN_NAME, tfidf_run.records, tfidf_summary, lds_hearings)
    assert verification.problems == {}


def test_encoder_independent_fields_match_udv_v1(
    tfidf_run: UdvRun, udv_v1_subset: list[UdvRecord]
) -> None:
    assert [record.id for record in tfidf_run.records] == [record.id for record in udv_v1_subset]
    quotes = [record for record in tfidf_run.records if record.tier == "quote_found"]
    published_quotes = [record for record in udv_v1_subset if record.tier == "quote_found"]
    assert [record.evidence for record in quotes] == [
        record.evidence for record in published_quotes
    ]

    def family(record: UdvRecord) -> str:
        return "semantic" if record.provenance == "model" else record.tier

    assert Counter(map(family, tfidf_run.records)) == Counter(map(family, udv_v1_subset))


def test_tfidf_counts_are_stable(tfidf_summary: dict[str, Any]) -> None:
    assert tfidf_summary["people"] == {"total": 108, "resolved": 102}
    assert tfidf_summary["opinions"]["by_tier"] == TFIDF_CHARACTERIZATION["by_tier"]
    assert (
        tfidf_summary["evidence_support_types"] == TFIDF_CHARACTERIZATION["evidence_support_types"]
    )
    assert tfidf_summary["evidence_offsets"] == TFIDF_CHARACTERIZATION["evidence_offsets"]
