import json
from pathlib import Path
from typing import Any

import pytest
from conftest import cache_only_encoder, directory_state, udv_artifact_path

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    UdvRun,
    build_udvs,
    summarize_run,
)
from bookworm.udv.verify import coverage_hearings

pytestmark = pytest.mark.dataset

RUN_NAMES = ("udv_v1", "udv_v1_pre")
RECOUNTED_SECTIONS = (
    "hearings",
    "people",
    "opinions",
    "evidence_offsets",
    "evidence_support_types",
    "pipeline",
)


@pytest.fixture(scope="module", params=RUN_NAMES)
def run_name(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


@pytest.fixture(scope="module")
def coverage(run_name: str, udv_artifacts_dir: Path) -> dict[str, Any]:
    path = udv_artifact_path(udv_artifacts_dir, f"{run_name}_coverage.json")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module")
def rebuilt(
    coverage: dict[str, Any], lds_hearings: list[HearingRecord], embedding_cache_dir: Path
) -> UdvRun:
    before = directory_state(embedding_cache_dir)
    run = build_udvs(
        coverage_hearings(coverage, lds_hearings),
        CachedEncoder(cache_only_encoder(coverage), embedding_cache_dir, read_only=True),
        EvidenceSettings(coverage["config"]["evidence"]["embedding_threshold"]),
    )
    assert directory_state(embedding_cache_dir) == before
    return run


def test_cached_rebuild_reproduces_the_artifact_bytes(
    rebuilt: UdvRun, run_name: str, udv_artifacts_dir: Path
) -> None:
    artifact = udv_artifact_path(udv_artifacts_dir, f"{run_name}.jsonl").read_text(encoding="utf-8")
    rebuilt_text = "".join(record.to_json_line() + "\n" for record in rebuilt.records)
    assert rebuilt_text == artifact


def test_cached_rebuild_reproduces_the_coverage_sections(
    rebuilt: UdvRun, coverage: dict[str, Any], run_name: str
) -> None:
    summary = summarize_run(
        run_name,
        rebuilt.records,
        rebuilt.people,
        rebuilt.hearings,
        encoder_runtime=coverage["encoder_runtime"],
        hearing_seconds=rebuilt.hearing_seconds,
        config_source=coverage["config"],
    )
    for section in RECOUNTED_SECTIONS:
        assert summary[section] == coverage[section], section
    assert list(summary) == list(coverage)
