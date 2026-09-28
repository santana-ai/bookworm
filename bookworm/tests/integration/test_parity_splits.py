import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from conftest import import_reference_module
from typer.testing import CliRunner

from bookworm import (
    SPLIT_NAMES,
    HearingRecord,
    SplitArtifacts,
    SplitConfig,
    UdvRecord,
    build_temporal_split,
    check_article_date,
    compute_temporal_split,
    load_jsonl,
    load_split_config,
    load_udv_jsonl,
    summarize_date_extraction,
    verify_split_run,
)
from bookworm.cli import app
from bookworm.data.io import JsonObject
from bookworm.data.splits import choose_boundary, cut_candidates, dates_by_hearing

pytestmark = pytest.mark.dataset

Mutation = Callable[[JsonObject, JsonObject], None]
EXPECTED_HEARINGS = {"train": 144, "validation": 32, "test": 30}
EXPECTED_UDVS = {"train": 1536, "validation": 308, "test": 359}
EXPECTED_BOUNDARIES = {
    "train_end": "2023-11-13",
    "validation_start": "2023-11-21",
    "validation_end": "2023-12-20",
    "test_start": "2024-03-05",
    "train_gap_days": 8,
    "test_gap_days": 76,
    "train_fraction_reached": 0.699,
    "validation_end_fraction_reached": 0.8544,
}


def serialize(payload: JsonObject) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@pytest.fixture(scope="module")
def config(experiments_split_config_path: Path) -> SplitConfig:
    return load_split_config(experiments_split_config_path)


@pytest.fixture(scope="module")
def udvs(udv_artifacts_dir: Path) -> list[UdvRecord]:
    return load_udv_jsonl(udv_artifacts_dir / "udv_v0.jsonl")


@pytest.fixture(scope="module")
def manifest_text(split_artifacts_dir: Path) -> str:
    return (split_artifacts_dir / "temporal_v1.json").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def report_text(split_artifacts_dir: Path) -> str:
    return (split_artifacts_dir / "temporal_v1_report.json").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def published_manifest(manifest_text: str) -> JsonObject:
    payload: JsonObject = json.loads(manifest_text)
    return payload


@pytest.fixture(scope="module")
def published_report(report_text: str) -> JsonObject:
    payload: JsonObject = json.loads(report_text)
    return payload


@pytest.fixture(scope="module")
def rebuilt(
    lds_hearings: list[HearingRecord],
    config: SplitConfig,
    udvs: list[UdvRecord],
    published_report: JsonObject,
) -> SplitArtifacts:
    return build_temporal_split(
        lds_hearings, config, udvs, environment=published_report["environment"]
    )


def test_manifest_equals_the_artifact_except_created_at(
    rebuilt: SplitArtifacts, published_manifest: JsonObject, manifest_text: str
) -> None:
    differing = [
        key
        for key in published_manifest
        if json.loads(json.dumps(rebuilt.manifest.get(key))) != published_manifest[key]
    ]
    assert differing == ["created_at"]
    assert list(rebuilt.manifest) == list(published_manifest)
    aligned = {**rebuilt.manifest, "created_at": published_manifest["created_at"]}
    assert serialize(aligned) == manifest_text


def test_report_equals_the_artifact_except_created_at_and_environment(
    lds_hearings: list[HearingRecord],
    config: SplitConfig,
    udvs: list[UdvRecord],
    published_report: JsonObject,
    report_text: str,
) -> None:
    fresh = build_temporal_split(lds_hearings, config, udvs).report
    differing = {
        key
        for key in published_report
        if json.loads(json.dumps(fresh.get(key))) != published_report[key]
    }
    assert differing <= {"created_at", "environment"}
    assert "created_at" in differing
    assert list(fresh) == list(published_report)
    aligned = {
        **fresh,
        "created_at": published_report["created_at"],
        "environment": published_report["environment"],
    }
    assert serialize(aligned) == report_text


def test_published_split_sizes(rebuilt: SplitArtifacts) -> None:
    assert {name: len(rebuilt.manifest[name]) for name in SPLIT_NAMES} == EXPECTED_HEARINGS
    assert {name: rebuilt.report["splits"][name]["udvs"] for name in SPLIT_NAMES} == EXPECTED_UDVS
    assert rebuilt.manifest["boundaries"] == EXPECTED_BOUNDARIES
    assert rebuilt.report["cut_candidates_considered"] == 65


def test_published_artifacts_verify_clean(
    published_manifest: JsonObject,
    published_report: JsonObject,
    lds_hearings: list[HearingRecord],
    udvs: list[UdvRecord],
    config: SplitConfig,
) -> None:
    verification = verify_split_run(
        published_manifest, published_report, lds_hearings, udvs, config
    )
    assert verification.problems == []
    assert verification.hearings == EXPECTED_HEARINGS
    assert verification.udvs == EXPECTED_UDVS
    assert verification.extraction.max_lag_days == 1
    assert verification.extraction.confirmed_by_at_least_one_mention == 155


def hearing_totals(hearing: HearingRecord, udvs: list[UdvRecord]) -> tuple[int, int, int]:
    people = len(hearing.metadados.envolvidos)
    opinions = sum(len(person.opinioes) for person in hearing.metadados.envolvidos)
    return people, opinions, sum(1 for udv in udvs if udv.hearing_id == hearing.id)


def test_injected_defects_on_the_published_manifest(
    published_manifest: JsonObject,
    published_report: JsonObject,
    lds_hearings: list[HearingRecord],
    udvs: list[UdvRecord],
    config: SplitConfig,
) -> None:
    by_id = {hearing.id: hearing for hearing in lds_hearings}
    moved = published_manifest["train"][0]
    removed = published_manifest["test"][-1]
    people, opinions, removed_udvs = hearing_totals(by_id[removed], udvs)

    def run(mutation: Mutation) -> list[str]:
        manifest = copy.deepcopy(published_manifest)
        report = copy.deepcopy(published_report)
        mutation(manifest, report)
        return verify_split_run(manifest, report, lds_hearings, udvs, config).problems

    def swap(manifest: JsonObject, report: JsonObject) -> None:
        manifest["train"].remove(moved)
        manifest["test"].append(moved)

    def remove(manifest: JsonObject, report: JsonObject) -> None:
        manifest["test"].remove(removed)

    def alter_date(manifest: JsonObject, report: JsonObject) -> None:
        manifest["article_dates"]["1"] = "2024-04-15"

    def alter_boundary(manifest: JsonObject, report: JsonObject) -> None:
        manifest["boundaries"]["test_start"] = "2024-03-06"

    def alter_counter(manifest: JsonObject, report: JsonObject) -> None:
        report["splits"]["test"]["udvs"] = 360

    swapped = run(swap)
    assert "not_chronological:validation->test" in swapped
    assert f"assignment_mismatch:{moved}" in swapped
    assert "report.train.hearings: reported 144, recomputed 143" in swapped
    assert "report.test.hearings: reported 30, recomputed 31" in swapped
    assert run(remove) == [
        f"unassigned_hearing:{removed}",
        "report.test.hearings: reported 30, recomputed 29",
        f"report.test.people: reported 159, recomputed {159 - people}",
        f"report.test.opinions: reported 359, recomputed {359 - opinions}",
        f"report.test.udvs: reported 359, recomputed {359 - removed_udvs}",
    ]
    assert run(alter_date) == ["article_date_mismatch:1"]
    assert run(alter_boundary) == [
        "boundary_mismatch:test_start: manifest 2024-03-06, recomputed 2024-03-05"
    ]
    assert run(alter_counter) == [
        "report.test.udvs: reported 360, recomputed 359",
        "report.udvs_total_mismatch",
    ]


def test_cli_rebuilds_the_artifacts_from_a_consumer_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiments_split_config_path: Path,
    config: SplitConfig,
    lds_path: Path,
    udv_artifacts_dir: Path,
    published_manifest: JsonObject,
    published_report: JsonObject,
    manifest_text: str,
    report_text: str,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "splits.toml").write_bytes(experiments_split_config_path.read_bytes())
    (tmp_path / config.lds_path).parent.mkdir(parents=True)
    (tmp_path / config.lds_path).symlink_to(lds_path.resolve())
    (tmp_path / config.udv_path).parent.mkdir(parents=True)
    (tmp_path / config.udv_path).symlink_to((udv_artifacts_dir / "udv_v0.jsonl").resolve())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    built = runner.invoke(app, ["build-splits", "--config", "configs/splits.toml"])
    assert built.exit_code == 0, built.output
    manifest = json.loads((tmp_path / config.manifest_path).read_text(encoding="utf-8"))
    report = json.loads((tmp_path / config.report_path).read_text(encoding="utf-8"))
    manifest["created_at"] = published_manifest["created_at"]
    report["created_at"] = published_report["created_at"]
    report["environment"] = published_report["environment"]
    assert serialize(manifest) == manifest_text
    assert serialize(report) == report_text
    verified = runner.invoke(app, ["verify-splits", "--config", "configs/splits.toml"])
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["problems"] == []


@pytest.fixture(scope="module")
def reference_dates(experiments_dir: Path) -> ModuleType:
    return import_reference_module(experiments_dir, "experiments.data.legacy_splits")


@pytest.fixture(scope="module")
def reference_splits(experiments_dir: Path) -> ModuleType:
    return import_reference_module(experiments_dir, "experiments.data.legacy_splits")


@pytest.fixture(scope="module")
def raw_records(lds_path: Path) -> list[JsonObject]:
    return load_jsonl(lds_path)


def test_date_checks_match_the_reference_on_every_hearing(
    reference_dates: ModuleType,
    lds_hearings: list[HearingRecord],
    raw_records: list[JsonObject],
) -> None:
    mismatched = [
        hearing.id
        for hearing, raw in zip(lds_hearings, raw_records, strict=True)
        if check_article_date(hearing.materia).to_dict()
        != reference_dates.check_article_date(raw["materia"])
    ]
    assert mismatched == []
    mentions = sum(len(check_article_date(h.materia).mentions) for h in lds_hearings)
    assert mentions > 0


def test_split_steps_match_the_reference(
    reference_splits: ModuleType,
    lds_hearings: list[HearingRecord],
    raw_records: list[JsonObject],
    config: SplitConfig,
) -> None:
    dates = dates_by_hearing(lds_hearings)
    assert dates == reference_splits.dates_by_hearing(raw_records)
    ours = cut_candidates(dates, config.min_boundary_gap_days)
    theirs = reference_splits.cut_candidates(dates, config.min_boundary_gap_days)
    assert [
        {
            "date": c.day,
            "next_date": c.next_day,
            "gap_days": c.gap_days,
            "cumulative": c.cumulative,
            "fraction": c.fraction,
        }
        for c in ours
    ] == theirs
    for target in (0.5, config.train_fraction, 0.85, 0.95):
        assert (
            choose_boundary(ours, target, None).day
            == (reference_splits.choose_boundary(theirs, target, None)["date"])
        )
    split = compute_temporal_split(lds_hearings, config)
    reference_groups = reference_splits.assign_splits(
        dates, split.boundaries.train_end, split.boundaries.validation_end
    )
    assert {name: list(ids) for name, ids in split.groups.items()} == reference_groups
    assert summarize_date_extraction(
        lds_hearings
    ).to_dict() == reference_splits.summarize_date_extraction(raw_records)
