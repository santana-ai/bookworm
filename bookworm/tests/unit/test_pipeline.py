import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import typer
from conftest import FIXTURES_DIR, MINI_THRESHOLD, StubEncoder
from typer.testing import CliRunner

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    SentenceEncoder,
    UdvConfig,
    build_udvs,
    load_hearings,
)
from bookworm.actors import UdvActorLink, read_actor_speeches, read_udv_actor_links
from bookworm.actors.config import load_actors_config
from bookworm.actors.speeches import collect_actor_speeches
from bookworm.cli import create_app
from bookworm.pipeline import PendingLink, resolve_link, run_pipeline

UDV_CONFIG = "udv_actors.toml"
ACTORS_CONFIG = "actors_mini.toml"
FIXTURE_FILES = ("lds_actors.jsonl", UDV_CONFIG, ACTORS_CONFIG)
ACTOR_FILES = (
    "actors/single.jsonl",
    "actors/multi.jsonl",
    "actors/ambiguous_names.json",
    "actors/stats.json",
    "out/run_actor_links.jsonl",
)
EXPECTED_LINKS = [
    ("udv-10-0-0", "JOAO SILVA", "João Silva", 1, 1),
    ("udv-10-1-0", "MARCOS PEREIRA", "MARCOS PEREIRA", 2, 1),
    ("udv-10-1-1", "MARCOS PEREIRA", "MARCOS PEREIRA", 2, 1),
    ("udv-10-2-0", None, None, 0, 0),
    ("udv-10-3-0", "DEP. HELENA PRADO", "Dep. Helena Prado", 2, 1),
    ("udv-10-4-0", None, None, 1, 0),
    ("udv-11-0-0", "MARCOS PEREIRA", "MARCOS PEREIRA", 1, 1),
    ("udv-11-1-0", "ANA SOUZA", "ANA SOUZA", 2, 1),
]

runner = CliRunner()


def stub_factory(config: UdvConfig, hearings: Sequence[HearingRecord]) -> SentenceEncoder:
    return StubEncoder()


stub_app = create_app(encoder_factory=stub_factory)


def link_rows(links: Sequence[UdvActorLink]) -> list[tuple[Any, ...]]:
    return [
        (link.udv_id, link.actor_key, link.actor, link.matched_turns, link.linked_turns)
        for link in links
    ]


def build(application: typer.Typer, *options: str) -> Any:
    return runner.invoke(
        application, ["build-udvs", "--config", UDV_CONFIG, "--run-name", "run", *options]
    )


@pytest.fixture(scope="module")
def hearings() -> list[HearingRecord]:
    return load_hearings(FIXTURES_DIR / "lds_actors.jsonl")


@pytest.fixture
def actors_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in FIXTURE_FILES:
        shutil.copy(FIXTURES_DIR / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_pipeline_without_actors_matches_build_udvs(hearings: list[HearingRecord]) -> None:
    settings = EvidenceSettings(MINI_THRESHOLD)
    run = run_pipeline(hearings, CachedEncoder(StubEncoder()), settings)
    expected = build_udvs(hearings, CachedEncoder(StubEncoder()), settings)
    assert run.actors is None
    assert run.links == []
    assert run.udv.records == expected.records
    assert run.udv.people == expected.people


def test_pipeline_with_actors_keeps_the_udvs_and_links_them(
    hearings: list[HearingRecord],
) -> None:
    settings = EvidenceSettings(MINI_THRESHOLD)
    config = load_actors_config(FIXTURES_DIR / ACTORS_CONFIG)
    progress: list[tuple[int, int]] = []
    run = run_pipeline(
        hearings,
        CachedEncoder(StubEncoder()),
        settings,
        config,
        on_hearing=lambda number, hearing, records, seconds: progress.append((number, records)),
    )
    expected = build_udvs(hearings, CachedEncoder(StubEncoder()), settings)
    assert run.udv.records == expected.records
    assert progress == [(1, 6), (2, 2)]
    assert run.actors == collect_actor_speeches(hearings, config)
    assert [link.udv_id for link in run.links] == [record.id for record in expected.records]
    assert link_rows(run.links) == EXPECTED_LINKS


def test_unresolved_people_have_no_actor(hearings: list[HearingRecord]) -> None:
    run = run_pipeline(
        hearings,
        CachedEncoder(StubEncoder()),
        EvidenceSettings(MINI_THRESHOLD),
        load_actors_config(FIXTURES_DIR / ACTORS_CONFIG),
    )
    tiers = {record.id: record.tier for record in run.udv.records}
    for link in run.links:
        assert (tiers[link.udv_id] == "person_not_resolved") == (link.matched_turns == 0)


def test_resolve_link_takes_the_most_frequent_key_then_the_first_sorted() -> None:
    names = {"A": "Ana", "B": "Beto"}
    majority = resolve_link(PendingLink("udv-1-0-0", 1, 4, ("B", "A", "B")), names)
    assert (majority.actor_key, majority.actor, majority.linked_turns) == ("B", "Beto", 2)
    tie = resolve_link(PendingLink("udv-1-0-0", 1, 2, ("B", "A")), names)
    assert (tie.actor_key, tie.linked_turns, tie.matched_turns) == ("A", 1, 2)
    empty = resolve_link(PendingLink("udv-1-0-0", 1, 3, ()), names)
    assert (empty.actor_key, empty.actor, empty.linked_turns) == (None, None, 0)


def test_build_without_actors_config_writes_only_the_udv_run(actors_workdir: Path) -> None:
    result = build(stub_app)
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in (actors_workdir / "out").iterdir()) == [
        "run.jsonl",
        "run_coverage.json",
    ]
    assert not (actors_workdir / "actors").exists()
    assert "actors" not in json.loads(result.stdout)


def test_build_with_actors_config_writes_every_file(
    actors_workdir: Path, hearings: list[HearingRecord]
) -> None:
    plain = build(stub_app)
    assert plain.exit_code == 0, plain.output
    plain_run = (actors_workdir / "out" / "run.jsonl").read_bytes()
    result = build(stub_app, "--actors-config", ACTORS_CONFIG, "--overwrite")
    assert result.exit_code == 0, result.output
    assert (actors_workdir / "out" / "run.jsonl").read_bytes() == plain_run
    for name in ACTOR_FILES:
        assert (actors_workdir / name).is_file(), name
    summary = json.loads(result.stdout)
    assert summary["actors"] == 6
    assert summary["actor_turns_kept"] == 9
    assert summary["udvs_linked_to_actor"] == 6
    links = read_udv_actor_links(actors_workdir / "out" / "run_actor_links.jsonl")
    assert link_rows(links) == EXPECTED_LINKS
    config = load_actors_config(actors_workdir / ACTORS_CONFIG)
    speeches = collect_actor_speeches(hearings, config)
    assert read_actor_speeches(actors_workdir / "actors" / "multi.jsonl") == (
        speeches.multi_hearing
    )
    stats = json.loads((actors_workdir / "actors" / "stats.json").read_text("utf-8"))
    assert stats == speeches.stats


def test_build_refuses_an_actors_config_for_another_lds(actors_workdir: Path) -> None:
    path = actors_workdir / ACTORS_CONFIG
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("bb3b7cdd", "00000000"), encoding="utf-8")
    result = build(stub_app, "--actors-config", ACTORS_CONFIG)
    assert result.exit_code == 2
    assert "differs from the UDV config" in result.stderr
    assert not (actors_workdir / "out").exists()


def test_build_refuses_to_replace_existing_actor_files(actors_workdir: Path) -> None:
    (actors_workdir / "actors").mkdir()
    (actors_workdir / "actors" / "stats.json").write_text("{}", encoding="utf-8")
    result = build(stub_app, "--actors-config", ACTORS_CONFIG)
    assert result.exit_code == 2
    assert "actors/stats.json: run file already exists" in result.stderr
    assert not (actors_workdir / "out").exists()


def test_build_reports_a_missing_actors_config(actors_workdir: Path) -> None:
    result = build(stub_app, "--actors-config", "missing.toml")
    assert result.exit_code == 2
    assert "missing.toml: config file not found" in result.stderr


def test_build_reports_unused_merges_for_a_partial_selection(actors_workdir: Path) -> None:
    result = build(stub_app, "--actors-config", ACTORS_CONFIG, "--ids", "10")
    assert result.exit_code == 2
    assert "matched no kept turn" in result.stderr
