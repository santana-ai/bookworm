import dataclasses
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import (
    FIXTURES_DIR,
    cache_only_encoder,
    directory_state,
    import_challenge_module,
    udv_artifact_path,
)

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    UdvRecord,
    load_hearings,
    load_jsonl,
    load_udv_jsonl,
    resolve_hearing_people,
    split_into_turns,
)
from bookworm.actors import ActorSpeechRecord, UdvActorLink
from bookworm.actors.config import ActorsConfig, load_actors_config
from bookworm.actors.speeches import (
    AMBIGUITY_CRITERION,
    ActorCollector,
    ActorSpeeches,
    collect_actor_speeches,
    write_actor_outputs,
)
from bookworm.pipeline import PendingLink, pending_links, resolve_link, run_pipeline
from bookworm.udv.verify import coverage_hearings

REFERENCE_MODULE = "utils.build_actor_speeches"
ACTORS_CONFIG_NAME = "hearing_actors.toml"
STATS_NAME = "actor_speeches_stats.json"
AMBIGUOUS_NAME = "ambiguous_names.json"
UDV_RUN = "udv_v1"


@pytest.fixture(scope="module")
def reference(challenge_dir: Path) -> ModuleType:
    return import_challenge_module(challenge_dir, REFERENCE_MODULE)


@pytest.fixture(scope="module")
def challenge_actors_config_path(challenge_dir: Path) -> Path:
    path = challenge_dir / "configs" / ACTORS_CONFIG_NAME
    if not path.is_file():
        pytest.skip(f"actors config not found at {path}")
    return path


@pytest.fixture(scope="module")
def actors_config(challenge_actors_config_path: Path) -> ActorsConfig:
    return load_actors_config(challenge_actors_config_path)


@pytest.fixture(scope="module")
def hearing_actors_dir(artifacts_dir: Path) -> Path:
    path = artifacts_dir / "hearing_actors"
    if not path.is_dir():
        pytest.skip(f"hearing actor artifacts not found at {path}")
    return path


@pytest.fixture(scope="module")
def speeches(lds_hearings: list[HearingRecord], actors_config: ActorsConfig) -> ActorSpeeches:
    return collect_actor_speeches(lds_hearings, actors_config)


def artifact_text(directory: Path, name: str) -> str:
    path = directory / name
    if not path.is_file():
        pytest.skip(f"artifact not found at {path}")
    return path.read_text(encoding="utf-8")


def reference_records(
    reference: ModuleType, config_path: Path, hearings: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    config = reference.load_config(config_path)
    actors, _, usage = reference.collect_actors(hearings, config)
    reference.check_merge_usage(config, usage)
    return {key: reference.actor_record(actor) for key, actor in sorted(actors.items())}


def assert_same_records(
    records: dict[str, ActorSpeechRecord], expected: dict[str, dict[str, Any]]
) -> None:
    assert list(records) == list(expected)
    for key, record in records.items():
        assert record.model_dump() == expected[key], key


def redirected_paths(directory: Path) -> dict[str, Path]:
    return {
        "single_hearing_path": directory / "single.jsonl",
        "multi_hearing_path": directory / "multi.jsonl",
        "ambiguous_names_path": directory / "ambiguous_names.json",
        "stats_path": directory / "stats.json",
    }


def test_fixture_speeches_match_the_reference_script(reference: ModuleType) -> None:
    config_path = FIXTURES_DIR / "actors_mini.toml"
    hearings_path = FIXTURES_DIR / "lds_actors.jsonl"
    speeches = collect_actor_speeches(load_hearings(hearings_path), load_actors_config(config_path))
    expected = reference_records(reference, config_path, load_jsonl(hearings_path))
    assert_same_records(speeches.records, expected)
    expected_pairs = reference.ambiguous_name_pairs(expected)
    assert speeches.pairs == expected_pairs


@pytest.mark.dataset
def test_stats_match_the_versioned_artifact(
    speeches: ActorSpeeches, hearing_actors_dir: Path
) -> None:
    text = artifact_text(hearing_actors_dir, STATS_NAME)
    assert speeches.stats == json.loads(text)
    assert json.dumps(speeches.stats, ensure_ascii=False, indent=2) == text


@pytest.mark.dataset
def test_ambiguous_names_match_the_versioned_artifact(
    speeches: ActorSpeeches, hearing_actors_dir: Path
) -> None:
    text = artifact_text(hearing_actors_dir, AMBIGUOUS_NAME)
    payload = {"criterion": AMBIGUITY_CRITERION, "pairs": speeches.pairs}
    assert json.dumps(payload, ensure_ascii=False, indent=2) == text


@pytest.mark.dataset
def test_speech_records_match_the_reference_script(
    speeches: ActorSpeeches,
    reference: ModuleType,
    challenge_actors_config_path: Path,
    lds_path: Path,
) -> None:
    expected = reference_records(reference, challenge_actors_config_path, load_jsonl(lds_path))
    assert_same_records(speeches.records, expected)
    assert len(speeches.multi_hearing) == 301
    assert len(speeches.single_hearing) == 1550


@pytest.mark.dataset
def test_written_files_are_byte_identical_to_the_reference_script(
    lds_hearings: list[HearingRecord],
    reference: ModuleType,
    actors_config: ActorsConfig,
    challenge_actors_config_path: Path,
    lds_path: Path,
    tmp_path: Path,
) -> None:
    paths = redirected_paths(tmp_path)
    reference_config = dataclasses.replace(
        reference.load_config(challenge_actors_config_path), lds_path=lds_path, **paths
    )
    reference.build(reference_config)
    reference_bytes = {name: path.read_bytes() for name, path in paths.items()}
    for path in paths.values():
        path.unlink()
    library_config = actors_config.model_copy(update={"lds_path": lds_path, **paths})
    write_actor_outputs(collect_actor_speeches(lds_hearings, library_config))
    for name, path in paths.items():
        assert path.read_bytes() == reference_bytes[name], name


def encoder_free_links(
    hearings: list[HearingRecord], config: ActorsConfig
) -> tuple[ActorSpeeches, list[UdvActorLink]]:
    collector = ActorCollector(config)
    pending: list[PendingLink] = []
    for hearing in hearings:
        turns = split_into_turns(hearing.transcricao)
        people = resolve_hearing_people(hearing, turns)
        kept = collector.add_hearing(hearing.id, hearing.transcricao, turns)
        pending.extend(pending_links(hearing.id, people, kept))
    speeches = collector.finish()
    names = speeches.display_names()
    return speeches, [resolve_link(link, names) for link in pending]


@pytest.fixture(scope="module")
def udv_v1_records(udv_artifacts_dir: Path) -> list[UdvRecord]:
    return load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, f"{UDV_RUN}.jsonl"))


@pytest.mark.dataset
def test_links_cover_every_udv_of_the_versioned_run(
    lds_hearings: list[HearingRecord],
    actors_config: ActorsConfig,
    udv_v1_records: list[UdvRecord],
) -> None:
    speeches, links = encoder_free_links(lds_hearings, actors_config)
    assert [link.udv_id for link in links] == [record.id for record in udv_v1_records]
    names = set(speeches.display_names().values())
    tiers = {record.id: record.tier for record in udv_v1_records}
    for link in links:
        assert (tiers[link.udv_id] == "person_not_resolved") == (link.matched_turns == 0)
        assert link.linked_turns <= link.matched_turns
        assert (link.actor_key is None) == (link.linked_turns == 0)
        assert link.actor_key is None or speeches.display_names()[link.actor_key] == link.actor
        assert link.actor is None or link.actor in names
    multi_hearing_actors = {record.actor for record in speeches.multi_hearing}
    assert len(links) == 2203
    assert sum(link.matched_turns == 0 for link in links) == 90
    assert sum(link.matched_turns > 0 and link.actor is None for link in links) == 9
    assert sum(link.actor is not None for link in links) == 2104
    assert sum(0 < link.linked_turns < link.matched_turns for link in links) == 355
    assert sum(link.actor in multi_hearing_actors for link in links) == 726
    assert len({link.actor for link in links if link.actor is not None}) == 827


@pytest.mark.dataset
def test_one_pass_rebuild_reproduces_udvs_and_actor_files(
    lds_hearings: list[HearingRecord],
    actors_config: ActorsConfig,
    udv_artifacts_dir: Path,
    embedding_cache_dir: Path,
    hearing_actors_dir: Path,
) -> None:
    coverage_path = udv_artifact_path(udv_artifacts_dir, f"{UDV_RUN}_coverage.json")
    coverage: dict[str, Any] = json.loads(coverage_path.read_text(encoding="utf-8"))
    hearings = coverage_hearings(coverage, lds_hearings)
    if len(hearings) != len(lds_hearings):
        pytest.skip(f"{UDV_RUN} does not cover the whole LDS")
    before = directory_state(embedding_cache_dir)
    run = run_pipeline(
        hearings,
        CachedEncoder(cache_only_encoder(coverage), embedding_cache_dir, read_only=True),
        EvidenceSettings(coverage["config"]["evidence"]["embedding_threshold"]),
        actors_config,
    )
    assert directory_state(embedding_cache_dir) == before
    artifact = udv_artifact_path(udv_artifacts_dir, f"{UDV_RUN}.jsonl").read_text(encoding="utf-8")
    assert "".join(record.to_json_line() + "\n" for record in run.udv.records) == artifact
    assert run.actors is not None
    assert run.actors.stats == json.loads(artifact_text(hearing_actors_dir, STATS_NAME))
    _, links = encoder_free_links(lds_hearings, actors_config)
    assert run.links == links
