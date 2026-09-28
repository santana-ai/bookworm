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
    ConfigError,
    EvidenceSettings,
    HearingRecord,
    SentenceEncoder,
    UdvConfig,
    UdvRecord,
    build_udvs,
    export_site,
    load_hearings,
    pipeline_description,
)
from bookworm.actors.config import load_actors_config
from bookworm.actors.schemas import ActorSpeechRecord
from bookworm.actors.speeches import collect_actor_speeches
from bookworm.cli import create_app
from bookworm.profiles import ProfileRecord, write_profiles
from bookworm.profiles.site import (
    ACTORS_FILE_NAME,
    PROFILES_DIR_NAME,
    SAME_SENTENCE_RULE,
    SIMILAR_TEXT_RULE,
    UNTITLED_SECTION,
    ProfileSection,
    ProfileSiteBuilder,
    actor_slug,
    assign_slugs,
    parse_profile,
    trim_passage,
)

JsonObject = dict[str, Any]

UDV_CONFIG = "udv_actors.toml"
ACTORS_CONFIG = "actors_mini.toml"
FIXTURE_FILES = ("lds_actors.jsonl", UDV_CONFIG, ACTORS_CONFIG)
PIPELINE = pipeline_description()
JOAO_PROFILE = (
    "## Posições\n"
    "- Defende reforçar a fiscalização das cooperativas antes de qualquer mudança no texto.\n"
    "- Tema sem fala correspondente: orçamento militar.\n"
    "\n"
    "## Forma de argumentar\n"
    "- Usa frases curtas\n"
    "  e objetivas.\n"
)
MARCOS_PROFILE = (
    "## Posições\n"
    "- Pede prazos de transição mais longos e regras claras para as cooperativas.\n"
    "## Alinhamentos declarados\n"
    "- Retoma o ponto das pequenas cooperativas.\n"
)

runner = CliRunner()


def stub_factory(config: UdvConfig, hearings: Sequence[HearingRecord]) -> SentenceEncoder:
    return StubEncoder()


stub_app = create_app(encoder_factory=stub_factory)


def profile(actor: str, text: str, hearing_ids: list[int], statements: int) -> ProfileRecord:
    return ProfileRecord(
        actor=actor,
        profile=text,
        model="stub-model",
        prompt_version="abc123def456",
        n_statements=statements,
        n_hearings=len(hearing_ids),
        hearing_ids=hearing_ids,
        input_tokens=100,
        output_tokens=20,
        generated_at="2026-01-01T00:00:00+00:00",
        duration_seconds=1.5,
    )


def fixture_profiles() -> list[ProfileRecord]:
    return [
        profile("João Silva", JOAO_PROFILE, [10], 1),
        profile("MARCOS PEREIRA", MARCOS_PROFILE, [10, 11], 2),
    ]


@pytest.fixture(scope="module")
def hearings() -> list[HearingRecord]:
    return load_hearings(FIXTURES_DIR / "lds_actors.jsonl")


@pytest.fixture(scope="module")
def records(hearings: list[HearingRecord]) -> list[UdvRecord]:
    return build_udvs(
        hearings, CachedEncoder(StubEncoder()), EvidenceSettings(MINI_THRESHOLD)
    ).records


@pytest.fixture(scope="module")
def speeches(hearings: list[HearingRecord]) -> dict[str, ActorSpeechRecord]:
    config = load_actors_config(FIXTURES_DIR / ACTORS_CONFIG)
    return collect_actor_speeches(hearings, config).records


def builder(
    speeches: dict[str, ActorSpeechRecord], profiles: Sequence[ProfileRecord] | None = None
) -> ProfileSiteBuilder:
    return ProfileSiteBuilder(
        fixture_profiles() if profiles is None else profiles,
        speeches,
        run_name="stub_run",
        source_sha256="0" * 64,
    )


def export(
    output: Path,
    hearings: list[HearingRecord],
    records: list[UdvRecord],
    profiles: ProfileSiteBuilder | None,
) -> Any:
    return export_site(
        hearings,
        records,
        CachedEncoder(StubEncoder()),
        output,
        run_name="run",
        pipeline=PIPELINE,
        profiles=profiles,
    )


def read(path: Path) -> JsonObject:
    data: JsonObject = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_parse_profile_splits_sections_and_joins_continuations() -> None:
    assert parse_profile(JOAO_PROFILE) == [
        ProfileSection(
            "Posições",
            (
                "Defende reforçar a fiscalização das cooperativas antes de qualquer mudança no "
                "texto.",
                "Tema sem fala correspondente: orçamento militar.",
            ),
        ),
        ProfileSection("Forma de argumentar", ("Usa frases curtas e objetivas.",)),
    ]


def test_parse_profile_keeps_text_before_any_heading_and_drops_empty_sections() -> None:
    sections = parse_profile("Texto solto.\n## Vazia\n\n## Posições\n* Item com asterisco.\n")
    assert sections == [
        ProfileSection(UNTITLED_SECTION, ("Texto solto.",)),
        ProfileSection("Posições", ("Item com asterisco.",)),
    ]


def test_slugs_fold_accents_and_break_collisions() -> None:
    assert actor_slug("Érika  Kokay (PT-DF)") == "erika-kokay-pt-df"
    assert actor_slug("???") == "ator"
    assert assign_slugs(["João Silva", "Joao Silva", "Ana"]) == {
        "Ana": "ana",
        "Joao Silva": "joao-silva",
        "João Silva": "joao-silva-2",
    }


def test_trim_passage_cuts_at_a_word_boundary() -> None:
    assert trim_passage("curto") == "curto"
    assert trim_passage("aaaa bbbb cccc", 11) == "aaaa bbbb […]"


def test_export_site_writes_the_profile_pages(
    tmp_path: Path,
    hearings: list[HearingRecord],
    records: list[UdvRecord],
    speeches: dict[str, ActorSpeechRecord],
) -> None:
    site = export(tmp_path / "site", hearings, records, builder(speeches))
    assert site.profiles is not None
    assert sorted(site.profiles.profile_bytes) == ["joao-silva", "marcos-pereira"]
    assert site.total_bytes == sum(
        path.stat().st_size for path in (tmp_path / "site").rglob("*.json")
    )
    actors = read(tmp_path / "site" / ACTORS_FILE_NAME)
    assert actors["run"] == site.index["run"]
    assert actors["profiles"]["run"] == "stub_run"
    assert actors["profiles"]["models"] == ["stub-model"]
    assert actors["people"] == {
        "10": {"João Silva": "joao-silva", "Marcos Pereira": "marcos-pereira"},
        "11": {"Marcos": "marcos-pereira"},
    }
    assert actors["udvs"] == {
        "udv-10-0-0": "joao-silva",
        "udv-10-1-0": "marcos-pereira",
        "udv-10-1-1": "marcos-pereira",
        "udv-11-0-0": "marcos-pereira",
    }
    summary = {actor["slug"]: actor for actor in actors["actors"]}
    assert summary["joao-silva"]["claims"] == {
        "claims": 3,
        "with_udv": 1,
        "passage_only": 0,
        "without_evidence": 2,
    }
    assert summary["joao-silva"]["n_hearings"] == 2
    assert summary["joao-silva"]["n_hearings_in_profile"] == 1
    assert summary["joao-silva"]["n_turns"] == 3
    assert summary["marcos-pereira"]["n_udvs"] == 3


def test_profile_page_links_claims_to_passages_and_udvs(
    tmp_path: Path,
    hearings: list[HearingRecord],
    records: list[UdvRecord],
    speeches: dict[str, ActorSpeechRecord],
) -> None:
    export(tmp_path / "site", hearings, records, builder(speeches))
    page = read(tmp_path / "site" / PROFILES_DIR_NAME / "joao-silva.json")
    assert page["actor"] == {
        "slug": "joao-silva",
        "name": "João Silva",
        "role": "Deputado (PT-SP)",
        "article_names": ["João Silva"],
        "party_uf": ["Bloco/PT - SP"],
    }
    assert page["provenance"]["model"] == "stub-model"
    assert page["provenance"]["hearing_ids"] == [10]
    assert [hearing["id"] for hearing in page["hearings"]] == [10, 11]
    assert [hearing["in_profile"] for hearing in page["hearings"]] == [True, False]
    assert [hearing["turns"] for hearing in page["hearings"]] == [1, 2]
    first, unsupported = page["sections"][0]["claims"]
    assert first["passage"]["hearing_id"] == 10
    assert first["passage"]["turn"] == 1
    assert first["passage"]["text"] == (
        "A fiscalização das cooperativas precisa ser reforçada antes de qualquer mudança no texto."
    )
    assert first["udv"]["id"] == "udv-10-0-0"
    assert first["udv"]["rule"] == SAME_SENTENCE_RULE
    assert unsupported == {
        "text": "Tema sem fala correspondente: orçamento militar.",
        "passage": None,
        "udv": None,
    }
    assert page["udvs"][0]["id"] == "udv-10-0-0"
    assert page["udvs"][0]["n"] == 1
    assert page["udvs"][0]["verifier"] is None
    assert page["udvs"][0]["in_profile"] is True
    assert "match_text" not in page["udvs"][0]
    assert page["verifier_threshold"] is None


def test_profile_page_reads_only_the_hearings_of_the_profile(
    tmp_path: Path,
    hearings: list[HearingRecord],
    records: list[UdvRecord],
    speeches: dict[str, ActorSpeechRecord],
) -> None:
    only_first = [profile("MARCOS PEREIRA", MARCOS_PROFILE, [10], 1)]
    export(tmp_path / "site", hearings, records, builder(speeches, only_first))
    page = read(tmp_path / "site" / PROFILES_DIR_NAME / "marcos-pereira.json")
    passages = [
        claim["passage"]
        for section in page["sections"]
        for claim in section["claims"]
        if claim["passage"] is not None
    ]
    assert passages
    assert {passage["hearing_id"] for passage in passages} == {10}
    assert [udv["in_profile"] for udv in page["udvs"]] == [True, True, False]


def test_similar_text_links_a_udv_without_a_shared_sentence(
    tmp_path: Path,
    hearings: list[HearingRecord],
    records: list[UdvRecord],
    speeches: dict[str, ActorSpeechRecord],
) -> None:
    text = "## Posições\n- Pediu reforço na fiscalização das cooperativas.\n"
    site_builder = ProfileSiteBuilder(
        [profile("João Silva", text, [10], 1)],
        speeches,
        run_name="stub_run",
        source_sha256="0" * 64,
        passage_match_min=1.0,
        udv_match_min=0.1,
    )
    export(tmp_path / "site", hearings, records, site_builder)
    page = read(tmp_path / "site" / PROFILES_DIR_NAME / "joao-silva.json")
    claim = page["sections"][0]["claims"][0]
    assert claim["passage"] is None
    assert claim["udv"]["id"] == "udv-10-0-0"
    assert claim["udv"]["rule"] == SIMILAR_TEXT_RULE
    assert claim["udv"]["score"] >= 0.1


def test_export_site_without_profiles_removes_stale_profile_files(
    tmp_path: Path,
    hearings: list[HearingRecord],
    records: list[UdvRecord],
    speeches: dict[str, ActorSpeechRecord],
) -> None:
    output = tmp_path / "site"
    export(output, hearings, records, builder(speeches))
    first = {path.name: path.read_bytes() for path in output.rglob("*.json")}
    export(output, hearings, records, builder(speeches))
    assert {path.name: path.read_bytes() for path in output.rglob("*.json")} == first
    site = export(output, hearings, records, None)
    assert site.profiles is None
    assert not (output / ACTORS_FILE_NAME).exists()
    assert list((output / PROFILES_DIR_NAME).iterdir()) == []


def test_builder_refuses_profiles_from_another_actor_pass(
    speeches: dict[str, ActorSpeechRecord],
) -> None:
    with pytest.raises(ConfigError, match="actors without speeches"):
        builder(speeches, [profile("Ninguém", JOAO_PROFILE, [10], 1)])
    with pytest.raises(ConfigError, match="are not in the actor speeches"):
        builder(speeches, [profile("João Silva", JOAO_PROFILE, [99], 1)])
    with pytest.raises(ConfigError, match="the profiles come from another actor pass"):
        builder(speeches, [profile("João Silva", JOAO_PROFILE, [10, 11], 1)])


@pytest.fixture
def actors_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in FIXTURE_FILES:
        shutil.copy(FIXTURES_DIR / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    write_profiles(fixture_profiles(), tmp_path / "profiles.jsonl")
    return tmp_path


def site_command(application: typer.Typer, *options: str) -> Any:
    return runner.invoke(
        application,
        ["export-site", "--config", UDV_CONFIG, "--run-name", "run", "--output", "site", *options],
    )


def test_cli_export_site_with_profiles(actors_workdir: Path) -> None:
    build = runner.invoke(stub_app, ["build-udvs", "--config", UDV_CONFIG, "--run-name", "run"])
    assert build.exit_code == 0, build.output
    result = site_command(
        stub_app,
        "--profiles",
        "profiles.jsonl",
        "--actors-config",
        ACTORS_CONFIG,
        "--profiles-run",
        "stub_run",
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["profiles"] == {
        "actors": 2,
        "linked_udvs": 4,
        "claims": 5,
        "with_udv": 3,
        "passage_only": 0,
        "without_evidence": 2,
    }
    assert summary["bytes"] == sum(
        path.stat().st_size for path in (actors_workdir / "site").rglob("*.json")
    )
    assert read(actors_workdir / "site" / ACTORS_FILE_NAME)["profiles"]["run"] == "stub_run"
    again = site_command(stub_app, "--profiles", "profiles.jsonl", "--actors-config", ACTORS_CONFIG)
    assert again.exit_code == 2
    assert "site export already exists" in again.stderr


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--profiles", "absent.jsonl", "--actors-config", ACTORS_CONFIG], "file not found"),
        (["--profiles", "profiles.jsonl", "--actors-config", "absent.toml"], "absent.toml"),
    ],
)
def test_cli_export_site_profile_input_errors_exit_two(
    actors_workdir: Path, options: list[str], message: str
) -> None:
    build = runner.invoke(stub_app, ["build-udvs", "--config", UDV_CONFIG, "--run-name", "run"])
    assert build.exit_code == 0, build.output
    result = site_command(stub_app, *options)
    assert result.exit_code == 2
    assert message in result.stderr
    assert not (actors_workdir / "site").exists()
