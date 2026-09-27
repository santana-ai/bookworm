import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bookworm.actors.schemas import (
    ActorHearing,
    ActorSpeechRecord,
    ActorTurn,
    TurnRole,
    read_actor_speeches,
    write_actor_speeches,
)
from bookworm.cli import create_app
from bookworm.data.io import sha256_of_file, write_json, write_jsonl
from bookworm.errors import ConfigError
from bookworm.profiles.config import (
    PACKAGED_PROMPTS_DIR,
    ProfilesConfig,
    load_split_filter_config,
)
from bookworm.profiles.generate import GenerateRequest, load_done_actors, run_generate_profiles
from bookworm.profiles.llm import ChatResult
from bookworm.profiles.prompts import load_prompts, prompt_version
from bookworm.profiles.schemas import ProfileRecord, read_profiles
from bookworm.profiles.split_filter import LINK_DESCRIPTION, build_split_filter
from bookworm.udv.schemas import Actor, Evidence, Method, Tier, UdvRecord, write_udv_jsonl

runner = CliRunner()

TRAIN_LATE = 1
TRAIN_EARLY = 2
TEST_HEARING = 3
HEARING_DATES = {TRAIN_LATE: "10/03/2024", TRAIN_EARLY: "05/02/2024", TEST_HEARING: "01/06/2024"}
HEARING_TOPICS = {
    TRAIN_LATE: "reforma do ensino médio",
    TRAIN_EARLY: "transporte escolar rural",
    TEST_HEARING: "merenda escolar",
}
ANA_EARLY = "O transporte escolar das crianças do campo precisa de ônibus novos."
ANA_LATE = "A reforma do ensino médio exige professores formados e bem pagos."
ANA_TEST = "A merenda escolar deve comprar da agricultura familiar."
BRUNO_TEST = "Declaro aberta a reunião sobre a merenda escolar e passo a palavra."
CAIO_CHAIR = "Como presidente desta comissão, defendo mais recursos para o transporte escolar."


def turn(index: int, text: str, role: TurnRole = "speaker") -> ActorTurn:
    return ActorTurn(turn_index=index, role=role, start_char=0, end_char=len(text), text=text)


def hearing(hearing_id: int, *turns: ActorTurn) -> ActorHearing:
    return ActorHearing(
        hearing_id=hearing_id,
        full_speech="\n\n".join(item.text for item in turns),
        turns=list(turns),
    )


def speech(actor: str, *hearings: ActorHearing) -> ActorSpeechRecord:
    return ActorSpeechRecord(
        actor=actor, has_party_header=False, party_uf=[], hearings=list(hearings)
    )


SPEECHES = [
    speech(
        "Ana Silva",
        hearing(TRAIN_LATE, turn(3, ANA_LATE)),
        hearing(TEST_HEARING, turn(1, ANA_TEST)),
        hearing(TRAIN_EARLY, turn(2, ANA_EARLY)),
    ),
    speech("Bruno Costa", hearing(TEST_HEARING, turn(0, BRUNO_TEST, "chair"))),
    speech("Caio Lima", hearing(TRAIN_EARLY, turn(0, CAIO_CHAIR, "chair"))),
]


def udv(
    udv_id: str, hearing_id: int, tier: Tier, speaker_turn: int | None, has_evidence: bool = True
) -> UdvRecord:
    evidence = Evidence(
        text="Trecho.",
        support_type="direct_quote",
        score=None,
        quote_prefix=None,
        start_char=None,
        end_char=None,
        speaker_turn=speaker_turn,
    )
    return UdvRecord(
        id=udv_id,
        hearing_id=hearing_id,
        actor=Actor(name="Pessoa", role="Convidada"),
        proposition="Opinião.",
        evidence=evidence if has_evidence else None,
        tier=tier,
        provenance=None,
        method=Method(encoder="stub-encoder", revision="stub-revision-1", embedding_threshold=0.6),
    )


UDVS = [
    udv("udv-1-0-0", TRAIN_LATE, "quote_found", 3),
    udv("udv-3-0-0", TEST_HEARING, "quote_found", 1),
    udv("udv-3-0-1", TEST_HEARING, "semantic_match_high", 7),
    udv("udv-3-0-2", TEST_HEARING, "no_evidence", None, has_evidence=False),
    udv("udv-3-1-0", TEST_HEARING, "semantic_match_weak", 0),
]


def lds_record(hearing_id: int) -> dict[str, Any]:
    return {
        "id": hearing_id,
        "materia": f"{HEARING_DATES[hearing_id]} - 18:30\nMatéria da audiência {hearing_id}.",
        "metadados": {"assunto": HEARING_TOPICS[hearing_id], "envolvidos": []},
        "transcricao": "",
    }


@dataclass
class FakeClient:
    fail_for: set[str] = field(default_factory=set)
    empty_for: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def chat(self, system: str, user: str) -> ChatResult:
        self.calls.append((system, user))
        label = user.split("\n", 1)[0]
        if any(name in label for name in self.fail_for):
            raise RuntimeError("generation reached max_output_tokens=10")
        text = "" if any(name in label for name in self.empty_for) else f"Perfil: {label}"
        return ChatResult(text=text, input_tokens=len(user), output_tokens=7)


@dataclass
class FakeFactory:
    client: FakeClient = field(default_factory=FakeClient)
    loads: int = 0

    def __call__(self, config: ProfilesConfig) -> FakeClient:
        self.loads += 1
        return self.client


@dataclass(frozen=True)
class Workspace:
    root: Path
    config: Path
    speeches: Path
    train_speeches: Path
    profiles: Path
    stats: Path
    manifest: Path


def write_config(
    root: Path, lds_sha256: str, extra: str = "", eval_splits: str = '["test"]'
) -> Path:
    path = root / "actor_profiles.toml"
    path.write_text(
        f"""
[input]
speeches_path = "speeches.jsonl"
lds_path = "lds.jsonl"
lds_sha256 = "{lds_sha256}"

[output]
profiles_path = "out/profiles.jsonl"

[split_filter]
manifest_path = "manifest.json"
splits = ["train"]
eval_splits = {eval_splits}
udv_path = "udvs.jsonl"
speeches_path = "cache/speeches_train.jsonl"
stats_path = "out/stats.json"

{extra}

[model]
name = ""
device_map = "auto"
temperature = 0.2
top_p = 0.9
max_output_tokens = 3000
seed = 42
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    monkeypatch.chdir(tmp_path)
    write_jsonl([lds_record(item) for item in HEARING_DATES], tmp_path / "lds.jsonl")
    lds_sha256 = sha256_of_file(tmp_path / "lds.jsonl")
    write_json(
        {
            "split_version": "toy_v1",
            "dataset": {"path": "lds.jsonl", "sha256": lds_sha256},
            "train": [TRAIN_LATE, TRAIN_EARLY],
            "validation": [],
            "test": [TEST_HEARING],
        },
        tmp_path / "manifest.json",
    )
    write_actor_speeches(SPEECHES, tmp_path / "speeches.jsonl")
    write_udv_jsonl(UDVS, tmp_path / "udvs.jsonl")
    return Workspace(
        root=tmp_path,
        config=write_config(tmp_path, lds_sha256),
        speeches=tmp_path / "speeches.jsonl",
        train_speeches=tmp_path / "cache" / "speeches_train.jsonl",
        profiles=tmp_path / "out" / "profiles.jsonl",
        stats=tmp_path / "out" / "stats.json",
        manifest=tmp_path / "manifest.json",
    )


@pytest.fixture
def filtered(workspace: Workspace) -> Workspace:
    build_split_filter(load_split_filter_config(workspace.config))
    return workspace


def request(workspace: Workspace, **options: Any) -> GenerateRequest:
    values: dict[str, Any] = {"model": "toy-model", "speeches_path": workspace.train_speeches}
    return GenerateRequest(workspace.config, **{**values, **options})


def cli_args(workspace: Workspace, *extra: str) -> list[str]:
    return ["generate-profiles", "--config", str(workspace.config), *extra]


def test_prompt_version_hashes_sorted_names_and_bytes() -> None:
    names = ("user_profile.md.j2", "system_profile.md")
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update((PACKAGED_PROMPTS_DIR / name).read_bytes())
    assert prompt_version(PACKAGED_PROMPTS_DIR, names) == digest.hexdigest()[:12]
    assert load_prompts(PACKAGED_PROMPTS_DIR, *reversed(names)).version == digest.hexdigest()[:12]


def test_prompt_version_changes_when_a_prompt_changes(tmp_path: Path) -> None:
    for name in ("system_profile.md", "user_profile.md.j2"):
        (tmp_path / name).write_bytes((PACKAGED_PROMPTS_DIR / name).read_bytes())
    before = prompt_version(tmp_path, ("system_profile.md", "user_profile.md.j2"))
    with (tmp_path / "system_profile.md").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert prompt_version(tmp_path, ("system_profile.md", "user_profile.md.j2")) != before


def test_missing_prompt_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="prompt file not found"):
        load_prompts(tmp_path, "system_profile.md", "user_profile.md.j2")


def test_split_filter_keeps_only_train_hearings(workspace: Workspace) -> None:
    stats = build_split_filter(load_split_filter_config(workspace.config))
    records = read_actor_speeches(workspace.train_speeches)
    assert [record.actor for record in records] == ["Ana Silva", "Caio Lima"]
    assert records[0].hearing_ids == [TRAIN_LATE, TRAIN_EARLY]
    assert records[0].hearings[0] == SPEECHES[0].hearings[0]
    assert json.loads(workspace.stats.read_text(encoding="utf-8")) == stats
    assert stats["split_manifest"] == {
        "path": "manifest.json",
        "sha256": sha256_of_file(workspace.manifest),
        "split_version": "toy_v1",
        "splits": ["train"],
        "hearings": 2,
    }
    assert stats["input"] == {
        "path": "speeches.jsonl",
        "sha256": sha256_of_file(workspace.speeches),
        "actors": 3,
        "hearings": 3,
        "turns": 5,
        "hearings_per_actor": {"1": 2, "3": 1},
    }
    assert stats["output"] == {
        "path": "cache/speeches_train.jsonl",
        "sha256": sha256_of_file(workspace.train_speeches),
        "actors": 2,
        "hearings": 2,
        "turns": 3,
        "hearings_per_actor": {"1": 1, "2": 1},
        "actors_without_split_hearings": 1,
    }
    assert list(stats) == ["split_manifest", "input", "output", "evaluation"]
    assert stats["evaluation"] == {
        "splits": ["test"],
        "udv_path": "udvs.jsonl",
        "udv_sha256": sha256_of_file(workspace.root / "udvs.jsonl"),
        "link": LINK_DESCRIPTION,
        "hearings": 1,
        "profiled_actors_speaking": 1,
        "udvs": 4,
        "linked_udvs": 1,
        "linked_actors": 1,
        "linked_hearings": 1,
        "linked_udvs_by_tier": {"quote_found": 1},
    }


def test_split_filter_refuses_eval_splits_that_overlap_the_profile_splits(
    workspace: Workspace,
) -> None:
    write_config(
        workspace.root, sha256_of_file(workspace.root / "lds.jsonl"), eval_splits='["train"]'
    )
    with pytest.raises(ConfigError, match="overlap the profile splits"):
        load_split_filter_config(workspace.config)


def test_split_filter_needs_the_udv_file(workspace: Workspace) -> None:
    (workspace.root / "udvs.jsonl").unlink()
    with pytest.raises(ConfigError, match="UDV file not found"):
        build_split_filter(load_split_filter_config(workspace.config))


def test_split_filter_refuses_a_manifest_of_another_lds(workspace: Workspace) -> None:
    manifest = json.loads(workspace.manifest.read_text(encoding="utf-8"))
    manifest["dataset"]["sha256"] = "0" * 64
    write_json(manifest, workspace.manifest)
    with pytest.raises(ConfigError, match="another LDS file"):
        build_split_filter(load_split_filter_config(workspace.config))
    assert not workspace.train_speeches.exists()


def test_filter_cli_output_override(workspace: Workspace) -> None:
    args = ["filter-actor-speeches", "--config", str(workspace.config), "--output", "x.jsonl"]
    result = runner.invoke(create_app(), args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["output"]["path"] == "x.jsonl"
    assert len(read_actor_speeches(Path("x.jsonl"))) == 2


def test_dry_run_renders_without_a_model(filtered: Workspace) -> None:
    factory = FakeFactory()
    outcome = run_generate_profiles(request(filtered, model=None, dry_run=True), factory)
    assert factory.loads == 0
    assert not filtered.profiles.exists()
    summary = outcome.summary
    assert outcome.ok
    assert summary["prompts"] == 2
    assert summary["max_prompt_actor"] == "Ana Silva"
    assert (
        summary["total_characters"]
        == summary["user_characters"] + 2 * (summary["system_characters"])
    )


def test_generation_writes_chronological_profiles(filtered: Workspace) -> None:
    factory = FakeFactory()
    outcome = run_generate_profiles(request(filtered), factory)
    assert outcome.ok
    assert outcome.summary["generated"] == 2
    records = read_profiles(filtered.profiles)
    assert [record.actor for record in records] == ["Ana Silva", "Caio Lima"]
    ana = records[0]
    assert (ana.hearing_ids, ana.n_hearings, ana.n_statements) == ([1, 2], 2, 2)
    assert ana.model == "toy-model"
    assert ana.prompt_version == outcome.summary["prompt_version"]
    assert list(json.loads(filtered.profiles.read_text(encoding="utf-8").splitlines()[0])) == [
        "actor",
        "profile",
        "model",
        "prompt_version",
        "n_statements",
        "n_hearings",
        "hearing_ids",
        "input_tokens",
        "output_tokens",
        "generated_at",
        "duration_seconds",
    ]
    ana_prompt = factory.client.calls[0][1]
    assert ANA_TEST not in ana_prompt
    assert ana_prompt.index(ANA_EARLY) < ana_prompt.index(ANA_LATE)
    assert "=== Audiência de 05/02/2024 sobre: transporte escolar rural ===" in ana_prompt
    assert "2 audiência(s)" in ana_prompt
    assert f"[presidência da sessão]\n{CAIO_CHAIR}" in factory.client.calls[1][1]


def test_generation_reads_the_speeches_it_is_given(workspace: Workspace) -> None:
    factory = FakeFactory()
    run_generate_profiles(request(workspace, speeches_path=None), factory)
    assert ANA_TEST in factory.client.calls[0][1]
    assert [record.actor for record in read_profiles(workspace.profiles)] == [
        "Ana Silva",
        "Bruno Costa",
        "Caio Lima",
    ]


def test_resume_skips_existing_actors(filtered: Workspace) -> None:
    run_generate_profiles(request(filtered, limit=1), FakeFactory())
    assert [record.actor for record in read_profiles(filtered.profiles)] == ["Ana Silva"]
    second = FakeFactory()
    outcome = run_generate_profiles(request(filtered), second)
    assert (outcome.summary["skipped_existing"], outcome.summary["generated"]) == (1, 1)
    assert len(second.client.calls) == 1
    third = FakeFactory()
    outcome = run_generate_profiles(request(filtered), third)
    assert third.loads == 0
    assert (outcome.summary["skipped_existing"], outcome.summary["generated"]) == (2, 0)


def test_resume_ignores_an_invalid_line(filtered: Workspace) -> None:
    run_generate_profiles(request(filtered, limit=1), FakeFactory())
    with filtered.profiles.open("a", encoding="utf-8") as handle:
        handle.write('{"actor": "Caio Lima"}\n{"act\n')
    messages: list[str] = []
    outcome = run_generate_profiles(request(filtered), FakeFactory(), messages.append)
    assert (outcome.summary["skipped_existing"], outcome.summary["generated"]) == (1, 1)
    assert "ignoring invalid line 2 in out/profiles.jsonl" in messages
    assert "ignoring invalid line 3 in out/profiles.jsonl" in messages


def profile_row(actor: str, prompt_version: str, model: str) -> str:
    return ProfileRecord(
        actor=actor,
        profile="Texto.",
        model=model,
        prompt_version=prompt_version,
        n_statements=1,
        n_hearings=1,
        hearing_ids=[1],
        input_tokens=1,
        output_tokens=1,
        generated_at="2026-09-26T00:00:00+00:00",
        duration_seconds=0.1,
    ).to_json_line()


def test_load_done_actors_accepts_rows_of_the_current_run(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    path.write_text(
        profile_row("Ana Silva", "v1", "m") + "\n\n" + profile_row("Caio Lima", "v1", "m") + "\n",
        encoding="utf-8",
    )
    assert load_done_actors(path, "v1", "m") == {"Ana Silva", "Caio Lima"}
    assert load_done_actors(tmp_path / "missing.jsonl", "v1", "m") == set()


@pytest.mark.parametrize(("version", "model"), [("v0", "m"), ("v1", "other"), ("v0", "other")])
def test_load_done_actors_refuses_rows_of_another_run(
    tmp_path: Path, version: str, model: str
) -> None:
    path = tmp_path / "profiles.jsonl"
    path.write_text(
        profile_row("Ana Silva", "v1", "m")
        + "\n"
        + profile_row("Caio Lima", version, model)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"other \(prompt_version, model\) pairs") as error:
        load_done_actors(path, "v1", "m")
    assert f"[('{version}', '{model}')]" in str(error.value)
    assert "current run is ('v1', 'm')" in str(error.value)


def test_resume_refuses_profiles_of_another_prompt_version(filtered: Workspace) -> None:
    filtered.profiles.parent.mkdir(parents=True, exist_ok=True)
    filtered.profiles.write_text(profile_row("Ana Silva", "0" * 12, "toy-model") + "\n")
    before = filtered.profiles.read_bytes()
    factory = FakeFactory()
    with pytest.raises(ConfigError, match="move the file away"):
        run_generate_profiles(request(filtered), factory)
    assert factory.loads == 0
    assert filtered.profiles.read_bytes() == before


def test_resume_refuses_another_model_before_reading_speeches(filtered: Workspace) -> None:
    run_generate_profiles(request(filtered, limit=1), FakeFactory())
    missing = filtered.root / "missing.jsonl"
    with pytest.raises(ConfigError, match="other \\(prompt_version, model\\) pairs"):
        run_generate_profiles(
            request(filtered, model="another-model", speeches_path=missing), FakeFactory()
        )


def test_dry_run_ignores_profiles_of_another_run(filtered: Workspace) -> None:
    filtered.profiles.parent.mkdir(parents=True, exist_ok=True)
    filtered.profiles.write_text(profile_row("Ana Silva", "0" * 12, "toy-model") + "\n")
    outcome = run_generate_profiles(request(filtered, dry_run=True), FakeFactory())
    assert outcome.summary["prompts"] == 2


def test_failures_are_counted_and_retried(filtered: Workspace) -> None:
    factory = FakeFactory(FakeClient(fail_for={"Ana Silva"}, empty_for={"Caio Lima"}))
    messages: list[str] = []
    outcome = run_generate_profiles(request(filtered), factory, messages.append)
    assert not outcome.ok
    assert outcome.failures == ["Ana Silva", "Caio Lima"]
    assert outcome.summary["failed"] == 2
    assert any("max_output_tokens" in message for message in messages)
    assert any("empty profile for Caio Lima" in message for message in messages)
    assert read_profiles(filtered.profiles) == []
    assert run_generate_profiles(request(filtered), FakeFactory()).summary["generated"] == 2


def test_actor_selection_by_exact_name(filtered: Workspace) -> None:
    run_generate_profiles(request(filtered, actors=["Caio Lima", "Ana Silva"]), FakeFactory())
    assert [record.actor for record in read_profiles(filtered.profiles)] == [
        "Caio Lima",
        "Ana Silva",
    ]
    with pytest.raises(ConfigError, match="Bruno Costa"):
        run_generate_profiles(request(filtered, actors=["Bruno Costa"]), FakeFactory())


def test_real_run_needs_a_model_name(filtered: Workspace) -> None:
    factory = FakeFactory()
    with pytest.raises(ConfigError, match="--model"):
        run_generate_profiles(request(filtered, model=None), factory)
    assert factory.loads == 0


def test_config_can_point_to_another_prompt_dir(filtered: Workspace) -> None:
    prompts = filtered.root / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("Sistema.", encoding="utf-8")
    (prompts / "user.j2").write_text("Ator {{ actor_label }}", encoding="utf-8")
    write_config(
        filtered.root,
        sha256_of_file(filtered.root / "lds.jsonl"),
        extra='[prompts]\ndir = "prompts"\nsystem_profile = "system.md"\nuser_profile = "user.j2"',
    )
    factory = FakeFactory()
    outcome = run_generate_profiles(request(filtered), factory)
    assert factory.client.calls[0] == ("Sistema.", "Ator Ana Silva")
    assert outcome.summary["prompt_version"] == prompt_version(prompts, ("system.md", "user.j2"))


def test_cli_dry_run_prints_counts(filtered: Workspace) -> None:
    args = cli_args(filtered, "--dry-run", "--input", str(filtered.train_speeches))
    result = runner.invoke(create_app(client_factory=FakeFactory()), args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["prompts"] == 2


def test_cli_exit_code_is_one_on_failures(filtered: Workspace) -> None:
    factory = FakeFactory(FakeClient(fail_for={"Caio Lima"}))
    args = cli_args(
        filtered,
        "--model",
        "toy-model",
        "--speeches",
        str(filtered.train_speeches),
        "--output",
        "other.jsonl",
    )
    result = runner.invoke(create_app(client_factory=factory), args)
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert (summary["generated"], summary["failed_actors"]) == (1, ["Caio Lima"])
    assert [record.actor for record in read_profiles(Path("other.jsonl"))] == ["Ana Silva"]


def test_cli_config_error_exits_two(filtered: Workspace) -> None:
    result = runner.invoke(create_app(client_factory=FakeFactory()), cli_args(filtered))
    assert result.exit_code == 2
    assert "--model" in result.stderr
