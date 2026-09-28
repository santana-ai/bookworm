"""Profile generation: prompt rendering, dry run and resumable runs."""

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from bookworm.actors.schemas import ActorSpeechRecord
from bookworm.data.dates import article_date
from bookworm.data.io import JsonObject, load_hearings
from bookworm.data.schemas import HearingRecord
from bookworm.errors import ConfigError
from bookworm.profiles.config import ProfilesConfig, load_profiles_config
from bookworm.profiles.llm import ChatClient, load_transformers_client
from bookworm.profiles.prompts import PromptHearing, PromptSet, PromptTurn, load_prompts
from bookworm.profiles.schemas import ProfileRecord, append_profile
from bookworm.profiles.split_filter import load_speeches

ClientFactory = Callable[[ProfilesConfig], ChatClient]
Reporter = Callable[[str], None]

DATE_BR_FORMAT = "%d/%m/%Y"


def default_client_factory(config: ProfilesConfig) -> ChatClient:
    return load_transformers_client(config.model)


def ignore_message(message: str) -> None:
    return None


@dataclass(frozen=True)
class HearingInfo:
    date: date
    assunto: str

    @property
    def date_br(self) -> str:
        return self.date.strftime(DATE_BR_FORMAT)


def hearing_metadata(hearings: Iterable[HearingRecord]) -> dict[int, HearingInfo]:
    metadata: dict[int, HearingInfo] = {}
    for hearing in hearings:
        published = article_date(hearing.materia)
        if published is None:
            raise ConfigError(f"hearing {hearing.id} has no article date")
        metadata[hearing.id] = HearingInfo(published, hearing.metadados.assunto)
    return metadata


def load_lds_metadata(config: ProfilesConfig) -> dict[int, HearingInfo]:
    if not config.lds_path.is_file():
        raise ConfigError(f"{config.lds_path}: LDS file not found")
    try:
        hearings = load_hearings(config.lds_path, config.lds_sha256)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{config.lds_path}: cannot read LDS file: {error}") from error
    return hearing_metadata(hearings)


def select_actors(
    records: Sequence[ActorSpeechRecord], actors: Sequence[str] | None
) -> list[ActorSpeechRecord]:
    if not actors:
        return list(records)
    by_name = {record.actor: record for record in records}
    missing = [name for name in actors if name not in by_name]
    if missing:
        raise ConfigError(f"actors not found in input: {missing}")
    return [by_name[name] for name in actors]


def prompt_hearings(
    record: ActorSpeechRecord, metadata: Mapping[int, HearingInfo]
) -> list[PromptHearing]:
    missing = sorted(set(record.hearing_ids) - set(metadata))
    if missing:
        raise ConfigError(f"{record.actor} speaks in hearings missing from the LDS: {missing}")
    hearings = sorted(
        record.hearings,
        key=lambda hearing: (metadata[hearing.hearing_id].date, hearing.hearing_id),
    )
    return [
        PromptHearing(
            date=metadata[hearing.hearing_id].date_br,
            assunto=metadata[hearing.hearing_id].assunto,
            turns=tuple(PromptTurn(turn.role, turn.text) for turn in hearing.turns),
        )
        for hearing in hearings
    ]


@dataclass(frozen=True)
class ProfilePrompt:
    record: ActorSpeechRecord
    system: str
    user: str

    @property
    def characters(self) -> int:
        return len(self.system) + len(self.user)


def render_prompt(
    record: ActorSpeechRecord, prompts: PromptSet, metadata: Mapping[int, HearingInfo]
) -> ProfilePrompt:
    user = prompts.render_user(record.actor, prompt_hearings(record, metadata))
    return ProfilePrompt(record, prompts.system_profile, user)


@dataclass(frozen=True)
class GenerationContext:
    config: ProfilesConfig
    prompts: PromptSet
    metadata: Mapping[int, HearingInfo]


def generate_profile(
    prompt: ProfilePrompt, client: ChatClient, context: GenerationContext
) -> ProfileRecord:
    started = time.monotonic()
    result = client.chat(prompt.system, prompt.user)
    record = prompt.record
    if not result.text:
        raise ValueError(f"empty profile for {record.actor}")
    return ProfileRecord(
        actor=record.actor,
        profile=result.text,
        model=context.config.model.name,
        prompt_version=context.prompts.version,
        n_statements=sum(len(hearing.turns) for hearing in record.hearings),
        n_hearings=len(record.hearings),
        hearing_ids=sorted(record.hearing_ids),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_seconds=round(time.monotonic() - started, 1),
    )


def load_done_actors(
    path: Path, prompt_version: str, model: str, report: Reporter = ignore_message
) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    stale: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                actor, version = row["actor"], (row["prompt_version"], row["model"])
            except (json.JSONDecodeError, KeyError, TypeError):
                report(f"ignoring invalid line {line_number} in {path}")
                continue
            if version != (prompt_version, model):
                stale.add(version)
            done.add(actor)
    if stale:
        raise ConfigError(
            f"{path} has profiles from other (prompt_version, model) pairs {sorted(stale)},"
            f" current run is {(prompt_version, model)}; move the file away to regenerate"
            " every profile, or pass a new --output"
        )
    return done


@dataclass(frozen=True)
class GenerateRequest:
    config_path: Path
    speeches_path: Path | None = None
    output_path: Path | None = None
    model: str | None = None
    actors: list[str] | None = None
    limit: int | None = None
    dry_run: bool = False


@dataclass
class GenerationOutcome:
    summary: JsonObject
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class PreparedRun:
    context: GenerationContext
    selected: list[ActorSpeechRecord]
    done: frozenset[str] = frozenset()


def prepare_run(request: GenerateRequest, report: Reporter = ignore_message) -> PreparedRun:
    config = load_profiles_config(request.config_path).with_overrides(
        request.speeches_path, request.output_path, request.model
    )
    if not request.dry_run and not config.model.name:
        raise ConfigError("set [model] name in the config, or pass --model")
    prompts = load_prompts(config.prompts_dir, config.system_profile_file, config.user_profile_file)
    done = (
        frozenset()
        if request.dry_run
        else frozenset(
            load_done_actors(config.profiles_path, prompts.version, config.model.name, report)
        )
    )
    selected = select_actors(load_speeches(config.speeches_path), request.actors)
    context = GenerationContext(config, prompts, load_lds_metadata(config))
    return PreparedRun(context, selected, done)


def dry_run(run: PreparedRun, limit: int | None) -> GenerationOutcome:
    todo = run.selected if limit is None else run.selected[:limit]
    prompts = [render_prompt(record, run.context.prompts, run.context.metadata) for record in todo]
    largest = max(prompts, key=lambda prompt: prompt.characters, default=None)
    return GenerationOutcome(
        {
            "dry_run": True,
            "prompt_version": run.context.prompts.version,
            "speeches": str(run.context.config.speeches_path),
            "prompts": len(prompts),
            "system_characters": len(run.context.prompts.system_profile),
            "user_characters": sum(len(prompt.user) for prompt in prompts),
            "total_characters": sum(prompt.characters for prompt in prompts),
            "max_prompt_characters": largest.characters if largest else 0,
            "max_prompt_actor": largest.record.actor if largest else None,
        }
    )


@dataclass
class GenerationTally:
    generated: list[ProfileRecord] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def write_new_profiles(
    todo: Sequence[ActorSpeechRecord],
    client: ChatClient,
    context: GenerationContext,
    report: Reporter,
) -> GenerationTally:
    tally = GenerationTally()
    path = context.config.profiles_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for index, record in enumerate(todo, start=1):
            progress = f"[{index}/{len(todo)}] {record.actor}"
            try:
                profile = generate_profile(
                    render_prompt(record, context.prompts, context.metadata), client, context
                )
            except Exception as error:
                tally.failures.append(record.actor)
                report(f"{progress} failed: {error}")
                continue
            append_profile(profile, output)
            tally.generated.append(profile)
            report(
                f"{progress}: {profile.input_tokens:,} in / {profile.output_tokens:,} out tokens, "
                f"{profile.duration_seconds:.1f}s"
            )
    return tally


def generate_profiles(
    run: PreparedRun,
    client_factory: ClientFactory,
    limit: int | None,
    report: Reporter = ignore_message,
) -> GenerationOutcome:
    context = run.context
    path = context.config.profiles_path
    todo = [record for record in run.selected if record.actor not in run.done]
    skipped = len(run.selected) - len(todo)
    if skipped:
        report(f"resuming: {skipped} actors already in {path}")
    if limit is not None:
        todo = todo[:limit]
    started = time.monotonic()
    tally = (
        write_new_profiles(todo, client_factory(context.config), context, report)
        if todo
        else GenerationTally()
    )
    return GenerationOutcome(
        {
            "dry_run": False,
            "prompt_version": context.prompts.version,
            "model": context.config.model.name,
            "output": str(path),
            "generated": len(tally.generated),
            "failed": len(tally.failures),
            "skipped_existing": skipped,
            "input_tokens": sum(profile.input_tokens for profile in tally.generated),
            "output_tokens": sum(profile.output_tokens for profile in tally.generated),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "failed_actors": tally.failures,
        },
        tally.failures,
    )


def run_generate_profiles(
    request: GenerateRequest,
    client_factory: ClientFactory = default_client_factory,
    report: Reporter = ignore_message,
) -> GenerationOutcome:
    """Generate the missing profiles of a speeches file, or render the prompts only."""
    run = prepare_run(request, report)
    report(f"{len(run.selected)} actors selected, prompts {run.context.prompts.version}")
    if request.dry_run:
        return dry_run(run, request.limit)
    return generate_profiles(run, client_factory, request.limit, report)
