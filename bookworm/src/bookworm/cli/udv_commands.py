"""``build-udvs`` and ``verify-udvs``."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from bookworm.actors.schemas import UdvActorLink, write_udv_actor_links
from bookworm.actors.speeches import ActorSpeeches, write_actor_outputs
from bookworm.cli.common import (
    DEFAULT_CONFIG_PATH,
    RunNameOption,
    UdvConfigOption,
    echo_json,
    evidence_settings,
    exit_on_problems,
    guarded,
    load_lds,
    load_run_actors_config,
    output_paths,
    read_run_records,
    refuse_existing,
    run_paths,
    seed_everything,
)
from bookworm.config import TfidfSettings, UdvConfig, load_udv_config
from bookworm.data.io import read_json_object, write_json
from bookworm.data.schemas import HearingRecord
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder, SentenceEncoder
from bookworm.features.loading import load_encoder
from bookworm.features.tfidf import TfidfEncoder
from bookworm.pipeline import PipelineRun, run_pipeline
from bookworm.udv.build import HearingProgress, select_hearings, udv_corpus
from bookworm.udv.coverage import summarize_run
from bookworm.udv.schemas import read_udv_run, write_udv_jsonl
from bookworm.udv.verify import compare_with_baseline, verify_udv_run

EncoderFactory = Callable[[UdvConfig, Sequence[HearingRecord]], SentenceEncoder]


def fit_tfidf_encoder(settings: TfidfSettings, hearings: Sequence[HearingRecord]) -> TfidfEncoder:
    return TfidfEncoder.fit(udv_corpus(hearings), max_features=settings.max_features)


def default_encoder_factory(
    config: UdvConfig, hearings: Sequence[HearingRecord]
) -> SentenceEncoder:
    return load_encoder(config.encoder, config.seed, lambda: udv_corpus(hearings))


@dataclass(frozen=True)
class BuildRequest:
    config_path: Path
    run_name: str
    limit: int | None
    ids: list[int] | None
    overwrite: bool
    actors_config_path: Path | None = None


def actor_links_path(config: UdvConfig, run_name: str) -> Path:
    return config.output_dir / f"{run_name}_actor_links.jsonl"


def actor_summary(actors: ActorSpeeches, links: Sequence[UdvActorLink]) -> dict[str, Any]:
    return {
        "actors": len(actors.records),
        "actor_turns_kept": actors.stats["turns"]["kept"],
        "udvs_linked_to_actor": sum(link.actor_key is not None for link in links),
    }


def selected_hearings(config: UdvConfig, request: BuildRequest) -> list[HearingRecord]:
    hearings = select_hearings(
        load_lds(config.lds_path, config.expected_sha256), request.limit, request.ids or None
    )
    if not hearings:
        raise ConfigError("the hearing selection is empty")
    return hearings


def write_pipeline_run(
    pipeline_run: PipelineRun,
    summary: dict[str, Any],
    run_paths: tuple[Path, Path],
    links_path: Path,
) -> dict[str, Any]:
    records_path, coverage_path = run_paths
    write_udv_jsonl(pipeline_run.udv.records, records_path)
    write_json(summary, coverage_path)
    report = {**summary["opinions"], **summary["timing"]}
    if pipeline_run.actors is None:
        return report
    write_actor_outputs(pipeline_run.actors)
    write_udv_actor_links(pipeline_run.links, links_path)
    return {**report, **actor_summary(pipeline_run.actors, pipeline_run.links)}


def echo_build_start(hearings: Sequence[HearingRecord], encoder: SentenceEncoder) -> None:
    typer.echo(
        f"{len(hearings)} hearings | {encoder.name}@{encoder.revision[:7]} "
        f"({encoder.cache_identity})",
        err=True,
    )


def hearing_progress(total: int) -> HearingProgress:
    def report_progress(number: int, hearing: HearingRecord, records: int, seconds: float) -> None:
        typer.echo(
            f"[{number}/{total}] hearing {hearing.id}: {records} opinions in {seconds:.1f}s",
            err=True,
        )

    return report_progress


def run_build(request: BuildRequest, encoder_factory: EncoderFactory) -> None:
    config = load_udv_config(request.config_path)
    actors_config = load_run_actors_config(request.actors_config_path, config)
    seed_everything(config.seed)
    hearings = selected_hearings(config, request)
    records_path, coverage_path = output_paths(config, request.run_name)
    links_path = actor_links_path(config, request.run_name)
    actor_paths = () if actors_config is None else (*actors_config.output_paths, links_path)
    refuse_existing(
        (records_path, coverage_path, *actor_paths),
        request.overwrite,
        "run file already exists; pass --overwrite to replace it",
    )
    encoder = encoder_factory(config, hearings)
    echo_build_start(hearings, encoder)
    settings = evidence_settings(config)
    pipeline_run = run_pipeline(
        hearings,
        CachedEncoder(encoder, config.cache_dir),
        settings,
        actors_config,
        on_hearing=hearing_progress(len(hearings)),
    )
    run = pipeline_run.udv
    summary = summarize_run(
        request.run_name,
        run.records,
        run.people,
        run.hearings,
        encoder_runtime=encoder.runtime_info(),
        hearing_seconds=run.hearing_seconds,
        config_source=config.source,
        quote_policy=settings.quote_policy,
        settings=settings,
    )
    echo_json(write_pipeline_run(pipeline_run, summary, (records_path, coverage_path), links_path))


def run_verify(config_path: Path, run_name: str, baseline: Path | None) -> bool:
    config = load_udv_config(config_path)
    records_path, coverage_path = run_paths(config, run_name)
    parsed = read_run_records(records_path)
    coverage = read_json_object(coverage_path, "coverage file")
    hearings = load_lds(config.lds_path, config.expected_sha256)
    try:
        verification = verify_udv_run(
            run_name, parsed.records, coverage, hearings, schema_errors=parsed.errors
        )
    except ConfigError as error:
        raise ConfigError(f"{coverage_path}: {error}") from error
    report = verification.to_report()
    if baseline is not None:
        report["baseline_diff"] = compare_with_baseline(
            parsed.records, read_udv_run(baseline, "baseline run")
        )
    echo_json(report)
    return verification.ok


def add_udv_commands(app: typer.Typer, encoder_factory: EncoderFactory) -> None:
    @app.command(
        "build-udvs", help="Build UDV records (opinion to transcript evidence) from the LDS file."
    )
    def build_udvs_command(
        run_name: Annotated[str, typer.Option("--run-name", help="Basename of the output files.")],
        config_path: UdvConfigOption = DEFAULT_CONFIG_PATH,
        limit: Annotated[
            int | None, typer.Option("--limit", min=1, help="Only the first N LDS records.")
        ] = None,
        ids: Annotated[
            list[int] | None,
            typer.Option("--ids", help="Only these hearing ids; repeat the option for more."),
        ] = None,
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace the files of an existing run.")
        ] = False,
        actors_config_path: Annotated[
            Path | None,
            typer.Option(
                "--actors-config",
                help=(
                    "Actor speeches TOML config; also writes per-actor speeches and the "
                    "UDV to actor links from the same pass over the transcripts."
                ),
            ),
        ] = None,
    ) -> None:
        request = BuildRequest(config_path, run_name, limit, ids, overwrite, actors_config_path)
        guarded(lambda: run_build(request, encoder_factory))

    @app.command("verify-udvs", help="Recompute and cross-check a UDV run against the LDS file.")
    def verify_udvs_command(
        run_name: RunNameOption,
        config_path: UdvConfigOption = DEFAULT_CONFIG_PATH,
        baseline: Annotated[
            Path | None, typer.Option("--baseline", help="Previous <run>.jsonl to diff against.")
        ] = None,
    ) -> None:
        exit_on_problems(guarded(lambda: run_verify(config_path, run_name, baseline)))
