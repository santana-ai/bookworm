import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from bookworm import __version__
from bookworm.actors.config import ActorsConfig, load_actors_config
from bookworm.actors.schemas import UdvActorLink, write_udv_actor_links
from bookworm.actors.speeches import ActorSpeeches, write_actor_outputs
from bookworm.config import TfidfSettings, UdvConfig, load_split_config, load_udv_config
from bookworm.data.io import json_path, load_hearings, sha256_of_file, write_json
from bookworm.data.schemas import HearingRecord
from bookworm.data.splits import SPLIT_NAMES, build_temporal_split
from bookworm.data.verify_splits import verify_split_run
from bookworm.errors import BookwormError, ConfigError
from bookworm.features.encoders import CachedEncoder, RunCacheEncoder, SentenceEncoder
from bookworm.features.tfidf import TfidfEncoder
from bookworm.pipeline import run_pipeline
from bookworm.profiles.config import load_split_filter_config
from bookworm.profiles.generate import (
    ClientFactory,
    GenerateRequest,
    default_client_factory,
    run_generate_profiles,
)
from bookworm.profiles.review import run_sample_profile_review, run_score_profile_review
from bookworm.profiles.split_filter import build_split_filter
from bookworm.profiles.validate import run_validate_profiles
from bookworm.udv.build import EvidenceSettings, select_hearings, udv_corpus
from bookworm.udv.coverage import summarize_run
from bookworm.udv.export import DEFAULT_TOP_K, check_run_pipeline, export_hearing, split_of
from bookworm.udv.schemas import (
    ParsedUdvLines,
    UdvRecord,
    load_udv_jsonl,
    read_udv_jsonl,
    write_udv_jsonl,
)
from bookworm.udv.signals import SiteSignals, load_site_signals
from bookworm.udv.site import HEARINGS_DIR_NAME, INDEX_FILE_NAME, export_site
from bookworm.udv.verify import compare_with_baseline, coverage_hearings, verify_udv_run

EncoderFactory = Callable[[UdvConfig, Sequence[HearingRecord]], SentenceEncoder]

ERROR_EXIT_CODE = 2
PROBLEMS_EXIT_CODE = 1
DEFAULT_CONFIG_PATH = Path("configs/udv.toml")
DEFAULT_SPLIT_CONFIG_PATH = Path("configs/splits.toml")
SPLIT_SUMMARY_KEYS = ("hearings", "share_of_hearings", "first_date", "last_date", "udvs")
PACKAGE_DIR = Path(__file__).resolve().parent
SITE_DATA_PARTS = ("web", "app", "data")


def fit_tfidf_encoder(settings: TfidfSettings, hearings: Sequence[HearingRecord]) -> TfidfEncoder:
    return TfidfEncoder.fit(udv_corpus(hearings), max_features=settings.max_features)


def default_encoder_factory(
    config: UdvConfig, hearings: Sequence[HearingRecord]
) -> SentenceEncoder:
    settings = config.encoder
    if isinstance(settings, TfidfSettings):
        return fit_tfidf_encoder(settings, hearings)
    try:
        from bookworm.features import sentence_transformer
    except ModuleNotFoundError as error:
        raise ConfigError(
            f"encoder kind {settings.kind!r} needs the optional 'embeddings' extra: {error}"
        ) from error
    sentence_transformer.seed_torch(config.seed)
    return sentence_transformer.SentenceTransformerEncoder(
        settings.name, settings.revision, settings.device, settings.batch_size
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def echo_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def fail(error: BookwormError) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(ERROR_EXIT_CODE)


def show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def load_lds(path: Path, expected_sha256: str) -> list[HearingRecord]:
    if not path.is_file():
        raise ConfigError(f"{path}: LDS file not found")
    return load_hearings(path, expected_sha256)


def load_baseline(path: Path) -> list[UdvRecord]:
    try:
        return load_udv_jsonl(path)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read baseline run: {error}") from error


def load_split_udvs(path: Path) -> list[UdvRecord] | None:
    if not path.exists():
        return None
    try:
        return load_udv_jsonl(path)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read UDV run: {error}") from error


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{path}: {description} not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError(f"{path}: {description} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: invalid JSON: {error}") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot read {description}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: {description} is not a JSON object")
    return payload


def read_run_records(path: Path) -> ParsedUdvLines:
    try:
        return read_udv_jsonl(path)
    except UnicodeDecodeError as error:
        raise ConfigError(f"{path}: run file is not UTF-8: {error}") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot read run file: {error}") from error


def output_paths(config: UdvConfig, run_name: str) -> tuple[Path, Path]:
    return (
        config.output_dir / f"{run_name}.jsonl",
        config.output_dir / f"{run_name}_coverage.json",
    )


@dataclass(frozen=True)
class BuildRequest:
    config_path: Path
    run_name: str
    limit: int | None
    ids: list[int] | None
    overwrite: bool
    actors_config_path: Path | None = None


def check_new_run(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ConfigError(f"{existing[0]}: run file already exists; pass --overwrite to replace it")


def actor_links_path(config: UdvConfig, run_name: str) -> Path:
    return config.output_dir / f"{run_name}_actor_links.jsonl"


def load_run_actors_config(path: Path | None, config: UdvConfig) -> ActorsConfig | None:
    if path is None:
        return None
    actors_config = load_actors_config(path)
    if actors_config.expected_sha256 != config.expected_sha256:
        raise ConfigError(
            f"{path}: LDS sha256 {actors_config.expected_sha256} differs from the UDV config "
            f"{config.expected_sha256}"
        )
    return actors_config


def actor_summary(actors: ActorSpeeches, links: Sequence[UdvActorLink]) -> dict[str, Any]:
    return {
        "actors": len(actors.records),
        "actor_turns_kept": actors.stats["turns"]["kept"],
        "udvs_linked_to_actor": sum(link.actor_key is not None for link in links),
    }


def run_build(request: BuildRequest, encoder_factory: EncoderFactory) -> None:
    config = load_udv_config(request.config_path)
    actors_config = load_run_actors_config(request.actors_config_path, config)
    seed_everything(config.seed)
    hearings = select_hearings(
        load_lds(config.lds_path, config.expected_sha256), request.limit, request.ids or None
    )
    if not hearings:
        raise ConfigError("the hearing selection is empty")
    records_path, coverage_path = output_paths(config, request.run_name)
    links_path = actor_links_path(config, request.run_name)
    actor_paths = () if actors_config is None else (*actors_config.output_paths, links_path)
    check_new_run((records_path, coverage_path, *actor_paths), request.overwrite)
    encoder = encoder_factory(config, hearings)
    typer.echo(
        f"{len(hearings)} hearings | {encoder.name}@{encoder.revision[:7]} "
        f"({encoder.cache_identity})",
        err=True,
    )
    cached = CachedEncoder(encoder, config.cache_dir)

    def report_progress(number: int, hearing: HearingRecord, records: int, seconds: float) -> None:
        typer.echo(
            f"[{number}/{len(hearings)}] hearing {hearing.id}: "
            f"{records} opinions in {seconds:.1f}s",
            err=True,
        )

    settings = EvidenceSettings(embedding_threshold=config.embedding_threshold)
    pipeline_run = run_pipeline(
        hearings, cached, settings, actors_config, on_hearing=report_progress
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
    )
    write_udv_jsonl(run.records, records_path)
    write_json(summary, coverage_path)
    if pipeline_run.actors is None:
        echo_json({**summary["opinions"], **summary["timing"]})
        return
    write_actor_outputs(pipeline_run.actors)
    write_udv_actor_links(pipeline_run.links, links_path)
    echo_json(
        {
            **summary["opinions"],
            **summary["timing"],
            **actor_summary(pipeline_run.actors, pipeline_run.links),
        }
    )


def run_paths(config: UdvConfig, run_name: str) -> tuple[Path, Path]:
    records_path, coverage_path = output_paths(config, run_name)
    for path in (records_path, coverage_path):
        if not path.is_file():
            raise ConfigError(f"{path}: run file not found")
    return records_path, coverage_path


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
        report["baseline_diff"] = compare_with_baseline(parsed.records, load_baseline(baseline))
    echo_json(report)
    return verification.ok


def run_device(coverage: dict[str, Any]) -> str:
    found, device = json_path(coverage, ("encoder_runtime", "device"))
    if not found or not isinstance(device, str):
        raise ConfigError(
            "encoder_runtime.device is missing or not text, so the embedding cache files of "
            "the run cannot be named"
        )
    return device


def run_cache_encoder(
    config: UdvConfig, hearings: Sequence[HearingRecord], coverage: dict[str, Any]
) -> SentenceEncoder:
    settings = config.encoder
    if isinstance(settings, TfidfSettings):
        return fit_tfidf_encoder(settings, hearings)
    return RunCacheEncoder(settings.name, settings.revision, run_device(coverage))


@dataclass(frozen=True)
class ExportRun:
    records: list[UdvRecord]
    hearings: list[HearingRecord]
    pipeline: object
    encoder: CachedEncoder
    records_path: Path


def load_export_run(config_path: Path, run_name: str) -> ExportRun:
    config = load_udv_config(config_path)
    records_path, coverage_path = run_paths(config, run_name)
    parsed = read_run_records(records_path)
    if parsed.errors:
        raise ConfigError(f"{records_path}: {parsed.errors[0]}")
    coverage = read_json_object(coverage_path, "coverage file")
    hearings = load_lds(config.lds_path, config.expected_sha256)
    try:
        run_hearings = coverage_hearings(coverage, hearings)
        check_run_pipeline(coverage.get("pipeline"))
        encoder = run_cache_encoder(config, run_hearings, coverage)
    except ConfigError as error:
        raise ConfigError(f"{coverage_path}: {error}") from error
    seed_everything(config.seed)
    return ExportRun(
        parsed.records,
        run_hearings,
        coverage.get("pipeline"),
        CachedEncoder(encoder, config.cache_dir, cache_only=True),
        records_path,
    )


def load_run_signals(report_path: Path | None, run: ExportRun) -> SiteSignals | None:
    if report_path is None:
        return None
    return load_site_signals(report_path, run.records, sha256_of_file(run.records_path))


@dataclass(frozen=True)
class ExportRequest:
    config_path: Path
    run_name: str
    hearing_id: int
    output: Path
    top_k: int
    split_manifest: Path | None
    verifier_report: Path | None = None


def run_export(request: ExportRequest) -> None:
    run = load_export_run(request.config_path, request.run_name)
    hearing = next((item for item in run.hearings if item.id == request.hearing_id), None)
    if hearing is None:
        raise ConfigError(f"hearing {request.hearing_id} is not part of run {request.run_name}")
    signals = load_run_signals(request.verifier_report, run)
    split = None
    if request.split_manifest is not None:
        split = split_of(
            read_json_object(request.split_manifest, "split manifest"), request.hearing_id
        )
    payload = export_hearing(
        hearing,
        [record for record in run.records if record.hearing_id == request.hearing_id],
        run.encoder,
        run_name=request.run_name,
        pipeline=run.pipeline,
        top_k=request.top_k,
        split=split,
        signals=signals,
    )
    write_json(payload, request.output)
    echo_json(
        {
            "hearing": request.hearing_id,
            "split": split,
            "turns": len(payload["turns"]),
            "udvs": len(payload["udvs"]),
            "output": str(request.output),
        }
    )


def default_site_dir(package_dir: Path = PACKAGE_DIR) -> Path:
    project_dir = package_dir.parent.parent
    if not (project_dir / "pyproject.toml").is_file():
        raise ConfigError(
            f"{package_dir}: bookworm is not running from its source tree, so there is no "
            "default web/app/data directory; pass --output"
        )
    return project_dir.joinpath(*SITE_DATA_PARTS)


@dataclass(frozen=True)
class SiteRequest:
    config_path: Path
    run_name: str
    output: Path | None
    top_k: int
    split_manifest: Path | None
    overwrite: bool
    verifier_report: Path | None = None


def check_new_site(output_dir: Path, overwrite: bool) -> None:
    existing = [
        path
        for path in (output_dir / INDEX_FILE_NAME, output_dir / HEARINGS_DIR_NAME)
        if path.exists()
    ]
    if existing and not overwrite:
        raise ConfigError(
            f"{existing[0]}: site export already exists; pass --overwrite to replace its files"
        )


def run_export_site(request: SiteRequest) -> None:
    output_dir = request.output if request.output is not None else default_site_dir()
    check_new_site(output_dir, request.overwrite)
    manifest = (
        None
        if request.split_manifest is None
        else read_json_object(request.split_manifest, "split manifest")
    )
    run = load_export_run(request.config_path, request.run_name)
    signals = load_run_signals(request.verifier_report, run)
    total = len(run.hearings)

    def report_progress(number: int, entry: dict[str, Any], size: int) -> None:
        typer.echo(
            f"[{number}/{total}] hearing {entry['id']}: {entry['n_udvs']} UDVs, {size} bytes",
            err=True,
        )

    site = export_site(
        run.hearings,
        run.records,
        run.encoder,
        output_dir,
        run_name=request.run_name,
        pipeline=run.pipeline,
        top_k=request.top_k,
        split_manifest=manifest,
        on_hearing=report_progress,
        signals=signals,
    )
    echo_json(
        {
            "run": request.run_name,
            "hearings": len(site.index["hearings"]),
            "udvs": sum(entry["n_udvs"] for entry in site.index["hearings"]),
            "bytes": site.total_bytes,
            "output": str(output_dir),
        }
    )


def run_build_splits(config_path: Path) -> None:
    config = load_split_config(config_path)
    hearings = load_lds(config.lds_path, config.expected_sha256)
    artifacts = build_temporal_split(hearings, config, load_split_udvs(config.udv_path))
    write_json(artifacts.manifest, config.manifest_path)
    write_json(artifacts.report, config.report_path)
    splits = artifacts.report["splits"]
    echo_json(
        {
            "split_version": config.split_version,
            "splits": {
                name: {key: splits[name][key] for key in SPLIT_SUMMARY_KEYS} for name in SPLIT_NAMES
            },
            "boundaries": artifacts.manifest["boundaries"],
        }
    )


def run_verify_splits(config_path: Path) -> bool:
    config = load_split_config(config_path)
    manifest = read_json_object(config.manifest_path, "split manifest")
    report = read_json_object(config.report_path, "split report")
    hearings = load_lds(config.lds_path, config.expected_sha256)
    udvs = load_split_udvs(config.udv_path) or []
    verification = verify_split_run(manifest, report, hearings, udvs, config)
    echo_json(verification.to_report())
    return verification.ok


def echo_progress(message: str) -> None:
    typer.echo(message, err=True)


def create_app(
    encoder_factory: EncoderFactory = default_encoder_factory,
    client_factory: ClientFactory = default_client_factory,
) -> typer.Typer:
    app = typer.Typer(
        name="bookworm",
        help=(
            "Build and verify evidence units (UDVs) and temporal splits of the "
            "PublicHearingBR hearings."
        ),
        no_args_is_help=True,
        add_completion=False,
    )

    @app.callback()
    def main(
        version: Annotated[
            bool,
            typer.Option(
                "--version",
                callback=show_version,
                is_eager=True,
                help="Show the version and exit.",
            ),
        ] = False,
    ) -> None:
        pass

    @app.command(
        "build-udvs", help="Build UDV records (opinion to transcript evidence) from the LDS file."
    )
    def build_udvs_command(
        run_name: Annotated[str, typer.Option("--run-name", help="Basename of the output files.")],
        config_path: Annotated[
            Path, typer.Option("--config", help="UDV TOML config.")
        ] = DEFAULT_CONFIG_PATH,
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
        try:
            run_build(request, encoder_factory)
        except BookwormError as error:
            raise fail(error) from error

    @app.command("verify-udvs", help="Recompute and cross-check a UDV run against the LDS file.")
    def verify_udvs_command(
        run_name: Annotated[str, typer.Option("--run-name", help="Basename of the run files.")],
        config_path: Annotated[
            Path, typer.Option("--config", help="UDV TOML config.")
        ] = DEFAULT_CONFIG_PATH,
        baseline: Annotated[
            Path | None, typer.Option("--baseline", help="Previous <run>.jsonl to diff against.")
        ] = None,
    ) -> None:
        try:
            ok = run_verify(config_path, run_name, baseline)
        except BookwormError as error:
            raise fail(error) from error
        if not ok:
            raise typer.Exit(PROBLEMS_EXIT_CODE)

    @app.command(
        "export-hearing",
        help="Write one hearing of a UDV run, with ranked candidate sentences, as demo JSON.",
    )
    def export_hearing_command(
        run_name: Annotated[str, typer.Option("--run-name", help="Basename of the run files.")],
        hearing_id: Annotated[int, typer.Option("--hearing", help="Hearing id to export.")],
        output: Annotated[Path, typer.Option("--output", help="Path of the JSON to write.")],
        config_path: Annotated[
            Path, typer.Option("--config", help="UDV TOML config.")
        ] = DEFAULT_CONFIG_PATH,
        top_k: Annotated[
            int, typer.Option("--top-k", min=1, help="Candidate sentences kept per opinion.")
        ] = DEFAULT_TOP_K,
        split_manifest: Annotated[
            Path | None,
            typer.Option("--split-manifest", help="Split manifest that names the hearing split."),
        ] = None,
        verifier_report: Annotated[
            Path | None,
            typer.Option(
                "--verifier-report",
                help=(
                    "Verifier report of the run; adds the verifier, question and translation "
                    "signals of each UDV."
                ),
            ),
        ] = None,
    ) -> None:
        request = ExportRequest(
            config_path, run_name, hearing_id, output, top_k, split_manifest, verifier_report
        )
        try:
            run_export(request)
        except BookwormError as error:
            raise fail(error) from error

    @app.command(
        "export-site",
        help=(
            "Write every hearing of a UDV run as demo JSON, plus an index.json, read only "
            "from the embedding cache."
        ),
    )
    def export_site_command(
        run_name: Annotated[str, typer.Option("--run-name", help="Basename of the run files.")],
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                help="Directory to write; defaults to web/app/data of the bookworm source tree.",
            ),
        ] = None,
        config_path: Annotated[
            Path, typer.Option("--config", help="UDV TOML config.")
        ] = DEFAULT_CONFIG_PATH,
        top_k: Annotated[
            int, typer.Option("--top-k", min=1, help="Candidate sentences kept per opinion.")
        ] = DEFAULT_TOP_K,
        split_manifest: Annotated[
            Path | None,
            typer.Option("--split-manifest", help="Split manifest that names each hearing split."),
        ] = None,
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace the files of an existing export.")
        ] = False,
        verifier_report: Annotated[
            Path | None,
            typer.Option(
                "--verifier-report",
                help=(
                    "Verifier report of the run; adds the verifier, question and translation "
                    "signals of each UDV."
                ),
            ),
        ] = None,
    ) -> None:
        request = SiteRequest(
            config_path, run_name, output, top_k, split_manifest, overwrite, verifier_report
        )
        try:
            run_export_site(request)
        except BookwormError as error:
            raise fail(error) from error

    @app.command(
        "build-splits", help="Build the temporal split manifest and report from the LDS file."
    )
    def build_splits_command(
        config_path: Annotated[
            Path, typer.Option("--config", help="Split TOML config.")
        ] = DEFAULT_SPLIT_CONFIG_PATH,
    ) -> None:
        try:
            run_build_splits(config_path)
        except BookwormError as error:
            raise fail(error) from error

    @app.command(
        "verify-splits", help="Recompute and cross-check a split manifest against the LDS file."
    )
    def verify_splits_command(
        config_path: Annotated[
            Path, typer.Option("--config", help="Split TOML config.")
        ] = DEFAULT_SPLIT_CONFIG_PATH,
    ) -> None:
        try:
            ok = run_verify_splits(config_path)
        except BookwormError as error:
            raise fail(error) from error
        if not ok:
            raise typer.Exit(PROBLEMS_EXIT_CODE)

    @app.command(
        "filter-actor-speeches",
        help="Keep only the hearings of the configured splits in the actor speeches file.",
    )
    def filter_actor_speeches_command(
        config_path: Annotated[Path, typer.Option("--config", help="Actor profiles TOML config.")],
        output: Annotated[
            Path | None,
            typer.Option("--output", help="Filtered speeches JSONL (overrides config)."),
        ] = None,
    ) -> None:
        try:
            config = load_split_filter_config(config_path).with_output(output)
            stats = build_split_filter(config)
        except BookwormError as error:
            raise fail(error) from error
        echo_json(stats)

    @app.command(
        "generate-profiles",
        help="Write one LLM-written profile per actor from an actor speeches file.",
    )
    def generate_profiles_command(
        config_path: Annotated[Path, typer.Option("--config", help="Actor profiles TOML config.")],
        speeches: Annotated[
            Path | None,
            typer.Option("--speeches", "--input", help="Actor speeches JSONL (overrides config)."),
        ] = None,
        output: Annotated[
            Path | None, typer.Option("--output", help="Profiles JSONL (overrides config).")
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Hugging Face model id or local path (overrides config)."),
        ] = None,
        actors: Annotated[
            list[str] | None,
            typer.Option("--actors", help="Only these actors, by exact name; repeat for more."),
        ] = None,
        limit: Annotated[
            int | None, typer.Option("--limit", min=0, help="Process at most N actors this run.")
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run", help="Render every prompt and report sizes, without a model."
            ),
        ] = False,
    ) -> None:
        request = GenerateRequest(config_path, speeches, output, model, actors, limit, dry_run)
        try:
            outcome = run_generate_profiles(request, client_factory, echo_progress)
        except BookwormError as error:
            raise fail(error) from error
        echo_json(outcome.summary)
        if not outcome.ok:
            raise typer.Exit(PROBLEMS_EXIT_CODE)

    @app.command(
        "validate-profiles",
        help=(
            "Score each verified UDV against the profile of its actor and of every other "
            "actor, separating hearings seen at generation from held-out ones."
        ),
    )
    def validate_profiles_command(
        config_path: Annotated[
            Path, typer.Option("--config", help="Profile validation TOML config.")
        ],
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace existing pairs and report files.")
        ] = False,
    ) -> None:
        try:
            echo_json(run_validate_profiles(config_path, overwrite=overwrite))
        except BookwormError as error:
            raise fail(error) from error

    @app.command(
        "sample-profile-review",
        help="Draw a seeded, stratified sample of profile pairs as a CSV with empty judgments.",
    )
    def sample_profile_review_command(
        config_path: Annotated[
            Path, typer.Option("--config", help="Profile validation TOML config.")
        ],
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace an existing review sample.")
        ] = False,
    ) -> None:
        try:
            echo_json(run_sample_profile_review(config_path, overwrite=overwrite))
        except BookwormError as error:
            raise fail(error) from error

    @app.command(
        "score-profile-review",
        help="Compute support proportions with Wilson intervals from a filled review CSV.",
    )
    def score_profile_review_command(
        config_path: Annotated[
            Path, typer.Option("--config", help="Profile validation TOML config.")
        ],
        annotations: Annotated[
            Path, typer.Option("--annotations", help="Review CSV with the judgments filled in.")
        ],
    ) -> None:
        try:
            echo_json(run_score_profile_review(config_path, annotations))
        except BookwormError as error:
            raise fail(error) from error

    return app


app = create_app()
