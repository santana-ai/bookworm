import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from bookworm import __version__
from bookworm.config import TfidfSettings, UdvConfig, load_split_config, load_udv_config
from bookworm.data.io import load_hearings, write_json
from bookworm.data.schemas import HearingRecord
from bookworm.data.splits import SPLIT_NAMES, build_temporal_split
from bookworm.data.verify_splits import verify_split_run
from bookworm.errors import BookwormError, ConfigError
from bookworm.features.encoders import CachedEncoder, SentenceEncoder
from bookworm.features.tfidf import TfidfEncoder
from bookworm.udv.build import EvidenceSettings, build_udvs, select_hearings, udv_corpus
from bookworm.udv.coverage import summarize_run
from bookworm.udv.export import DEFAULT_TOP_K, check_run_pipeline, export_hearing, split_of
from bookworm.udv.schemas import (
    ParsedUdvLines,
    UdvRecord,
    load_udv_jsonl,
    read_udv_jsonl,
    write_udv_jsonl,
)
from bookworm.udv.verify import compare_with_baseline, coverage_hearings, verify_udv_run

EncoderFactory = Callable[[UdvConfig, Sequence[HearingRecord]], SentenceEncoder]

ERROR_EXIT_CODE = 2
PROBLEMS_EXIT_CODE = 1
DEFAULT_CONFIG_PATH = Path("configs/udv.toml")
DEFAULT_SPLIT_CONFIG_PATH = Path("configs/splits.toml")
SPLIT_SUMMARY_KEYS = ("hearings", "share_of_hearings", "first_date", "last_date", "udvs")


def default_encoder_factory(
    config: UdvConfig, hearings: Sequence[HearingRecord]
) -> SentenceEncoder:
    settings = config.encoder
    if isinstance(settings, TfidfSettings):
        return TfidfEncoder.fit(udv_corpus(hearings), max_features=settings.max_features)
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


def check_new_run(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ConfigError(f"{existing[0]}: run file already exists; pass --overwrite to replace it")


def run_build(request: BuildRequest, encoder_factory: EncoderFactory) -> None:
    config = load_udv_config(request.config_path)
    seed_everything(config.seed)
    hearings = select_hearings(
        load_lds(config.lds_path, config.expected_sha256), request.limit, request.ids or None
    )
    if not hearings:
        raise ConfigError("the hearing selection is empty")
    records_path, coverage_path = output_paths(config, request.run_name)
    check_new_run((records_path, coverage_path), request.overwrite)
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
    run = build_udvs(hearings, cached, settings, on_hearing=report_progress)
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
    echo_json({**summary["opinions"], **summary["timing"]})


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


@dataclass(frozen=True)
class ExportRequest:
    config_path: Path
    run_name: str
    hearing_id: int
    output: Path
    top_k: int
    split_manifest: Path | None


def run_export(request: ExportRequest, encoder_factory: EncoderFactory) -> None:
    config = load_udv_config(request.config_path)
    records_path, coverage_path = run_paths(config, request.run_name)
    parsed = read_run_records(records_path)
    if parsed.errors:
        raise ConfigError(f"{records_path}: {parsed.errors[0]}")
    coverage = read_json_object(coverage_path, "coverage file")
    hearings = load_lds(config.lds_path, config.expected_sha256)
    try:
        run_hearings = coverage_hearings(coverage, hearings)
        check_run_pipeline(coverage.get("pipeline"))
    except ConfigError as error:
        raise ConfigError(f"{coverage_path}: {error}") from error
    hearing = next((item for item in run_hearings if item.id == request.hearing_id), None)
    if hearing is None:
        raise ConfigError(f"hearing {request.hearing_id} is not part of run {request.run_name}")
    split = None
    if request.split_manifest is not None:
        split = split_of(
            read_json_object(request.split_manifest, "split manifest"), request.hearing_id
        )
    seed_everything(config.seed)
    encoder = encoder_factory(config, run_hearings)
    payload = export_hearing(
        hearing,
        [record for record in parsed.records if record.hearing_id == request.hearing_id],
        CachedEncoder(encoder, config.cache_dir),
        run_name=request.run_name,
        pipeline=coverage.get("pipeline"),
        top_k=request.top_k,
        split=split,
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


def create_app(encoder_factory: EncoderFactory = default_encoder_factory) -> typer.Typer:
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
    ) -> None:
        request = BuildRequest(config_path, run_name, limit, ids, overwrite)
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
    ) -> None:
        request = ExportRequest(config_path, run_name, hearing_id, output, top_k, split_manifest)
        try:
            run_export(request, encoder_factory)
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

    return app


app = create_app()
