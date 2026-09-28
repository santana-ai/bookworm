"""``export-hearing`` and ``export-site``: demo JSON read only from the embedding cache."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from bookworm.actors.speeches import collect_actor_speeches
from bookworm.cli.common import (
    DEFAULT_ACTORS_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    RunNameOption,
    UdvConfigOption,
    echo_json,
    evidence_settings,
    guarded,
    load_lds,
    load_run_actors_config,
    read_run_records,
    refuse_existing,
    run_paths,
    seed_everything,
)
from bookworm.cli.udv_commands import fit_tfidf_encoder
from bookworm.config import TfidfSettings, UdvConfig, load_udv_config
from bookworm.data.io import JsonObject, json_path, read_json_object, sha256_of_file, write_json
from bookworm.data.schemas import HearingRecord
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder, RunCacheEncoder, SentenceEncoder
from bookworm.profiles.schemas import read_profiles
from bookworm.profiles.site import ACTORS_FILE_NAME, PROFILES_DIR_NAME, ProfileSiteBuilder
from bookworm.udv.build import EvidenceSettings
from bookworm.udv.export import DEFAULT_TOP_K, check_run_pipeline, export_hearing, split_of
from bookworm.udv.schemas import UdvRecord
from bookworm.udv.signals import SiteSignals, load_site_signals
from bookworm.udv.site import (
    HEARINGS_DIR_NAME,
    INDEX_FILE_NAME,
    HearingCallback,
    SiteExport,
    export_site,
)
from bookworm.udv.site_validation import SiteValidation, load_site_validation
from bookworm.udv.verify import coverage_hearings, coverage_threshold

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SITE_DATA_PARTS = ("web", "app", "data")
PROFILE_CLAIM_KEYS = ("claims", "with_udv", "passage_only", "without_evidence")

TopKOption = Annotated[
    int, typer.Option("--top-k", min=1, help="Candidate units kept per opinion.")
]
VerifierReportOption = Annotated[
    Path | None,
    typer.Option(
        "--verifier-report",
        help=(
            "Verifier report of the run; adds the verifier, question and translation "
            "signals of each UDV."
        ),
    ),
]


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
    """The encoder whose identity names the cache files of the run; it never loads a model."""
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
    settings: EvidenceSettings


def check_run_threshold(coverage: Mapping[str, Any], config: UdvConfig) -> None:
    recorded = coverage_threshold(coverage)
    if recorded != config.embedding_threshold:
        raise ConfigError(
            f"the run was built with embedding_threshold {recorded}, but the config has "
            f"{config.embedding_threshold}"
        )


def load_export_run(config_path: Path, run_name: str) -> ExportRun:
    """Read a finished run and refuse it unless it matches the pipeline of its config."""
    config = load_udv_config(config_path)
    settings = evidence_settings(config)
    records_path, coverage_path = run_paths(config, run_name)
    parsed = read_run_records(records_path)
    if parsed.errors:
        raise ConfigError(f"{records_path}: {parsed.errors[0]}")
    coverage = read_json_object(coverage_path, "coverage file")
    hearings = load_lds(config.lds_path, config.expected_sha256)
    try:
        run_hearings = coverage_hearings(coverage, hearings)
        check_run_pipeline(coverage.get("pipeline"), settings)
        check_run_threshold(coverage, config)
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
        settings,
    )


def load_run_signals(report_path: Path | None, run: ExportRun) -> SiteSignals | None:
    if report_path is None:
        return None
    return load_site_signals(report_path, run.records, sha256_of_file(run.records_path))


def read_split_manifest(path: Path | None) -> JsonObject | None:
    return None if path is None else read_json_object(path, "split manifest")


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
    manifest = read_split_manifest(request.split_manifest)
    split = None if manifest is None else split_of(manifest, request.hearing_id)
    payload = export_hearing(
        hearing,
        [record for record in run.records if record.hearing_id == request.hearing_id],
        run.encoder,
        run_name=request.run_name,
        pipeline=run.pipeline,
        top_k=request.top_k,
        split=split,
        settings=run.settings,
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
    """``web/app/data`` of the bookworm source tree the package runs from."""
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
    profiles: Path | None = None
    actors_config: Path = DEFAULT_ACTORS_CONFIG_PATH
    profiles_run: str | None = None
    human_validation: Path | None = None


def load_run_validation(request: SiteRequest, run: ExportRun) -> SiteValidation | None:
    if request.human_validation is None:
        return None
    return load_site_validation(request.human_validation, run.records, request.run_name)


def load_profile_site(request: SiteRequest, run: ExportRun) -> ProfileSiteBuilder | None:
    if request.profiles is None:
        return None
    if not request.profiles.is_file():
        raise ConfigError(f"{request.profiles}: profile file not found")
    config = load_udv_config(request.config_path)
    actors_config = load_run_actors_config(request.actors_config, config)
    if actors_config is None:
        return None
    speeches = collect_actor_speeches(run.hearings, actors_config)
    return ProfileSiteBuilder(
        read_profiles(request.profiles),
        speeches.records,
        run_name=request.profiles_run or request.profiles.stem,
        source_sha256=sha256_of_file(request.profiles),
    )


def check_new_site(output_dir: Path, overwrite: bool) -> None:
    refuse_existing(
        [
            output_dir / name
            for name in (INDEX_FILE_NAME, HEARINGS_DIR_NAME, ACTORS_FILE_NAME, PROFILES_DIR_NAME)
        ],
        overwrite,
        "site export already exists; pass --overwrite to replace its files",
    )


def site_summary(site: SiteExport, run_name: str, output_dir: Path) -> dict[str, Any]:
    entries = site.index["hearings"]
    summary: dict[str, Any] = {
        "run": run_name,
        "hearings": len(entries),
        "udvs": sum(entry["n_udvs"] for entry in entries),
        "bytes": site.total_bytes,
        "output": str(output_dir),
    }
    if site.profiles is not None:
        actors = site.profiles.actors["actors"]
        summary["profiles"] = {
            "actors": len(actors),
            "linked_udvs": len(site.profiles.actors["udvs"]),
            **{key: sum(actor["claims"][key] for actor in actors) for key in PROFILE_CLAIM_KEYS},
        }
    validation = site.index.get("validation")
    if isinstance(validation, dict):
        summary["validation"] = {
            "judged_udvs": len(validation["udvs"]),
            "bands": validation["verifier_bands"] is not None,
        }
    return summary


def site_progress(total: int) -> HearingCallback:
    def report_progress(number: int, entry: JsonObject, size: int) -> None:
        typer.echo(
            f"[{number}/{total}] hearing {entry['id']}: {entry['n_udvs']} UDVs, {size} bytes",
            err=True,
        )

    return report_progress


def run_export_site(request: SiteRequest) -> None:
    output_dir = request.output if request.output is not None else default_site_dir()
    check_new_site(output_dir, request.overwrite)
    manifest = read_split_manifest(request.split_manifest)
    run = load_export_run(request.config_path, request.run_name)
    signals = load_run_signals(request.verifier_report, run)
    profiles = load_profile_site(request, run)
    validation = load_run_validation(request, run)
    site = export_site(
        run.hearings,
        run.records,
        run.encoder,
        output_dir,
        run_name=request.run_name,
        pipeline=run.pipeline,
        top_k=request.top_k,
        split_manifest=manifest,
        on_hearing=site_progress(len(run.hearings)),
        signals=signals,
        profiles=profiles,
        settings=run.settings,
        validation=validation,
    )
    echo_json(site_summary(site, request.run_name, output_dir))


def add_export_commands(app: typer.Typer) -> None:
    @app.command(
        "export-hearing",
        help="Write one hearing of a UDV run, with ranked candidate units, as demo JSON.",
    )
    def export_hearing_command(
        run_name: RunNameOption,
        hearing_id: Annotated[int, typer.Option("--hearing", help="Hearing id to export.")],
        output: Annotated[Path, typer.Option("--output", help="Path of the JSON to write.")],
        config_path: UdvConfigOption = DEFAULT_CONFIG_PATH,
        top_k: TopKOption = DEFAULT_TOP_K,
        split_manifest: Annotated[
            Path | None,
            typer.Option("--split-manifest", help="Split manifest that names the hearing split."),
        ] = None,
        verifier_report: VerifierReportOption = None,
    ) -> None:
        request = ExportRequest(
            config_path, run_name, hearing_id, output, top_k, split_manifest, verifier_report
        )
        guarded(lambda: run_export(request))

    @app.command(
        "export-site",
        help=(
            "Write every hearing of a UDV run as demo JSON, plus an index.json, read only "
            "from the embedding cache."
        ),
    )
    def export_site_command(
        run_name: RunNameOption,
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                help="Directory to write; defaults to web/app/data of the bookworm source tree.",
            ),
        ] = None,
        config_path: UdvConfigOption = DEFAULT_CONFIG_PATH,
        top_k: TopKOption = DEFAULT_TOP_K,
        split_manifest: Annotated[
            Path | None,
            typer.Option("--split-manifest", help="Split manifest that names each hearing split."),
        ] = None,
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace the files of an existing export.")
        ] = False,
        verifier_report: VerifierReportOption = None,
        profiles: Annotated[
            Path | None,
            typer.Option(
                "--profiles",
                help=(
                    "Actor profiles JSONL; adds actors.json and one profiles/<actor>.json per "
                    "profiled actor."
                ),
            ),
        ] = None,
        actors_config: Annotated[
            Path,
            typer.Option(
                "--actors-config",
                help="Hearing actors TOML config, used with --profiles to rebuild the speeches.",
            ),
        ] = DEFAULT_ACTORS_CONFIG_PATH,
        profiles_run: Annotated[
            str | None,
            typer.Option(
                "--profiles-run",
                help="Name of the profile run shown on the page; defaults to the file name.",
            ),
        ] = None,
        human_validation: Annotated[
            Path | None,
            typer.Option(
                "--human-validation",
                help=(
                    "Final precision report of the run; adds the human judgments, the "
                    "precision per tier and the judgments per verifier band to index.json."
                ),
            ),
        ] = None,
    ) -> None:
        request = SiteRequest(
            config_path,
            run_name,
            output,
            top_k,
            split_manifest,
            overwrite,
            verifier_report,
            profiles,
            actors_config,
            profiles_run,
            human_validation,
        )
        guarded(lambda: run_export_site(request))
