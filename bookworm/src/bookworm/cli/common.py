"""Output, errors, exit codes and shared options of the ``bookworm`` commands."""

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from bookworm.actors.config import ActorsConfig, load_actors_config
from bookworm.config import UdvConfig
from bookworm.data.io import load_hearings
from bookworm.data.schemas import HearingRecord
from bookworm.errors import BookwormError, ConfigError
from bookworm.udv.build import EvidenceSettings
from bookworm.udv.schemas import ParsedUdvLines, read_udv_jsonl

ERROR_EXIT_CODE = 2
PROBLEMS_EXIT_CODE = 1
DEFAULT_CONFIG_PATH = Path("configs/udv.toml")
DEFAULT_SPLIT_CONFIG_PATH = Path("configs/splits.toml")
DEFAULT_ACTORS_CONFIG_PATH = Path("configs/hearing_actors.toml")

UdvConfigOption = Annotated[Path, typer.Option("--config", help="UDV TOML config.")]
SplitConfigOption = Annotated[Path, typer.Option("--config", help="Split TOML config.")]
ProfilesConfigOption = Annotated[Path, typer.Option("--config", help="Actor profiles TOML config.")]
ProfileValidationConfigOption = Annotated[
    Path, typer.Option("--config", help="Profile validation TOML config.")
]
RunNameOption = Annotated[str, typer.Option("--run-name", help="Basename of the run files.")]


def echo_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def echo_progress(message: str) -> None:
    typer.echo(message, err=True)


def fail(error: BookwormError) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(ERROR_EXIT_CODE)


def guarded[T](action: Callable[[], T]) -> T:
    """Run a command body, turning a ``BookwormError`` into a message and exit code 2."""
    try:
        return action()
    except BookwormError as error:
        raise fail(error) from error


def exit_on_problems(ok: bool) -> None:
    if not ok:
        raise typer.Exit(PROBLEMS_EXIT_CODE)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_lds(path: Path, expected_sha256: str) -> list[HearingRecord]:
    if not path.is_file():
        raise ConfigError(f"{path}: LDS file not found")
    return load_hearings(path, expected_sha256)


def evidence_settings(config: UdvConfig) -> EvidenceSettings:
    return EvidenceSettings(
        embedding_threshold=config.embedding_threshold,
        semantic_unit=config.semantic_unit,
        quote_extent=config.quote_extent,
    )


def output_paths(config: UdvConfig, run_name: str) -> tuple[Path, Path]:
    return (
        config.output_dir / f"{run_name}.jsonl",
        config.output_dir / f"{run_name}_coverage.json",
    )


def run_paths(config: UdvConfig, run_name: str) -> tuple[Path, Path]:
    """Records and coverage files of an existing run; raises ``ConfigError`` if one is missing."""
    records_path, coverage_path = output_paths(config, run_name)
    for path in (records_path, coverage_path):
        if not path.is_file():
            raise ConfigError(f"{path}: run file not found")
    return records_path, coverage_path


def refuse_existing(paths: Sequence[Path], overwrite: bool, reason: str) -> None:
    """Raise ``ConfigError`` naming the first existing path, unless ``overwrite`` is set."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ConfigError(f"{existing[0]}: {reason}")


def read_run_records(path: Path) -> ParsedUdvLines:
    try:
        return read_udv_jsonl(path)
    except UnicodeDecodeError as error:
        raise ConfigError(f"{path}: run file is not UTF-8: {error}") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot read run file: {error}") from error


def load_run_actors_config(path: Path | None, config: UdvConfig) -> ActorsConfig | None:
    """Actor speeches config of a UDV run, refused when it reads another LDS file."""
    if path is None:
        return None
    actors_config = load_actors_config(path)
    if actors_config.expected_sha256 != config.expected_sha256:
        raise ConfigError(
            f"{path}: LDS sha256 {actors_config.expected_sha256} differs from the UDV config "
            f"{config.expected_sha256}"
        )
    return actors_config
