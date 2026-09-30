"""``build-splits`` and ``verify-splits``."""

from pathlib import Path

import typer

from bookworm.cli.common import (
    DEFAULT_SPLIT_CONFIG_PATH,
    SplitConfigOption,
    echo_json,
    exit_on_problems,
    guarded,
    load_lds,
)
from bookworm.config import load_split_config
from bookworm.data.io import read_json_object, write_json
from bookworm.data.splits import SPLIT_NAMES, build_temporal_split
from bookworm.data.verify_splits import verify_split_run
from bookworm.udv.schemas import UdvRecord, read_udv_run

SPLIT_SUMMARY_KEYS = ("hearings", "share_of_hearings", "first_date", "last_date", "udvs")


def load_split_udvs(path: Path) -> list[UdvRecord] | None:
    return read_udv_run(path, "UDV run") if path.exists() else None


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


def add_split_commands(app: typer.Typer) -> None:
    @app.command(
        "build-splits", help="Build the temporal split manifest and report from the LDS file."
    )
    def build_splits_command(config_path: SplitConfigOption = DEFAULT_SPLIT_CONFIG_PATH) -> None:
        guarded(lambda: run_build_splits(config_path))

    @app.command(
        "verify-splits", help="Recompute and cross-check a split manifest against the LDS file."
    )
    def verify_splits_command(config_path: SplitConfigOption = DEFAULT_SPLIT_CONFIG_PATH) -> None:
        exit_on_problems(guarded(lambda: run_verify_splits(config_path)))
