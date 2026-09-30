"""Command-line interface of the translation step."""

import argparse
from pathlib import Path

import transformers

from experiments.common.hub_offline import enforce_offline
from experiments.common.udv_run import seed_everything
from experiments.verifier.translate.commands import (
    command_fetch,
    command_plan,
    command_report,
    command_translate,
)
from experiments.verifier.translate.config import load_config
from experiments.verifier.translate.report import code_hashes
from experiments.verifier.translate.spot_check import command_spot_check
from experiments.verifier.translate.store import selected_model

DEVICES = ("auto", "cpu", "mps", "cuda")


def add_selection_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--run-name", required=True)
    command.add_argument("--output-dir", type=Path, default=None)
    command.add_argument("--splits", nargs="+", default=None)
    command.add_argument("--limit", type=int, default=None, help="first N opinions per split")
    command.add_argument(
        "--final-test",
        action="store_true",
        help="also read the final_test splits; never used for choices",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portuguese to English machine translation of the E3 premises and "
        "hypotheses (benchmark opinions and chunk parts), with a resumable cache."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/translation.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "plan", "translate", "report", "spot-check"):
        command = commands.add_parser(name)
        command.add_argument(
            "--model",
            default=None,
            help="translation condition of [models] (default: models.default, nllb)",
        )
        if name == "fetch":
            continue
        command.add_argument("--cache-dir", type=Path, default=None)
        if name in ("plan", "translate", "report"):
            add_selection_arguments(command)
        if name in ("translate", "spot-check"):
            command.add_argument("--device", choices=DEVICES, default=None)
        if name in ("translate", "report"):
            command.add_argument("--estimate-from-plan", type=Path, default=None)
        if name == "spot-check":
            command.add_argument("--cached-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    code_hashes()
    args = parse_args()
    config = load_config(args.config)
    selected_model(config, args.model)
    if args.command != "fetch" and config.hf_hub_offline:
        enforce_offline()
    seed_everything(config.seed)
    transformers.logging.set_verbosity_error()
    commands = {
        "fetch": command_fetch,
        "plan": command_plan,
        "translate": command_translate,
        "report": command_report,
        "spot-check": command_spot_check,
    }
    commands[args.command](args, config)
