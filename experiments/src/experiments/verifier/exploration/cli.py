"""Command-line interface of the verifier exploration."""

import argparse
from pathlib import Path

from experiments.verifier.exploration.config import load_exploration_config
from experiments.verifier.exploration.confirm import command_confirm
from experiments.verifier.exploration.explore import command_explore
from experiments.verifier.exploration.final_test import command_final_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploration of the stored NLI verifier scores: grouped cross-validation "
        "on train to choose one candidate, then one confirmation on validation."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nli_verifier_exploration.toml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "explore", help="cross-validate every candidate on train (reads train only)"
    )
    commands.add_parser("confirm", help="evaluate the selected candidate once on validation")
    commands.add_parser("final-test", help="evaluate the declared final candidates once on test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_exploration_config(args.config)
    if args.command == "explore":
        command_explore(args, config)
    elif args.command == "confirm":
        command_confirm(args, config)
    else:
        command_final_test(args, config)
