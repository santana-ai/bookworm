"""Command-line interface of the confidence experiments."""

import argparse
from pathlib import Path

from experiments.verifier.confidence.collect import command_collect
from experiments.verifier.confidence.config import load_config
from experiments.verifier.confidence.pairs import command_pairs
from experiments.verifier.confidence.report import command_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confidence policies for retrieved top-1 evidence: collect per-query signals "
        "from the retrieval harness, build entailment pairs for the NLI verifier, fit policies on "
        "train and evaluate them on validation."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/confidence_policies.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "pairs", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-name", default=None, help="default from run.run_name")
        command.add_argument("--targets", default=None, help="comma list of <retriever>.<unit>")
        command.add_argument(
            "--final-test",
            action="store_true",
            help="also read and evaluate the final_test splits; nothing is fitted on them",
        )
        if name == "collect":
            command.add_argument(
                "--hearing-ids", default=None, help="comma list of hearings (smoke tests only)"
            )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = {"collect": command_collect, "pairs": command_pairs, "evaluate": command_evaluate}
    commands[args.command](args, config)
