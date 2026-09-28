"""Command-line interface of the verifier over the UDVs."""

import argparse
from pathlib import Path

from experiments.verifier.nli.config import (
    DEVICES,
)
from experiments.verifier.udv_scores.apply import command_apply
from experiments.verifier.udv_scores.config import load_config
from experiments.verifier.udv_scores.scoring import command_score, command_translate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entailment-style verifier scores on the UDV evidence: translate, score "
        "with the E3 scorers, then apply the E3x primary refitted on the benchmark train split."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv_verifier.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("translate", "score"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=DEVICES, default=None)
    commands.add_parser("apply", help="refit the primary on train and write the UDV scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = {
        "translate": command_translate,
        "score": command_score,
        "apply": command_apply,
    }
    commands[args.command](args, config)
