"""Command-line interface of the grounding comparison."""

import argparse
from pathlib import Path

from experiments.verifier.grounding.config import load_config
from experiments.verifier.grounding.evaluate import command_evaluate_ea
from experiments.verifier.grounding.laya import command_laya_score, command_laya_smoke
from experiments.verifier.grounding.scoring import (
    command_decide,
    command_score,
    command_smoke,
    command_translate,
)
from experiments.verifier.nli.config import DEVICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confidence signals for UDV evidence: literature grounding scorers against "
        "the serafim cosine and the E3x primary, on the train and validation opinions of the "
        "NLI benchmark (E-A)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/confidence_v2.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("translate", "smoke", "score"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=DEVICES, default="mps")
        if name != "translate":
            command.add_argument("--candidates", nargs="+", default=None)
        if name == "smoke":
            command.add_argument("--fallback", action="store_true")
    for name in ("laya-smoke", "laya-score"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=DEVICES, default="mps")
        command.add_argument("--scorers", nargs="+", default=None)
    commands.add_parser("decide")
    commands.add_parser("evaluate-ea")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = {
        "translate": command_translate,
        "smoke": command_smoke,
        "decide": command_decide,
        "laya-smoke": command_laya_smoke,
        "laya-score": command_laya_score,
        "score": command_score,
        "evaluate-ea": command_evaluate_ea,
    }
    commands[args.command](args, config)
