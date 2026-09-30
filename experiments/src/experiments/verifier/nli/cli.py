"""Command-line interface of the NLI verifier experiments."""

import argparse
from pathlib import Path

import transformers

from experiments.common.hub_offline import enforce_offline
from experiments.common.udv_run import seed_everything
from experiments.verifier.decision.laya import fetch_laya
from experiments.verifier.nli.config import DEVICES, VerifierConfig, load_config
from experiments.verifier.nli.pairs import command_pairs
from experiments.verifier.nli.plan import command_plan
from experiments.verifier.nli.provenance import code_hashes
from experiments.verifier.nli.report import command_evaluate
from experiments.verifier.nli.scoring import command_score

SUBSET_RULES = ("random", "first")
DECISION_MODES = ("live", "replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLI verifier experiment on the NLI benchmark (train fit, validation "
        "evaluation) and entailment scoring of UDV-style (proposition, evidence) pairs."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/nli_verifier.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "fetch", help="download the pinned Laya checkpoints of the laya scorers (network)"
    )
    for name in ("score", "plan", "evaluate", "pairs"):
        command = commands.add_parser(name)
        command.add_argument("--run-name", required=True)
        command.add_argument("--output-dir", type=Path, default=None)
        command.add_argument(
            "--final-test",
            action="store_true",
            help="also read, score and evaluate the final_test splits; never used for choices",
        )
        if name in ("score", "pairs", "plan"):
            command.add_argument("--scorers", nargs="+", default=None)
        if name in ("score", "pairs"):
            command.add_argument("--device", choices=DEVICES, default=None)
            command.add_argument("--decision-mode", choices=DECISION_MODES, default="live")
        if name in ("score", "plan"):
            command.add_argument("--limit-per-split", type=int, default=None)
            command.add_argument("--subset-rule", choices=SUBSET_RULES, default="random")
            command.add_argument("--translation-cache-dir", type=Path, default=None)
        if name == "score":
            command.add_argument(
                "--splits",
                nargs="+",
                default=None,
                help="score only these of the fit and evaluate splits (a smoke run)",
            )
        if name == "plan":
            command.add_argument(
                "--timing-from",
                type=Path,
                default=None,
                help="a run directory whose score reports give the measured throughput",
            )
        if name == "evaluate":
            command.add_argument(
                "--declaration",
                default=None,
                help="evaluate with [declarations.<name>] (default: the run name, if declared)",
            )
        if name == "pairs":
            command.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def command_fetch(args: argparse.Namespace, config: VerifierConfig) -> None:
    targets = dict.fromkeys(
        (spec.name, spec.revision, spec.source["subfolder"])
        for spec in config.scorers.values()
        if spec.kind == "laya"
    )
    for name, revision, subfolder in targets:
        directory = fetch_laya(name, revision, subfolder)
        print(f"{name}@{revision} {subfolder or 'root'}: {directory}", flush=True)


def main() -> None:
    code_hashes()
    args = parse_args()
    config = load_config(args.config)
    if config.hf_hub_offline and args.command != "fetch":
        enforce_offline()
    seed_everything(config.seed)
    transformers.logging.set_verbosity_error()
    commands = {
        "fetch": command_fetch,
        "score": command_score,
        "plan": command_plan,
        "evaluate": command_evaluate,
        "pairs": command_pairs,
    }
    commands[args.command](args, config)
