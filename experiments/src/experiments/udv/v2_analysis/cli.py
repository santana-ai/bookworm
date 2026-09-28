"""Command line of the udv_v2 analysis: ``analyze``, ``score-annotation``, ``supplement-sheet``."""

import argparse
from pathlib import Path

from experiments.udv.v2_analysis.analysis import command_analyze
from experiments.udv.v2_analysis.paths import (
    DEFAULT_ANALYSIS,
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    DEFAULT_PLAN,
)
from experiments.udv.v2_analysis.scoring import command_score
from experiments.udv.v2_analysis.supplement import SUPPLEMENT_NAME, command_supplement

COMMANDS = {
    "analyze": command_analyze,
    "score-annotation": command_score,
    "supplement-sheet": command_supplement,
}


def add_analyze(commands: argparse._SubParsersAction) -> None:
    analyze = commands.add_parser("analyze", help="diff, lengths, verifier shares and the plan")
    analyze.add_argument("--output", default=DEFAULT_ANALYSIS)
    analyze.add_argument("--plan-output", default=DEFAULT_PLAN)


def add_score(commands: argparse._SubParsersAction) -> None:
    score = commands.add_parser(
        "score-annotation", help="precision per stratum from the filled annotation.csv"
    )
    score.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    score.add_argument("--sample-dir", default=DEFAULT_PATHS["sample_dir"])
    score.add_argument("--plan", default=DEFAULT_PLAN)
    score.add_argument("--final-test", action="store_true")
    score.add_argument("--annotation", default=None)
    score.add_argument("--output", default=None)
    score.add_argument(
        "--supplement-dir",
        default=None,
        help="folder of the udv_v2 supplementary sheet; adds the combined udv_v2 precision",
    )
    score.add_argument(
        "--supplement-annotation",
        default=None,
        help="filled supplementary sheet (default: annotation.csv of --supplement-dir)",
    )


def add_supplement(commands: argparse._SubParsersAction) -> None:
    supplement = commands.add_parser(
        "supplement-sheet",
        help="write the udv_v2 sheet of the sampled items whose evidence changed, judgments empty",
    )
    supplement.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    supplement.add_argument("--sample-dir", default=DEFAULT_PATHS["sample_dir"])
    supplement.add_argument("--plan", default=DEFAULT_PLAN)
    supplement.add_argument("--udv-v1", default=DEFAULT_PATHS["v1"])
    supplement.add_argument("--udv-v2", default=DEFAULT_PATHS["v2"])
    supplement.add_argument("--final-test", action="store_true")
    supplement.add_argument(
        "--output-dir", default=f"artifacts/validation/{SUPPLEMENT_NAME}", help="sheet and key"
    )
    supplement.add_argument(
        "--transcripts-dir",
        default=f"artifacts/cache/validation/{SUPPLEMENT_NAME}/transcripts",
        help="full transcripts of the hearings in the sheet (raw dataset text, not versioned)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare udv_v2 with udv_v1 and score the validation sample for both runs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_analyze(commands)
    add_score(commands)
    add_supplement(commands)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    COMMANDS[args.command](args)
