import argparse
from pathlib import Path

from experiments.udv.fuzzy_matching.config import load_config
from experiments.udv.fuzzy_matching.run import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure what approximate string matching (rapidfuzz) would recover over the "
        "exact quote and name rules, and at what risk, without changing the pipeline."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/fuzzy_matching.toml"))
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="also read the test split; only for the final, pre-registered evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config), args.final_test)
