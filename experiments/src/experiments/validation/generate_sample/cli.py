"""Command line of the validation sample: ``--stage sample`` or ``--stage repeat``."""

import argparse
from pathlib import Path

from experiments.validation.generate_sample.config import load_validation_config
from experiments.validation.generate_sample.repeat_stage import run_repeat_stage
from experiments.validation.generate_sample.sample_stage import run_sample_stage

STAGES = {"sample": run_sample_stage, "repeat": run_repeat_stage}
DEFAULT_CONFIG = Path("configs/validation_sample.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the stratified human validation sample of a UDV run (stage sample) or, "
        "after the first pass is complete, the blind repeat sheet (stage repeat)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=tuple(STAGES), default="sample")
    parser.add_argument("--run-name", required=True, help="UDV run under source.udv_dir")
    parser.add_argument(
        "--final-test", action="store_true", help="allow reading test hearings (final sample)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write outside the repository (a new temp dir unless --output-dir is given)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="dry runs only")
    parser.add_argument(
        "--splits", nargs="+", default=None, help="dry runs only: splits to sample instead"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    STAGES[args.stage](args, load_validation_config(args.config))
