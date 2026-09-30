"""Command-line pieces shared by the actor modules."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from bookworm import write_json

Record = dict[str, Any]

MODEL_HELP = "Hugging Face model id or local path (overrides config)"
MISSING_MODEL = "set [model] name in the config, or pass --model"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def require_model(parser: argparse.ArgumentParser, model: str, dry_run: bool) -> None:
    """A model run needs a model name; a dry run does not."""
    if not model and not dry_run:
        parser.error(MISSING_MODEL)


def write_report(report: Record, path: Path) -> None:
    """Write a dry-run report to `path` and print it."""
    write_json(report, path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
