"""Sections the sample and repeat reports share: code, environment, seeds and output files."""

import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from bookworm import sha256_of_file

from experiments.common import transcript, udv_run
from experiments.common.provenance import code_section
from experiments.validation.generate_sample.config import ValidationConfig

Record = dict[str, Any]

PACKAGE_DIR = Path(__file__).parent
GENERATOR = "numpy PCG64"


def code_hashes() -> Record:
    return code_section(*transcript.CODE_SOURCES, udv_run, PACKAGE_DIR)


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def file_entry(path: Path) -> Record:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_of_file(path)}


def seeds_section(config: ValidationConfig) -> Record:
    return {"seed": config.seed, "streams": config.seed_streams, "generator": GENERATOR}


def sample_name_of(config: ValidationConfig, run_name: str) -> str:
    return f"{config.version}_{run_name}"
