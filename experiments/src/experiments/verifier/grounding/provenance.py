"""Source hashes and the run record of the grounding reports."""

from typing import Any

from experiments.common.provenance import Source, source_hashes
from experiments.common.reporting import file_record
from experiments.verifier import (
    confidence_policies,
    nli_experiments,
    nli_exploration,
    translation,
    udv_verifier,
)
from experiments.verifier.grounding.config import Config
from experiments.verifier.nli.provenance import environment
from experiments.verifier.runtime import package_sources

Record = dict[str, Any]

SOURCES = package_sources(__file__, "confidence_v2.py")
CODE_MODULES: tuple[Source, ...] = (
    *confidence_policies.SOURCES,
    *udv_verifier.SOURCES,
    *nli_experiments.SOURCES,
    *nli_exploration.SOURCES,
    *translation.SOURCES,
)


def code_hashes() -> Record:
    return source_hashes(*SOURCES, *CODE_MODULES)


def run_record(config: Config) -> Record:
    return {
        "config": file_record(config.path),
        "code": code_hashes(),
        "environment": environment(),
    }
