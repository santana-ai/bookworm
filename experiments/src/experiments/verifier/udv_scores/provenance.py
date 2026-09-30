"""Source hashes recorded in the reports of the verifier run over the UDVs."""

from typing import Any

from experiments.common.provenance import source_hashes
from experiments.verifier import nli_exploration
from experiments.verifier.nli.provenance import code_hashes as nli_code_hashes
from experiments.verifier.runtime import package_sources

Record = dict[str, Any]

SOURCES = package_sources(__file__, "udv_verifier.py")


def code_hashes() -> Record:
    return {
        **source_hashes(*SOURCES),
        **source_hashes(*nli_exploration.SOURCES),
        **nli_code_hashes(),
    }
