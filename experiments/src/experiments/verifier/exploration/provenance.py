"""Source hashes recorded in the exploration reports."""

from typing import Any

from experiments.common.provenance import source_hashes
from experiments.verifier.nli.provenance import code_hashes as nli_code_hashes
from experiments.verifier.runtime import package_files

Record = dict[str, Any]

SOURCE_FILES = package_files(__file__, "nli_exploration.py")


def code_hashes() -> Record:
    """Hashes of the exploration modules and of the verifier modules that wrote the scores."""
    return {**source_hashes(*SOURCE_FILES), **nli_code_hashes()}
