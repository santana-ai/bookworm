"""Source hashes and library versions recorded in the confidence reports."""

import platform
from typing import Any

import bookworm.data.io
import numpy as np
import scipy
import sklearn

from experiments import retrieval
from experiments.common import reporting, transcript, udv_run
from experiments.common import splits as common_splits
from experiments.common import stats as retrieval_stats
from experiments.common.provenance import Source, source_hashes
from experiments.udv import calibrate_threshold
from experiments.verifier.runtime import package_sources

Record = dict[str, Any]

SOURCES = package_sources(__file__, "confidence_policies.py")
ENTRY_POINT = SOURCES[0]
CODE_MODULES: tuple[Source, ...] = (
    transcript,
    *transcript.SOURCES,
    udv_run,
    reporting,
    common_splits,
    calibrate_threshold,
    bookworm.data.io,
    retrieval,
    retrieval_stats,
)


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def code_hashes() -> Record:
    return dict(sorted(source_hashes(*CODE_MODULES, *SOURCES).items()))
