"""Source hashes and runtime environment recorded in every report."""

import functools
import os
import platform
from typing import Any

import bookworm.data.io
import huggingface_hub
import laya
import numpy as np
import sentence_transformers
import sklearn
import torch
import transformers

from experiments.common import cache_lock, hub_offline, reporting, transcript, udv_run
from experiments.common import splits as common_splits
from experiments.common import stats as retrieval_stats
from experiments.common.hub_offline import offline_state
from experiments.common.provenance import Source, source_hashes
from experiments.data import nli_benchmark
from experiments.udv import calibrate_threshold
from experiments.verifier import (
    benchmark_inputs,
    decision,
    decision_models,
    decision_scoring,
    runtime,
    stats,
    translation,
)
from experiments.verifier.runtime import package_sources

Record = dict[str, Any]

SOURCES = package_sources(__file__, "nli_experiments.py")


@functools.cache
def code_hashes() -> Record:
    """Hashes of the entry point, every module of this package and the modules they rely on."""
    modules: tuple[Source, ...] = (
        transcript,
        *transcript.SOURCES,
        udv_run,
        nli_benchmark,
        cache_lock,
        calibrate_threshold,
        bookworm.data.io,
        hub_offline,
        reporting,
        common_splits,
        decision_models,
        decision_scoring,
        decision,
        stats,
        runtime,
        benchmark_inputs,
        retrieval_stats,
        *translation.SOURCES,
    )
    return source_hashes(*SOURCES, *modules)


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "laya": laya.__version__,
        "torch_threads": torch.get_num_threads(),
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "mps_available": torch.backends.mps.is_available(),
        "platform": platform.platform(),
        "hub_offline": offline_state(),
    }
