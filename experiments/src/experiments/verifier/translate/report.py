"""Provenance and the translation report."""

import argparse
import functools
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import bookworm.data.io
import huggingface_hub
import numpy as np
import tokenizers
import torch
import transformers

from experiments.common import cache_lock, hub_offline, transcript, udv_run
from experiments.common.hub_offline import offline_state
from experiments.common.provenance import source_hashes
from experiments.common.transcript import (
    normalize_whitespace,
)
from experiments.data import nli_benchmark
from experiments.udv import calibrate_threshold
from experiments.verifier import runtime
from experiments.verifier.runtime import now, package_files
from experiments.verifier.translate.config import ModelSpec, TranslationConfig
from experiments.verifier.translate.selection import OpinionUnit, split_records, unit_selection
from experiments.verifier.translate.statistics import (
    cache_timing,
    chunk_statistics,
    estimate_from_plan,
    record_statistics,
    selection_counts,
)
from experiments.verifier.translate.store import TranslationStore, english_probe_pairs, model_record

Record = dict[str, Any]

SOURCE_FILES = package_files(__file__, "translation.py")


@functools.cache
def code_hashes() -> Record:
    """Hashes of the entry point, every module of this package and the modules they rely on."""
    modules = (
        transcript,
        *transcript.SOURCES,
        udv_run,
        nli_benchmark,
        cache_lock,
        calibrate_threshold,
        bookworm.data.io,
        hub_offline,
        runtime,
    )
    return source_hashes(*SOURCE_FILES, *modules)


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "platform": platform.platform(),
        "hub_offline": offline_state(),
    }


def run_directory(config: TranslationConfig, args: argparse.Namespace) -> Path:
    base = args.output_dir if args.output_dir is not None else config.output_dir
    return base / args.run_name


def texts_digest(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def build_report(
    args: argparse.Namespace,
    config: TranslationConfig,
    spec: ModelSpec,
    units: list[OpinionUnit],
    context: Record,
    store: TranslationStore,
    run: Record,
    splits: tuple[str, ...],
) -> Record:
    selection = unit_selection(units, config)
    probe_chunks = {normalize_whitespace(p) for p in config.probe_premises}
    probes_ready = not store.missing(
        [*config.probe_hypotheses, *(s for p in probe_chunks for s in store.segmenter.segments(p))]
    )
    timing = cache_timing(selection, store)
    report = {
        "experiment": "translation",
        "command": args.command,
        "run_name": args.run_name,
        "created_at": now(),
        "splits": split_records(splits, args.final_test),
        "model": model_record(config, spec),
        "segmentation": config.segmenter.describe(),
        "counts": selection_counts(units, selection, config.segmenter),
        "coverage": {
            "model_texts_missing": len(store.missing(selection.model_texts())),
            "complete": not store.missing(selection.model_texts()),
        },
        "opinions": record_statistics(config, selection.opinions, store),
        "segments": record_statistics(config, selection.segments, store),
        "chunks": chunk_statistics(
            [chunk for chunk in selection.chunks if chunk not in probe_chunks], store
        ),
        "timing": {"this_run": run, "cache": timing},
        "label_probes": english_probe_pairs(config, store) if probes_ready else None,
        "store": store.summary(),
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    if getattr(args, "estimate_from_plan", None) is not None:
        report["estimate"] = estimate_from_plan(args.estimate_from_plan, timing)
    return report


def print_report(report: Record) -> None:
    for kind in ("opinions", "segments"):
        part = report[kind]
        print(
            f"{kind}: {part['translated']}/{part['texts']} translated, flags {part['flags']}, "
            f"token ratio median {part['token_ratio'].get('q50')}",
            flush=True,
        )
    print(f"timing {json.dumps(report['timing'])}", flush=True)
    if "estimate" in report:
        print(f"estimate {json.dumps(report['estimate']['by_device'])}", flush=True)
