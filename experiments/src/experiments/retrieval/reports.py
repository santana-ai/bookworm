"""The run and summary reports of the retrieval harness, with their provenance sections."""

import importlib.metadata
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bookworm.data.io
import huggingface_hub
import numpy as np
import rank_bm25
import scipy
import sentence_transformers
import sklearn
import torch
import transformers

from experiments.common import hub_offline, splits, stats, transcript, udv_run
from experiments.common.provenance import code_section
from experiments.common.reporting import utc_timestamp
from experiments.retrieval.config import ExperimentConfig, RunPlan, SummaryPlan
from experiments.retrieval.summary import DEFINITIONS
from experiments.retrieval.workload import Workload, workload_counts
from experiments.verifier import decision_models, decision_scoring

Record = dict[str, Any]

RETRIEVAL_DIR = Path(__file__).parent
CODE_SOURCES = (
    *transcript.CODE_SOURCES,
    udv_run,
    splits,
    bookworm.data.io,
    decision_models,
    decision_scoring,
    hub_offline,
    stats,
    RETRIEVAL_DIR,
)
RANK_BM25_VERSION = "0.2.2 (no __version__ attribute)"
QUESTION = (
    "given an opinion and the speaker's candidate units, which representation ranks a "
    "supporting passage first"
)
BASELINE_ROLES = {
    "retriever_role": "reference retriever of E1, compared within each unit kind",
    "unit_role": "reference unit kind of E2, compared within each retriever",
}


@dataclass(frozen=True)
class RunContext:
    """The runtime facts every run report of one ``run`` command shares."""

    device: str
    offline: Record
    code: Record
    load_seconds: float


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rank_bm25": getattr(rank_bm25, "__version__", RANK_BM25_VERSION),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "laya": importlib.metadata.version("laya"),
        "platform": platform.platform(),
    }


def code_hashes() -> Record:
    return dict(sorted(code_section(*CODE_SOURCES).items()))


def run_report(
    config: ExperimentConfig,
    plan: RunPlan,
    context: RunContext,
    workload: Workload,
    retriever_id: str,
    details: Record,
    written: list[Record],
) -> Record:
    """The report of one retriever over the workload of a ``run`` command."""
    return {
        "run_name": plan.run_name,
        "retriever": retriever_id,
        "created_at": utc_timestamp(),
        "splits_used": list(plan.splits),
        "final_test": plan.final_test,
        "benches": list(plan.benches),
        "units": list(plan.kinds),
        "limits": {"hearings": plan.limit_hearings, "queries_per_bench": plan.limit_queries},
        "device": context.device,
        "hub_offline": {"configured": config.run["hf_hub_offline"], **context.offline},
        "workload": {
            "hearings": len(workload.hearings),
            "queries": workload_counts(workload),
            "query_checks": {bench: dict(counter) for bench, counter in workload.checks.items()},
            "sentence_span_checks": dict(workload.span_checks),
            "load_seconds": round(context.load_seconds, 1),
        },
        "retriever_details": details["details"],
        "timing": details["timing"],
        "outputs": written,
        "sources": workload.sources,
        "code": context.code,
        "environment": environment(),
        "config": config.source,
    }


def run_report_name(plan: RunPlan, retriever_id: str) -> str:
    selection = "__".join("+".join(items) for items in (plan.splits, plan.kinds, plan.benches))
    return f"{retriever_id}__{selection}_report.json"


def summary_report(
    config: ExperimentConfig,
    plan: SummaryPlan,
    summaries: Record,
    comparisons: Record,
    inputs: list[Record],
    code: Record,
) -> Record:
    """The report of the ``summarize`` command: summaries, E1/E2 comparisons and sources."""
    evaluation = config.evaluation
    return {
        "run_name": plan.run_name,
        "created_at": utc_timestamp(),
        "splits_used": list(plan.splits),
        "final_test": plan.final_test,
        "baseline": {
            "retriever": plan.baseline_retriever,
            "unit": plan.baseline_unit,
            **BASELINE_ROLES,
            "configured": plan.baseline
            == (evaluation["baseline_unit"], evaluation["baseline_retriever"]),
        },
        "question": QUESTION,
        "declaration": config.source.get("declarations", {}).get(plan.run_name),
        "benchmarks": config.benches,
        "definitions": DEFINITIONS,
        "summaries": summaries,
        "comparisons": comparisons,
        "inputs": inputs,
        "run_reports": sorted(str(path) for path in (plan.run_dir / "runs").glob("*_report.json")),
        "code": code,
        "environment": environment(),
        "config": config.source,
    }


def summary_report_name(plan: SummaryPlan, configured: bool) -> str:
    suffix = "final_test_report" if "test" in plan.splits else "report"
    if not configured:
        suffix = f"{suffix}_vs_{plan.baseline_retriever}_{plan.baseline_unit}"
    return f"{plan.run_name}_{suffix}.json"
