"""Entry point kept for the documented ``python -m experiments.retrieval.experiments`` commands.

The harness lives in ``run`` (commands), ``config``, ``specs``, ``workload``, ``scoring``,
``summary`` and ``reports``; the names other modules import from here are re-exported.
"""

from experiments.retrieval.config import ExperimentConfig, load_config
from experiments.retrieval.run import main
from experiments.retrieval.scoring import (
    build_retriever,
    evaluate_query,
    query_file,
    unit_mean_chars,
)
from experiments.retrieval.specs import decision_rerank_spec, retriever_ids
from experiments.retrieval.workload import Workload, build_workload

__all__ = [
    "ExperimentConfig",
    "Workload",
    "build_retriever",
    "build_workload",
    "decision_rerank_spec",
    "evaluate_query",
    "load_config",
    "main",
    "query_file",
    "retriever_ids",
    "unit_mean_chars",
]

if __name__ == "__main__":
    main()
