"""Rank every query of a workload with one retriever and write one row per query and unit kind."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import write_jsonl

from experiments.common.reporting import file_record
from experiments.retrieval.config import ExperimentConfig
from experiments.retrieval.data import HearingData, Query, Unit
from experiments.retrieval.models import (
    Bm25Retriever,
    DecisionRerankRetriever,
    DenseRetriever,
    Ranking,
    RerankRetriever,
    Retriever,
    RrfRetriever,
    Runtime,
    TfidfRetriever,
)
from experiments.retrieval.specs import (
    BM25_PARAMS,
    decision_rerank_spec,
    dense_spec,
    rerank_spec,
    retriever_ids,
)
from experiments.retrieval.workload import Workload

Record = dict[str, Any]

MEAN_CHARS_DECIMALS = 1
RANDOM_ACC_DECIMALS = 6
TIMING_DECIMALS = 2


def build_retriever(rid: str, config: ExperimentConfig, runtime: Runtime) -> Retriever:
    name, scope = retriever_ids(config.retrievers)[rid]
    spec = config.retrievers[name]
    kind = spec["kind"]
    if kind == "tfidf" and scope is not None:
        return TfidfRetriever(rid, {k: v for k, v in spec.items() if k != "kind"}, scope)
    if kind == "bm25" and scope is not None:
        return Bm25Retriever(rid, {k: spec[k] for k in BM25_PARAMS}, scope)
    if kind == "dense":
        return DenseRetriever(dense_spec(name, spec), runtime)
    if kind == "rrf":
        components = [
            build_retriever(component, config, runtime) for component in spec["components"]
        ]
        return RrfRetriever(rid, components, spec["k"])
    base = build_retriever(spec["base"], config, runtime)
    if kind == "decision_rerank":
        decision = decision_rerank_spec(
            name, spec, runtime.device, Path(config.cache["decision_dir"])
        )
        return DecisionRerankRetriever(decision, base, runtime)
    return RerankRetriever(rerank_spec(name, spec), base, runtime)


def unit_mean_chars(units: list[Unit]) -> float:
    return float(np.mean([len(unit.text) for unit in units])) if units else 0.0


def top_scores(units: list[Unit], ranking: Ranking, top: np.ndarray, decimals: int) -> list[Any]:
    return [
        [
            units[int(index)].unit_id,
            None
            if np.isnan(ranking.display[index])
            else round(float(ranking.display[index]), decimals),
        ]
        for index in top
    ]


def evaluate_query(
    query: Query,
    kind: str,
    units: list[Unit],
    ranking: Ranking,
    retriever_id: str,
    mean_chars: float,
    evaluation: Record,
) -> Record:
    """The rank of the first relevant unit, its tie bounds and the stored top of the order."""
    relevant = np.array(query.relevant[kind], dtype=np.int64)
    is_relevant = np.zeros(len(units), dtype=bool)
    is_relevant[relevant] = True
    first_position = int(np.flatnonzero(is_relevant[ranking.order])[0])
    first_index = int(ranking.order[first_position])
    best = ranking.sort_key[relevant].max()
    others = ranking.sort_key[~is_relevant]
    top1 = units[int(ranking.order[0])]
    return {
        "bench": query.bench,
        "query_id": query.query_id,
        "split": query.split,
        "hearing_id": query.hearing_id,
        "retriever": retriever_id,
        "unit": kind,
        "n_units": len(units),
        "n_relevant": len(relevant),
        "rank": first_position + 1,
        "rank_optimistic": 1 + int((others > best).sum()),
        "rank_pessimistic": 1 + int((others >= best).sum()),
        "random_acc_at_1": round(len(relevant) / len(units), RANDOM_ACC_DECIMALS),
        "query_chars": len(query.text),
        "top1_id": top1.unit_id,
        "top1_chars": len(top1.text),
        "mean_unit_chars": round(mean_chars, MEAN_CHARS_DECIMALS),
        "first_relevant_id": units[first_index].unit_id,
        "top": top_scores(
            units,
            ranking,
            ranking.order[: evaluation["top_n_stored"]],
            evaluation["score_decimals"],
        ),
    }


def query_file(run_dir: Path, bench: str, kind: str, retriever_id: str, split: str) -> Path:
    return run_dir / "queries" / bench / kind / retriever_id / f"{split}.jsonl"


def hearing_rows(
    retriever: Retriever,
    retriever_id: str,
    hearing: HearingData,
    queries: list[Query],
    kinds: tuple[str, ...],
    evaluation: Record,
) -> list[Record]:
    rows: list[Record] = []
    for kind in kinds:
        rankings = retriever.rank_hearing(hearing, kind, queries)
        means: dict[str, float] = {}
        for query in queries:
            units = hearing.contexts[query.context_key].units[kind]
            if query.context_key not in means:
                means[query.context_key] = unit_mean_chars(units)
            rows.append(
                evaluate_query(
                    query,
                    kind,
                    units,
                    rankings[query.query_id],
                    retriever_id,
                    means[query.context_key],
                    evaluation,
                )
            )
    return rows


def timing_summary(seconds: list[float]) -> Record:
    return {
        "elapsed_seconds": round(sum(seconds), 1),
        "mean_seconds_per_hearing": round(float(np.mean(seconds)), TIMING_DECIMALS)
        if seconds
        else None,
        "max_seconds_per_hearing": round(max(seconds), TIMING_DECIMALS) if seconds else None,
    }


def run_retriever(
    retriever_id: str,
    workload: Workload,
    kinds: tuple[str, ...],
    config: ExperimentConfig,
    runtime: Runtime,
) -> tuple[list[Record], Record]:
    retriever = build_retriever(retriever_id, config, runtime)
    rows: list[Record] = []
    seconds: list[float] = []
    for number, hearing in enumerate(workload.hearings, start=1):
        started = time.perf_counter()
        queries = workload.queries[hearing.hearing_id]
        rows.extend(
            hearing_rows(retriever, retriever_id, hearing, queries, kinds, config.evaluation)
        )
        seconds.append(time.perf_counter() - started)
        print(
            f"[{retriever_id}] [{number}/{len(workload.hearings)}] hearing {hearing.hearing_id}: "
            f"{len(queries)} queries in {seconds[-1]:.1f}s",
            flush=True,
        )
    retriever.close()
    return rows, {"details": retriever.describe(), "timing": timing_summary(seconds)}


def write_query_rows(
    rows: list[Record], run_dir: Path, retriever_id: str, splits: tuple[str, ...]
) -> list[Record]:
    grouped: dict[tuple[str, str, str], list[Record]] = {}
    for row in rows:
        grouped.setdefault((row["bench"], row["unit"], row["split"]), []).append(row)
    written = []
    for (bench, kind, split), members in sorted(grouped.items()):
        if split not in splits:
            raise SystemExit(f"row of split {split} outside the requested splits")
        path = query_file(run_dir, bench, kind, retriever_id, split)
        write_jsonl(members, path)
        written.append(file_record(path, len(members)))
    return written
