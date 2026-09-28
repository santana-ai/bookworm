"""Guards around the retrieval harness and the ranking signals read from its rows."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file

from experiments.retrieval.data import Unit
from experiments.retrieval.experiments import (
    ExperimentConfig,
    query_file,
    retriever_ids,
)
from experiments.retrieval.models import (
    DenseRetriever,
    Ranking,
    RerankRetriever,
    Retriever,
    RrfRetriever,
    prefix_directory,
)
from experiments.retrieval.store import VectorStore, slug
from experiments.verifier.confidence.config import ZERO_SPREAD_RELATIVE, Target, text_sha256

Record = dict[str, Any]


def store_directories(retriever: str, harness: ExperimentConfig) -> list[Path]:
    name, _ = retriever_ids(harness.retrievers)[retriever]
    spec = harness.retrievers[name]
    if spec["kind"] == "dense":
        base = Path(harness.cache["embeddings_dir"]) / slug(spec["model"]) / spec["revision"][:12]
        prefixes = dict.fromkeys((spec["query_prefix"], spec["passage_prefix"]))
        return [
            base / prefix_directory(prefix) / f"msl{spec['max_seq_length']}" for prefix in prefixes
        ]
    if spec["kind"] == "rrf":
        return [path for part in spec["components"] for path in store_directories(part, harness)]
    if spec["kind"] == "rerank":
        own = (
            Path(harness.cache["rerank_dir"])
            / slug(spec["model"])
            / spec["revision"][:12]
            / f"maxlen{spec['max_length']}"
        )
        return [own, *store_directories(spec["base"], harness)]
    return []


def model_revisions(retriever: str, harness: ExperimentConfig) -> list[Record]:
    name, _ = retriever_ids(harness.retrievers)[retriever]
    spec = harness.retrievers[name]
    if spec["kind"] in ("dense", "rerank"):
        own = [{"retriever": name, "model": spec["model"], "revision": spec["revision"]}]
        return own + (model_revisions(spec["base"], harness) if "base" in spec else [])
    if spec["kind"] == "rrf":
        return [item for part in spec["components"] for item in model_revisions(part, harness)]
    return []


def missing_inputs(
    target: Target,
    harness: ExperimentConfig,
    harness_dir: Path,
    benches: tuple[str, ...],
    splits: tuple[str, ...],
) -> list[str]:
    covered = harness_coverage(harness_dir, target)
    missing = [
        str(query_file(harness_dir, bench, target.unit, target.retriever, split))
        for bench in benches
        for split in splits
        if not query_file(harness_dir, bench, target.unit, target.retriever, split).exists()
        and (bench, split) not in covered
    ]
    missing += [
        str(path / "store.json")
        for path in store_directories(target.retriever, harness)
        if not (path / "store.json").exists()
    ]
    return missing


def refuse(what: str) -> Callable[..., Any]:
    def raise_error(*args: Any, **kwargs: Any) -> Any:
        raise SystemExit(
            f"{what} was requested but is not in the harness cache: run the harness step of this "
            "retriever first (collect never encodes, scores or writes cache files)"
        )

    return raise_error


def keep_in_memory(*args: Any, **kwargs: Any) -> None:
    return None


def replace_methods(target: object, replacements: dict[str, Callable[..., Any]]) -> None:
    for name, replacement in replacements.items():
        setattr(target, name, replacement)


def guard_store(store: VectorStore, what: str) -> None:
    if len(store) == 0:
        raise SystemExit(f"{store.directory}: the harness cache is empty")
    replace_methods(store, {"add": refuse(what), "flush": keep_in_memory})


def guard_retriever(retriever: Retriever) -> None:
    if isinstance(retriever, DenseRetriever):
        what = f"a {retriever.spec.model} vector"
        replace_methods(retriever.encoder, {"encode": refuse(what), "load": refuse(what)})
        for store in retriever.encoder.stores.values():
            guard_store(store, what)
    elif isinstance(retriever, RerankRetriever):
        what = f"a {retriever.spec.model} score"
        replace_methods(retriever, {"load": refuse(what)})
        guard_store(retriever.store, what)
        guard_retriever(retriever.base)
    elif isinstance(retriever, RrfRetriever):
        for component in retriever.components:
            guard_retriever(component)


def load_harness_rows(
    harness_dir: Path, target: Target, benches: tuple[str, ...], splits: tuple[str, ...]
) -> tuple[dict[tuple[str, str], Record], list[Record]]:
    rows: dict[tuple[str, str], Record] = {}
    files: list[Record] = []
    for bench in benches:
        for split in splits:
            path = query_file(harness_dir, bench, target.unit, target.retriever, split)
            if not path.exists():
                files.append({"path": str(path), "rows": 0, "sha256": None, "empty_run": True})
                continue
            members = load_jsonl(path)
            for row in members:
                expected = (bench, split, target.retriever, target.unit)
                if (row["bench"], row["split"], row["retriever"], row["unit"]) != expected:
                    raise SystemExit(f"{path}: row {row['query_id']} belongs to another file")
                rows[(bench, row["query_id"])] = row
            files.append({"path": str(path), "rows": len(members), "sha256": sha256_of_file(path)})
    return rows, files


def harness_coverage(harness_dir: Path, target: Target) -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for path in sorted((harness_dir / "runs").glob(f"{target.retriever}__*_report.json")):
        with open(path) as f:
            report = json.load(f)
        if report.get("retriever") == target.retriever and target.unit in report["units"]:
            covered.update(
                (bench, split) for bench in report["benches"] for split in report["splits_used"]
            )
    return covered


def harness_run_reports(harness_dir: Path, target: Target) -> list[Record]:
    reports = []
    for path in sorted((harness_dir / "runs").glob(f"{target.retriever}__*_report.json")):
        with open(path) as f:
            report = json.load(f)
        if report.get("retriever") != target.retriever or target.unit not in report["units"]:
            continue
        reports.append(
            {
                "path": str(path),
                "sha256": sha256_of_file(path),
                "created_at": report["created_at"],
                "splits_used": report["splits_used"],
                "units": report["units"],
                "device": report["device"],
                "code": report["code"],
            }
        )
    return reports


def ranking_signals(ranking: Ranking, zero_value: float) -> Record:
    display = np.asarray(ranking.display, dtype=np.float64)
    order = ranking.order
    top1 = float(display[order[0]])
    second = float(display[order[1]]) if len(order) > 1 else math.nan
    scored = display[~np.isnan(display)]
    mean, std = float(scored.mean()), float(scored.std())
    zero_spread = bool(std <= ZERO_SPREAD_RELATIVE * max(1.0, abs(mean)))
    defined = len(scored) >= 2 and not math.isnan(second)
    zscore = None
    if defined:
        zscore = zero_value if zero_spread else (top1 - mean) / std
    return {
        "top1_score": top1,
        "top2_score": None if math.isnan(second) else second,
        "margin": None if math.isnan(second) else top1 - second,
        "zscore": zscore,
        "score_mean": mean,
        "score_std": std,
        "scored_units": int(len(scored)),
        "zero_spread": zero_spread,
    }


def row_differences(built: Record, stored: Record, tolerance: float) -> list[str]:
    if set(built) != set(stored):
        return ["fields"]
    problems = [key for key in built if key != "top" and built[key] != stored[key]]
    built_ids = [item[0] for item in built["top"]]
    if built_ids != [item[0] for item in stored["top"]]:
        return [*problems, "top_ids"]
    for (_, mine), (_, theirs) in zip(built["top"], stored["top"], strict=True):
        if (mine is None) != (theirs is None) or (
            mine is not None and abs(mine - theirs) > tolerance
        ):
            return [*problems, "top_scores"]
    return problems


def feature_row(
    built: Record, target: Target, units: list[Unit], ranking: Ranking, zero_value: float
) -> Record:
    top1 = units[int(ranking.order[0])]
    return {
        "bench": built["bench"],
        "query_id": built["query_id"],
        "split": built["split"],
        "hearing_id": built["hearing_id"],
        "target": target.name,
        "retriever": target.retriever,
        "unit": target.unit,
        "n_units": built["n_units"],
        "n_relevant": built["n_relevant"],
        "rank": built["rank"],
        "rank_optimistic": built["rank_optimistic"],
        "rank_pessimistic": built["rank_pessimistic"],
        "correct": built["rank"] == 1,
        "correct_optimistic": built["rank_optimistic"] == 1,
        "correct_pessimistic": built["rank_pessimistic"] == 1,
        "top1_id": top1.unit_id,
        "top1_chars": len(top1.text),
        "top1_text_sha256": text_sha256(top1.text),
        **ranking_signals(ranking, zero_value),
    }
