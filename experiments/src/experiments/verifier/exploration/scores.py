"""Benchmark labels and per-item scorer outputs read from the verifier score files."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file

from experiments.data.nli_benchmark import load_split_lookup
from experiments.udv.calibrate_threshold import rounded
from experiments.verifier.exploration.config import (
    EMPTY_SCORE,
    OUTPUT_ROOT,
    Candidate,
    ExplorationConfig,
    ScorerData,
)

Record = dict[str, Any]


SCORES_ROOT = Path("artifacts/experiments/nli_verifier")


def load_labels(config: ExplorationConfig, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    verifier = config.verifier
    benchmark = Path(verifier["benchmark"]["path"])
    with open(verifier["benchmark"]["report_path"]) as f:
        recorded = json.load(f)["artifact"]["sha256"]
    if sha256_of_file(benchmark) != recorded:
        raise SystemExit(f"{benchmark}: sha256 differs from its report")
    split_of, _ = load_split_lookup(
        Path(verifier["splits"]["manifest_path"]), verifier["dataset"]["lds_sha256"]
    )
    ids, labels, hearings = [], [], []
    for row in load_jsonl(benchmark):
        if split_of.get(row["hearing_id"]) != row["split"]:
            raise SystemExit(f"{row['id']}: split differs from the manifest")
        if row["split"] == split:
            ids.append(row["id"])
            labels.append(bool(row["label_inferable"]))
            hearings.append(int(row["hearing_id"]))
    return ids, np.array(labels), np.array(hearings)


def item_signals(kind: str, item: Record | None, config: ExplorationConfig) -> dict[str, float]:
    if not item:
        return {}
    if kind == "decision":
        signals = item.get("signals") or {}
        values = {name: float(value) for name, value in signals.items()}
        for name, (first, second) in config.order_pairs.items():
            if first in values and second in values:
                values[name] = (values[first] + values[second]) / 2
        return values
    if kind == "nli":
        return {"entailment": float(item["probabilities"]["entailment"])}
    return {"cosine": float(item["cosine"]), "sentence_max": float(item["sentence_max"])}


def load_scorer(
    key: str, kind: str, run: str, split: str, ids: list[str], config: ExplorationConfig
) -> ScorerData | None:
    path = SCORES_ROOT / run / "scores" / f"{key}_{split}.jsonl"
    if not path.exists():
        return None
    rows = {row["id"]: row for row in load_jsonl(path)}
    if set(rows) != set(ids):
        raise SystemExit(f"{path}: opinions differ from the benchmark {split} rows")
    per_chunk: dict[str, list[list[float]]] = {}
    concatenated: dict[str, list[float]] = {}
    stored: dict[str, list[float]] = {}
    for index, opinion in enumerate(ids):
        row = rows[opinion]
        for name, value in row["scores"].items():
            stored.setdefault(name, []).append(float(value))
        signals = [item_signals(kind, item, config) for item in row.get("items") or []]
        signals = [values for values in signals if values]
        names = set().union(*signals) if signals else set()
        for name in names:
            per_chunk.setdefault(name, [[] for _ in ids])[index] = [v[name] for v in signals]
        joined = item_signals(kind, row.get("concatenated"), config)
        for name, value in joined.items():
            concatenated.setdefault(name, [float("nan")] * len(ids))[index] = value
    return ScorerData(
        key=key,
        kind=kind,
        path=path,
        chunks=per_chunk,
        concatenated={name: np.array(values) for name, values in concatenated.items()},
        stored={name: np.array(values) for name, values in stored.items()},
    )


def pool_values(values: list[float], pool: str, empty: float = 0.0) -> float:
    if not values:
        return empty
    array = np.asarray(values, dtype=float)
    if pool == "max":
        return float(array.max())
    if pool == "mean":
        return float(array.mean())
    if pool == "top2_mean":
        return float(np.sort(array)[-2:].mean())
    if pool == "noisy_or":
        return float(1.0 - np.prod(1.0 - np.clip(array, 0.0, 1.0)))
    raise ValueError(pool)


def panel_chunks(data: ScorerData, components: tuple[str, ...]) -> list[list[float]]:
    count = len(next(iter(data.chunks.values())))
    merged = []
    for index in range(count):
        columns = [data.chunks[name][index] for name in components]
        merged.append([float(np.mean(values)) for values in zip(*columns, strict=True)])
    return merged


def candidate_scores(candidate: Candidate, data: dict[str, ScorerData]) -> np.ndarray:
    scorer = data[candidate.scorer]
    if candidate.pool == "concatenated":
        columns = [scorer.concatenated[name] for name in candidate.components]
        return np.nan_to_num(np.mean(columns, axis=0), nan=EMPTY_SCORE[scorer.kind])
    if candidate.kind == "signal":
        chunks = list(scorer.chunks[candidate.components[0]])
    else:
        chunks = panel_chunks(scorer, candidate.components)
    empty = EMPTY_SCORE[scorer.kind]
    return np.array([pool_values(values, candidate.pool, empty) for values in chunks])


def output_dir(config: ExplorationConfig, override: Path | None) -> Path:
    return (override or OUTPUT_ROOT) / config.name


def load_all(
    config: ExplorationConfig, split: str
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, ScorerData], list[str]]:
    ids, labels, hearings = load_labels(config, split)
    data: dict[str, ScorerData] = {}
    missing = []
    for key, (kind, run) in config.scorers.items():
        scorer = load_scorer(key, kind, run, split, ids, config)
        if scorer is None:
            missing.append(key)
        else:
            data[key] = scorer
    return ids, labels, hearings, data, missing


def write_csv(path: Path, rows: list[Record]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: rounded(v) if isinstance(v, float) else v for k, v in row.items()})
