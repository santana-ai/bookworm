"""Exploration configuration and the candidate and scorer records it works on."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Record = dict[str, Any]
OUTPUT_ROOT = Path("artifacts/experiments/nli_verifier_exploration")
KIND_ORDER = {"signal": 0, "panel": 1, "learned": 2}
CHUNK_POOLS = ("max", "mean", "top2_mean", "noisy_or")
BATTERY_SIGNALS = (
    "p1_nli",
    "p2_nli_reversed",
    "p3_inferable",
    "p4_supports",
    "p5_position",
    "p6_position_reversed",
    "p7_coverage",
    "p8_similarity",
)
LEARNED_POOLS = ("max", "mean", "concatenated")
MATCH_TOLERANCE = 1e-9
EMPTY_SCORE = {"decision": 0.0, "nli": 0.0, "cosine": -1.0}


@dataclass(frozen=True)
class ExplorationConfig:
    name: str
    raw: Record
    path: Path
    verifier: Record
    scorers: dict[str, tuple[str, str]]
    decision_signals: tuple[str, ...]
    order_pairs: dict[str, tuple[str, str]]
    panel_components: tuple[str, ...]
    panel_min_size: int
    pools: tuple[str, ...]
    c_grid: tuple[float, ...]
    inner_splits: int
    cross_models: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    n_splits: int
    repeats: int
    cv_seed: int
    references: tuple[str, ...]
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass(frozen=True)
class Candidate:
    key: str
    kind: str
    scorer: str
    components: tuple[str, ...]
    pool: str
    features: tuple[str, ...] = ()


@dataclass
class ScorerData:
    key: str
    kind: str
    path: Path
    chunks: dict[str, list[list[float]]]
    concatenated: dict[str, np.ndarray]
    stored: dict[str, np.ndarray]


def load_exploration_config(path: Path) -> ExplorationConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)["exploration"]
    with open(raw["verifier_config"], "rb") as f:
        verifier = tomllib.load(f)
    inputs, candidates = raw["inputs"], raw["candidates"]
    scorers: dict[str, tuple[str, str]] = {}
    for kind, table in (
        ("decision", inputs["decision_scorers"]),
        ("nli", inputs["nli_scorers"]),
        ("cosine", inputs["cosine_scorers"]),
    ):
        for key, run in table.items():
            scorers[key] = (kind, run)
    order_pairs = {name: (pair[0], pair[1]) for name, pair in candidates["order_pairs"].items()}
    entries = [
        {"name": "cross_model", "bases": [raw_cross["base"]], "extra": raw_cross["extra"]}
        for raw_cross in [candidates["cross_model"]]
        if "cross_model" in candidates
    ] + list(candidates.get("cross_models", []))
    cross_models = []
    for entry in entries:
        if any(len(item.split(":")) != 3 for item in entry["extra"]):
            raise SystemExit("cross model extra entries must be scorer:signal:pool")
        cross_models.append((entry["name"], tuple(entry["bases"]), tuple(entry["extra"])))
    selection = raw["selection"]
    if selection["metric"] != "roc_auc":
        raise SystemExit("only roc_auc is supported as the selection metric")
    for pool in candidates["pools"]:
        if pool not in (*CHUNK_POOLS, "concatenated"):
            raise SystemExit(f"unknown pool {pool!r}")
    return ExplorationConfig(
        name=raw["name"],
        raw=raw,
        path=path,
        verifier=verifier,
        scorers=scorers,
        decision_signals=tuple(candidates["decision_signals"]),
        order_pairs=order_pairs,
        panel_components=tuple(candidates["panel_components"]),
        panel_min_size=int(candidates["panel_min_size"]),
        pools=tuple(candidates["pools"]),
        c_grid=tuple(float(c) for c in candidates["c_grid"]),
        inner_splits=int(candidates["inner_splits"]),
        cross_models=tuple(cross_models),
        n_splits=int(raw["cv"]["n_splits"]),
        repeats=int(raw["cv"]["repeats"]),
        cv_seed=int(raw["cv"]["seed"]),
        references=tuple(raw["confirmation"]["references"]),
        bootstrap_samples=int(raw["confirmation"]["bootstrap_samples"]),
        bootstrap_seed=int(raw["confirmation"]["seed"]),
    )
