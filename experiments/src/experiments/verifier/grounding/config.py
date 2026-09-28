"""Configuration of the grounding comparison (E-A) and the systems it reports."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.verifier.grounding.scorers import (
    CandidateSpec,
    parse_candidate,
)

Record = dict[str, Any]

EA_SPLITS = ("train", "validation")
COVERAGES = (0.8, 0.9)
REPORTED_METRICS = (
    "roc_auc",
    "average_precision",
    "aurc",
    "e_aurc",
    "precision_at_0.8",
    "precision_at_0.9",
)
COMPARED_METRICS = ("roc_auc", "aurc")
REFERENCE = "cosine_serafim"
PRIMARY = "e3x_primary"
EXISTING = ("cosine_serafim", "e3x_primary", "laya_multi_pt_p4", "xnli_mdeberta")
COMBINATIONS = ("rank_mean", "rank_max")
EXISTING_FEATURES = {
    "cosine_serafim": "cosine_serafim:cosine:max",
    "laya_multi_pt_p4": "laya_multi_pt:p4_supports:max",
    "xnli_mdeberta": "xnli_mdeberta:entailment:max",
}
POOLS = ("max", "concatenated")
LAYA_RUN = "nli_verifier_v2"
LAYA_AGGREGATES = ("mean", "median", "min")
TRUE_OPTION = "true"


@dataclass(frozen=True)
class Config:
    path: Path
    raw: Record
    name: str
    verifier_config: Path
    exploration_config: Path
    udv_verifier_config: Path
    translation_config: Path
    translation_model: str
    candidates: dict[str, CandidateSpec]
    order: tuple[str, ...]
    smoke_items: int
    smoke_seed: int
    budget_hours: float
    bootstrap_samples: int
    level: float
    seed: int
    output_dir: Path
    cache_dir: Path

    @property
    def scores_dir(self) -> Path:
        return self.output_dir / "scores"

    @property
    def smoke_dir(self) -> Path:
        return self.output_dir / "smoke"


def load_config(path: Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    inputs, new, smoke, evaluation = raw["inputs"], raw["new"], raw["smoke"], raw["evaluation"]
    order = tuple(new["order"])
    candidates = {key: parse_candidate(key, new[key]) for key in order}
    return Config(
        path=path,
        raw=raw,
        name=raw["experiment"]["name"],
        verifier_config=Path(inputs["verifier_config"]),
        exploration_config=Path(inputs["exploration_config"]),
        udv_verifier_config=Path(inputs["udv_verifier_config"]),
        translation_config=Path(inputs["translation_config"]),
        translation_model=inputs["translation_model"],
        candidates=candidates,
        order=order,
        smoke_items=int(smoke["items"]),
        smoke_seed=int(smoke["seed"]),
        budget_hours=float(smoke["budget_hours"]),
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        level=float(evaluation["confidence_level"]),
        seed=int(evaluation["seed"]),
        output_dir=Path(raw["outputs"]["output_dir"]),
        cache_dir=Path(raw["outputs"]["cache_dir"]),
    )
