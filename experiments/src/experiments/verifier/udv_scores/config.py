"""Configuration of the verifier run over the UDVs and the scorers it declares."""

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.verifier.exploration import scores as exploration_scores
from experiments.verifier.exploration.candidates import (
    learned_candidates,
)
from experiments.verifier.exploration.config import (
    Candidate,
    ExplorationConfig,
    ScorerData,
)
from experiments.verifier.exploration.cross_validation import feature_matrix
from experiments.verifier.nli.config import (
    ScorerSpec,
    VerifierConfig,
    selected_scorers,
)

Record = dict[str, Any]
ScoreFunction = Callable[..., tuple[list[Record], Record]]

UDV_SPLIT = "udv"
POOLS = ("max", "mean", "concatenated")
SUPPORT_TYPE_NONE = "none"
SENTENCE_UNIT = "sentence"
EVIDENCE_COSINE_FEATURES = {
    SENTENCE_UNIT: "cosine_serafim:sentence_max:max",
    "window": "cosine_serafim:cosine:max",
}
SENTENCE_CHECK_RULE = (
    "cosine_serafim sentence_max.cosine of each semantic UDV against evidence.score"
)
WINDOW_CHECK_RULE = (
    "cosine_serafim max.cosine (the cosine of the whole premise, the window the verifier read) of"
    " each semantic UDV against evidence.score (the cosine of the window the UDV builder encoded);"
    " n, max_abs_gap and within_1e-4 cover the UDVs whose premise equals the encoded text, the"
    " candidate sentences of the window joined by a space; encoded_text_differs counts the UDVs"
    " whose evidence span also holds parts that are not candidate sentences (under 4 words or"
    " stage directions), which the verifier read and the encoder did not"
)


@dataclass(frozen=True)
class UdvVerifierConfig:
    path: Path
    raw: Record
    name: str
    udv_path: Path
    verifier_config: Path
    exploration_config: Path
    selection_path: Path
    final_test_path: Path
    evidence_tiers: tuple[str, ...]
    primary: str
    secondary: str
    scorers: tuple[str, ...]
    device: str
    quantiles: tuple[float, ...]
    semantic_tiers: tuple[str, ...]
    lowest_tiers: tuple[str, ...]
    lowest_count: int
    output_path: Path
    report_path: Path
    udv_threshold_path: Path | None = None

    @property
    def run_dir(self) -> Path:
        return exploration_scores.SCORES_ROOT / self.name


@dataclass(frozen=True)
class UdvThreshold:
    value: float
    path: Path
    sha256: str
    rule: str
    record: Record


@dataclass(frozen=True)
class FittedPrimary:
    candidate: Candidate
    scaler: StandardScaler
    model: LogisticRegression
    threshold: float
    record: Record

    def probabilities(self, data: dict[str, ScorerData]) -> np.ndarray:
        matrix = feature_matrix(self.candidate, data)
        return self.model.predict_proba(self.scaler.transform(matrix))[:, 1]


def load_config(path: Path) -> UdvVerifierConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)["udv_verifier"]
    inputs, pairs, scoring = raw["inputs"], raw["pairs"], raw["scoring"]
    report, outputs = raw["report"], raw["outputs"]
    if outputs["score_run"] != raw["name"]:
        raise SystemExit("outputs.score_run must equal the run name")
    return UdvVerifierConfig(
        path=path,
        raw=raw,
        name=raw["name"],
        udv_path=Path(inputs["udv_path"]),
        verifier_config=Path(inputs["verifier_config"]),
        exploration_config=Path(inputs["exploration_config"]),
        selection_path=Path(inputs["selection_path"]),
        final_test_path=Path(inputs["final_test_path"]),
        evidence_tiers=tuple(pairs["evidence_tiers"]),
        primary=scoring["primary"],
        secondary=scoring["secondary"],
        scorers=tuple(scoring["scorers"]),
        device=scoring["device"],
        quantiles=tuple(float(q) for q in report["quantiles"]),
        semantic_tiers=tuple(report["semantic_tiers"]),
        lowest_tiers=tuple(report["lowest_tiers"]),
        lowest_count=int(report["lowest_count"]),
        output_path=Path(outputs["output_path"]),
        report_path=Path(outputs["report_path"]),
        udv_threshold_path=(
            Path(scoring["udv_threshold_path"]) if "udv_threshold_path" in scoring else None
        ),
    )


def declared_scorers(config: ExplorationConfig) -> dict[str, ScorerData]:
    return {
        key: ScorerData(key, kind, Path(), {}, {}, {}) for key, (kind, _) in config.scorers.items()
    }


def primary_candidate(config: ExplorationConfig, key: str) -> Candidate:
    for candidate in learned_candidates(config, declared_scorers(config)):
        if candidate.key == key:
            return candidate
    raise SystemExit(f"{key} is not a learned candidate of {config.path}")


def candidate_scorers(candidate: Candidate) -> tuple[str, ...]:
    return tuple(dict.fromkeys(feature.split(":")[0] for feature in candidate.features))


def check_scorer_list(config: UdvVerifierConfig, candidate: Candidate) -> None:
    needed = set(candidate_scorers(candidate)) | {config.secondary.split(":")[0]}
    if needed != set(config.scorers):
        raise SystemExit(
            f"scoring.scorers {sorted(config.scorers)} differs from the scorers the primary and "
            f"secondary read {sorted(needed)}"
        )


def selected_specs(config: UdvVerifierConfig, verifier: VerifierConfig) -> list[ScorerSpec]:
    return selected_scorers(verifier, list(config.scorers))


def english_specs(specs: list[ScorerSpec]) -> list[ScorerSpec]:
    return [spec for spec in specs if spec.language == "en"]
