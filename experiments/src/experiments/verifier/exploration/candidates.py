"""Fixed and learned candidate systems built from the scorer signals."""

import itertools
from typing import Any

import numpy as np

from experiments.verifier.exploration.config import (
    BATTERY_SIGNALS,
    LEARNED_POOLS,
    MATCH_TOLERANCE,
    Candidate,
    ExplorationConfig,
    ScorerData,
)
from experiments.verifier.exploration.scores import candidate_scores

Record = dict[str, Any]


def fixed_candidates(config: ExplorationConfig, data: dict[str, ScorerData]) -> list[Candidate]:
    candidates = []
    for key, scorer in data.items():
        pools = [p for p in config.pools if p != "concatenated" or scorer.concatenated]
        if scorer.kind == "decision":
            signals = config.decision_signals
        elif scorer.kind == "nli":
            signals = tuple(config.raw["candidates"]["nli_signals"])
        else:
            signals = tuple(config.raw["candidates"]["cosine_signals"])
        for signal in signals:
            for pool in pools:
                candidates.append(
                    Candidate(f"{key}:{signal}:{pool}", "signal", key, (signal,), pool)
                )
        if scorer.kind != "decision":
            continue
        for size in range(config.panel_min_size, len(config.panel_components) + 1):
            for subset in itertools.combinations(config.panel_components, size):
                label = "+".join(subset)
                for pool in pools:
                    candidates.append(
                        Candidate(f"{key}:panel[{label}]:{pool}", "panel", key, subset, pool)
                    )
    return candidates


def learned_candidates(config: ExplorationConfig, data: dict[str, ScorerData]) -> list[Candidate]:
    candidates = []
    for key, scorer in data.items():
        if scorer.kind != "decision":
            continue
        features = tuple(f"{key}:{s}:{p}" for s in BATTERY_SIGNALS for p in LEARNED_POOLS)
        candidates.append(Candidate(f"{key}:learned", "learned", key, (), "learned", features))
    for name, bases, extra in config.cross_models:
        if not all(scorer in data for scorer in (*bases, *(e.split(":")[0] for e in extra))):
            continue
        features = tuple(
            f"{base}:{s}:{p}" for base in bases for s in BATTERY_SIGNALS for p in LEARNED_POOLS
        )
        candidates.append(
            Candidate(
                f"{bases[0]}:learned_{name}", "learned", bases[0], (), "learned", features + extra
            )
        )
    return candidates


def feature_candidate(feature: str) -> Candidate:
    scorer, signal, pool = feature.split(":")
    return Candidate(feature, "signal", scorer, (signal,), pool)


def reference_candidate(reference: str) -> Candidate:
    scorer, signal, pool = reference.split(":")
    if signal == "panel":
        return Candidate(reference, "panel", scorer, (), pool)
    return feature_candidate(reference)


def resolve_reference(
    reference: str, config: ExplorationConfig, data: dict[str, ScorerData]
) -> np.ndarray:
    candidate = reference_candidate(reference)
    if candidate.kind == "panel":
        declared = tuple(config.verifier["decision_battery"]["aggregation"]["panel_components"])
        candidate = Candidate(reference, "panel", candidate.scorer, declared, candidate.pool)
    return candidate_scores(candidate, data)


def check_reading(config: ExplorationConfig, data: dict[str, ScorerData]) -> Record:
    checked: Record = {}
    declared = tuple(config.verifier["decision_battery"]["aggregation"]["panel_components"])
    for key, scorer in data.items():
        pairs: list[tuple[str, Candidate]] = []
        if scorer.kind == "decision":
            pairs += [
                (f"max.{s}", Candidate("", "signal", key, (s,), "max")) for s in BATTERY_SIGNALS
            ]
            pairs.append(("max.panel", Candidate("", "panel", key, declared, "max")))
            if scorer.concatenated:
                pairs += [
                    (f"concatenated.{s}", Candidate("", "signal", key, (s,), "concatenated"))
                    for s in BATTERY_SIGNALS
                ]
        elif scorer.kind == "nli":
            pairs.append(("max.entailment", Candidate("", "signal", key, ("entailment",), "max")))
            if scorer.concatenated:
                pairs.append(
                    (
                        "concatenated.entailment",
                        Candidate("", "signal", key, ("entailment",), "concatenated"),
                    )
                )
        else:
            pairs.append(("max.cosine", Candidate("", "signal", key, ("cosine",), "max")))
            pairs.append(
                ("sentence_max.cosine", Candidate("", "signal", key, ("sentence_max",), "max"))
            )
        worst = 0.0
        for stored_name, candidate in pairs:
            gap = float(
                np.max(
                    np.abs(candidate_scores(candidate, {key: scorer}) - scorer.stored[stored_name])
                )
            )
            if gap > MATCH_TOLERANCE:
                raise SystemExit(
                    f"{key}: recomputed {stored_name} differs from the score file by {gap}"
                )
            worst = max(worst, gap)
        checked[key] = {"stored_scores_checked": [name for name, _ in pairs], "max_abs_gap": worst}
    return checked
