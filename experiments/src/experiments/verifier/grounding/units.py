"""Train and validation opinions of the NLI benchmark as premise units and scorer pairs."""

from typing import Any

from experiments.verifier.grounding.config import EA_SPLITS, Config
from experiments.verifier.grounding.scorers import (
    CandidateSpec,
)
from experiments.verifier.nli.benchmark import PremiseUnit, concatenated_premise, distinct_items
from experiments.verifier.nli.config import load_config as load_verifier_config
from experiments.verifier.nli.scoring import prepare_benchmark_units
from experiments.verifier.nli.translated import english_units
from experiments.verifier.translate import config as translate_config
from experiments.verifier.translate import store as translate_store

Record = dict[str, Any]


def ea_units(config: Config) -> tuple[list[PremiseUnit], Record]:
    verifier = load_verifier_config(config.verifier_config)
    units, context = prepare_benchmark_units(verifier, EA_SPLITS, None)
    if {unit.split for unit in units} - set(EA_SPLITS):
        raise SystemExit("E-A units outside train and validation")
    return units, context


def open_store(config: Config, writable: bool = False) -> tuple[Any, Any]:
    translation_config = translate_config.load_config(config.translation_config)
    store = translate_store.open_store(
        translation_config, config.translation_model, writable=writable
    )
    return translation_config, store


def language_units(units: list[PremiseUnit], spec: CandidateSpec, store: Any) -> list[PremiseUnit]:
    if spec.language == "pt":
        return units
    if store is None:
        raise SystemExit(f"{spec.key}: no translation store opened")
    return english_units(units, store)


def unit_pairs(unit: PremiseUnit, concatenated: bool) -> tuple[list[str], str | None]:
    items = distinct_items(unit.items)
    joined = concatenated_premise(unit, " ", False) if concatenated else None
    return items, joined


def all_pairs(units: list[PremiseUnit], concatenated: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for unit in units:
        items, joined = unit_pairs(unit, concatenated)
        pairs.extend((item, unit.hypothesis) for item in items)
        if joined is not None:
            pairs.append((joined, unit.hypothesis))
    return pairs
