"""English premise units and probes read from the translation caches."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from bookworm import sha256_of_file

from experiments.common.transcript import normalize_whitespace
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.config import ScorerSpec, VerifierConfig
from experiments.verifier.nli.cross_encoder import portuguese_probes
from experiments.verifier.translate.config import TranslationConfig, load_config
from experiments.verifier.translate.store import TranslationStore, model_record, open_store

Record = dict[str, Any]


@dataclass(frozen=True)
class Translations:
    config: TranslationConfig
    stores: dict[str, TranslationStore]

    def store(self, spec: ScorerSpec) -> TranslationStore:
        if spec.translation_model not in self.stores:
            raise SystemExit(f"{spec.key}: no store opened for {spec.translation_model}")
        return self.stores[spec.translation_model]


def load_translation_config(config: VerifierConfig) -> TranslationConfig:
    if config.translation_config_path is None:
        raise SystemExit("language en needs [translation] config_path")
    translation_config = load_config(config.translation_config_path)
    same_probes = (translation_config.probe_premises, translation_config.probe_hypotheses) == (
        config.probe_premises,
        config.probe_hypotheses,
    )
    if not same_probes or translation_config.nli_sha256 != config.nli_sha256:
        raise SystemExit(
            f"{config.translation_config_path} reads another E3 config: its NLI file or label "
            "probes differ from this one"
        )
    return translation_config


def open_translations(
    config: VerifierConfig, specs: list[ScorerSpec], cache_dir: Path | None
) -> Translations | None:
    models = list(dict.fromkeys(s.translation_model for s in specs if s.language == "en"))
    if not models:
        return None
    translation_config = load_translation_config(config)
    unknown = [
        f"{spec.key}: {spec.translation_model}"
        for spec in specs
        if spec.language == "en" and spec.translation_model not in translation_config.models
    ]
    if unknown:
        raise SystemExit(
            f"translation_model must be a condition of {config.translation_config_path} "
            f"{list(translation_config.models)}: {unknown}"
        )
    stores = {key: open_store(translation_config, key, cache_dir=cache_dir) for key in models}
    return Translations(translation_config, cast(dict[str, TranslationStore], stores))


def translation_summary(
    config: VerifierConfig, translations: Translations, spec: ScorerSpec
) -> Record:
    path = config.translation_config_path
    model = translations.config.models[str(spec.translation_model)]
    return {
        "config": {"path": str(path), "sha256": sha256_of_file(path) if path else None},
        "model": model_record(translations.config, model),
        "store": translations.store(spec).summary(),
        "rule": config.source["translation"]["rule"],
    }


def translation_texts(
    units: list[PremiseUnit], probes: list[tuple[str, str]], store: TranslationStore
) -> list[str]:
    opinions = [*(unit.hypothesis for unit in units), *(hypothesis for _, hypothesis in probes)]
    chunks = dict.fromkeys(
        [*(item for unit in units for item in unit.items if item), *(p for p, _ in probes)]
    )
    segments = [segment for chunk in chunks for segment in store.segmenter.segments(chunk)]
    return list(dict.fromkeys(normalize_whitespace(text) for text in [*opinions, *segments]))


def check_translations(
    spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig, store: TranslationStore
) -> Record:
    texts = translation_texts(units, portuguese_probes(config), store)
    missing = store.missing(texts)
    model = spec.translation_model
    if missing:
        raise SystemExit(
            f"{spec.key} (language en, translation model {model}, "
            f"{store.signature.get('model')}): {len(missing)} of {len(texts)} distinct texts of "
            f"the selected opinions and label probes have no translation in {store.path} "
            f"(signature {store.digest[:16]}); run python -m experiments.verifier.translation "
            f"translate --model "
            f"{model} for these splits, or point --translation-cache-dir at a cache that has them"
        )
    return {"translation_model": model, "distinct_texts": len(texts), "missing": 0}


def english_units(units: list[PremiseUnit], store: TranslationStore) -> list[PremiseUnit]:
    english = []
    for unit in units:
        items = tuple(
            normalize_whitespace(store.translate_chunk(item)) if item else "" for item in unit.items
        )
        if any(bool(pt) != bool(en) for pt, en in zip(unit.items, items, strict=True)):
            raise SystemExit(f"{unit.unit_id}: a chunk and its translation differ in emptiness")
        hypothesis = normalize_whitespace(store.translate_opinion(unit.hypothesis))
        english.append(PremiseUnit(unit.unit_id, unit.hearing_id, unit.split, hypothesis, items))
    return english


def english_probes(config: VerifierConfig, store: TranslationStore) -> list[tuple[str, str]]:
    return [
        (
            normalize_whitespace(store.translate_chunk(premise)),
            normalize_whitespace(store.translate_opinion(hypothesis)),
        )
        for premise, hypothesis in portuguese_probes(config)
    ]
