"""The translate and score commands over the UDV premise units."""

import argparse
import json
from collections import Counter
from typing import Any

import transformers
from bookworm import sha256_of_file, write_json, write_jsonl

from experiments.common.hub_offline import enforce_offline
from experiments.common.udv_run import seed_everything, select_device
from experiments.verifier.exploration.config import (
    ExplorationConfig,
    load_exploration_config,
)
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.config import (
    DECISION_KINDS,
    ScorerSpec,
    VerifierConfig,
)
from experiments.verifier.nli.config import load_config as load_verifier_config
from experiments.verifier.nli.cross_encoder import portuguese_probes, release_device
from experiments.verifier.nli.decision import check_decision_scorer
from experiments.verifier.nli.scoring import (
    check_score_names,
    score_file,
    score_report,
    score_report_file,
    score_units,
)
from experiments.verifier.nli.translated import (
    Translations,
    check_translations,
    english_probes,
    english_units,
    load_translation_config,
    open_translations,
    translation_summary,
    translation_texts,
)
from experiments.verifier.runtime import now
from experiments.verifier.translate.seq2seq import load_translator, translate_missing
from experiments.verifier.translate.store import model_record, open_store, selected_model
from experiments.verifier.udv_scores.config import (
    UDV_SPLIT,
    ScoreFunction,
    UdvVerifierConfig,
    check_scorer_list,
    english_specs,
    primary_candidate,
    selected_specs,
)
from experiments.verifier.udv_scores.units import load_udvs

Record = dict[str, Any]


def open_run(config: UdvVerifierConfig) -> tuple[VerifierConfig, ExplorationConfig]:
    verifier = load_verifier_config(config.verifier_config)
    exploration_config = load_exploration_config(config.exploration_config)
    if exploration_config.raw["verifier_config"] != str(config.verifier_config):
        raise SystemExit("the exploration config reads another verifier config")
    check_scorer_list(config, primary_candidate(exploration_config, config.primary))
    return verifier, exploration_config


def prepare_models(verifier: VerifierConfig) -> None:
    if verifier.hf_hub_offline:
        enforce_offline()
    seed_everything(verifier.seed)
    transformers.logging.set_verbosity_error()


def command_translate(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    verifier, _ = open_run(config)
    _, units, _, _, _ = load_udvs(config, verifier)
    specs = english_specs(selected_specs(config, verifier))
    translation_config = load_translation_config(verifier)
    report: Record = {"created_at": now(), "models": {}}
    for model_key in dict.fromkeys(str(spec.translation_model) for spec in specs):
        spec = selected_model(translation_config, model_key)
        store = open_store(translation_config, model_key, writable=True)
        texts = translation_texts(units, portuguese_probes(verifier), store)
        missing = store.missing(texts)
        run: Record = {"requested": len(texts), "already_cached": len(texts) - len(missing)}
        info = None
        if missing:
            prepare_models(verifier)
            device = select_device(args.device or config.device)
            translator = load_translator(spec, translation_config.decoding, device)
            print(f"{spec.name} on {device}: {len(missing)} texts to translate", flush=True)
            run |= translate_missing(
                translator,
                store,
                missing,
                translation_config.progress_every,
                translation_config.decoding.batch_token_budget,
            )
            info = translator.info
            del translator
            release_device(device)
        report["models"][model_key] = {
            "model": model_record(translation_config, spec),
            "run": run,
            "model_info": info,
            "store": store.summary(),
            "still_missing": len(store.missing(texts)),
        }
    write_json(report, config.run_dir / "translate_report.json")
    print(json.dumps({k: v["run"] for k, v in report["models"].items()}, indent=2), flush=True)


def score_scorers(
    units: list[PremiseUnit],
    specs: list[ScorerSpec],
    verifier: VerifierConfig,
    device: str,
    translations: Translations | None,
    prefix: str,
    score_function: ScoreFunction = score_units,
) -> dict[str, tuple[list[Record], Record]]:
    results: dict[str, tuple[list[Record], Record]] = {}
    for spec in specs:
        spec_units, probes, extra = units, portuguese_probes(verifier), {}
        if spec.language == "en":
            if translations is None:
                raise SystemExit(f"{spec.key}: no translation store opened")
            store = translations.store(spec)
            spec_units, probes = english_units(units, store), english_probes(verifier, store)
            extra["translation"] = translation_summary(verifier, translations, spec)
        rows, details = score_function(
            spec_units, spec, verifier, device, True, prefix, probes, "live"
        )
        results[spec.key] = (rows, details | extra)
        print(f"{spec.key}: {len(rows)} UDVs, timing {json.dumps(details['timing'])}", flush=True)
    return results


def write_scores(
    results: dict[str, tuple[list[Record], Record]],
    specs: list[ScorerSpec],
    verifier: VerifierConfig,
    config: UdvVerifierConfig,
    context: Record,
) -> None:
    by_key = {spec.key: spec for spec in specs}
    for key, (rows, details) in results.items():
        spec = by_key[key]
        check_score_names(rows, spec)
        path = score_file(config.run_dir, key, UDV_SPLIT)
        write_jsonl(rows, path)
        files = {UDV_SPLIT: {"path": str(path), "rows": len(rows), "sha256": sha256_of_file(path)}}
        splits = {"hearing_splits": dict(sorted(Counter(row["split"] for row in rows).items()))}
        report = score_report(
            "udv", config.name, spec, rows, details, context, splits, verifier, files
        )
        write_json(report, score_report_file(config.run_dir, key))


def command_score(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    verifier, _ = open_run(config)
    _, units, _, _, split_source = load_udvs(config, verifier)
    specs = selected_specs(config, verifier)
    translations = open_translations(verifier, specs, None)
    for spec in specs:
        if spec.kind in DECISION_KINDS:
            check_decision_scorer(spec, verifier, "live")
        if spec.language == "en" and translations is not None:
            check_translations(spec, units, verifier, translations.store(spec))
    prepare_models(verifier)
    device = select_device(args.device or config.device)
    context = {
        "sources": {
            "udv": {"path": str(config.udv_path), "sha256": sha256_of_file(config.udv_path)},
            "splits": split_source,
        },
        "premise": {"rule": config.raw["pairs"]["premise"], "concatenate_single": True},
        "subset": None,
    }
    print(f"{len(units)} UDVs on {device} -> {config.run_dir}", flush=True)
    results = score_scorers(units, specs, verifier, device, translations, config.name)
    write_scores(results, specs, verifier, config, context)
