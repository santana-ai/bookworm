"""The fetch, plan, translate and report commands."""

import argparse
import json
from typing import Any

from bookworm import write_json
from huggingface_hub import snapshot_download

from experiments.common.transcript import (
    normalize_whitespace,
)
from experiments.common.udv_run import select_device
from experiments.verifier.runtime import now
from experiments.verifier.translate.config import ModelSpec, TranslationConfig
from experiments.verifier.translate.report import (
    build_report,
    code_hashes,
    environment,
    print_report,
    run_directory,
    texts_digest,
)
from experiments.verifier.translate.selection import (
    load_units,
    resolve_splits,
    split_records,
    unit_selection,
)
from experiments.verifier.translate.seq2seq import (
    check_tokenizer,
    load_tokenizer,
    load_translator,
    translate_missing,
)
from experiments.verifier.translate.statistics import (
    segmentation_check,
    selection_counts,
    token_plan,
)
from experiments.verifier.translate.store import (
    TranslationStore,
    model_record,
    model_signature,
    open_store,
    selected_model,
    signature_digest,
)

Record = dict[str, Any]


FETCH_PATTERNS = ["*.json", "*.model", "README.md", "pytorch_model.bin", "model.safetensors"]


def command_fetch(args: argparse.Namespace, config: TranslationConfig) -> None:
    spec = selected_model(config, args.model)
    path = snapshot_download(spec.name, revision=spec.revision, allow_patterns=FETCH_PATTERNS)
    print(f"{spec.key}: {spec.name}@{spec.revision} -> {path}", flush=True)


def command_plan(args: argparse.Namespace, config: TranslationConfig) -> None:
    splits = resolve_splits(config, args.splits, args.final_test)
    units, context = load_units(config, splits, args.limit)
    selection = unit_selection(units, config)
    spec = selected_model(config, args.model)
    tokenizer = load_tokenizer(spec)
    tokenizer_info = check_tokenizer(tokenizer, spec)
    special = tokenizer_info["special_tokens_per_input"]
    limit = config.decoding.max_input_tokens
    store = open_store(config, spec.key, cache_dir=args.cache_dir)
    probe_chunks = {normalize_whitespace(p) for p in config.probe_premises}
    report = {
        "experiment": "translation",
        "command": "plan",
        "run_name": args.run_name,
        "created_at": now(),
        "splits": split_records(splits, args.final_test),
        "model": {**model_record(config, spec), "tokenizer": tokenizer_info},
        "segmentation": config.segmenter.describe(),
        "counts": selection_counts(units, selection, config.segmenter),
        "model_texts": {
            "distinct": len(selection.model_texts()),
            "sha256": texts_digest(selection.model_texts()),
            "rule": (
                "sha256 of the JSON list of the distinct model texts (opinions, then chunk units, "
                "each once, in first-occurrence order); it depends on the units only, not on the "
                "model, so two conditions with the same sha256 translate the same texts"
            ),
        },
        "tokens": {
            "opinions": token_plan(tokenizer, selection.opinions, limit, special),
            "segments": token_plan(tokenizer, selection.segments, limit, special),
            "all": token_plan(tokenizer, selection.model_texts(), limit, special),
        },
        "segmentation_check": segmentation_check(
            [chunk for chunk in selection.chunks if chunk not in probe_chunks]
        ),
        "cache": {
            **store.summary(),
            "model_texts_missing": len(store.missing(selection.model_texts())),
        },
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    path = run_directory(config, args) / "plan.json"
    write_json(report, path)
    print(json.dumps({key: report[key] for key in ("counts", "tokens")}, indent=2))
    print(json.dumps(report["segmentation_check"], indent=2))
    print(f"-> {path}", flush=True)


def translate_selection(
    args: argparse.Namespace,
    config: TranslationConfig,
    store: TranslationStore,
    spec: ModelSpec,
    texts: list[str],
) -> tuple[Record, Record | None]:
    missing = store.missing(texts)
    run: Record = {"requested": len(texts), "already_cached": len(texts) - len(missing)}
    if not missing:
        return {**run, "translated": 0, "model_loaded": False}, None
    if store.digest != signature_digest(model_signature(spec, config.decoding)):
        raise SystemExit(f"{store.path} was opened for another model than {spec.key}")
    device = select_device(args.device if args.device is not None else config.device)
    translator = load_translator(spec, config.decoding, device)
    print(f"{spec.name} on {device}: {len(missing)} texts to translate", flush=True)
    result = translate_missing(
        translator, store, missing, config.progress_every, config.decoding.batch_token_budget
    )
    return {**run, **result, "model_loaded": True}, translator.info


def command_translate(args: argparse.Namespace, config: TranslationConfig) -> None:
    spec = selected_model(config, args.model)
    splits = resolve_splits(config, args.splits, args.final_test)
    units, context = load_units(config, splits, args.limit)
    selection = unit_selection(units, config)
    store = open_store(config, spec.key, writable=True, cache_dir=args.cache_dir)
    run, model_info = translate_selection(args, config, store, spec, selection.model_texts())
    report = build_report(args, config, spec, units, context, store, run, splits)
    report["model_info"] = model_info
    path = run_directory(config, args) / "report.json"
    write_json(report, path)
    print_report(report)
    print(f"-> {path}", flush=True)


def command_report(args: argparse.Namespace, config: TranslationConfig) -> None:
    spec = selected_model(config, args.model)
    splits = resolve_splits(config, args.splits, args.final_test)
    units, context = load_units(config, splits, args.limit)
    store = open_store(config, spec.key, cache_dir=args.cache_dir)
    report = build_report(
        args, config, spec, units, context, store, {"model_loaded": False}, splits
    )
    path = run_directory(config, args) / "report.json"
    write_json(report, path)
    print_report(report)
    print(f"-> {path}", flush=True)
