"""Scorer dispatch, score files and the score command."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import sha256_of_file, write_json, write_jsonl

from experiments.common.reporting import utc_timestamp
from experiments.common.splits import SPLIT_NAMES, load_split_lookup
from experiments.common.udv_run import select_device
from experiments.udv.calibrate_threshold import describe
from experiments.verifier.benchmark_inputs import (
    check_benchmark_file,
    load_nli_chunks,
    split_records,
)
from experiments.verifier.nli.benchmark import (
    PremiseUnit,
    benchmark_units,
    load_benchmark_rows,
    load_transcripts,
    sample_rows,
    verify_chunk_sources,
)
from experiments.verifier.nli.config import (
    DECISION_KINDS,
    ScorerSpec,
    VerifierConfig,
    narrowed_splits,
    selected_scorers,
)
from experiments.verifier.nli.cosine import score_units_cosine
from experiments.verifier.nli.cross_encoder import portuguese_probes, score_units_nli
from experiments.verifier.nli.decision import (
    check_decision_scorer,
    decision_truncation_summary,
    score_units_decision,
)
from experiments.verifier.nli.provenance import code_hashes, environment
from experiments.verifier.nli.translated import (
    Translations,
    check_translations,
    english_probes,
    english_units,
    open_translations,
    translation_summary,
)

Record = dict[str, Any]


def score_units(
    units: list[PremiseUnit],
    spec: ScorerSpec,
    config: VerifierConfig,
    device: str,
    concatenate_single: bool,
    prefix: str,
    probe_pairs: list[tuple[str, str]] | None = None,
    decision_mode: str = "live",
) -> tuple[list[Record], Record]:
    pairs = portuguese_probes(config) if probe_pairs is None else probe_pairs
    if spec.kind == "cosine":
        return score_units_cosine(units, spec, config, device, prefix)
    if spec.kind in DECISION_KINDS:
        return score_units_decision(
            units, spec, config, device, concatenate_single, pairs, decision_mode
        )
    return score_units_nli(units, spec, config, device, concatenate_single, pairs)


def truncation_summary(rows: list[Record]) -> Record:
    items = [item for row in rows for item in row["items"] if item is not None]
    concatenated = [row["concatenated"] for row in rows if row["concatenated"] is not None]
    truncated = [entry for entry in concatenated if entry["truncated"]]
    summary: Record = {
        "items_scored": len(items),
        "items_truncated": sum(1 for item in items if item["truncated"]),
        "item_tokens": describe(np.array([item["tokens"] for item in items], dtype=np.float64)),
        "concatenated_premises": len(concatenated),
        "concatenated_truncated": len(truncated),
    }
    if concatenated:
        summary["concatenated_tokens"] = describe(
            np.array([entry["tokens"] for entry in concatenated], dtype=np.float64)
        )
    if truncated:
        summary["kept_premise_fraction_when_truncated"] = describe(
            np.array([e["premise_tokens_kept"] / e["premise_tokens"] for e in truncated])
        )
    if items and "sentences" in items[0]:
        summary["sentences"] = sum(item["sentences"] for item in items)
        summary["sentences_truncated"] = sum(item["sentences_truncated"] for item in items)
        summary["items_without_sentence"] = sum(1 for item in items if item["sentence_fallback"])
    return summary


def unit_counts(rows: list[Record]) -> Record:
    return {
        "units": len(rows),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "hearings": len({row["hearing_id"] for row in rows}),
        "no_premise": sum(1 for row in rows if row["no_premise"]),
        "no_premise_ids": [row["id"] for row in rows if row["no_premise"]],
        "item_count_histogram": dict(sorted(Counter(row["item_count"] for row in rows).items())),
    }


def check_score_names(rows: list[Record], spec: ScorerSpec) -> None:
    for row in rows:
        if set(row["scores"]) != set(spec.scores):
            raise SystemExit(f"{spec.key} {row['id']}: scores {sorted(row['scores'])}")


def score_file(run_dir: Path, scorer: str, split: str) -> Path:
    return run_dir / "scores" / f"{scorer}_{split}.jsonl"


def score_report_file(run_dir: Path, scorer: str) -> Path:
    return run_dir / "scores" / f"{scorer}_report.json"


def run_directory(config: VerifierConfig, args: argparse.Namespace) -> Path:
    base = args.output_dir if args.output_dir is not None else config.output_dir
    return base / args.run_name


def subset_record(limit: int | None, rule: str, seed: int) -> Record | None:
    if limit is None:
        return None
    if rule == "first":
        return {"limit_per_split": limit, "rule": "first rows of each split in file order"}
    return {"limit_per_split": limit, "seed": seed}


def prepare_benchmark_units(
    config: VerifierConfig, splits: tuple[str, ...], limit: int | None, rule: str = "random"
) -> tuple[list[PremiseUnit], Record]:
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    benchmark = check_benchmark_file(config)
    rows = sample_rows(load_benchmark_rows(config, split_of, splits), limit, config.seed, rule)
    nli_chunks = load_nli_chunks(config, rows)
    transcripts = load_transcripts(config, {row["hearing_id"] for row in rows})
    verification = verify_chunk_sources(rows, nli_chunks, transcripts)
    units = benchmark_units(rows, nli_chunks, transcripts, config.text_source)
    context = {
        "sources": {
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "nli": {"path": str(config.nli_path), "sha256": config.nli_sha256},
            "benchmark": benchmark,
            "splits": split_source,
        },
        "premise": {
            "text_source": config.text_source,
            "concat_separator": config.concat_separator,
            "chunk_source_verification": verification,
        },
        "subset": subset_record(limit, rule, config.seed),
    }
    return units, context


def score_report(
    mode: str,
    run_name: str,
    spec: ScorerSpec,
    rows: list[Record],
    details: Record,
    context: Record,
    splits: Record,
    config: VerifierConfig,
    files: Record,
) -> Record:
    return {
        "experiment": "nli_verifier",
        "mode": mode,
        "run_name": run_name,
        "created_at": utc_timestamp(),
        "scorer": spec.key,
        "kind": spec.kind,
        "splits": splits,
        "language": spec.language,
        "counts": unit_counts(rows),
        "truncation": (
            decision_truncation_summary(rows, spec.questions)
            if spec.kind in DECISION_KINDS
            else truncation_summary(rows)
        ),
        **details,
        **context,
        "files": files,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }


def write_split_scores(rows: list[Record], run_dir: Path, scorer: str) -> Record:
    files: Record = {}
    for split in SPLIT_NAMES:
        members = [row for row in rows if row["split"] == split]
        if not members:
            continue
        path = score_file(run_dir, scorer, split)
        write_jsonl(members, path)
        files[split] = {"path": str(path), "rows": len(members), "sha256": sha256_of_file(path)}
    return files


def scorer_checks(
    spec: ScorerSpec,
    units: list[PremiseUnit],
    config: VerifierConfig,
    translations: Translations | None,
    mode: str,
) -> Record:
    checks: Record = {}
    if spec.kind in DECISION_KINDS:
        checks |= check_decision_scorer(spec, config, mode)
    if spec.language == "en":
        if translations is None:
            raise SystemExit(f"{spec.key}: no translation store opened")
        checks["translation"] = check_translations(spec, units, config, translations.store(spec))
    return checks


def device_of(args: argparse.Namespace, config: VerifierConfig) -> str:
    requested = getattr(args, "device", None)
    return select_device(requested if requested is not None else config.device)


def command_score(args: argparse.Namespace, config: VerifierConfig) -> None:
    splits = narrowed_splits(config, args.final_test, args.splits)
    specs = selected_scorers(config, args.scorers)
    device = device_of(args, config)
    units, context = prepare_benchmark_units(config, splits, args.limit_per_split, args.subset_rule)
    run_dir = run_directory(config, args)
    translations = open_translations(config, specs, args.translation_cache_dir)
    checks = {
        spec.key: scorer_checks(spec, units, config, translations, args.decision_mode)
        for spec in specs
    }
    print(f"{len(units)} opinions ({', '.join(splits)}) on {device} -> {run_dir}", flush=True)
    for spec in specs:
        spec_units, probes, extra = units, portuguese_probes(config), {}
        if checks[spec.key]:
            extra["checks"] = checks[spec.key]
        if spec.language == "en" and translations is not None:
            store = translations.store(spec)
            spec_units, probes = english_units(units, store), english_probes(config, store)
            extra["translation"] = translation_summary(config, translations, spec)
        rows, details = score_units(
            spec_units, spec, config, device, True, "b2", probes, args.decision_mode
        )
        details |= extra
        check_score_names(rows, spec)
        files = write_split_scores(rows, run_dir, spec.key)
        report = score_report(
            "benchmark",
            args.run_name,
            spec,
            rows,
            details,
            context,
            split_records(splits, args.final_test),
            config,
            files,
        )
        write_json(report, score_report_file(run_dir, spec.key))
        print(
            f"{spec.key}: {len(rows)} opinions, timing {json.dumps(details['timing'])}, "
            f"truncation {json.dumps(report['truncation'])}",
            flush=True,
        )
