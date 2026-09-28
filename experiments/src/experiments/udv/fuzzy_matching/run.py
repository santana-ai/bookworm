"""One run of the experiment: collect the rows, write rows, blind sheets, keys and the report."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm import load_gated_jsonl, sha256_of_file, write_json, write_jsonl

from experiments.common import udv_run
from experiments.common.reporting import file_record
from experiments.common.splits import load_split_lookup
from experiments.udv.fuzzy_matching.config import REVIEW_KINDS, FuzzyConfig, run_name_for
from experiments.udv.fuzzy_matching.inputs import select_hearings, select_reference_run
from experiments.udv.fuzzy_matching.names import collect_names
from experiments.udv.fuzzy_matching.quotes import collect_quotes
from experiments.udv.fuzzy_matching.report import RunSources, build_report, print_summary
from experiments.udv.fuzzy_matching.review import (
    blind_sheet,
    hearing_text,
    name_review_items,
    quote_review_items,
    refuse_filled_sheet,
    review_key,
    review_summary,
    sheet_rows,
)

Record = dict[str, Any]

PRECISION_COMMAND = "uv run --no-sync python -m experiments.validation.fuzzy_review_precision"


@dataclass(frozen=True)
class ReviewPaths:
    rows: Path
    sheet: Path
    key: Path


def review_paths(config: FuzzyConfig, run_name: str, kind: str) -> ReviewPaths:
    return ReviewPaths(
        rows=config.output_dir / f"{run_name}_{kind}.jsonl",
        sheet=config.output_dir / f"{run_name}_{kind}_review.jsonl",
        key=config.output_dir / f"{run_name}_{kind}_review_key.json",
    )


def encoder_source(config: FuzzyConfig, udv_config: udv_run.UdvConfig) -> Record:
    return {
        "udv_config_path": str(config.udv_config_path),
        "udv_config_sha256": sha256_of_file(config.udv_config_path),
        "encoder": udv_config.model_name,
        "encoder_revision": udv_config.model_revision,
        "cache_dir": str(udv_config.cache_dir),
        "cache_device_label": config.cache_device_label,
        "encoding_done_here": False,
    }


def write_reviews(
    rows_by_kind: dict[str, list[Record]],
    sheets: dict[str, dict[str, Record]],
    sources: RunSources,
    config: FuzzyConfig,
) -> tuple[Record, Record]:
    """Write every kind's rows, blind sheet and key; the artifacts and review sections."""
    paths = {kind: review_paths(config, sources.run_name, kind) for kind in REVIEW_KINDS}
    for kind in REVIEW_KINDS:
        refuse_filled_sheet(paths[kind].sheet)
    artifacts: Record = {}
    review: Record = {
        "blinding_rule": config.source["review"]["blinding_rule"],
        "known_leaks": config.source["review"]["known_leaks"],
        "reviewer_opens": [str(paths[kind].sheet) for kind in REVIEW_KINDS],
        "precision_command": PRECISION_COMMAND + (" --final-test" if sources.final_test else ""),
    }
    for kind in REVIEW_KINDS:
        kind_paths = paths[kind]
        write_jsonl(rows_by_kind[kind], kind_paths.rows)
        write_jsonl(sheet_rows(kind, sheets[kind]), kind_paths.sheet)
        key = review_key(
            kind,
            sheets[kind],
            sources.run_name,
            sources.splits,
            sources.final_test,
            kind_paths.rows,
            kind_paths.sheet,
            config,
        )
        write_json(key, kind_paths.key)
        artifacts[kind] = file_record(kind_paths.rows, len(rows_by_kind[kind]))
        artifacts[f"{kind}_review"] = file_record(kind_paths.sheet, len(sheets[kind]))
        artifacts[f"{kind}_review_key"] = file_record(kind_paths.key, len(sheets[kind]))
        review[kind] = review_summary(kind, key, kind_paths.key)
    return artifacts, review


def run(config: FuzzyConfig, final_test: bool) -> None:
    started = time.perf_counter()
    splits = (*config.splits, "test") if final_test else config.splits
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    hearings = select_hearings(lds, split_of, splits)
    reference, reference_source = select_reference_run(config.reference_runs)
    udv_config = udv_run.load_config(config.udv_config_path)
    sources = RunSources(
        run_name=run_name_for(config, final_test),
        splits=splits,
        final_test=final_test,
        split_source=split_source,
        reference_source=reference_source,
        encoder_source=encoder_source(config, udv_config),
    )
    print(
        f"{len(hearings)} hearings ({', '.join(splits)}) | reference {reference_source['run_name']}"
        f"{' (fallback)' if reference_source['fallback_used'] else ''}",
        flush=True,
    )
    quotes = collect_quotes(hearings, split_of, reference, udv_config, config)
    name_rows, impostors = collect_names(hearings, split_of, config)
    texts = {hearing["id"]: hearing_text(hearing) for hearing in hearings}
    sheets = {
        "quotes": blind_sheet("quotes", quote_review_items(quotes.rows, texts, config), config),
        "names": blind_sheet("names", name_review_items(name_rows, config), config),
    }
    rows_by_kind = {"quotes": quotes.rows, "names": name_rows}
    artifacts, review = write_reviews(rows_by_kind, sheets, sources, config)
    report = build_report(
        sources,
        quotes,
        name_rows,
        impostors,
        artifacts,
        review,
        time.perf_counter() - started,
        config,
    )
    write_json(report, config.output_dir / f"{sources.run_name}_report.json")
    print_summary(report, config)
    print(f"elapsed {report['timing']['elapsed_seconds']}s", flush=True)
