"""Blinded spot-check sheet of translations for manual review."""

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import sha256_of_file, write_json

from experiments.common.reporting import file_record, utc_timestamp
from experiments.common.transcript import is_sentence
from experiments.verifier.benchmark_inputs import split_records
from experiments.verifier.translate.commands import translate_selection
from experiments.verifier.translate.config import Segmenter, SpotCheckSpec, TranslationConfig
from experiments.verifier.translate.report import code_hashes, environment
from experiments.verifier.translate.selection import OpinionUnit, load_units, resolve_splits
from experiments.verifier.translate.statistics import FLAG_NAMES, output_flags
from experiments.verifier.translate.store import (
    TranslationStore,
    has_content,
    model_record,
    open_store,
    selected_model,
)

Record = dict[str, Any]

BLINDING_RULE = (
    "every non-alphanumeric, non-space character of the shown translations that occurs in the rows "
    "of one model only; such a row can be told apart by that character, and so can the other row "
    "of its unit, which shows the same texto_pt"
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def spot_check_population(
    units: list[OpinionUnit], segmenter: Segmenter
) -> tuple[list[str], dict[str, list[str]]]:
    occurrences: dict[str, list[str]] = {}
    for unit in units:
        for chunk in unit.chunks:
            for segment in segmenter.segments(chunk):
                if is_sentence(segment) and has_content(segment):
                    ids = occurrences.setdefault(segment, [])
                    if unit.unit_id not in ids:
                        ids.append(unit.unit_id)
    return list(occurrences), occurrences


def existing_judgments(path: Path, spot: SpotCheckSpec) -> int:
    if not path.exists():
        return 0
    with open(path, encoding=spot.encoding, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=spot.delimiter))
    return sum(
        1 for row in rows for column in spot.judgment_columns if (row.get(column) or "").strip()
    )


def write_spot_check_csv(path: Path, spot: SpotCheckSpec, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=spot.encoding, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(spot.columns), delimiter=spot.delimiter, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in spot.columns})


def spot_check_draws(
    units: list[OpinionUnit], config: TranslationConfig
) -> tuple[list[int], list[str], int, dict[str, list[str]]]:
    spot = config.spot_check
    population, occurrences = spot_check_population(units, config.segmenter)
    if len(population) < spot.size:
        raise SystemExit(f"population of {len(population)} sentences is smaller than {spot.size}")
    rng = np.random.default_rng(spot.seed)
    draws = [int(index) for index in rng.choice(len(population), size=spot.size, replace=False)]
    return draws, [population[index] for index in draws], len(population), occurrences


def spot_check_order(rows: int, seed: int) -> list[int]:
    return [int(index) for index in np.random.default_rng(seed).permutation(rows)]


def blinding_check(sheet: list[Record], items: list[Record]) -> Record:
    model_of = {item["item_id"]: item["model"] for item in items}
    unit_of = {item["item_id"]: item["unit"] for item in items}
    rows: dict[str, dict[str, set[str]]] = {}
    for row in sheet:
        for character in row["traducao_en"]:
            if not character.isalnum() and not character.isspace():
                by_model = rows.setdefault(character, {})
                by_model.setdefault(model_of[row["item_id"]], set()).add(row["item_id"])
    single = {
        f"U+{ord(character):04X}": {model: sorted(ids) for model, ids in by_model.items()}
        for character, by_model in sorted(rows.items())
        if len(by_model) == 1
    }
    identifiable = sorted(
        {item_id for by_model in single.values() for ids in by_model.values() for item_id in ids}
    )
    units = sorted({unit_of[item_id] for item_id in identifiable})
    return {
        "rule": BLINDING_RULE,
        "characters_of_one_model": single,
        "identifiable_items": identifiable,
        "identifiable_units": units,
        "items_of_identifiable_units": sorted(i for i, u in unit_of.items() if u in units),
    }


def spot_check_stores(
    args: argparse.Namespace, config: TranslationConfig, sources: list[str]
) -> tuple[dict[str, TranslationStore], dict[str, Record], dict[str, Record | None]]:
    spot = config.spot_check
    writer = None if args.cached_only else selected_model(config, args.model).key
    if writer is not None and writer not in spot.models:
        raise SystemExit(f"--model {writer} is not among spot_check.models {list(spot.models)}")
    stores = {key: open_store(config, key, cache_dir=args.cache_dir) for key in spot.models}
    for key, store in stores.items():
        missing = store.missing(sources)
        if missing and key != writer:
            raise SystemExit(
                f"{len(missing)} of {len(sources)} spot-check units have no {key} translation in "
                f"{store.path} (signature {store.digest[:16]}); spot-check translates only the "
                f"model of --model and opens every other cache read-only, so nothing was written; "
                f"run spot-check --model {key} for it first"
            )
    runs: dict[str, Record] = {}
    infos: dict[str, Record | None] = {}
    for key in spot.models:
        store = stores[key]
        missing = store.missing(sources)
        if key == writer and missing:
            stores[key] = open_store(config, key, writable=True, cache_dir=args.cache_dir)
            runs[key], infos[key] = translate_selection(
                args, config, stores[key], config.models[key], sources
            )
            continue
        runs[key] = {
            "requested": len(sources),
            "already_cached": len(sources) - len(missing),
            "translated": 0,
            "model_loaded": False,
            "store_opened": "read-only",
        }
        infos[key] = None
    return stores, runs, infos


def replaced_sheet_record(csv_path: Path, spot: SpotCheckSpec, filled: int) -> Record:
    return {
        "path": str(csv_path),
        "existed": csv_path.exists(),
        "sha256": sha256_of_file(csv_path) if csv_path.exists() else None,
        "filled_judgment_cells": filled,
        "judgment_columns": list(spot.judgment_columns),
    }


def item_record(
    item_id: str,
    unit_label: str,
    built_row: int,
    key: str,
    source: str,
    record: Record,
    shown: str,
    store: TranslationStore,
    population_index: int,
    occurrences: dict[str, list[str]],
    config: TranslationConfig,
) -> Record:
    """The key-file record of one sheet row: what the reviewer does not see."""
    return {
        "item_id": item_id,
        "unit": unit_label,
        "built_row": built_row,
        "model": key,
        "model_name": config.models[key].name,
        "model_revision": config.models[key].revision,
        "signature_sha256": store.digest,
        "population_index": population_index,
        "cache_key": record["key"],
        "source_sha256": text_sha256(source),
        "translation_sha256": text_sha256(record["translation"]),
        "display_normalized": shown != record["translation"],
        "sheet_translation_sha256": text_sha256(shown),
        "opinion_ids": occurrences[source],
        "hearing_ids": sorted({int(i.split("-")[1]) for i in occurrences[source]}),
        "input_tokens": record["input_tokens"],
        "output_tokens": record["output_tokens"],
        "device": record["device"],
        "flags": output_flags(source, record, config),
    }


def unit_label(unit: int, units: int) -> str:
    return f"U{unit + 1:0{len(str(units))}d}"


def build_sheet(
    sources: list[str],
    draws: list[int],
    occurrences: dict[str, list[str]],
    stores: dict[str, TranslationStore],
    config: TranslationConfig,
) -> tuple[list[Record], list[Record], dict[tuple[int, str], str], list[int]]:
    """The shuffled sheet rows, their key records and the item id of each (unit, model)."""
    spot = config.spot_check
    built = [(unit, key) for unit in range(len(sources)) for key in spot.models]
    order = spot_check_order(len(built), spot.order_seed)
    width = len(str(len(built)))
    sheet, items = [], []
    item_of: dict[tuple[int, str], str] = {}
    for number, index in enumerate(order, start=1):
        unit, key = built[index]
        source = sources[unit]
        record = stores[key].record(source)
        if record is None:
            raise SystemExit(f"unit {unit + 1} has no {key} translation after translate")
        item_id = f"{spot.item_id_prefix}{number:0{width}d}"
        item_of[(unit, key)] = item_id
        shown = spot.display(record["translation"])
        sheet.append({"item_id": item_id, "texto_pt": source, "traducao_en": shown})
        items.append(
            item_record(
                item_id,
                unit_label(unit, len(sources)),
                index,
                key,
                source,
                record,
                shown,
                stores[key],
                draws[unit],
                occurrences,
                config,
            )
        )
    return sheet, items, item_of, order


def unit_records(
    sources: list[str],
    draws: list[int],
    stores: dict[str, TranslationStore],
    item_of: dict[tuple[int, str], str],
    spot: SpotCheckSpec,
) -> list[Record]:
    distinct = [
        {stores[key].entries[stores[key].key(source)]["translation"] for key in spot.models}
        for source in sources
    ]
    return [
        {
            "unit": unit_label(unit, len(sources)),
            "population_index": draws[unit],
            "source_sha256": text_sha256(sources[unit]),
            "items": {key: item_of[(unit, key)] for key in spot.models},
            "translations_identical": len(distinct[unit]) == 1,
        }
        for unit in range(len(sources))
    ]


def counts_record(
    sources: list[str], items: list[Record], units: list[Record], spot: SpotCheckSpec
) -> Record:
    return {
        "units": len(sources),
        "rows": len(items),
        "units_with_identical_translations": sum(
            1 for record in units if record["translations_identical"]
        ),
        "flags_by_model": {
            key: {
                name: sum(1 for item in items if item["model"] == key and item["flags"][name])
                for name in FLAG_NAMES
            }
            for key in spot.models
        },
    }


def command_spot_check(args: argparse.Namespace, config: TranslationConfig) -> None:
    spot = config.spot_check
    splits = resolve_splits(config, [spot.split], False)
    units, context = load_units(config, splits, None)
    draws, sources, population_size, occurrences = spot_check_draws(units, config)
    out_dir = spot.output_dir / spot.name
    csv_path = out_dir / "spot_check.csv"
    filled = existing_judgments(csv_path, spot)
    if filled:
        raise SystemExit(f"{csv_path} has {filled} judgment cells filled in; move it away first")
    replaced = replaced_sheet_record(csv_path, spot, filled)
    stores, runs, infos = spot_check_stores(args, config, sources)
    sheet, items, item_of, order = build_sheet(sources, draws, occurrences, stores, config)
    unit_list = unit_records(sources, draws, stores, item_of, spot)
    write_spot_check_csv(csv_path, spot, sheet)
    (out_dir / "README.md").write_text(spot.readme)
    raw = config.source["translation"]["spot_check"]
    key_report = {
        "experiment": "translation_spot_check",
        "name": spot.name,
        "created_at": utc_timestamp(),
        "supersedes": raw.get("supersedes"),
        "splits": split_records(splits, False),
        "population": {"size": population_size, "rule": raw["population"]},
        "draw": {"seed": spot.seed, "size": spot.size, "rule": raw["draw_rule"], "indices": draws},
        "order": {"seed": spot.order_seed, "rule": raw["order_rule"], "permutation": order},
        "display": {
            "normalization": {
                f"U+{ord(glyph):04X}": shown for glyph, shown in spot.display_normalization
            },
            "rule": raw.get("display_rule"),
            "rows_changed": sum(1 for item in items if item["display_normalized"]),
        },
        "blinding_check": blinding_check(sheet, items),
        "replaced_sheet": replaced,
        "models": {key: model_record(config, config.models[key]) for key in spot.models},
        "counts": counts_record(sources, items, unit_list, spot),
        "units": unit_list,
        "items": items,
        "sheet": file_record(csv_path),
        "runs": runs,
        "model_info": infos,
        "stores": {key: store.summary() for key, store in stores.items()},
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    write_json(key_report, out_dir / "spot_check_key.json")
    print(
        f"{len(sources)} of {population_size} units x {len(spot.models)} models = {len(items)} "
        f"rows -> {csv_path}",
        flush=True,
    )
