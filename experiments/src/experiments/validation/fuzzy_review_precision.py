import argparse
import json
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bookworm.data.io
import numpy as np
import rapidfuzz
import scipy
from bookworm import load_jsonl, sha256_of_file, write_json

from experiments.common.provenance import source_hashes
from experiments.udv import fuzzy_matching
from experiments.udv.fuzzy_matching import (
    DISPLAY_FIELDS,
    NAME_METRICS,
    QUOTE_METHODS,
    REVIEW_KINDS,
    FuzzyConfig,
    accepted_result,
    bootstrap_ratio,
    group_members,
    load_config,
    name_decisions,
    name_rules,
    resolved_names,
    rounded,
)
from experiments.validation import generate_sample
from experiments.validation.generate_sample import canonical_sha256, wilson_interval

Record = dict[str, Any]
Entry = tuple[int, str | None]


def run_name_for(config: FuzzyConfig, final_test: bool) -> str:
    return f"{config.version}_final_test" if final_test else config.version


def load_key(path: Path, kind: str, final_test: bool) -> Record:
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run experiments.udv.fuzzy_matching first")
    with open(path) as f:
        key: Record = json.load(f)
    if key.get("role") != f"fuzzy_{kind}_review_key":
        raise SystemExit(f"{path} is not a {kind} review key")
    if (key["final_test"] or "test" in key["splits_used"]) and not final_test:
        raise SystemExit(f"{path} covers test hearings; pass --final-test")
    if key["display_fields"] != list(DISPLAY_FIELDS[kind]):
        raise SystemExit(f"{path} was built with other sheet fields than this code shows")
    return key


def normalized_judgment(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    return value.strip().lower() or None


def validate_sheet(rows: list[Record], key: Record) -> tuple[dict[str, str], list[str]]:
    expected = ["item_id", *key["display_fields"], *key["fill_fields"]]
    labels = set(key["labels"])
    judgments: dict[str, str] = {}
    present: set[str] = set()
    problems: list[str] = []
    for line, row in enumerate(rows, start=1):
        item_id = row.get("item_id")
        if sorted(row) != sorted(expected) or not isinstance(item_id, str):
            problems.append(f"line {line} ({item_id}): fields {sorted(row)} differ from {expected}")
            continue
        if item_id in present:
            problems.append(f"{item_id}: appears more than once")
            continue
        present.add(item_id)
        item = key["items"].get(item_id)
        if item is None:
            problems.append(f"{item_id}: not an item of the key")
            continue
        shown = {name: row[name] for name in key["display_fields"]}
        if canonical_sha256(shown) != item["display_sha256"]:
            problems.append(f"{item_id}: a shown field was edited")
        judgment = normalized_judgment(row["judgment"])
        if judgment is None:
            problems.append(f"{item_id}: judgment is empty")
        elif judgment not in labels:
            problems.append(f"{item_id}: judgment {row['judgment']!r} is not in {sorted(labels)}")
        else:
            judgments[item_id] = judgment
    problems += [
        f"{item_id}: missing from the sheet" for item_id in key["items"] if item_id not in present
    ]
    return judgments, problems


def precision_entry(entries: list[Entry], key: Record, config: FuzzyConfig, stream: str) -> Record:
    positive, unsure = key["positive_label"], key["unsure_label"]
    counts = Counter(
        judgment if judgment is not None else "not_reviewed" for _, judgment in entries
    )
    hits: dict[int, float] = {}
    decided: dict[int, float] = {}
    judged: dict[int, float] = {}
    for hearing_id, judgment in entries:
        if judgment is None:
            continue
        judged[hearing_id] = judged.get(hearing_id, 0.0) + 1
        if judgment != unsure:
            decided[hearing_id] = decided.get(hearing_id, 0.0) + 1
        if judgment == positive:
            hits[hearing_id] = hits.get(hearing_id, 0.0) + 1
    hearing_ids = sorted(judged)
    wilson = wilson_interval(
        int(sum(hits.values())), int(sum(decided.values())), config.confidence_level
    )
    return {
        "accepted": len(entries),
        "hearings": len({hearing_id for hearing_id, _ in entries}),
        "judgments": dict(sorted(counts.items())),
        "precision": bootstrap_ratio(hits, decided, hearing_ids, config, stream),
        "precision_unsure_as_error": bootstrap_ratio(
            hits, judged, hearing_ids, config, f"{stream}/unsure_as_error"
        ),
        "wilson_ignoring_hearings": {
            name: rounded(value) if isinstance(value, float) else value
            for name, value in wilson.items()
        },
    }


def quote_lookup(key: Record) -> dict[tuple[str, str, int], str]:
    lookup: dict[tuple[str, str, int], str] = {}
    for item_id, item in key["items"].items():
        for proposal in item["proposals"]:
            lookup[(item["row_id"], proposal["method"], proposal["level"])] = item_id
    return lookup


def quote_entries(
    rows: list[Record],
    method: str,
    low: float,
    high: float | None,
    lookup: dict[tuple[str, str, int], str],
    judgments: dict[str, str],
    config: FuzzyConfig,
) -> list[Entry]:
    entries: list[Entry] = []
    for row in rows:
        result = accepted_result(row, method, low, config.prefix_levels)
        if result is None:
            continue
        if high is not None and accepted_result(row, method, high, config.prefix_levels):
            continue
        item_id = lookup.get((row["id"], method, result["level"]))
        entries.append((row["hearing_id"], judgments[item_id] if item_id else None))
    return entries


def quote_precision(
    rows: list[Record],
    key: Record,
    judgments: dict[str, str],
    config: FuzzyConfig,
    group: str,
) -> Record:
    lookup = quote_lookup(key)
    thresholds = config.quote_thresholds
    result: Record = {}
    for method in QUOTE_METHODS:
        by_threshold = {
            str(threshold): precision_entry(
                quote_entries(rows, method, threshold, None, lookup, judgments, config),
                key,
                config,
                f"review/{group}/quotes/{method}/{threshold}",
            )
            for threshold in thresholds
        }
        bands = []
        for position, low in enumerate(thresholds):
            high = thresholds[position + 1] if position + 1 < len(thresholds) else None
            entry = precision_entry(
                quote_entries(rows, method, low, high, lookup, judgments, config),
                key,
                config,
                f"review/{group}/quotes/{method}/band/{low}",
            )
            bands.append({"low": low, "high": high, **entry})
        result[method] = {"at_or_above_threshold": by_threshold, "bands": bands}
    return result


def name_precision(
    rows: list[Record],
    key: Record,
    judgments: dict[str, str],
    config: FuzzyConfig,
    group: str,
) -> Record:
    lookup = {(item["row_id"], item["speaker"]): item_id for item_id, item in key["items"].items()}
    result: Record = {}
    for rule in name_rules(config):
        result[rule] = {}
        for threshold in config.name_thresholds:
            resolved, _ = resolved_names(name_decisions(rows, rule, threshold, config))
            entries: list[Entry] = []
            for row, decision in resolved:
                item_id = lookup.get((row["id"], decision["result"]["best"]["speaker"]))
                entries.append((row["hearing_id"], judgments[item_id] if item_id else None))
            result[rule][str(threshold)] = precision_entry(
                entries, key, config, f"review/{group}/names/{rule}/{threshold}"
            )
    return result


def load_review(
    kind: str, review_dir: Path, run_name: str, final_test: bool
) -> tuple[Record, list[Record], dict[str, str], list[str], Record]:
    key_path = review_dir / f"{run_name}_{kind}_review_key.json"
    sheet_path = review_dir / f"{run_name}_{kind}_review.jsonl"
    key = load_key(key_path, kind, final_test)
    rows_path = Path(key["rows_file"]["path"])
    problems: list[str] = []
    if not rows_path.exists() or sha256_of_file(rows_path) != key["rows_file"]["sha256"]:
        problems.append(f"{rows_path} is not the rows file the {kind} key was built from")
    if not sheet_path.exists():
        raise SystemExit(f"{sheet_path} does not exist")
    judgments, sheet_problems = validate_sheet(load_jsonl(sheet_path), key)
    problems += [f"{sheet_path.name}: {problem}" for problem in sheet_problems]
    rows = load_jsonl(rows_path) if not problems else []
    inputs = {
        "sheet": {
            "path": str(sheet_path),
            "sha256": sha256_of_file(sheet_path),
            "sha256_at_creation": key["sheet"]["sha256_at_creation"],
        },
        "key": {"path": str(key_path), "sha256": sha256_of_file(key_path)},
        "rows": {"path": str(rows_path), "sha256": key["rows_file"]["sha256"]},
    }
    return key, rows, judgments, problems, inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join the filled blind review sheets of the fuzzy matching experiment with "
        "their keys and report quote and name precision by method, rule and score band."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/fuzzy_matching.toml"))
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=None,
        help="folder with the sheets and keys (default: run.output_dir of the config)",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="read the review of the final-test run, which covers test hearings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    review_dir = args.review_dir or config.output_dir
    run_name = run_name_for(config, args.final_test)
    loaded = {
        kind: load_review(kind, review_dir, run_name, args.final_test) for kind in REVIEW_KINDS
    }
    problems = [problem for kind in REVIEW_KINDS for problem in loaded[kind][3]]
    if problems:
        print("\n".join(problems[:50]))
        raise SystemExit(f"{len(problems)} problems in the review sheets; no report written")
    splits = tuple(loaded["quotes"][0]["splits_used"])
    if tuple(loaded["names"][0]["splits_used"]) != splits:
        raise SystemExit("the quote and name keys cover different splits")
    groups = group_members(splits)
    precision: Record = {"quotes": {}, "names": {}}
    for kind, compute in (("quotes", quote_precision), ("names", name_precision)):
        key, rows, judgments, _, _ = loaded[kind]
        for group, members in groups.items():
            members_rows = [row for row in rows if row["split"] in members]
            precision[kind][group] = compute(members_rows, key, judgments, config, group)
    report = {
        "run_name": run_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": (
            "how often a fuzzy quote match points at the passage where the quotation starts, and "
            "how often a fuzzy name match picks the participant, by method, rule and score band, "
            "from blind human judgments"
        ),
        "splits_used": list(splits),
        "final_test": args.final_test,
        "rule": config.source["review"]["precision"]["precision_rule"],
        "label_definitions": {
            kind: config.source["review"][kind]["label_definitions"] for kind in REVIEW_KINDS
        },
        "inputs": {kind: loaded[kind][4] for kind in REVIEW_KINDS},
        "judgment_counts": {
            kind: dict(sorted(Counter(loaded[kind][2].values()).items())) for kind in REVIEW_KINDS
        },
        "quotes": precision["quotes"],
        "names": precision["names"],
        "name_metrics": list(NAME_METRICS),
        "code": {
            **source_hashes(Path(__file__)),
            **source_hashes(fuzzy_matching),
            **source_hashes(generate_sample),
            **source_hashes(bookworm.data.io),
        },
        "timing": {"elapsed_seconds": round(time.perf_counter() - started, 1)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rapidfuzz": rapidfuzz.__version__,
            "platform": platform.platform(),
        },
        "config": config.source,
    }
    output = review_dir / f"{run_name}_{config.source['review']['precision']['output_suffix']}.json"
    write_json(report, output)
    for group, methods in precision["quotes"].items():
        for method, entry in methods.items():
            cells = [
                f"t={threshold}: {value['judgments']} p={value['precision']['value']}"
                for threshold, value in entry["at_or_above_threshold"].items()
                if float(threshold) in (config.quote_thresholds[0], 90.0, 100.0)
            ]
            print(f"quotes {group:10s} {method:5s} " + " | ".join(cells))
    for group, rules in precision["names"].items():
        for rule, by_threshold in rules.items():
            cells = [
                f"t={threshold}: {value['judgments']} p={value['precision']['value']}"
                for threshold, value in by_threshold.items()
                if float(threshold) in (config.name_thresholds[0], 90.0)
            ]
            print(f"names  {group:10s} {rule:15s} " + " | ".join(cells))
    print(f"report -> {output}")


if __name__ == "__main__":
    main()
