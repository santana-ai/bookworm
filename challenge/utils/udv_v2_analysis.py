import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from utils import precision_report as precision
from utils.dataset_io import load_jsonl, sha256_of_file, write_json
from utils.generate_validation_sample import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    load_key,
    load_validation_config,
    read_annotation_csv,
    require_final_test,
    validate_annotation,
)
from utils.udv_pipeline import (
    SENTENCE_BOUNDARY_PATTERN,
    extract_quotes,
    quote_prefix_pattern,
    quote_prefixes,
)

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
EVIDENCE_TIERS = ("quote_found", "semantic_match_high", "semantic_match_weak")
SEMANTIC_TIERS = ("semantic_match_high", "semantic_match_weak")
IDENTITY_FIELDS = ("text", "start_char", "end_char", "speaker_turn")
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CLOSING_WORDS = 3
NO_JUDGMENTS = "no judgments yet"
DEFAULT_PATHS: Record = {
    "v1": "artifacts/udv/udv_v1.jsonl",
    "v2": "artifacts/udv/udv_v2.jsonl",
    "v1_coverage": "artifacts/udv/udv_v1_coverage.json",
    "v2_coverage": "artifacts/udv/udv_v2_coverage.json",
    "v1_verifier": "artifacts/udv/udv_v1_verifier.jsonl",
    "v2_verifier": "artifacts/udv/udv_v2_verifier.jsonl",
    "v2_verifier_report": "artifacts/udv/udv_v2_verifier_report.json",
    "cosine_calibration": "artifacts/calibration/threshold_v2.json",
    "verifier_calibration": "artifacts/calibration/udv_verifier_threshold_v1.json",
    "splits": "artifacts/splits/temporal_v1.json",
    "sample_dir": "artifacts/validation/human_validation_v1_udv_v1",
}


def load_split_lookup(path: Path) -> dict[int, str]:
    with open(path) as f:
        manifest = json.load(f)
    return {hearing: name for name in SPLIT_NAMES for hearing in manifest[name]}


def evidence_identity(record: Record) -> tuple[Any, ...] | None:
    evidence = record.get("evidence")
    if evidence is None:
        return None
    return tuple(evidence[field] for field in IDENTITY_FIELDS)


def same_evidence(first: Record, second: Record) -> bool:
    return evidence_identity(first) == evidence_identity(second)


def part_count(text: str) -> int:
    return len([part for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()])


def describe(values: list[float]) -> Record:
    if not values:
        return {"n": 0}
    array = np.array(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": round(float(array.mean()), 4),
        "quantiles": {str(q): round(float(np.quantile(array, q)), 4) for q in QUANTILES},
    }


def chosen_quote(record: Record) -> str | None:
    prefix = record["evidence"]["quote_prefix"]
    for quote in extract_quotes(record["proposition"]):
        if any(candidate == prefix for candidate, _ in quote_prefixes(quote)):
            return quote
    return None


def closing_words_found(quote: str, text: str) -> bool:
    words = quote.split()[-CLOSING_WORDS:]
    closing = " ".join(words).rstrip(".,;:!?…\"'”’)] ")
    if len(closing) < 3:
        return False
    return quote_prefix_pattern(closing).search(text) is not None


def quote_lengths(records: list[Record]) -> Record:
    quotes = [record for record in records if record["tier"] == "quote_found"]
    matched = [(record, chosen_quote(record)) for record in quotes]
    with_quote = [(record, quote) for record, quote in matched if quote is not None]
    return {
        "records": len(quotes),
        "chars": describe([len(record["evidence"]["text"]) for record in quotes]),
        "words": describe([len(record["evidence"]["text"].split()) for record in quotes]),
        "sentence_parts": describe([part_count(record["evidence"]["text"]) for record in quotes]),
        "quote_identified": len(with_quote),
        "quote_parts": describe([part_count(quote) for _, quote in with_quote]),
        "multi_part_quotes": sum(1 for _, quote in with_quote if part_count(quote) > 1),
        "closing_words_in_evidence": sum(
            1
            for record, quote in with_quote
            if closing_words_found(quote, record["evidence"]["text"])
        ),
        "closing_words_rule": (
            f"the last {CLOSING_WORDS} words of the quote whose prefix is evidence.quote_prefix, "
            "trailing punctuation removed, found in the evidence text with the prefix pattern"
        ),
    }


def change_kind(old: Record, new: Record) -> str:
    if same_evidence(old, new):
        return "same_tier_same_evidence" if old["tier"] == new["tier"] else "tier_only"
    if old["tier"] == "quote_found" and new["tier"] == "quote_found":
        old_text, new_text = old["evidence"]["text"], new["evidence"]["text"]
        return "quote_extended" if old_text in new_text else "quote_changed"
    if old["tier"] in SEMANTIC_TIERS and new["tier"] in SEMANTIC_TIERS:
        if old["evidence"]["text"] in new["evidence"]["text"]:
            return "window_contains_v1_sentence"
        return "window_elsewhere"
    return "other"


def record_diff(v1: dict[str, Record], v2: dict[str, Record], split_of: dict[int, str]) -> Record:
    if set(v1) != set(v2):
        raise SystemExit("udv_v1 and udv_v2 hold different UDV ids")
    kinds: Counter[str] = Counter()
    by_tier: dict[str, Counter[str]] = {}
    by_split: dict[str, Counter[str]] = {}
    moves: Counter[str] = Counter()
    for udv_id, old in v1.items():
        new = v2[udv_id]
        kind = change_kind(old, new)
        kinds[kind] += 1
        by_tier.setdefault(old["tier"], Counter())[kind] += 1
        by_split.setdefault(split_of[old["hearing_id"]], Counter())[kind] += 1
        moves[f"{old['tier']}->{new['tier']}"] += 1
    changed = sum(count for kind, count in kinds.items() if kind != "same_tier_same_evidence")
    evidence_changed = sum(1 for udv_id in v1 if not same_evidence(v1[udv_id], v2[udv_id]))
    return {
        "records": len(v1),
        "evidence_identical": len(v1) - evidence_changed,
        "evidence_changed": evidence_changed,
        "records_changed_any": changed,
        "identity_fields": list(IDENTITY_FIELDS),
        "kinds": dict(sorted(kinds.items())),
        "kinds_by_v1_tier": {tier: dict(sorted(c.items())) for tier, c in sorted(by_tier.items())},
        "kinds_by_split": {split: dict(sorted(c.items())) for split, c in sorted(by_split.items())},
        "tier_moves": dict(sorted(moves.items())),
        "kind_rules": {
            "same_tier_same_evidence": "evidence text and offsets identical, same tier",
            "tier_only": "evidence identical, tier changed",
            "quote_extended": "quote_found in both, the v1 text is contained in the v2 text",
            "quote_changed": "quote_found in both, the v1 text is not contained in the v2 text",
            "window_contains_v1_sentence": "semantic in both, the v2 window holds the v1 sentence",
            "window_elsewhere": "semantic in both, the v2 window does not hold the v1 sentence",
            "other": "any other change",
        },
    }


def tier_counts(records: list[Record], split_of: dict[int, str]) -> Record:
    result: Record = {"all": dict(Counter(record["tier"] for record in records))}
    for split in SPLIT_NAMES:
        result[split] = dict(
            Counter(record["tier"] for record in records if split_of[record["hearing_id"]] == split)
        )
    return result


def semantic_lengths(records: list[Record]) -> Record:
    semantic = [record for record in records if record["tier"] in SEMANTIC_TIERS]
    return {
        "records": len(semantic),
        "chars": describe([len(record["evidence"]["text"]) for record in semantic]),
        "scores": describe([record["evidence"]["score"] for record in semantic]),
    }


def shares(rows: list[Record], field: str, split_of: dict[int, str]) -> Record:
    scored = [row for row in rows if row["scored"]]
    result: Record = {}
    groups = [("all", lambda row: True)]
    groups += [(tier, lambda row, tier=tier: row["tier"] == tier) for tier in EVIDENCE_TIERS]
    for name, keep in groups:
        chosen = [row for row in scored if keep(row)]
        result[name] = {
            "n": len(chosen),
            "passing": sum(1 for row in chosen if row[field]),
            "share": round(sum(1 for row in chosen if row[field]) / len(chosen), 4)
            if chosen
            else None,
            "by_split": {
                split: {
                    "n": sum(1 for row in chosen if split_of[row["hearing_id"]] == split),
                    "passing": sum(
                        1 for row in chosen if split_of[row["hearing_id"]] == split and row[field]
                    ),
                }
                for split in SPLIT_NAMES
            },
        }
    return result


def distributions(rows: list[Record], split_of: dict[int, str]) -> Record:
    scored = [row for row in rows if row["scored"]]
    result: Record = {}
    for tier in EVIDENCE_TIERS:
        tier_rows = [row for row in scored if row["tier"] == tier]
        result[tier] = {
            "all": describe([row["primary_probability"] for row in tier_rows]),
            **{
                split: describe(
                    [
                        row["primary_probability"]
                        for row in tier_rows
                        if split_of[row["hearing_id"]] == split
                    ]
                )
                for split in SPLIT_NAMES
            },
        }
    return result


def agreement(rows: list[Record], field: str) -> Record:
    semantic = [row for row in rows if row["scored"] and row["tier"] in SEMANTIC_TIERS]
    high = [row["tier"] == "semantic_match_high" for row in semantic]
    passing = [bool(row[field]) for row in semantic]
    table = Counter(zip(high, passing, strict=True))
    agree = sum(1 for first, second in zip(high, passing, strict=True) if first == second)
    kappa = (
        cohen_kappa_score(high, passing) if len(set(high)) > 1 or len(set(passing)) > 1 else None
    )
    return {
        "n": len(semantic),
        "table": {
            "high_and_passing": table[(True, True)],
            "high_not_passing": table[(True, False)],
            "weak_and_passing": table[(False, True)],
            "weak_not_passing": table[(False, False)],
        },
        "raw_agreement": round(agree / len(semantic), 4) if semantic else None,
        "cohen_kappa": None if kappa is None else round(float(kappa), 4),
    }


def verifier_comparison(
    v1: dict[str, Record], v2: dict[str, Record], v1_rows: list[Record], v2_rows: list[Record]
) -> Record:
    first = {row["udv_id"]: row for row in v1_rows if row["scored"]}
    second = {row["udv_id"]: row for row in v2_rows if row["scored"]}
    common = sorted(set(first) & set(second))
    changed = [udv for udv in common if not same_evidence(v1[udv], v2[udv])]
    unchanged = [udv for udv in common if same_evidence(v1[udv], v2[udv])]

    def group(ids: list[str]) -> Record:
        if not ids:
            return {"n": 0}
        old = np.array([first[udv]["primary_probability"] for udv in ids])
        new = np.array([second[udv]["primary_probability"] for udv in ids])
        return {
            "n": len(ids),
            "v1_mean": round(float(old.mean()), 4),
            "v2_mean": round(float(new.mean()), 4),
            "v2_higher": int((new > old).sum()),
            "v1_train_threshold_passing": sum(
                1 for udv in ids if first[udv]["supported_at_train_threshold"]
            ),
            "v2_train_threshold_passing": sum(
                1 for udv in ids if second[udv]["supported_at_train_threshold"]
            ),
            "max_abs_gap": round(float(np.abs(new - old).max()), 6),
        }

    by_tier = {
        tier: group([udv for udv in changed if v2[udv]["tier"] == tier]) for tier in EVIDENCE_TIERS
    }
    return {
        "evidence_changed": group(changed),
        "evidence_unchanged": group(unchanged),
        "by_v2_tier": by_tier,
    }


def v2_stratum(records: list[Record], strata: list[Record]) -> str | None:
    for stratum in strata:
        if all(
            record["tier"] in stratum["tiers"]
            and ((record.get("evidence") or {}).get("support_type") or "none")
            in stratum["support_types"]
            for record in records
        ):
            return str(stratum["name"])
    return None


def stratum_population(
    records: list[Record], strata: list[Record], split_of: dict[int, str], splits: list[str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if split_of[record["hearing_id"]] not in splits:
            continue
        name = v2_stratum([record], strata)
        if name is not None:
            counts[name] += 1
    return {str(stratum["name"]): counts[str(stratum["name"])] for stratum in strata}


def annotation_plan(key: Record, v1: dict[str, Record], v2: dict[str, Record]) -> Record:
    items: dict[str, Record] = {}
    for item_id, item in sorted(key["items"].items()):
        new = [v2[udv] for udv in item["udv_ids"]]
        identical = all(same_evidence(v1[udv], v2[udv]) for udv in item["udv_ids"])
        items[item_id] = {
            "question": item["question"],
            "v1_stratum": item["stratum"],
            "udv_ids": item["udv_ids"],
            "v2_tiers": [record["tier"] for record in new],
            "v2_stratum": v2_stratum(new, key["strata"]),
            "evidence_identical": identical,
            "action": "inherit" if identical else "reannotate",
        }
    summary: dict[str, Counter[str]] = {}
    for entry in items.values():
        summary.setdefault(entry["v1_stratum"], Counter())[entry["action"]] += 1
    inherited: Counter[str] = Counter(
        str(entry["v2_stratum"]) for entry in items.values() if entry["action"] == "inherit"
    )
    return {
        "rule": (
            "an item inherits its udv_v1 label for udv_v2 when every UDV of the item has the same "
            "evidence text, start_char, end_char and speaker_turn in both runs (or no evidence in "
            "both); otherwise the passage shown to the annotator is not the udv_v2 passage and the "
            "item needs a new judgment"
        ),
        "sample_name": key["sample_name"],
        "items": len(items),
        "by_v1_stratum": {name: dict(sorted(c.items())) for name, c in sorted(summary.items())},
        "inherited_by_v2_stratum": dict(sorted(inherited.items())),
        "caveat": (
            "the inherited items are the udv_v1 items whose evidence did not change; they are "
            "not a random sample of the udv_v2 strata, so their precision describes that subset "
            "only"
        ),
        "item_plan": items,
    }


def command_analyze(args: argparse.Namespace) -> None:
    paths = {name: Path(value) for name, value in DEFAULT_PATHS.items()}
    split_of = load_split_lookup(paths["splits"])
    v1_records = load_jsonl(paths["v1"])
    v2_records = load_jsonl(paths["v2"])
    v1 = {record["id"]: record for record in v1_records}
    v2 = {record["id"]: record for record in v2_records}
    v1_rows = load_jsonl(paths["v1_verifier"])
    v2_rows = load_jsonl(paths["v2_verifier"])
    v2_rows_by_id = {row["udv_id"]: row for row in v2_rows}
    for row in v2_rows:
        if v2[row["udv_id"]]["tier"] != row["tier"]:
            raise SystemExit(f"{row['udv_id']}: the verifier sidecar tier differs from udv_v2")
    if set(v2_rows_by_id) != set(v2):
        raise SystemExit("the verifier sidecar does not cover every udv_v2 record")
    with open(paths["cosine_calibration"]) as f:
        cosine = json.load(f)
    with open(paths["verifier_calibration"]) as f:
        verifier_cut = json.load(f)
    with open(paths["v2_verifier_report"]) as f:
        verifier_report = json.load(f)
    adopted = cosine["adopted_rule"]
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cuts": {
            "cosine": {
                "rule": adopted,
                "threshold": cosine["rules"][adopted]["threshold"],
                "threshold_rounded": cosine["rules"][adopted]["threshold_rounded"],
                "bootstrap": cosine["rules"][adopted]["bootstrap"]["threshold"],
                "other_rules": {
                    name: {
                        "threshold": cosine["rules"][name]["threshold"],
                        "bootstrap": cosine["rules"][name]["bootstrap"]["threshold"],
                    }
                    for name in ("hard_negative", "masked_hard_negative", "masked_top1_youden")
                },
                "leak_check_passed": cosine["leak_check"]["passed"],
                "hearings_used": cosine["leak_check"]["hearings_used"],
            },
            "verifier": {
                "train_threshold": verifier_cut["train_threshold"],
                "udv_threshold": verifier_cut["udv_threshold"],
                "udv_threshold_rule": verifier_cut["adopted_rule"],
                "udv_threshold_bootstrap": verifier_cut["rules"][verifier_cut["adopted_rule"]][
                    "bootstrap"
                ]["threshold"],
                "hard_negative_threshold": verifier_cut["rules"]["hard_negative"]["threshold"],
                "hard_negative_bootstrap": verifier_cut["rules"]["hard_negative"]["bootstrap"][
                    "threshold"
                ],
                "leak_check_passed": verifier_cut["leak_check"]["passed"],
            },
        },
        "tiers": {
            "udv_v1": tier_counts(v1_records, split_of),
            "udv_v2": tier_counts(v2_records, split_of),
        },
        "diff": record_diff(v1, v2, split_of),
        "quote_evidence": {
            "udv_v1": quote_lengths(v1_records),
            "udv_v2": quote_lengths(v2_records),
        },
        "semantic_evidence": {
            "udv_v1": semantic_lengths(v1_records),
            "udv_v2": semantic_lengths(v2_records),
        },
        "verifier": {
            "distributions_by_tier_and_split": distributions(v2_rows, split_of),
            "passing_train_threshold": shares(v2_rows, "supported_at_train_threshold", split_of),
            "passing_udv_threshold": shares(v2_rows, "supported_at_udv_threshold", split_of),
            "udv_v1_passing_train_threshold": shares(
                v1_rows, "supported_at_train_threshold", split_of
            ),
            "cosine_tier_agreement": {
                "rule": "semantic UDVs only: tier semantic_match_high (window cosine at or above "
                "the cosine cut) against the verifier decision; Cohen kappa of the two booleans",
                "train_threshold": agreement(v2_rows, "supported_at_train_threshold"),
                "udv_threshold": agreement(v2_rows, "supported_at_udv_threshold"),
            },
            "spearman_with_cosine": verifier_report["cosine_relation"]["spearman"],
            "v1_vs_v2": verifier_comparison(v1, v2, v1_rows, v2_rows),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_of_file(path)}
            for name, path in paths.items()
            if path.is_file()
        },
        "code": {"utils/udv_v2_analysis.py": sha256_of_file(Path(__file__))},
    }
    write_json(report, Path(args.output))
    key = load_key(paths["sample_dir"] / ANNOTATION_KEY, "annotation")
    plan = annotation_plan(key, v1, v2)
    v1_population = stratum_population(v1_records, key["strata"], split_of, key["splits_used"])
    if v1_population != key["population"]["sampled_splits"]:
        raise SystemExit("the udv_v1 strata sizes differ from the population frozen in the key")
    plan["v2_population_sampled_splits"] = stratum_population(
        v2_records, key["strata"], split_of, key["splits_used"]
    )
    plan["v2_population_rule"] = (
        "UDVs of udv_v2 in the sampled splits per stratum, with the tier and support type rules of "
        "the key; the same count on udv_v1 reproduces the population frozen in the key"
    )
    plan["inputs"] = {
        "annotation_key": {
            "path": str(paths["sample_dir"] / ANNOTATION_KEY),
            "sha256": sha256_of_file(paths["sample_dir"] / ANNOTATION_KEY),
        },
        "udv_v1": report["inputs"]["v1"],
        "udv_v2": report["inputs"]["v2"],
    }
    write_json(plan, Path(args.plan_output))
    print(json.dumps({"diff": report["diff"]["kinds"], "plan": plan["by_v1_stratum"]}, indent=2))


def tolerated_problems(item_id: str, raw_label: str) -> set[str]:
    return {
        f"{item_id}: julgamento is empty",
        f"{item_id}: existe_trecho_melhor is empty",
        f"{item_id}: julgamento {raw_label!r} is not one of",
    }


def is_tolerated(problem: str, tolerated: dict[str, set[str]]) -> bool:
    item_id = problem.split(":", 1)[0]
    return any(problem.startswith(prefix) for prefix in tolerated.get(item_id, set()))


def partial_judgments(rows: list[Record], key: Record, config: Any) -> Record:
    judged, problems = validate_annotation(rows, key, config)
    raw = {row.get("item_id", "").strip(): row.get("julgamento", "") for row in rows}
    tolerated = {item_id: tolerated_problems(item_id, raw.get(item_id, "")) for item_id in judged}
    fatal = [problem for problem in problems if not is_tolerated(problem, tolerated)]
    if fatal:
        raise SystemExit("\n".join(fatal))
    valid: dict[str, Record] = {}
    invalid: list[Record] = []
    for item_id, entry in sorted(judged.items()):
        if not entry["judgment"]:
            continue
        allowed = config.questions[key["items"][item_id]["question"]].judgments
        if entry["judgment"] in allowed:
            valid[item_id] = entry
        else:
            invalid.append({"item_id": item_id, "julgamento": raw[item_id].strip()})
    return {
        "valid": valid,
        "invalid_labels": invalid,
        "better_passage_missing": sorted(
            item_id for item_id, entry in valid.items() if entry["better_passage"] is None
        ),
    }


def subset_key(key: Record, items: dict[str, Record], population: dict[str, int]) -> Record:
    return {
        **key,
        "items": items,
        "population": {**key["population"], "sampled_splits": population},
    }


def run_results(
    key: Record,
    items: dict[str, Record],
    judged: dict[str, Record],
    population: dict[str, int],
    sample_items: dict[str, int],
    config: Any,
    rc: precision.ReportConfig,
    decision_valid: bool,
) -> Record:
    run_key = subset_key(key, items, population)
    run_judged = {item_id: judged[item_id] for item_id in items}
    units = precision.judged_units(run_key, run_judged)
    results = precision.stratum_results(run_key, units, run_judged, config, rc)
    for name, result in results.items():
        result["judged_items"] = sum(1 for item in items.values() if item["stratum"] == name)
        result["sample_items"] = sample_items.get(name, 0)
        result["judged_of_sample"] = f"{result['judged_items']} of {result['sample_items']}"
    rules = precision.evaluate_criteria(
        key["criteria"]["rules"], results, rc.confidence_level, not decision_valid
    )
    if not decision_valid:
        rules = [{**rule, "status": f"INTERIM_{rule['status']}"} for rule in rules]
    return {"strata": precision.round_floats(results), "criteria": precision.round_floats(rules)}


def inherited_items(key: Record, plan: Record) -> tuple[dict[str, Record], dict[str, int]]:
    items: dict[str, Record] = {}
    for item_id, entry in plan["item_plan"].items():
        if entry["action"] != "inherit" or entry["v2_stratum"] is None:
            continue
        tiers = set(entry["v2_tiers"])
        tier = entry["v2_tiers"][0] if len(tiers) == 1 else key["items"][item_id]["tier"]
        items[item_id] = {**key["items"][item_id], "stratum": entry["v2_stratum"], "tier": tier}
    counts = Counter(str(item["stratum"]) for item in items.values())
    return items, dict(counts)


def command_score(args: argparse.Namespace) -> None:
    config = load_validation_config(args.config)
    rc = precision.load_report_config(config)
    sample_dir = Path(args.sample_dir)
    annotation_path = Path(args.annotation) if args.annotation else sample_dir / ANNOTATION_CSV
    key = load_key(sample_dir / ANNOTATION_KEY, "annotation")
    require_final_test(key, args.final_test)
    precision.check_rules(
        key["criteria"]["rules"],
        {stratum["name"]: stratum["question"] for stratum in key["strata"]},
    )
    with open(args.plan) as f:
        plan = json.load(f)
    if plan["inputs"]["annotation_key"]["sha256"] != sha256_of_file(sample_dir / ANNOTATION_KEY):
        raise SystemExit(f"{args.plan} was built from another annotation key")
    judgments = partial_judgments(read_annotation_csv(annotation_path, config), key, config)
    valid = judgments["valid"]
    if not valid:
        raise SystemExit(
            f"{NO_JUDGMENTS}: {annotation_path} has no valid julgamento in any of its "
            f"{len(key['items'])} rows; fill the sheet and run this command again"
        )
    complete = len(valid) == len(key["items"])
    decision_valid = complete and not key["dry_run"]
    names = [stratum["name"] for stratum in key["strata"]]
    v1_items = {item_id: key["items"][item_id] for item_id in valid}
    v1_sample = dict(Counter(item["stratum"] for item in key["items"].values()))
    v2_all, v2_sample = inherited_items(key, plan)
    v2_items = {item_id: item for item_id, item in v2_all.items() if item_id in valid}
    v2_population = {name: plan["v2_population_sampled_splits"].get(name, 0) for name in names}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    status = "final" if complete else "interim"
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "status_note": (
            "every row of the sheet has a valid julgamento"
            if complete
            else "INTERIM: computed on the judged rows only; the criteria statuses below are not "
            "a decision and change as the sheet is filled"
        ),
        "sample_name": key["sample_name"],
        "splits_used": key["splits_used"],
        "judged": f"{len(valid)} of {len(key['items'])}",
        "judged_items": len(valid),
        "items": len(key["items"]),
        "invalid_labels": judgments["invalid_labels"],
        "invalid_labels_rule": (
            "a filled julgamento outside the labels of its question is not counted and is listed "
            "here; the report is interim while any row is empty or invalid"
        ),
        "better_passage_missing": len(judgments["better_passage_missing"]),
        "annotation_csv": {"path": str(annotation_path), "sha256": sha256_of_file(annotation_path)},
        "annotation_key": {
            "path": str(sample_dir / ANNOTATION_KEY),
            "sha256": sha256_of_file(sample_dir / ANNOTATION_KEY),
        },
        "plan": {"path": str(args.plan), "sha256": sha256_of_file(Path(args.plan))},
        "confidence_level": rc.confidence_level,
        "units": "precision per UDV; the judgment of an item counts for each of its UDVs",
        "label_semantics": precision.LABEL_SEMANTICS,
        "criteria_declaration": key["criteria"]["declaration"],
        "criteria_declared_on": key["criteria"]["declared_on"],
        "udv_v1": run_results(
            key,
            v1_items,
            valid,
            key["population"]["sampled_splits"],
            v1_sample,
            config,
            rc,
            decision_valid,
        ),
        "udv_v2_unchanged": {
            "rule": plan["rule"],
            "caveat": plan["caveat"],
            "items_in_sample": len(v2_all),
            **run_results(
                key, v2_items, valid, v2_population, v2_sample, config, rc, decision_valid
            ),
        },
        "code": {
            "utils/udv_v2_analysis.py": sha256_of_file(Path(__file__)),
            "utils/precision_report.py": sha256_of_file(Path(precision.__file__)),
        },
    }
    output = (
        Path(args.output)
        if args.output
        else Path(f"artifacts/udv/udv_v2_precision_{status}_{stamp}.json")
    )
    if output.exists():
        raise SystemExit(f"{output} exists; precision reports are never overwritten")
    write_json(report, output)
    print_score_summary(report)
    print(f"report -> {output}")


def print_score_summary(report: Record) -> None:
    print(f"{report['sample_name']}: {report['status'].upper()}, judged {report['judged']}")
    for label in report["invalid_labels"]:
        print(f"  not counted, label outside the allowed set: {label['item_id']}")
    for run in ("udv_v1", "udv_v2_unchanged"):
        print(f"  {run}")
        for name, result in report[run]["strata"].items():
            if result["question"] == precision.SUPPORT_QUESTION:
                strict, tolerant = result["strict_precision"], result["tolerant_precision"]
                print(
                    f"    {name:26s} judged {result['judged_of_sample']:9s} "
                    f"strict={strict['estimate']} [{strict['low']}, {strict['high']}] "
                    f"tolerant={tolerant['estimate']} [{tolerant['low']}, {tolerant['high']}]"
                )
            else:
                rate = result["persons"]["false_absence_rate"]
                print(
                    f"    {name:26s} judged {result['judged_of_sample']:9s} "
                    f"false_absence(persons)={rate['estimate']} [{rate['low']}, {rate['high']}]"
                )
        for rule in report[run]["criteria"]:
            print(
                f"    criterion {rule['name']}: {rule['status']} "
                f"(lower {rule['observed']['low']} vs {rule['min_wilson_lower']})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare udv_v2 with udv_v1 and score the validation sample for both runs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="diff, lengths, verifier shares and the plan")
    analyze.add_argument("--output", default="artifacts/udv/udv_v2_analysis.json")
    analyze.add_argument("--plan-output", default="artifacts/udv/udv_v2_annotation_plan.json")
    score = commands.add_parser(
        "score-annotation", help="precision per stratum from the filled annotation.csv"
    )
    score.add_argument("--config", type=Path, default=Path("configs/validation_sample.toml"))
    score.add_argument("--sample-dir", default=DEFAULT_PATHS["sample_dir"])
    score.add_argument("--plan", default="artifacts/udv/udv_v2_annotation_plan.json")
    score.add_argument("--final-test", action="store_true")
    score.add_argument("--annotation", default=None)
    score.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"analyze": command_analyze, "score-annotation": command_score}[args.command](args)


if __name__ == "__main__":
    main()
