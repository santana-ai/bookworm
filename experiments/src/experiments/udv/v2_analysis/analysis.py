"""Subcommand ``analyze``: how udv_v2 differs from udv_v1, and the annotation plan it implies."""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, write_json
from sklearn.metrics import cohen_kappa_score

from experiments.common.provenance import code_section
from experiments.common.reporting import file_record, rounded, utc_timestamp
from experiments.common.splits import SPLIT_NAMES, read_split_lookup
from experiments.common.transcript import (
    SENTENCE_BOUNDARY_PATTERN,
    extract_quotes,
    quote_prefix_pattern,
    quote_prefixes,
)
from experiments.udv.v2_analysis.paths import DEFAULT_PATHS, PACKAGE_DIR
from experiments.udv.v2_analysis.plan import POPULATION_RULE, annotation_plan, stratum_population
from experiments.udv.v2_analysis.relations import IDENTITY_FIELDS, same_evidence
from experiments.validation.generate_sample import ANNOTATION_KEY, load_key

Record = dict[str, Any]

EVIDENCE_TIERS = ("quote_found", "semantic_match_high", "semantic_match_weak")
SEMANTIC_TIERS = ("semantic_match_high", "semantic_match_weak")
HIGH_TIER = "semantic_match_high"
QUOTE_TIER = "quote_found"
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CLOSING_WORDS = 3
MIN_CLOSING_CHARS = 3
CLOSING_PUNCTUATION = ".,;:!?…\"'”’)] "
GAP_DECIMALS = 6
OTHER_COSINE_RULES = ("hard_negative", "masked_hard_negative", "masked_top1_youden")
KIND_RULES = {
    "same_tier_same_evidence": "evidence text and offsets identical, same tier",
    "tier_only": "evidence identical, tier changed",
    "quote_extended": "quote_found in both, the v1 text is contained in the v2 text",
    "quote_changed": "quote_found in both, the v1 text is not contained in the v2 text",
    "window_contains_v1_sentence": "semantic in both, the v2 window holds the v1 sentence",
    "window_elsewhere": "semantic in both, the v2 window does not hold the v1 sentence",
    "other": "any other change",
}
AGREEMENT_RULE = (
    "semantic UDVs only: tier semantic_match_high (window cosine at or above "
    "the cosine cut) against the verifier decision; Cohen kappa of the two booleans"
)


def part_count(text: str) -> int:
    return len([part for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()])


def describe(values: list[float]) -> Record:
    if not values:
        return {"n": 0}
    array = np.array(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": rounded(array.mean()),
        "quantiles": {str(q): rounded(np.quantile(array, q)) for q in QUANTILES},
    }


def chosen_quote(record: Record) -> str | None:
    prefix = record["evidence"]["quote_prefix"]
    for quote in extract_quotes(record["proposition"]):
        if any(candidate == prefix for candidate, _ in quote_prefixes(quote)):
            return quote
    return None


def closing_words_found(quote: str, text: str) -> bool:
    words = quote.split()[-CLOSING_WORDS:]
    closing = " ".join(words).rstrip(CLOSING_PUNCTUATION)
    if len(closing) < MIN_CLOSING_CHARS:
        return False
    return quote_prefix_pattern(closing).search(text) is not None


def quote_lengths(records: list[Record]) -> Record:
    quotes = [record for record in records if record["tier"] == QUOTE_TIER]
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
    if old["tier"] == QUOTE_TIER and new["tier"] == QUOTE_TIER:
        old_text, new_text = old["evidence"]["text"], new["evidence"]["text"]
        return "quote_extended" if old_text in new_text else "quote_changed"
    if old["tier"] in SEMANTIC_TIERS and new["tier"] in SEMANTIC_TIERS:
        if old["evidence"]["text"] in new["evidence"]["text"]:
            return "window_contains_v1_sentence"
        return "window_elsewhere"
    return "other"


def sorted_counts(counters: dict[str, Counter[str]]) -> Record:
    return {name: dict(sorted(counter.items())) for name, counter in sorted(counters.items())}


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
        "kinds_by_v1_tier": sorted_counts(by_tier),
        "kinds_by_split": sorted_counts(by_split),
        "tier_moves": dict(sorted(moves.items())),
        "kind_rules": KIND_RULES,
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


def has_tier(tier: str) -> Callable[[Record], bool]:
    return lambda row: row["tier"] == tier


def split_passing(rows: list[Record], field: str, split_of: dict[int, str]) -> Record:
    return {
        split: {
            "n": sum(1 for row in rows if split_of[row["hearing_id"]] == split),
            "passing": sum(
                1 for row in rows if split_of[row["hearing_id"]] == split and row[field]
            ),
        }
        for split in SPLIT_NAMES
    }


def shares(rows: list[Record], field: str, split_of: dict[int, str]) -> Record:
    """How many scored rows pass the verifier cut in ``field``, overall and per tier and split."""
    scored = [row for row in rows if row["scored"]]
    result: Record = {}
    groups: list[tuple[str, Callable[[Record], bool]]] = [("all", lambda row: True)]
    groups += [(tier, has_tier(tier)) for tier in EVIDENCE_TIERS]
    for name, keep in groups:
        chosen = [row for row in scored if keep(row)]
        passing = sum(1 for row in chosen if row[field])
        result[name] = {
            "n": len(chosen),
            "passing": passing,
            "share": rounded(passing / len(chosen)) if chosen else None,
            "by_split": split_passing(chosen, field, split_of),
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
    high = [row["tier"] == HIGH_TIER for row in semantic]
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
        "raw_agreement": rounded(agree / len(semantic)) if semantic else None,
        "cohen_kappa": None if kappa is None else rounded(kappa),
    }


def probability_change(
    first: dict[str, Record], second: dict[str, Record], ids: list[str]
) -> Record:
    if not ids:
        return {"n": 0}
    old = np.array([first[udv]["primary_probability"] for udv in ids])
    new = np.array([second[udv]["primary_probability"] for udv in ids])
    return {
        "n": len(ids),
        "v1_mean": rounded(old.mean()),
        "v2_mean": rounded(new.mean()),
        "v2_higher": int((new > old).sum()),
        "v1_train_threshold_passing": sum(
            1 for udv in ids if first[udv]["supported_at_train_threshold"]
        ),
        "v2_train_threshold_passing": sum(
            1 for udv in ids if second[udv]["supported_at_train_threshold"]
        ),
        "max_abs_gap": rounded(np.abs(new - old).max(), GAP_DECIMALS),
    }


def verifier_comparison(
    v1: dict[str, Record], v2: dict[str, Record], v1_rows: list[Record], v2_rows: list[Record]
) -> Record:
    first = {row["udv_id"]: row for row in v1_rows if row["scored"]}
    second = {row["udv_id"]: row for row in v2_rows if row["scored"]}
    common = sorted(set(first) & set(second))
    changed = [udv for udv in common if not same_evidence(v1[udv], v2[udv])]
    unchanged = [udv for udv in common if same_evidence(v1[udv], v2[udv])]
    by_tier = {
        tier: probability_change(first, second, [udv for udv in changed if v2[udv]["tier"] == tier])
        for tier in EVIDENCE_TIERS
    }
    return {
        "evidence_changed": probability_change(first, second, changed),
        "evidence_unchanged": probability_change(first, second, unchanged),
        "by_v2_tier": by_tier,
    }


def load_json(path: Path) -> Record:
    with open(path) as f:
        payload: Record = json.load(f)
    return payload


def cosine_cut(cosine: Record) -> Record:
    adopted = cosine["adopted_rule"]
    return {
        "rule": adopted,
        "threshold": cosine["rules"][adopted]["threshold"],
        "threshold_rounded": cosine["rules"][adopted]["threshold_rounded"],
        "bootstrap": cosine["rules"][adopted]["bootstrap"]["threshold"],
        "other_rules": {
            name: {
                "threshold": cosine["rules"][name]["threshold"],
                "bootstrap": cosine["rules"][name]["bootstrap"]["threshold"],
            }
            for name in OTHER_COSINE_RULES
        },
        "leak_check_passed": cosine["leak_check"]["passed"],
        "hearings_used": cosine["leak_check"]["hearings_used"],
    }


def verifier_cut(cut: Record) -> Record:
    rules = cut["rules"]
    return {
        "train_threshold": cut["train_threshold"],
        "udv_threshold": cut["udv_threshold"],
        "udv_threshold_rule": cut["adopted_rule"],
        "udv_threshold_bootstrap": rules[cut["adopted_rule"]]["bootstrap"]["threshold"],
        "hard_negative_threshold": rules["hard_negative"]["threshold"],
        "hard_negative_bootstrap": rules["hard_negative"]["bootstrap"]["threshold"],
        "leak_check_passed": cut["leak_check"]["passed"],
    }


def verifier_section(
    v1: dict[str, Record],
    v2: dict[str, Record],
    v1_rows: list[Record],
    v2_rows: list[Record],
    split_of: dict[int, str],
    verifier_report: Record,
) -> Record:
    return {
        "distributions_by_tier_and_split": distributions(v2_rows, split_of),
        "passing_train_threshold": shares(v2_rows, "supported_at_train_threshold", split_of),
        "passing_udv_threshold": shares(v2_rows, "supported_at_udv_threshold", split_of),
        "udv_v1_passing_train_threshold": shares(v1_rows, "supported_at_train_threshold", split_of),
        "cosine_tier_agreement": {
            "rule": AGREEMENT_RULE,
            "train_threshold": agreement(v2_rows, "supported_at_train_threshold"),
            "udv_threshold": agreement(v2_rows, "supported_at_udv_threshold"),
        },
        "spearman_with_cosine": verifier_report["cosine_relation"]["spearman"],
        "v1_vs_v2": verifier_comparison(v1, v2, v1_rows, v2_rows),
    }


def check_verifier_rows(v2: dict[str, Record], v2_rows: list[Record]) -> None:
    for row in v2_rows:
        if v2[row["udv_id"]]["tier"] != row["tier"]:
            raise SystemExit(f"{row['udv_id']}: the verifier sidecar tier differs from udv_v2")
    if {row["udv_id"] for row in v2_rows} != set(v2):
        raise SystemExit("the verifier sidecar does not cover every udv_v2 record")


def plan_with_population(
    paths: dict[str, Path],
    report: Record,
    v1_records: list[Record],
    v2_records: list[Record],
    split_of: dict[int, str],
) -> Record:
    key_path = paths["sample_dir"] / ANNOTATION_KEY
    key = load_key(key_path, "annotation")
    v1 = {record["id"]: record for record in v1_records}
    v2 = {record["id"]: record for record in v2_records}
    plan = annotation_plan(key, v1, v2)
    v1_population = stratum_population(v1_records, key["strata"], split_of, key["splits_used"])
    if v1_population != key["population"]["sampled_splits"]:
        raise SystemExit("the udv_v1 strata sizes differ from the population frozen in the key")
    plan["v2_population_sampled_splits"] = stratum_population(
        v2_records, key["strata"], split_of, key["splits_used"]
    )
    plan["v2_population_rule"] = POPULATION_RULE
    plan["inputs"] = {
        "annotation_key": file_record(key_path),
        "udv_v1": report["inputs"]["v1"],
        "udv_v2": report["inputs"]["v2"],
    }
    return plan


def command_analyze(args: argparse.Namespace) -> None:
    paths = {name: Path(value) for name, value in DEFAULT_PATHS.items()}
    split_of = read_split_lookup(paths["splits"])
    v1_records = load_jsonl(paths["v1"])
    v2_records = load_jsonl(paths["v2"])
    v1 = {record["id"]: record for record in v1_records}
    v2 = {record["id"]: record for record in v2_records}
    v1_rows = load_jsonl(paths["v1_verifier"])
    v2_rows = load_jsonl(paths["v2_verifier"])
    check_verifier_rows(v2, v2_rows)
    report: Record = {
        "created_at": utc_timestamp(),
        "cuts": {
            "cosine": cosine_cut(load_json(paths["cosine_calibration"])),
            "verifier": verifier_cut(load_json(paths["verifier_calibration"])),
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
        "verifier": verifier_section(
            v1, v2, v1_rows, v2_rows, split_of, load_json(paths["v2_verifier_report"])
        ),
        "inputs": {name: file_record(path) for name, path in paths.items() if path.is_file()},
        "code": code_section(PACKAGE_DIR),
    }
    write_json(report, Path(args.output))
    plan = plan_with_population(paths, report, v1_records, v2_records, split_of)
    write_json(plan, Path(args.plan_output))
    print(json.dumps({"diff": report["diff"]["kinds"], "plan": plan["by_v1_stratum"]}, indent=2))
