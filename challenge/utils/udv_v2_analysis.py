import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from utils import precision_report as precision
from utils.dataset_io import load_gated_jsonl, load_jsonl, sha256_of_file, write_json
from utils.generate_validation_sample import (
    ANNOTATION_CSV,
    ANNOTATION_KEY,
    CSV_COLUMNS,
    Stratum,
    ValidationConfig,
    assign_item_ids,
    blinding_section,
    canonical_sha256,
    ensure_new_dir,
    evidence_item,
    hearing_view,
    load_key,
    load_validation_config,
    read_annotation_csv,
    require_final_test,
    sheet_rows,
    validate_annotation,
    wilson_interval,
    write_annotation_csv,
    write_transcripts,
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
SUPPLEMENT_NAME = "human_validation_v1_udv_v2_supplement"
SUPPLEMENT_KIND = "udv_v2_supplement"
SUPPLEMENT_ITEM_PREFIX = "C"
SUPPLEMENT_ORDER_STREAM = 4
SUPPLEMENT_RULE = (
    "one row per item of the udv_v1 sample whose udv_v2 evidence is moved under the relation "
    "rules of the inheritance section: the same sampled opinion, shown with its udv_v2 evidence, "
    "context and link; the rows are put in a new random order and get new item ids, and the "
    "udv_v1 item id and relation are kept in the key only"
)
COMBINED_RULE = (
    "udv_v2 precision per stratum from every item of the udv_v1 sample: an item whose udv_v2 "
    "evidence is same, superset or no_evidence (inheritance section of the supplementary key) "
    "keeps its judgment from the udv_v1 sheet, and a moved item counts only with its judgment "
    "from the supplementary sheet; the stratum of an item is the stratum of its udv_v2 records, "
    "and the population is the udv_v2 count of the sampled splits"
)
COMBINED_CAVEATS = [
    "the items were drawn from the udv_v1 strata; an item that changed stratum in udv_v2 keeps "
    "the inclusion probability of its udv_v1 stratum, so the udv_v2 strata are not simple random "
    "samples of the udv_v2 populations",
    "udv_v2 UDVs of a stratum whose udv_v1 stratum was sampled at another rate, or not drawn, are "
    "represented only through the items that were drawn",
    "the supplementary rows show opinions the annotator already judged with their udv_v1 passage, "
    "so their judgments are not independent of the first sheet",
    "a superset item keeps a udv_v1 label given to a shorter passage; the inheritance section "
    "states the assumption this rests on",
]
RELATIONS = ("same", "superset", "moved", "no_evidence")
INHERITANCE_FIELDS = (
    "inherit_relations",
    "declared_on",
    "declaration",
    "relation_rules",
    "multi_udv_rule",
    "inheritance_rule",
    "label_independence",
    "justification",
    "consequence",
    "assumption",
    "sensitivity_rule",
)
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


def evidence_relation(old: Record, new: Record) -> str:
    first, second = old.get("evidence"), new.get("evidence")
    if first is None and second is None:
        return "no_evidence"
    if first is None or second is None or first["speaker_turn"] != second["speaker_turn"]:
        return "moved"
    if (first["start_char"], first["end_char"]) == (second["start_char"], second["end_char"]):
        return "same"
    if second["start_char"] <= first["start_char"] and second["end_char"] >= first["end_char"]:
        return "superset"
    return "moved"


def item_relation(relations: list[str]) -> str:
    found = set(relations)
    if "moved" in found or ("no_evidence" in found and len(found) > 1):
        return "moved"
    if found == {"no_evidence"}:
        return "no_evidence"
    if found == {"same"}:
        return "same"
    return "superset"


def load_inheritance(config: ValidationConfig) -> Record:
    section: Record = dict(config.source.get("udv_v2_supplement", {}))
    missing = [name for name in INHERITANCE_FIELDS if name not in section]
    if missing:
        raise SystemExit(f"udv_v2_supplement must define {missing}")
    inherit = section["inherit_relations"]
    if "same" not in inherit or not set(inherit) <= {"same", "superset"}:
        raise SystemExit("udv_v2_supplement.inherit_relations must hold same and at most superset")
    if set(section["relation_rules"]) != set(RELATIONS):
        raise SystemExit(f"udv_v2_supplement.relation_rules must define exactly {RELATIONS}")
    return section


def is_inherited(relation: str, inheritance: Record) -> bool:
    return relation == "no_evidence" or relation in inheritance["inherit_relations"]


def item_relations(
    key: Record, plan: Record, v1: dict[str, Record], v2: dict[str, Record], inheritance: Record
) -> dict[str, Record]:
    relations: dict[str, Record] = {}
    for item_id, item in sorted(key["items"].items()):
        by_udv = {udv: evidence_relation(v1[udv], v2[udv]) for udv in item["udv_ids"]}
        relation = item_relation(list(by_udv.values()))
        identical = plan["item_plan"][item_id]["action"] == "inherit"
        if identical != (relation in ("same", "no_evidence")):
            raise SystemExit(f"{item_id}: relation {relation} disagrees with the annotation plan")
        relations[item_id] = {
            "question": item["question"],
            "relation": relation,
            "udv_relations": by_udv,
            "inherited": is_inherited(relation, inheritance),
        }
    return relations


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


def has_tier(tier: str) -> Callable[[Record], bool]:
    return lambda row: row["tier"] == tier


def shares(rows: list[Record], field: str, split_of: dict[int, str]) -> Record:
    scored = [row for row in rows if row["scored"]]
    result: Record = {}
    groups: list[tuple[str, Callable[[Record], bool]]] = [("all", lambda row: True)]
    groups += [(tier, has_tier(tier)) for tier in EVIDENCE_TIERS]
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
    report: Record = {
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


def plan_items(key: Record, plan: Record, item_ids: set[str]) -> dict[str, Record]:
    items: dict[str, Record] = {}
    for item_id, entry in plan["item_plan"].items():
        if item_id not in item_ids or entry["v2_stratum"] is None:
            continue
        tiers = set(entry["v2_tiers"])
        tier = entry["v2_tiers"][0] if len(tiers) == 1 else key["items"][item_id]["tier"]
        items[item_id] = {**key["items"][item_id], "stratum": entry["v2_stratum"], "tier": tier}
    return items


def inherited_items(key: Record, plan: Record) -> tuple[dict[str, Record], dict[str, int]]:
    identical = {
        item_id for item_id, entry in plan["item_plan"].items() if entry["action"] == "inherit"
    }
    items = plan_items(key, plan, identical)
    counts = Counter(str(item["stratum"]) for item in items.values())
    return items, dict(counts)


def key_strata(key: Record) -> dict[str, Stratum]:
    return {
        stratum["name"]: Stratum(
            name=stratum["name"],
            question=stratum["question"],
            tiers=tuple(stratum["tiers"]),
            support_types=tuple(stratum["support_types"]),
            target=stratum.get("target", 0),
        )
        for stratum in key["strata"]
    }


def reannotated_records(
    plan: Record, relations: dict[str, Record], v2: dict[str, Record]
) -> list[tuple[str, Record, str, str]]:
    entries: list[tuple[str, Record, str, str]] = []
    for item_id, entry in sorted(plan["item_plan"].items()):
        if relations[item_id]["inherited"]:
            continue
        if entry["question"] != precision.SUPPORT_QUESTION or len(entry["udv_ids"]) != 1:
            raise SystemExit(f"{item_id}: only single-UDV evidence items can be reannotated")
        record = v2[entry["udv_ids"][0]]
        if record.get("evidence") is None or entry["v2_stratum"] is None:
            raise SystemExit(f"{item_id}: the udv_v2 record has no evidence stratum")
        entries.append((item_id, record, str(entry["v2_stratum"]), relations[item_id]["relation"]))
    return entries


def supplement_items(
    entries: list[tuple[str, Record, str, str]],
    views: dict[int, Record],
    strata: dict[str, Stratum],
    chars: int,
    rng: np.random.Generator,
) -> dict[str, Record]:
    items = [
        {
            **evidence_item(record, views[record["hearing_id"]], strata[name], chars),
            "v1_item_id": item_id,
            "v1_relation": relation,
        }
        for item_id, record, name, relation in entries
    ]
    return assign_item_ids(items, rng, SUPPLEMENT_ITEM_PREFIX)


def inherited_section(plan: Record, relations: dict[str, Record]) -> dict[str, Record]:
    return {
        item_id: {
            "relation": relation["relation"],
            "question": relation["question"],
            "v2_stratum": plan["item_plan"][item_id]["v2_stratum"],
            "v2_tiers": plan["item_plan"][item_id]["v2_tiers"],
        }
        for item_id, relation in sorted(relations.items())
        if relation["inherited"]
    }


def relation_counts(relations: dict[str, Record]) -> Record:
    by_question: dict[str, Counter[str]] = {}
    for relation in relations.values():
        by_question.setdefault(relation["question"], Counter())[relation["relation"]] += 1
    return {
        question: {name: counts[name] for name in RELATIONS if counts[name]}
        for question, counts in sorted(by_question.items())
    }


def supplement_rng(config: ValidationConfig) -> np.random.Generator:
    if SUPPLEMENT_ORDER_STREAM in config.seed_streams.values():
        raise SystemExit(f"seed stream {SUPPLEMENT_ORDER_STREAM} is already used by the sample")
    return np.random.default_rng([config.seed, SUPPLEMENT_ORDER_STREAM])


def load_plan(path: Path, key_path: Path) -> Record:
    with open(path) as f:
        plan: Record = json.load(f)
    if plan["inputs"]["annotation_key"]["sha256"] != sha256_of_file(key_path):
        raise SystemExit(f"{path} was built from another annotation key")
    return plan


def supplement_key(
    key: Record,
    key_path: Path,
    plan: Record,
    plan_path: Path,
    runs: dict[str, Path],
    inheritance: Record,
    final_test: bool,
) -> Record:
    return {
        "role": "annotation",
        "kind": SUPPLEMENT_KIND,
        "sample_name": SUPPLEMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": False,
        "final_test": final_test,
        "splits_used": key["splits_used"],
        "splits_declared": key["splits_declared"],
        "rule": SUPPLEMENT_RULE,
        "inheritance": inheritance,
        "inheritance_sha256": canonical_sha256(inheritance),
        "base_sample": {
            "sample_name": key["sample_name"],
            "annotation_key": {"path": str(key_path), "sha256": sha256_of_file(key_path)},
        },
        "plan": {"path": str(plan_path), "sha256": sha256_of_file(plan_path)},
        "runs": {
            name: {"path": str(path), "sha256": sha256_of_file(path)} for name, path in runs.items()
        },
        "population": {"unit": "UDV", "sampled_splits": plan["v2_population_sampled_splits"]},
        "strata": key["strata"],
        "criteria": key["criteria"],
        "criteria_sha256": key["criteria_sha256"],
        "code": {"utils/udv_v2_analysis.py": sha256_of_file(Path(__file__))},
    }


def command_supplement(args: argparse.Namespace) -> None:
    config = load_validation_config(args.config)
    inheritance = load_inheritance(config)
    sample_dir = Path(args.sample_dir)
    key_path = sample_dir / ANNOTATION_KEY
    key = load_key(key_path, "annotation")
    require_final_test(key, args.final_test)
    plan_path = Path(args.plan)
    plan = load_plan(plan_path, key_path)
    runs = {"udv_v1": Path(args.udv_v1), "udv_v2": Path(args.udv_v2)}
    for name, path in runs.items():
        if plan["inputs"][name]["sha256"] != sha256_of_file(path):
            raise SystemExit(f"{plan_path} was built from another {path}")
    output_dir = Path(args.output_dir)
    transcripts_dir = Path(args.transcripts_dir)
    ensure_new_dir(output_dir)
    ensure_new_dir(transcripts_dir)
    v1 = {record["id"]: record for record in load_jsonl(runs["udv_v1"])}
    v2 = {record["id"]: record for record in load_jsonl(runs["udv_v2"])}
    relations = item_relations(key, plan, v1, v2, inheritance)
    entries = reannotated_records(plan, relations, v2)
    hearing_ids = {record["hearing_id"] for _, record, _, _ in entries}
    views = {
        hearing["id"]: hearing_view(hearing)
        for hearing in load_gated_jsonl(config.lds_path, config.lds_sha256)
        if hearing["id"] in hearing_ids
    }
    items = supplement_items(
        entries, views, key_strata(key), config.context_chars, supplement_rng(config)
    )
    rows = sheet_rows(items)
    blinding = blinding_section(rows, items, config)
    csv_path = output_dir / ANNOTATION_CSV
    write_annotation_csv(rows, csv_path, config)
    rows_by_stratum = dict(sorted(Counter(item["stratum"] for item in items.values()).items()))
    supplement = {
        **supplement_key(key, key_path, plan, plan_path, runs, inheritance, args.final_test),
        "relation_counts": relation_counts(relations),
        "sample": {name: {"rows": count} for name, count in rows_by_stratum.items()},
        "seeds": {
            "seed": config.seed,
            "row_order_stream": SUPPLEMENT_ORDER_STREAM,
            "generator": "numpy PCG64",
        },
        "blinding": blinding,
        "csv": {
            "file": ANNOTATION_CSV,
            "delimiter": config.csv_delimiter,
            "encoding": config.csv_encoding,
            "columns": list(CSV_COLUMNS),
            "sha256_at_creation": sha256_of_file(csv_path),
        },
        "transcripts": write_transcripts(views, transcripts_dir),
        "transcripts_dir": str(transcripts_dir),
        "inherited_items": inherited_section(plan, relations),
        "items": items,
    }
    write_json(supplement, output_dir / ANNOTATION_KEY)
    print(f"{SUPPLEMENT_NAME}: {len(rows)} rows to judge -> {csv_path}")
    for question, counts in supplement["relation_counts"].items():
        print(f"  {question:26s} {counts}")
    for name, count in rows_by_stratum.items():
        print(f"  {name:26s} rows={count}")
    print(f"  transcripts in {transcripts_dir}")


def load_supplement(
    path: Path, key_path: Path, plan: Record, plan_path: Path, inheritance: Record
) -> Record:
    supplement = load_key(path / ANNOTATION_KEY, "annotation")
    if supplement.get("kind") != SUPPLEMENT_KIND:
        raise SystemExit(f"{path / ANNOTATION_KEY} is not a {SUPPLEMENT_KIND} key")
    if supplement["base_sample"]["annotation_key"]["sha256"] != sha256_of_file(key_path):
        raise SystemExit(f"{path} was drawn from another annotation key")
    if supplement["plan"]["sha256"] != sha256_of_file(plan_path):
        raise SystemExit(f"{path} was drawn from another annotation plan")
    if supplement["criteria_sha256"] != canonical_sha256(supplement["criteria"]):
        raise SystemExit(f"{path}: the criteria differ from the ones declared with the sample")
    if supplement["inheritance_sha256"] != canonical_sha256(supplement["inheritance"]):
        raise SystemExit(f"{path}: the inheritance section was edited after the sheet was drawn")
    if supplement["inheritance_sha256"] != canonical_sha256(inheritance):
        raise SystemExit(f"{path}: the inheritance section differs from the config")
    inherited = supplement["inherited_items"]
    covered = [item["v1_item_id"] for item in supplement["items"].values()]
    if len(set(covered)) != len(covered) or set(covered) & set(inherited):
        raise SystemExit(f"{path}: an item of the udv_v1 sample is covered more than once")
    if set(covered) | set(inherited) != set(plan["item_plan"]):
        raise SystemExit(f"{path}: the rows and the inherited items do not cover the plan")
    identical = {
        item_id for item_id, entry in plan["item_plan"].items() if entry["action"] == "inherit"
    }
    for item_id, entry in inherited.items():
        if not is_inherited(entry["relation"], inheritance):
            raise SystemExit(f"{path}: {item_id} is inherited with relation {entry['relation']}")
        if (item_id in identical) != (entry["relation"] in ("same", "no_evidence")):
            raise SystemExit(f"{path}: {item_id} relation disagrees with the annotation plan")
    for item in supplement["items"].values():
        if is_inherited(item["v1_relation"], inheritance) or item["v1_item_id"] in identical:
            raise SystemExit(f"{path}: {item['v1_item_id']} has a row but would be inherited")
    return supplement


def support_precision(
    items: dict[str, Record], judged: dict[str, Record], field: str, rc: precision.ReportConfig
) -> Record:
    groups: dict[str, list[str]] = {}
    for item_id, item in items.items():
        if item["question"] != precision.SUPPORT_QUESTION:
            continue
        for _ in item["udv_ids"]:
            groups.setdefault(str(item[field]), []).append(judged[item_id]["judgment"])
    result: Record = {}
    for name, labels in sorted(groups.items()):
        strict = sum(1 for label in labels if label in rc.strict_success)
        tolerant = sum(1 for label in labels if label in rc.tolerant_success)
        result[name] = {
            "judged_udvs": len(labels),
            "strict_precision": wilson_interval(strict, len(labels), rc.confidence_level),
            "tolerant_precision": wilson_interval(tolerant, len(labels), rc.confidence_level),
        }
    return precision.round_floats(result)


def item_sources(inherited: dict[str, Record], supplement: Record) -> dict[str, tuple[str, str]]:
    sources = {
        item_id: (supplement["inherited_items"][item_id]["relation"], "udv_v1 sheet")
        for item_id in inherited
    }
    for item_id, item in supplement["items"].items():
        sources[item_id] = (item["v1_relation"], "supplementary sheet")
    return sources


def source_counts(
    items: dict[str, Record],
    judged_items: dict[str, Record],
    sources: dict[str, tuple[str, str]],
    names: list[str],
) -> tuple[Record, Record]:
    totals: Record = {}
    for relation, sheet in sorted(set(sources.values()), key=lambda pair: RELATIONS.index(pair[0])):
        members = [item_id for item_id in items if sources[item_id] == (relation, sheet)]
        totals[relation] = {
            "label_from": sheet,
            "items": len(members),
            "judged": sum(1 for item_id in members if item_id in judged_items),
        }
    by_stratum = {
        name: {
            relation: sum(
                1
                for item_id, item in items.items()
                if item["stratum"] == name and sources[item_id][0] == relation
            )
            for relation in totals
        }
        for name in names
    }
    return totals, by_stratum


def sensitivity_section(
    judged_items: dict[str, Record],
    judged: dict[str, Record],
    sources: dict[str, tuple[str, str]],
    inheritance: Record,
    rc: precision.ReportConfig,
) -> Record:
    left_out = sorted(
        item_id
        for item_id in judged_items
        if sources[item_id] == ("superset", "udv_v1 sheet")
        and judged[item_id]["judgment"] not in rc.strict_success
    )
    kept = {item_id: item for item_id, item in judged_items.items() if item_id not in left_out}
    return {
        "rule": inheritance["sensitivity_rule"],
        "superset_judged": sum(
            1 for item_id in judged_items if sources[item_id] == ("superset", "udv_v1 sheet")
        ),
        "superset_left_out": len(left_out),
        "superset_left_out_items": left_out,
        "superset_inherited": {
            "by_stratum": support_precision(judged_items, judged, "stratum", rc),
            "by_tier": support_precision(judged_items, judged, "tier", rc),
        },
        "superset_only_if_v1_correta": {
            "by_stratum": support_precision(kept, judged, "stratum", rc),
            "by_tier": support_precision(kept, judged, "tier", rc),
        },
    }


def combined_results(
    key: Record,
    plan: Record,
    v1_valid: dict[str, Record],
    supplement: Record,
    supplement_valid: dict[str, Record],
    config: Any,
    rc: precision.ReportConfig,
) -> Record:
    inherited = plan_items(key, plan, set(supplement["inherited_items"]))
    items = {**inherited, **supplement["items"]}
    judged = {
        **{item_id: v1_valid[item_id] for item_id in inherited if item_id in v1_valid},
        **supplement_valid,
    }
    judged_items = {item_id: item for item_id, item in items.items() if item_id in judged}
    names = [stratum["name"] for stratum in key["strata"]]
    population = {name: plan["v2_population_sampled_splits"].get(name, 0) for name in names}
    sample_items = dict(Counter(str(item["stratum"]) for item in items.values()))
    sources = item_sources(inherited, supplement)
    totals, by_stratum = source_counts(items, judged_items, sources, names)
    complete = len(judged_items) == len(items)
    decision_valid = complete and not key["dry_run"] and not supplement["dry_run"]
    inheritance = supplement["inheritance"]
    return {
        "status": "final" if complete else "interim",
        "rule": COMBINED_RULE,
        "inheritance": {
            name: inheritance[name]
            for name in (
                "inherit_relations",
                "relation_rules",
                "inheritance_rule",
                "justification",
                "consequence",
                "assumption",
            )
        },
        "caveats": COMBINED_CAVEATS,
        "items_in_sample": len(items),
        "judged": f"{len(judged_items)} of {len(items)}",
        "judged_inherited": sum(1 for item_id in judged_items if item_id in inherited),
        "judged_reannotated": sum(1 for item_id in judged_items if item_id in supplement["items"]),
        "sources": totals,
        "sources_by_stratum": by_stratum,
        "by_tier": support_precision(judged_items, judged, "tier", rc),
        "sensitivity": sensitivity_section(judged_items, judged, sources, inheritance, rc),
        **run_results(
            {**key, "sample": by_stratum},
            judged_items,
            judged,
            population,
            sample_items,
            config,
            rc,
            decision_valid,
        ),
    }


def supplement_results(
    args: argparse.Namespace,
    key: Record,
    plan: Record,
    valid: dict[str, Record],
    config: Any,
    rc: precision.ReportConfig,
) -> tuple[Record, Record]:
    sample_dir = Path(args.sample_dir)
    supplement_dir = Path(args.supplement_dir)
    supplement = load_supplement(
        supplement_dir,
        sample_dir / ANNOTATION_KEY,
        plan,
        Path(args.plan),
        load_inheritance(config),
    )
    supplement_path = (
        Path(args.supplement_annotation)
        if args.supplement_annotation
        else supplement_dir / ANNOTATION_CSV
    )
    judgments = partial_judgments(read_annotation_csv(supplement_path, config), supplement, config)
    combined = combined_results(key, plan, valid, supplement, judgments["valid"], config, rc)
    combined["invalid_labels"] = judgments["invalid_labels"]
    source = {
        "annotation_csv": {"path": str(supplement_path), "sha256": sha256_of_file(supplement_path)},
        "annotation_key": {
            "path": str(supplement_dir / ANNOTATION_KEY),
            "sha256": sha256_of_file(supplement_dir / ANNOTATION_KEY),
        },
    }
    return combined, source


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
    plan = load_plan(Path(args.plan), sample_dir / ANNOTATION_KEY)
    judgments = partial_judgments(read_annotation_csv(annotation_path, config), key, config)
    valid = judgments["valid"]
    if not valid:
        raise SystemExit(
            f"{NO_JUDGMENTS}: {annotation_path} has no valid julgamento in any of its "
            f"{len(key['items'])} rows; fill the sheet and run this command again"
        )
    v1_complete = len(valid) == len(key["items"])
    decision_valid = v1_complete and not key["dry_run"]
    combined: tuple[Record, Record] | None = None
    if args.supplement_dir:
        combined = supplement_results(args, key, plan, valid, config, rc)
    complete = v1_complete and (combined is None or combined[0]["status"] == "final")
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
    if combined is not None:
        report["udv_v2"], report["supplement"] = combined
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
    for run in [name for name in ("udv_v1", "udv_v2_unchanged", "udv_v2") if name in report]:
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
    if "udv_v2" in report:
        print_combined_extras(report["udv_v2"])


def interval_text(interval: Record) -> str:
    return f"{interval['estimate']} [{interval['low']}, {interval['high']}]"


def print_combined_extras(combined: Record) -> None:
    for relation, counts in combined["sources"].items():
        print(
            f"    source {relation:12s} items={counts['items']} judged={counts['judged']} "
            f"label from the {counts['label_from']}"
        )
    for tier, result in combined["by_tier"].items():
        print(
            f"    tier {tier:21s} n={result['judged_udvs']} "
            f"strict={interval_text(result['strict_precision'])} "
            f"tolerant={interval_text(result['tolerant_precision'])}"
        )
    sensitivity = combined["sensitivity"]
    print(
        f"    sensitivity: {sensitivity['superset_left_out']} of "
        f"{sensitivity['superset_judged']} judged superset items left out "
        "(udv_v1 label not correta)"
    )
    inherited = sensitivity["superset_inherited"]["by_stratum"]
    for name, kept in sensitivity["superset_only_if_v1_correta"]["by_stratum"].items():
        print(
            f"    {name:26s} strict {interval_text(inherited[name]['strict_precision'])} "
            f"inherited vs {interval_text(kept['strict_precision'])} correta only; "
            f"tolerant {interval_text(inherited[name]['tolerant_precision'])} vs "
            f"{interval_text(kept['tolerant_precision'])}"
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
    score.add_argument(
        "--supplement-dir",
        default=None,
        help="folder of the udv_v2 supplementary sheet; adds the combined udv_v2 precision",
    )
    score.add_argument(
        "--supplement-annotation",
        default=None,
        help="filled supplementary sheet (default: annotation.csv of --supplement-dir)",
    )
    supplement = commands.add_parser(
        "supplement-sheet",
        help="write the udv_v2 sheet of the sampled items whose evidence changed, judgments empty",
    )
    supplement.add_argument("--config", type=Path, default=Path("configs/validation_sample.toml"))
    supplement.add_argument("--sample-dir", default=DEFAULT_PATHS["sample_dir"])
    supplement.add_argument("--plan", default="artifacts/udv/udv_v2_annotation_plan.json")
    supplement.add_argument("--udv-v1", default=DEFAULT_PATHS["v1"])
    supplement.add_argument("--udv-v2", default=DEFAULT_PATHS["v2"])
    supplement.add_argument("--final-test", action="store_true")
    supplement.add_argument(
        "--output-dir", default=f"artifacts/validation/{SUPPLEMENT_NAME}", help="sheet and key"
    )
    supplement.add_argument(
        "--transcripts-dir",
        default=f"artifacts/cache/validation/{SUPPLEMENT_NAME}/transcripts",
        help="full transcripts of the hearings in the sheet (raw dataset text, not versioned)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {
        "analyze": command_analyze,
        "score-annotation": command_score,
        "supplement-sheet": command_supplement,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
