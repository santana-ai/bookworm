"""The annotation plan: which sampled items keep their udv_v1 label for udv_v2."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from bookworm import sha256_of_file

from experiments.udv.v2_analysis.relations import same_evidence
from experiments.validation.generate_sample import Stratum

Record = dict[str, Any]

NO_EVIDENCE_SUPPORT = "none"
PLAN_RULE = (
    "an item inherits its udv_v1 label for udv_v2 when every UDV of the item has the same "
    "evidence text, start_char, end_char and speaker_turn in both runs (or no evidence in "
    "both); otherwise the passage shown to the annotator is not the udv_v2 passage and the "
    "item needs a new judgment"
)
PLAN_CAVEAT = (
    "the inherited items are the udv_v1 items whose evidence did not change; they are "
    "not a random sample of the udv_v2 strata, so their precision describes that subset "
    "only"
)
POPULATION_RULE = (
    "UDVs of udv_v2 in the sampled splits per stratum, with the tier and support type rules of "
    "the key; the same count on udv_v1 reproduces the population frozen in the key"
)


def support_type_of(record: Record) -> str:
    return (record.get("evidence") or {}).get("support_type") or NO_EVIDENCE_SUPPORT


def v2_stratum(records: list[Record], strata: list[Record]) -> str | None:
    for stratum in strata:
        if all(
            record["tier"] in stratum["tiers"]
            and support_type_of(record) in stratum["support_types"]
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


def item_plan_entry(
    item: Record, v1: dict[str, Record], v2: dict[str, Record], key: Record
) -> Record:
    new = [v2[udv] for udv in item["udv_ids"]]
    identical = all(same_evidence(v1[udv], v2[udv]) for udv in item["udv_ids"])
    return {
        "question": item["question"],
        "v1_stratum": item["stratum"],
        "udv_ids": item["udv_ids"],
        "v2_tiers": [record["tier"] for record in new],
        "v2_stratum": v2_stratum(new, key["strata"]),
        "evidence_identical": identical,
        "action": "inherit" if identical else "reannotate",
    }


def annotation_plan(key: Record, v1: dict[str, Record], v2: dict[str, Record]) -> Record:
    items = {
        item_id: item_plan_entry(item, v1, v2, key)
        for item_id, item in sorted(key["items"].items())
    }
    summary: dict[str, Counter[str]] = {}
    for entry in items.values():
        summary.setdefault(entry["v1_stratum"], Counter())[entry["action"]] += 1
    inherited: Counter[str] = Counter(
        str(entry["v2_stratum"]) for entry in items.values() if entry["action"] == "inherit"
    )
    return {
        "rule": PLAN_RULE,
        "sample_name": key["sample_name"],
        "items": len(items),
        "by_v1_stratum": {name: dict(sorted(c.items())) for name, c in sorted(summary.items())},
        "inherited_by_v2_stratum": dict(sorted(inherited.items())),
        "caveat": PLAN_CAVEAT,
        "item_plan": items,
    }


def load_plan(path: Path, key_path: Path) -> Record:
    with open(path) as f:
        plan: Record = json.load(f)
    if plan["inputs"]["annotation_key"]["sha256"] != sha256_of_file(key_path):
        raise SystemExit(f"{path} was built from another annotation key")
    return plan


def plan_items(key: Record, plan: Record, item_ids: set[str]) -> dict[str, Record]:
    """The key items among ``item_ids`` with their udv_v2 stratum and, when it is one, tier."""
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
