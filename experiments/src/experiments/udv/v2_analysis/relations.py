"""How the udv_v2 evidence of a sampled item relates to the udv_v1 evidence it was judged on."""

from typing import Any

from experiments.validation.generate_sample import ValidationConfig

Record = dict[str, Any]

IDENTITY_FIELDS = ("text", "start_char", "end_char", "speaker_turn")
RELATIONS = ("same", "superset", "moved", "no_evidence")
IDENTICAL_RELATIONS = ("same", "no_evidence")
INHERITABLE_RELATIONS = {"same", "superset"}
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
    """The relation of an item from those of its UDVs: any move, or a mix with none, moves it."""
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
    if "same" not in inherit or not set(inherit) <= INHERITABLE_RELATIONS:
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
        if identical != (relation in IDENTICAL_RELATIONS):
            raise SystemExit(f"{item_id}: relation {relation} disagrees with the annotation plan")
        relations[item_id] = {
            "question": item["question"],
            "relation": relation,
            "udv_relations": by_udv,
            "inherited": is_inherited(relation, inheritance),
        }
    return relations
