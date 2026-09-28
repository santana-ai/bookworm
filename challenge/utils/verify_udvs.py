import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from utils.build_udvs import SUPPORT_TYPES, TIERS, load_config, load_lds_records
from utils.dataset_io import load_jsonl
from utils.udv_pipeline import (
    find_opinion_turn_quote_match,
    is_trusted_quote,
    normalize_whitespace,
    resolve_person_speech,
    sentences_agree,
    split_into_turns,
    split_turn_sentences,
    turn_text,
)

Record = dict[str, Any]


def index_people(hearings: list[Record]) -> dict[tuple[int, int], Record]:
    people: dict[tuple[int, int], Record] = {}
    for hearing in hearings:
        turns = split_into_turns(hearing["transcricao"])
        for person_index, participant in enumerate(hearing["metadados"]["envolvidos"]):
            matched_turns, speech = resolve_person_speech(participant, turns)
            units = split_turn_sentences(matched_turns)
            people[(hearing["id"], person_index)] = {
                "hearing": hearing,
                "participant": participant,
                "matched_turns": matched_turns,
                "speech": speech,
                "sentences": [unit["text"] for unit in units],
                "sentence_turns": [unit["turn_index"] for unit in units],
            }
    return people


def expected_ids(people: dict[tuple[int, int], Record]) -> dict[str, tuple[Record, str]]:
    expected: dict[str, tuple[Record, str]] = {}
    for (hearing_id, person_index), person in people.items():
        for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"]):
            expected[f"udv-{hearing_id}-{person_index}-{opinion_index}"] = (person, opinion_text)
    return expected


def check_record(record: Record, person: Record, opinion_text: str, threshold: float) -> list[str]:
    problems: list[str] = []
    participant = person["participant"]
    if record["proposition"] != opinion_text:
        problems.append("proposition_mismatch")
    if record["actor"] != {"name": participant["nome"], "role": participant["cargo"]}:
        problems.append("actor_mismatch")
    if record["method"]["embedding_threshold"] != threshold:
        problems.append("method_threshold_mismatch")
    tier, evidence, provenance = record["tier"], record["evidence"], record["provenance"]
    if tier == "person_not_resolved":
        if person["matched_turns"]:
            problems.append("person_not_resolved_but_matched")
        if evidence is not None or provenance is not None:
            problems.append("person_not_resolved_shape")
        return problems
    if not person["matched_turns"]:
        problems.append("resolved_tier_but_unmatched")
    if tier == "no_evidence":
        if evidence is not None or provenance is not None or person["sentences"]:
            problems.append("no_evidence_shape")
        return problems
    if evidence is None:
        return problems + ["evidence_missing"]
    if evidence["support_type"] not in SUPPORT_TYPES:
        problems.append("unknown_support_type")
    problems.extend(check_single_turn(evidence, person))
    quote_match = find_opinion_turn_quote_match(opinion_text, person["matched_turns"])
    if tier == "quote_found":
        problems.extend(check_quote_evidence(evidence, quote_match, provenance))
    else:
        if provenance != "model" or evidence["support_type"] == "direct_quote":
            problems.append("semantic_shape")
        if is_trusted_quote(quote_match):
            problems.append("semantic_but_trusted_quote_findable")
        if evidence["text"] not in person["sentences"]:
            problems.append("evidence_not_person_sentence")
        problems.extend(check_short_quote_support(evidence, quote_match))
        score = evidence["score"]
        if score is None or not -1.0001 <= score <= 1.0001:
            problems.append("score_out_of_range")
        elif (tier == "semantic_match_high") != (score >= threshold):
            problems.append("tier_inconsistent_with_score")
    problems.extend(check_offsets(record, person))
    problems.extend(check_source_turn(evidence, quote_match, person, tier))
    return problems


def check_single_turn(evidence: Record, person: Record) -> list[str]:
    if any(evidence["text"] in turn_text(turn) for turn in person["matched_turns"]):
        return []
    return ["evidence_not_in_single_actor_turn"]


def check_quote_evidence(
    evidence: Record, quote_match: Record | None, provenance: str | None
) -> list[str]:
    problems: list[str] = []
    if provenance != "weak" or evidence["support_type"] != "direct_quote":
        problems.append("quote_shape")
    if evidence["score"] is not None:
        problems.append("quote_score_not_null")
    if not is_trusted_quote(quote_match):
        return problems + ["quote_not_trusted"]
    if evidence["quote_prefix"] != quote_match["prefix"]:
        return problems + ["quote_prefix_mismatch"]
    if quote_match["sentence"] != evidence["text"]:
        problems.append("quote_text_mismatch")
    return problems


def check_short_quote_support(evidence: Record, quote_match: Record | None) -> list[str]:
    supported = (
        quote_match is not None
        and not is_trusted_quote(quote_match)
        and sentences_agree(quote_match["sentence"], evidence["text"])
    )
    expected_support_type = "semantic_with_short_quote" if supported else "semantic_similarity"
    expected_prefix = cast(Record, quote_match)["prefix"] if supported else None
    problems: list[str] = []
    if evidence["support_type"] != expected_support_type:
        problems.append("short_quote_support_mismatch")
    if evidence["quote_prefix"] != expected_prefix:
        problems.append("short_quote_prefix_mismatch")
    return problems


def check_offsets(record: Record, person: Record) -> list[str]:
    evidence = record["evidence"]
    if evidence["start_char"] is None:
        if evidence["end_char"] is not None or evidence["speaker_turn"] is not None:
            return ["offset_shape", "evidence_offsets_missing"]
        return ["evidence_offsets_missing"]
    problems: list[str] = []
    transcript = person["hearing"]["transcricao"]
    span = transcript[evidence["start_char"] : evidence["end_char"]]
    if normalize_whitespace(span) != evidence["text"]:
        problems.append("offset_text_mismatch")
    turn = next(
        (t for t in person["matched_turns"] if t["turn_index"] == evidence["speaker_turn"]), None
    )
    if turn is None:
        problems.append("speaker_turn_not_actor")
    elif not (
        turn["start_char"] <= evidence["start_char"] and evidence["end_char"] <= turn["end_char"]
    ):
        problems.append("span_outside_turn")
    return problems


def check_source_turn(
    evidence: Record, quote_match: Record | None, person: Record, tier: str
) -> list[str]:
    if evidence["speaker_turn"] is None:
        return []
    if tier == "quote_found":
        if is_trusted_quote(quote_match) and evidence["speaker_turn"] != quote_match["turn_index"]:
            return ["quote_turn_mismatch"]
        return []
    source_turns = {
        turn_index
        for sentence, turn_index in zip(person["sentences"], person["sentence_turns"], strict=True)
        if sentence == evidence["text"]
    }
    if source_turns and evidence["speaker_turn"] not in source_turns:
        return ["semantic_turn_mismatch"]
    return []


def check_coverage(
    coverage: Record, records: list[Record], people: dict[tuple[int, int], Record]
) -> list[str]:
    problems: list[str] = []
    by_tier = Counter(record["tier"] for record in records)
    evidences = [record["evidence"] for record in records if record["evidence"] is not None]
    recount = {
        "opinions.total": (coverage["opinions"]["total"], len(records)),
        "people.total": (coverage["people"]["total"], len(people)),
        "people.resolved": (
            coverage["people"]["resolved"],
            sum(1 for person in people.values() if person["matched_turns"]),
        ),
        "evidence_offsets.total": (coverage["evidence_offsets"]["total"], len(evidences)),
        "evidence_offsets.located": (
            coverage["evidence_offsets"]["located"],
            sum(1 for evidence in evidences if evidence["start_char"] is not None),
        ),
        **{
            f"by_tier.{tier}": (coverage["opinions"]["by_tier"][tier], by_tier[tier])
            for tier in TIERS
        },
        **{
            f"evidence_support_types.{support_type}": (
                coverage["evidence_support_types"][support_type],
                sum(1 for evidence in evidences if evidence["support_type"] == support_type),
            )
            for support_type in SUPPORT_TYPES
        },
    }
    for name, (reported, recomputed) in recount.items():
        if reported != recomputed:
            problems.append(f"coverage.{name}: reported {reported}, recomputed {recomputed}")
    return problems


def compare_with_baseline(records: list[Record], baseline: list[Record]) -> Record:
    current = {record["id"]: record for record in records}
    moves: Counter[str] = Counter()
    evidence_changed = 0
    for old in baseline:
        new = current.get(old["id"])
        if new is None:
            moves["missing_in_run"] += 1
            continue
        if old["tier"] != new["tier"]:
            moves[f"{old['tier']}->{new['tier']}"] += 1
        if old["evidence"] != new["evidence"]:
            evidence_changed += 1
    return {
        "baseline_records": len(baseline),
        "evidence_changed": evidence_changed,
        "tier_moves": dict(moves),
        "extra_in_run": len(set(current) - {record["id"] for record in baseline}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and cross-check a UDV run against the LDS file."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    parser.add_argument("--run-name", default="udv_v0")
    parser.add_argument(
        "--baseline", type=Path, default=None, help="previous <run>.jsonl to diff against"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = load_jsonl(config.output_dir / f"{args.run_name}.jsonl")
    with open(config.output_dir / f"{args.run_name}_coverage.json") as f:
        coverage = json.load(f)
    threshold = coverage["config"]["evidence"]["embedding_threshold"]
    hearing_ids = set(coverage["hearings"]["ids"])
    hearings = [hearing for hearing in load_lds_records(config) if hearing["id"] in hearing_ids]
    people = index_people(hearings)
    expected = expected_ids(people)

    problems: dict[str, list[str]] = defaultdict(list)
    ids = Counter(record["id"] for record in records)
    for record_id, count in ids.items():
        if count > 1:
            problems["duplicate_id"].append(record_id)
    for record_id in sorted(set(expected) - set(ids)):
        problems["missing_id"].append(record_id)
    for record in records:
        if record["id"] not in expected:
            problems["unexpected_id"].append(record["id"])
            continue
        person, opinion_text = expected[record["id"]]
        for problem in check_record(record, person, opinion_text, threshold):
            problems[problem].append(record["id"])
    for problem in check_coverage(coverage, records, people):
        problems["coverage"].append(problem)

    report = {
        "run_name": args.run_name,
        "records": len(records),
        "expected_records": len(expected),
        "people_resolved": sum(1 for person in people.values() if person["matched_turns"]),
        "people_total": len(people),
        "by_tier": {tier: sum(1 for r in records if r["tier"] == tier) for tier in TIERS},
        "by_support_type": {
            support_type: sum(
                1
                for record in records
                if record["evidence"] is not None
                and record["evidence"]["support_type"] == support_type
            )
            for support_type in SUPPORT_TYPES
        },
        "problems": {
            kind: {"count": len(items), "examples": items[:5]} for kind, items in problems.items()
        },
    }
    if args.baseline is not None:
        report["baseline_diff"] = compare_with_baseline(records, load_jsonl(args.baseline))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
