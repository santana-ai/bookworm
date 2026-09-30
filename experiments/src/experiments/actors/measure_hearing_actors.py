"""Measurements that set the policy of the per-actor speech files: speaker recurrence across
hearings, the weight of chair turns, UDV evidence in chair turns, party headers and name
subsets."""

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm import load_gated_jsonl, load_jsonl, write_json

from experiments.actors.build_speeches import header_name, turn_words
from experiments.actors.io import read_toml
from experiments.common.transcript import (
    STAGE_DIRECTION_PATTERN,
    is_party_info,
    names_match,
    normalize_name,
    split_into_turns,
)

Record = dict[str, Any]

DEFAULT_CONFIG = Path("configs/hearing_actors.toml")
PARTY_UF_PATTERN = re.compile(r" - [A-Z]{2}$")
TITLED_HEADER_PATTERN = re.compile(
    r"(?:O\s+SR\.|A\s+SRA\.)\s*(SENADORA?|DEPUTAD[OA]|VEREADORA?|MINISTR[OA])\b"
)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
RECURRENCE_LEVELS = (2, 3, 5)
TOP_EXAMPLES = 10
SPEAKER_KEY_DESCRIPTION = (
    "turn header name, accents removed, upper case; chair turns use the name in parentheses"
)


@dataclass(frozen=True)
class HearingActorsConfig:
    lds_path: Path
    expected_sha256: str
    chair_names: tuple[str, ...]
    non_person_keys: tuple[str, ...]
    chair_min_words: int
    candidate_chair_cuts: tuple[int, ...]
    long_chair_turn_words: int
    udv_path: Path
    output_path: Path


def load_config(path: Path) -> HearingActorsConfig:
    raw = read_toml(path)
    speakers = raw["speakers"]
    return HearingActorsConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        expected_sha256=raw["dataset"]["sha256"],
        chair_names=tuple(speakers["chair_names"]),
        non_person_keys=tuple(speakers["non_person_keys"]),
        chair_min_words=speakers["chair_min_words"],
        candidate_chair_cuts=tuple(speakers["candidate_chair_cuts"]),
        long_chair_turn_words=speakers["long_chair_turn_words"],
        udv_path=Path(raw["measurement"]["udv_path"]),
        output_path=Path(raw["measurement"]["output_path"]),
    )


def speaker_key(turn: Record) -> str:
    return normalize_name(re.sub(r"\s+", " ", header_name(turn)))


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {f"p{round(q * 100)}": ordered[int(len(ordered) * q)] for q in QUANTILES}


def annotate_turns(hearings: list[Record], config: HearingActorsConfig) -> list[Record]:
    turns = []
    for hearing in hearings:
        for turn in split_into_turns(hearing["transcricao"]):
            turns.append(
                {
                    **turn,
                    "hearing_id": hearing["id"],
                    "key": speaker_key(turn),
                    "words": turn_words(turn),
                    "is_chair": turn["raw_name"] in config.chair_names,
                    "has_party": is_party_info(turn["party_info"]),
                    "is_stage_direction": bool(STAGE_DIRECTION_PATTERN.match(turn["speech"])),
                }
            )
    return turns


def aggregate_speakers(turns: list[Record]) -> dict[str, Record]:
    speakers: dict[str, Record] = defaultdict(
        lambda: {"hearings": set(), "words": 0, "chair_words": 0, "has_party": False}
    )
    for turn in turns:
        entry = speakers[turn["key"]]
        entry["hearings"].add(turn["hearing_id"])
        entry["words"] += turn["words"]
        entry["chair_words"] += turn["words"] if turn["is_chair"] else 0
        entry["has_party"] |= turn["has_party"]
    return dict(speakers)


def recurrence_summary(speakers: dict[str, Record], turns: list[Record]) -> Record:
    by_count = Counter(len(entry["hearings"]) for entry in speakers.values())
    levels = {}
    for level in RECURRENCE_LEVELS:
        group = [entry for entry in speakers.values() if len(entry["hearings"]) >= level]
        levels[f"at_least_{level}"] = {
            "speakers": len(group),
            "with_party_header": sum(entry["has_party"] for entry in group),
        }
    top = sorted(speakers.items(), key=lambda item: -len(item[1]["hearings"]))[:TOP_EXAMPLES]
    return {
        "turns": len(turns),
        "speaker_keys": len(speakers),
        "speakers_by_hearing_count": {str(k): v for k, v in sorted(by_count.items())},
        "recurrence": levels,
        "words_per_speaker": quantiles([entry["words"] for entry in speakers.values()]),
        "most_recurrent": [
            {"key": key, "hearings": len(entry["hearings"]), "words": entry["words"]}
            for key, entry in top
        ],
    }


def chair_summary(
    speakers: dict[str, Record], turns: list[Record], config: HearingActorsConfig
) -> Record:
    chair_lengths = [turn["words"] for turn in turns if turn["is_chair"]]
    recurrent = {key: entry for key, entry in speakers.items() if len(entry["hearings"]) >= 2}
    chair_words = sum(entry["chair_words"] for entry in recurrent.values())
    total_words = sum(entry["words"] for entry in recurrent.values())
    heaviest = sorted(
        (item for item in recurrent.items() if item[1]["words"] > 0),
        key=lambda item: -item[1]["chair_words"],
    )[:TOP_EXAMPLES]
    return {
        "chair_turns": len(chair_lengths),
        "chair_turn_words": quantiles(chair_lengths),
        "chair_turns_at_least_long": sum(n >= config.long_chair_turn_words for n in chair_lengths),
        "long_chair_turn_words": config.long_chair_turn_words,
        "recurrent_chair_words": chair_words,
        "recurrent_total_words": total_words,
        "recurrent_chair_share": round(chair_words / total_words, 4),
        "heaviest_chair_speakers": [
            {
                "key": key,
                "hearings": len(entry["hearings"]),
                "chair_share": round(entry["chair_words"] / entry["words"], 4),
            }
            for key, entry in heaviest
        ],
    }


def udv_chair_evidence(
    udvs: list[Record], turns: list[Record], config: HearingActorsConfig
) -> Record:
    turn_by_position = {(turn["hearing_id"], turn["turn_index"]): turn for turn in turns}
    located: Counter[str] = Counter()
    in_chair: Counter[str] = Counter()
    chair_actors = set()
    evidence_turn_words = []
    quote_examples: list[Record] = []
    for udv in udvs:
        evidence = udv.get("evidence") or {}
        if evidence.get("speaker_turn") is None:
            continue
        located[udv["tier"]] += 1
        turn = turn_by_position[(udv["hearing_id"], evidence["speaker_turn"])]
        if not turn["is_chair"]:
            continue
        in_chair[udv["tier"]] += 1
        chair_actors.add((udv["hearing_id"], udv["actor"]["name"]))
        evidence_turn_words.append(turn["words"])
        if udv["tier"] == "quote_found" and len(quote_examples) < TOP_EXAMPLES:
            quote_examples.append(
                {"id": udv["id"], "actor": udv["actor"]["name"], "evidence": evidence["text"]}
            )
    chair_lengths = [turn["words"] for turn in turns if turn["is_chair"]]
    total_chair_words = sum(chair_lengths)
    cuts = [
        {
            "min_words": cut,
            "chair_turns_kept": sum(n >= cut for n in chair_lengths),
            "chair_words_kept_share": round(
                sum(n for n in chair_lengths if n >= cut) / total_chair_words, 4
            ),
            "chair_evidence_udvs_kept": sum(n >= cut for n in evidence_turn_words),
        }
        for cut in config.candidate_chair_cuts
    ]
    return {
        "udv_path": str(config.udv_path),
        "udvs": len(udvs),
        "located_evidence_by_tier": dict(located),
        "located_evidence": sum(located.values()),
        "chair_evidence_by_tier": dict(in_chair),
        "chair_evidence": sum(in_chair.values()),
        "chair_evidence_actors": len({name for _, name in chair_actors}),
        "chair_quote_examples": quote_examples,
        "chair_cuts": cuts,
    }


def party_header_checks(hearings: list[Record], turns: list[Record]) -> Record:
    party_turns = [turn for turn in turns if turn["has_party"]]
    party_uf_shape = sum(
        bool(PARTY_UF_PATTERN.search(turn["party_info"].rsplit(". ", 1)[-1]))
        for turn in party_turns
    )
    titled: Counter[str] = Counter()
    for hearing in hearings:
        for match in TITLED_HEADER_PATTERN.finditer(hearing["transcricao"]):
            titled[match.group(1)] += 1
    titled_with_party = sum(
        1
        for turn in turns
        if TITLED_HEADER_PATTERN.match(f"O SR. {turn['raw_name']}") and turn["has_party"]
    )
    matched = 0
    matched_deputies = 0
    non_deputy_cargos = []
    turns_by_hearing: dict[int, set[str]] = defaultdict(set)
    for turn in party_turns:
        turns_by_hearing[turn["hearing_id"]].add(turn["key"])
    for hearing in hearings:
        party_keys = turns_by_hearing.get(hearing["id"], set())
        for person in hearing["metadados"]["envolvidos"]:
            if not any(names_match(person["nome"], key) for key in party_keys):
                continue
            matched += 1
            cargo = person.get("cargo") or ""
            if "deputad" in cargo.lower():
                matched_deputies += 1
            else:
                non_deputy_cargos.append(
                    {"hearing_id": hearing["id"], "nome": person["nome"], "cargo": cargo}
                )
    return {
        "party_header_turns": len(party_turns),
        "party_header_turns_ending_in_uf": party_uf_shape,
        "titled_headers": dict(titled),
        "titled_headers_with_party": titled_with_party,
        "involved_matched_to_party_speakers": matched,
        "involved_matched_with_deputy_cargo": matched_deputies,
        "involved_matched_other_cargo": non_deputy_cargos,
    }


def cross_hearing_name_subsets(speakers: dict[str, Record]) -> Record:
    keys = list(speakers)
    pairs = []
    for index, name_a in enumerate(keys):
        tokens_a = set(name_a.split())
        if len(tokens_a) < 2:
            continue
        for name_b in keys[index + 1 :]:
            tokens_b = set(name_b.split())
            if len(tokens_b) < 2 or not (tokens_a < tokens_b or tokens_b < tokens_a):
                continue
            if not speakers[name_a]["hearings"] & speakers[name_b]["hearings"]:
                pairs.append(
                    {
                        "a": name_a,
                        "b": name_b,
                        "a_has_party": speakers[name_a]["has_party"],
                        "b_has_party": speakers[name_b]["has_party"],
                    }
                )
    return {
        "pairs": len(pairs),
        "pairs_where_both_have_party": sum(p["a_has_party"] and p["b_has_party"] for p in pairs),
        "pairs_where_neither_has_party": sum(
            not p["a_has_party"] and not p["b_has_party"] for p in pairs
        ),
        "examples": pairs,
    }


def recurrent_without_party(
    speakers: dict[str, Record], hearings: list[Record], config: HearingActorsConfig
) -> Record:
    cargos: dict[str, set[str]] = defaultdict(set)
    for hearing in hearings:
        for person in hearing["metadados"]["envolvidos"]:
            cargos[normalize_name(person["nome"])].add(person.get("cargo") or "")
    group = sorted(
        (
            (key, entry)
            for key, entry in speakers.items()
            if len(entry["hearings"]) >= 2 and not entry["has_party"]
        ),
        key=lambda item: (-len(item[1]["hearings"]), item[0]),
    )
    return {
        "speakers": len(group),
        "non_person_keys": sum(key in config.non_person_keys for key, _ in group),
        "at_least_3_hearings": sum(len(entry["hearings"]) >= 3 for _, entry in group),
        "list": [
            {
                "key": key,
                "hearings": len(entry["hearings"]),
                "words": entry["words"],
                "cargo_in_article": sorted(cargos.get(key, set())),
            }
            for key, entry in group
        ],
    }


def is_kept(turn: Record, config: HearingActorsConfig) -> bool:
    if turn["key"] in config.non_person_keys or turn["is_stage_direction"]:
        return False
    return not (turn["is_chair"] and turn["words"] < config.chair_min_words)


def policy_projection(turns: list[Record], config: HearingActorsConfig) -> Record:
    kept = [turn for turn in turns if is_kept(turn, config)]
    speakers = aggregate_speakers(kept)
    by_count = Counter(len(entry["hearings"]) for entry in speakers.values())
    files = {}
    for label, selector in (
        ("single_hearing", lambda n: n == 1),
        ("multi_hearing", lambda n: n >= 2),
    ):
        group = [entry for entry in speakers.values() if selector(len(entry["hearings"]))]
        files[label] = {
            "actors": len(group),
            "with_party_header": sum(entry["has_party"] for entry in group),
            "speaker_words": sum(entry["words"] - entry["chair_words"] for entry in group),
            "chair_words": sum(entry["chair_words"] for entry in group),
        }
    return {
        "chair_min_words": config.chair_min_words,
        "turns_kept": len(kept),
        "chair_turns_dropped": sum(
            turn["is_chair"] and turn["words"] < config.chair_min_words for turn in turns
        ),
        "non_person_turns_dropped": sum(turn["key"] in config.non_person_keys for turn in turns),
        "stage_direction_turns_dropped": sum(turn["is_stage_direction"] for turn in turns),
        "actors_by_hearing_count": {str(k): v for k, v in sorted(by_count.items())},
        "files": files,
    }


def measure(config: HearingActorsConfig) -> Record:
    hearings = load_gated_jsonl(config.lds_path, config.expected_sha256)
    udvs = load_jsonl(config.udv_path)
    turns = annotate_turns(hearings, config)
    speakers = aggregate_speakers(turns)
    return {
        "dataset": {
            "path": str(config.lds_path),
            "sha256": config.expected_sha256,
            "hearings": len(hearings),
        },
        "speaker_key": SPEAKER_KEY_DESCRIPTION,
        "recurrence": recurrence_summary(speakers, turns),
        "chair": chair_summary(speakers, turns, config),
        "udv_chair_evidence": udv_chair_evidence(udvs, turns, config),
        "party_header": party_header_checks(hearings, turns),
        "name_subsets": cross_hearing_name_subsets(speakers),
        "recurrent_without_party": recurrent_without_party(speakers, hearings, config),
        "policy_projection": policy_projection(turns, config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure speaker recurrence across hearings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    result = measure(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, config.output_path)
    print(f"wrote {config.output_path}")


if __name__ == "__main__":
    main()
