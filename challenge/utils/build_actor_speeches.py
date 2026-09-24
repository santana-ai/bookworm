import argparse
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.dataset_io import load_gated_jsonl, write_json, write_jsonl
from utils.udv_pipeline import (
    STAGE_DIRECTION_PATTERN,
    is_party_info,
    normalize_name,
    normalize_whitespace,
    resolve_turn_name,
    split_into_turns,
)

Record = dict[str, Any]

DEFAULT_CONFIG = Path("configs/hearing_actors.toml")
CHAIR_ROLE = "chair"
SPEAKER_ROLE = "speaker"
MERGE_DESCRIPTION = (
    "actors merged across hearings by exact normalized name (accents removed, upper case,"
    " collapsed whitespace), plus the reviewed merge groups and hearing-scoped reassignments"
    " from the merges section of the config; remaining subset-name pairs are listed for review"
)
AMBIGUITY_CRITERION = (
    "pair of distinct actor keys whose word sets are equal or in strict subset relation"
)


@dataclass(frozen=True)
class ActorSpeechesConfig:
    lds_path: Path
    expected_sha256: str
    chair_names: tuple[str, ...]
    non_person_keys: tuple[str, ...]
    chair_min_words: int
    single_hearing_path: Path
    multi_hearing_path: Path
    ambiguous_names_path: Path
    stats_path: Path
    merge_aliases: dict[str, str]
    merge_reassignments: dict[tuple[str, int], str]


def load_config(path: Path) -> ActorSpeechesConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    merges = raw.get("merges", {})
    merge_aliases: dict[str, str] = {}
    for group in merges.get("groups", []):
        for alias in group["aliases"]:
            merge_aliases[alias] = group["canonical"]
    merge_reassignments = {
        (entry["key"], entry["hearing_id"]): entry["canonical"]
        for entry in merges.get("reassignments", [])
    }
    return ActorSpeechesConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        expected_sha256=raw["dataset"]["sha256"],
        chair_names=tuple(raw["speakers"]["chair_names"]),
        non_person_keys=tuple(raw["speakers"]["non_person_keys"]),
        chair_min_words=raw["speakers"]["chair_min_words"],
        single_hearing_path=Path(raw["speeches"]["single_hearing_path"]),
        multi_hearing_path=Path(raw["speeches"]["multi_hearing_path"]),
        ambiguous_names_path=Path(raw["speeches"]["ambiguous_names_path"]),
        stats_path=Path(raw["speeches"]["stats_path"]),
        merge_aliases=merge_aliases,
        merge_reassignments=merge_reassignments,
    )


def person_name(turn: Record) -> str:
    name = resolve_turn_name(turn)
    if name == turn["raw_name"] and turn["party_info"] and not is_party_info(turn["party_info"]):
        name = turn["party_info"]
    return normalize_whitespace(name)


def turn_party(turn: Record) -> str | None:
    if resolve_turn_name(turn) != turn["raw_name"]:
        return turn["party_info"].rsplit(". ", 1)[1].strip()
    if is_party_info(turn["party_info"]):
        return turn["party_info"]
    return None


def turn_words(turn: Record) -> int:
    return len(turn["speech"].split())


def turn_role(turn: Record, config: ActorSpeechesConfig) -> str:
    return CHAIR_ROLE if turn["raw_name"] in config.chair_names else SPEAKER_ROLE


def drop_reason(turn: Record, key: str, role: str, config: ActorSpeechesConfig) -> str | None:
    if key in config.non_person_keys:
        return "non_person_key"
    if STAGE_DIRECTION_PATTERN.match(turn["speech"]):
        return "stage_direction"
    if not turn["speech"]:
        return "empty"
    if role == CHAIR_ROLE and turn_words(turn) < config.chair_min_words:
        return "short_chair"
    return None


def apply_merges(
    key: str, hearing_id: int, config: ActorSpeechesConfig, usage: dict[str, Counter]
) -> str:
    target = config.merge_reassignments.get((key, hearing_id))
    if target is not None:
        usage["reassignments"][(key, hearing_id)] += 1
        key = target
    canonical = config.merge_aliases.get(key)
    if canonical is not None:
        usage["aliases"][key] += 1
        key = canonical
    return key


def collect_actors(
    hearings: list[Record], config: ActorSpeechesConfig
) -> tuple[dict[str, Record], Record, dict[str, Counter]]:
    actors: dict[str, Record] = {}
    usage: dict[str, Counter] = {"aliases": Counter(), "reassignments": Counter()}
    dropped: Record = {
        "non_person_key": [],
        "stage_direction": 0,
        "empty": 0,
        "short_chair": 0,
        "total_turns": 0,
    }
    for hearing in hearings:
        for turn in split_into_turns(hearing["transcricao"]):
            dropped["total_turns"] += 1
            name = person_name(turn)
            key = normalize_name(name)
            role = turn_role(turn, config)
            reason = drop_reason(turn, key, role, config)
            if reason == "non_person_key":
                dropped[reason].append(
                    {"key": key, "raw_name": turn["raw_name"], "hearing_id": hearing["id"]}
                )
                continue
            if reason is not None:
                dropped[reason] += 1
                continue
            key = apply_merges(key, hearing["id"], config, usage)
            actor = actors.setdefault(key, {"names": Counter(), "party_uf": {}, "hearings": {}})
            actor["names"][name] += 1
            party = turn_party(turn)
            if party is not None:
                actor["party_uf"].setdefault(party)
            actor["hearings"].setdefault(hearing["id"], []).append(
                {
                    "turn_index": turn["turn_index"],
                    "role": role,
                    "start_char": turn["start_char"],
                    "end_char": turn["end_char"],
                    "text": turn["speech"],
                }
            )
    return actors, dropped, usage


def check_merge_usage(config: ActorSpeechesConfig, usage: dict[str, Counter]) -> None:
    unused_aliases = sorted(set(config.merge_aliases) - set(usage["aliases"]))
    unused_reassignments = sorted(set(config.merge_reassignments) - set(usage["reassignments"]))
    if unused_aliases or unused_reassignments:
        raise ValueError(
            f"merge config entries matched no kept turn: aliases={unused_aliases},"
            f" reassignments={unused_reassignments}"
        )


def display_name(names: Counter) -> str:
    cased = Counter(
        {name: count for name, count in names.items() if any(c.islower() for c in name)}
    )
    pool = cased or names
    return sorted(pool.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[0][0]


def actor_record(actor: Record) -> Record:
    hearings = [
        {
            "hearing_id": hearing_id,
            "full_speech": "\n\n".join(turn["text"] for turn in turns),
            "turns": turns,
        }
        for hearing_id, turns in sorted(actor["hearings"].items())
    ]
    return {
        "actor": display_name(actor["names"]),
        "has_party_header": bool(actor["party_uf"]),
        "party_uf": list(actor["party_uf"]),
        "hearings": hearings,
    }


def verify_turn_texts(records: list[Record], transcripts: dict[int, str]) -> int:
    verified = 0
    for record in records:
        for hearing in record["hearings"]:
            transcript = transcripts[hearing["hearing_id"]]
            for turn in hearing["turns"]:
                if transcript[turn["start_char"] : turn["end_char"]] != turn["text"]:
                    raise ValueError(
                        f"turn text mismatch: hearing {hearing['hearing_id']}"
                        f" turn {turn['turn_index']}"
                    )
                verified += 1
    return verified


def pair_side(key: str, record: Record) -> Record:
    return {
        "key": key,
        "actor": record["actor"],
        "has_party_header": record["has_party_header"],
        "party_uf": record["party_uf"],
        "hearing_ids": [hearing["hearing_id"] for hearing in record["hearings"]],
        "turns": sum(len(hearing["turns"]) for hearing in record["hearings"]),
    }


def ambiguous_name_pairs(records: dict[str, Record]) -> list[Record]:
    keys = sorted(records)
    tokens = {key: set(key.split()) for key in keys}
    hearing_ids = {
        key: {hearing["hearing_id"] for hearing in record["hearings"]}
        for key, record in records.items()
    }
    pairs = []
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            tokens_a, tokens_b = tokens[key_a], tokens[key_b]
            if not (tokens_a <= tokens_b or tokens_b <= tokens_a):
                continue
            pairs.append(
                {
                    "relation": "equal_tokens" if tokens_a == tokens_b else "name_subset",
                    "a": pair_side(key_a, records[key_a]),
                    "b": pair_side(key_b, records[key_b]),
                    "shared_hearing_ids": sorted(hearing_ids[key_a] & hearing_ids[key_b]),
                }
            )
    return pairs


def words_by_role(records: list[Record]) -> dict[str, int]:
    words = {SPEAKER_ROLE: 0, CHAIR_ROLE: 0}
    for record in records:
        for hearing in record["hearings"]:
            for turn in hearing["turns"]:
                words[turn["role"]] += len(turn["text"].split())
    return words


def file_stats(records: list[Record], path: Path) -> Record:
    words = words_by_role(records)
    return {
        "path": str(path),
        "actors": len(records),
        "with_party_header": sum(record["has_party_header"] for record in records),
        "without_party_header": sum(not record["has_party_header"] for record in records),
        "speaker_words": words[SPEAKER_ROLE],
        "chair_words": words[CHAIR_ROLE],
    }


def non_person_summary(drops: list[Record]) -> Record:
    by_key = Counter(drop["key"] for drop in drops)
    return {
        "turns_dropped": len(drops),
        "by_key": {key: count for key, count in sorted(by_key.items())},
        "turns": drops,
    }


def build(config: ActorSpeechesConfig) -> Record:
    hearings = load_gated_jsonl(config.lds_path, config.expected_sha256)
    actors, dropped, usage = collect_actors(hearings, config)
    check_merge_usage(config, usage)
    records = {key: actor_record(actor) for key, actor in sorted(actors.items())}
    transcripts = {hearing["id"]: hearing["transcricao"] for hearing in hearings}
    verified = verify_turn_texts(list(records.values()), transcripts)
    single = [record for record in records.values() if len(record["hearings"]) == 1]
    multi = [record for record in records.values() if len(record["hearings"]) >= 2]
    write_jsonl(single, config.single_hearing_path)
    write_jsonl(multi, config.multi_hearing_path)
    pairs = ambiguous_name_pairs(records)
    write_json({"criterion": AMBIGUITY_CRITERION, "pairs": pairs}, config.ambiguous_names_path)
    by_hearing_count = Counter(len(record["hearings"]) for record in records.values())
    stats = {
        "dataset": {
            "path": str(config.lds_path),
            "sha256": config.expected_sha256,
            "hearings": len(hearings),
        },
        "policy": {
            "chair_min_words": config.chair_min_words,
            "non_person_keys": list(config.non_person_keys),
            "merge": MERGE_DESCRIPTION,
        },
        "turns": {
            "total": dropped["total_turns"],
            "kept": verified,
            "verified_against_transcript": verified,
            "dropped_non_person": non_person_summary(dropped["non_person_key"]),
            "dropped_stage_direction": dropped["stage_direction"],
            "dropped_empty": dropped["empty"],
            "dropped_short_chair": dropped["short_chair"],
        },
        "merges": {
            "groups": len(set(config.merge_aliases.values())),
            "alias_keys": len(config.merge_aliases),
            "alias_turns_kept": sum(usage["aliases"].values()),
            "reassignments": len(config.merge_reassignments),
            "reassigned_turns_kept": sum(usage["reassignments"].values()),
        },
        "hearings_per_actor": {str(k): v for k, v in sorted(by_hearing_count.items())},
        "files": {
            "single_hearing": file_stats(single, config.single_hearing_path),
            "multi_hearing": file_stats(multi, config.multi_hearing_path),
        },
        "ambiguous_name_pairs": {
            "path": str(config.ambiguous_names_path),
            "criterion": AMBIGUITY_CRITERION,
            "pairs": len(pairs),
            "equal_tokens": sum(pair["relation"] == "equal_tokens" for pair in pairs),
            "name_subset": sum(pair["relation"] == "name_subset" for pair in pairs),
        },
    }
    write_json(stats, config.stats_path)
    return stats


def print_report(stats: Record) -> None:
    turns = stats["turns"]
    print(f"turns: {turns['total']} total, {turns['kept']} kept,")
    print(f"  {turns['kept']} texts verified against transcricao[start_char:end_char]")
    print(f"  dropped: {turns['dropped_non_person']['turns_dropped']} non-person key,")
    print(f"  {turns['dropped_stage_direction']} stage direction, {turns['dropped_empty']} empty,")
    chair_cut = stats["policy"]["chair_min_words"]
    print(f"  {turns['dropped_short_chair']} chair under {chair_cut} words")
    for key, count in turns["dropped_non_person"]["by_key"].items():
        print(f"  non-person key {key!r}: {count} turns")
    merges = stats["merges"]
    print(
        f"merges: {merges['groups']} groups ({merges['alias_keys']} alias keys,"
        f" {merges['alias_turns_kept']} turns), {merges['reassignments']} reassignments"
        f" ({merges['reassigned_turns_kept']} turns)"
    )
    print(f"hearings per actor: {stats['hearings_per_actor']}")
    for label, entry in stats["files"].items():
        print(
            f"{label}: {entry['actors']} actors ({entry['with_party_header']} with party header,"
            f" {entry['without_party_header']} without),"
            f" {entry['speaker_words']} speaker words, {entry['chair_words']} chair words"
            f" -> {entry['path']}"
        )
    pairs = stats["ambiguous_name_pairs"]
    print(
        f"ambiguous name pairs: {pairs['pairs']} ({pairs['equal_tokens']} equal tokens,"
        f" {pairs['name_subset']} name subset) -> {pairs['path']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-actor full-speech files from hearing transcripts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    stats = build(config)
    print_report(stats)
    print(f"wrote {config.stats_path}")


if __name__ == "__main__":
    main()
