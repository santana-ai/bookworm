import argparse
import dataclasses
import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.build_splits import SPLIT_NAMES
from utils.dataset_io import load_jsonl, sha256_of_file, write_json, write_jsonl

Record = dict[str, Any]

DEFAULT_CONFIG = Path("configs/actor_profiles.toml")
LINK_DESCRIPTION = (
    "a UDV of an evaluation hearing is linked to the actor who owns its evidence turn"
    " (hearing_id, evidence.speaker_turn) in the unfiltered speech file; UDVs without an"
    " evidence turn, or whose turn was dropped when the speech file was built, stay unlinked"
)


@dataclass(frozen=True)
class SplitFilterConfig:
    speeches_path: Path
    lds_sha256: str
    manifest_path: Path
    splits: tuple[str, ...]
    eval_splits: tuple[str, ...]
    udv_path: Path
    output_path: Path
    stats_path: Path


def check_split_names(splits: tuple[str, ...]) -> None:
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"splits must be distinct names among {SPLIT_NAMES}, got {splits}")


def load_config(path: Path) -> SplitFilterConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    section = raw["split_filter"]
    splits = tuple(section["splits"])
    eval_splits = tuple(section["eval_splits"])
    check_split_names(splits)
    check_split_names(eval_splits)
    if set(splits) & set(eval_splits):
        raise SystemExit(f"eval_splits {eval_splits} overlap the profile splits {splits}")
    return SplitFilterConfig(
        speeches_path=Path(raw["input"]["speeches_path"]),
        lds_sha256=raw["input"]["lds_sha256"],
        manifest_path=Path(section["manifest_path"]),
        splits=splits,
        eval_splits=eval_splits,
        udv_path=Path(section["udv_path"]),
        output_path=Path(section["speeches_path"]),
        stats_path=Path(section["stats_path"]),
    )


def load_split_hearings(config: SplitFilterConfig) -> tuple[set[int], set[int], Record]:
    with open(config.manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != config.lds_sha256:
        raise SystemExit(f"{config.manifest_path} was built from another LDS file")
    hearings = {hearing_id for name in config.splits for hearing_id in manifest[name]}
    eval_hearings = {hearing_id for name in config.eval_splits for hearing_id in manifest[name]}
    info = {
        "path": str(config.manifest_path),
        "sha256": sha256_of_file(config.manifest_path),
        "split_version": manifest["split_version"],
        "splits": list(config.splits),
        "hearings": len(hearings),
    }
    return hearings, eval_hearings, info


def filter_record(record: Record, hearings: set[int]) -> Record | None:
    kept = [hearing for hearing in record["hearings"] if hearing["hearing_id"] in hearings]
    if not kept:
        return None
    return {**record, "hearings": kept}


def summarize(records: list[Record]) -> Record:
    return {
        "actors": len(records),
        "hearings": len({h["hearing_id"] for record in records for h in record["hearings"]}),
        "turns": sum(len(h["turns"]) for record in records for h in record["hearings"]),
        "hearings_per_actor": {
            str(count): actors
            for count, actors in sorted(Counter(len(r["hearings"]) for r in records).items())
        },
    }


def summarize_evaluation(
    records: list[Record],
    filtered: list[Record],
    udvs: list[Record],
    eval_hearings: set[int],
) -> Record:
    profiled = {record["actor"] for record in filtered}
    owners = {
        (hearing["hearing_id"], turn["turn_index"]): record["actor"]
        for record in records
        for hearing in record["hearings"]
        if hearing["hearing_id"] in eval_hearings
        for turn in hearing["turns"]
    }
    speaking = {actor for actor in owners.values() if actor in profiled}
    linked: list[tuple[Record, str]] = []
    for udv in udvs:
        evidence = udv["evidence"]
        if udv["hearing_id"] not in eval_hearings or not evidence:
            continue
        actor = owners.get((udv["hearing_id"], evidence["speaker_turn"]))
        if actor is not None and actor in profiled:
            linked.append((udv, actor))
    return {
        "hearings": len(eval_hearings),
        "profiled_actors_speaking": len(speaking),
        "udvs": sum(udv["hearing_id"] in eval_hearings for udv in udvs),
        "linked_udvs": len(linked),
        "linked_actors": len({actor for _, actor in linked}),
        "linked_hearings": len({udv["hearing_id"] for udv, _ in linked}),
        "linked_udvs_by_tier": dict(sorted(Counter(udv["tier"] for udv, _ in linked).items())),
    }


def build(config: SplitFilterConfig) -> Record:
    hearings, eval_hearings, manifest_info = load_split_hearings(config)
    records = load_jsonl(config.speeches_path)
    filtered = [
        kept for kept in (filter_record(record, hearings) for record in records) if kept is not None
    ]
    write_jsonl(filtered, config.output_path)
    stats = {
        "split_manifest": manifest_info,
        "input": {
            "path": str(config.speeches_path),
            "sha256": sha256_of_file(config.speeches_path),
            **summarize(records),
        },
        "output": {
            "path": str(config.output_path),
            "sha256": sha256_of_file(config.output_path),
            **summarize(filtered),
            "actors_without_split_hearings": len(records) - len(filtered),
        },
        "evaluation": {
            "splits": list(config.eval_splits),
            "udv_path": str(config.udv_path),
            "udv_sha256": sha256_of_file(config.udv_path),
            "link": LINK_DESCRIPTION,
            **summarize_evaluation(records, filtered, load_jsonl(config.udv_path), eval_hearings),
        },
    }
    write_json(stats, config.stats_path)
    return stats


def print_report(stats: Record) -> None:
    manifest = stats["split_manifest"]
    source = stats["input"]
    output = stats["output"]
    print(
        f"splits {manifest['splits']} of {manifest['split_version']}:"
        f" {manifest['hearings']} hearings"
    )
    print(
        f"input: {source['actors']} actors, {source['hearings']} hearings, {source['turns']} turns"
    )
    print(
        f"output: {output['actors']} actors, {output['hearings']} hearings,"
        f" {output['turns']} turns -> {output['path']}"
    )
    print(f"  {output['actors_without_split_hearings']} actors have no hearing in these splits")
    print(f"  hearings per actor: {output['hearings_per_actor']}")
    evaluation = stats["evaluation"]
    print(
        f"evaluation {evaluation['splits']}: {evaluation['hearings']} hearings,"
        f" {evaluation['profiled_actors_speaking']} profiled actors speak in them"
    )
    print(
        f"  {evaluation['linked_udvs']} of {evaluation['udvs']} UDVs linked to"
        f" {evaluation['linked_actors']} profiled actors"
        f" in {evaluation['linked_hearings']} hearings"
    )
    print(f"  linked UDVs by tier: {evaluation['linked_udvs_by_tier']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep only the hearings of the chosen splits in the per-actor speech file."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="filtered speeches JSONL (overrides config)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output is not None:
        config = dataclasses.replace(config, output_path=args.output)
    stats = build(config)
    print_report(stats)
    print(f"wrote {config.stats_path}")


if __name__ == "__main__":
    main()
