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


@dataclass(frozen=True)
class SplitFilterConfig:
    speeches_path: Path
    lds_sha256: str
    manifest_path: Path
    splits: tuple[str, ...]
    output_path: Path
    stats_path: Path


def load_config(path: Path) -> SplitFilterConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    section = raw["split_filter"]
    splits = tuple(section["splits"])
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"splits must be distinct names among {SPLIT_NAMES}, got {splits}")
    return SplitFilterConfig(
        speeches_path=Path(raw["input"]["speeches_path"]),
        lds_sha256=raw["input"]["lds_sha256"],
        manifest_path=Path(section["manifest_path"]),
        splits=splits,
        output_path=Path(section["speeches_path"]),
        stats_path=Path(section["stats_path"]),
    )


def load_split_hearings(config: SplitFilterConfig) -> tuple[set[int], Record]:
    with open(config.manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != config.lds_sha256:
        raise SystemExit(f"{config.manifest_path} was built from another LDS file")
    hearings = {hearing_id for name in config.splits for hearing_id in manifest[name]}
    info = {
        "path": str(config.manifest_path),
        "sha256": sha256_of_file(config.manifest_path),
        "split_version": manifest["split_version"],
        "splits": list(config.splits),
        "hearings": len(hearings),
    }
    return hearings, info


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


def build(config: SplitFilterConfig) -> Record:
    hearings, manifest_info = load_split_hearings(config)
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
