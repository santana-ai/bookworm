import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from bookworm.actors.schemas import ActorSpeechRecord, read_actor_speeches, write_actor_speeches
from bookworm.data.io import JsonObject, sha256_of_file, write_json
from bookworm.errors import ConfigError
from bookworm.profiles.config import SplitFilterConfig


@dataclass(frozen=True)
class SplitSelection:
    hearing_ids: frozenset[int]
    manifest: JsonObject


def read_manifest(path: Path) -> JsonObject:
    if not path.is_file():
        raise ConfigError(f"{path}: split manifest not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read split manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: split manifest is not a JSON object")
    return payload


def load_split_selection(path: Path, splits: Sequence[str], lds_sha256: str) -> SplitSelection:
    manifest = read_manifest(path)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("sha256") != lds_sha256:
        raise ConfigError(f"{path} was built from another LDS file")
    split_version = manifest.get("split_version")
    if not isinstance(split_version, str):
        raise ConfigError(f"{path}: split manifest has no split_version")
    hearing_ids: set[int] = set()
    for name in splits:
        ids = manifest.get(name)
        if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
            raise ConfigError(f"{path}: split manifest has no hearing list for {name!r}")
        hearing_ids.update(ids)
    return SplitSelection(
        hearing_ids=frozenset(hearing_ids),
        manifest={
            "path": str(path),
            "sha256": sha256_of_file(path),
            "split_version": split_version,
            "splits": list(splits),
            "hearings": len(hearing_ids),
        },
    )


def filter_speech(
    record: ActorSpeechRecord, hearing_ids: frozenset[int]
) -> ActorSpeechRecord | None:
    kept = [hearing for hearing in record.hearings if hearing.hearing_id in hearing_ids]
    if not kept:
        return None
    return record.model_copy(update={"hearings": kept})


def filter_speeches(
    records: Iterable[ActorSpeechRecord], hearing_ids: frozenset[int]
) -> list[ActorSpeechRecord]:
    return [
        kept
        for kept in (filter_speech(record, hearing_ids) for record in records)
        if kept is not None
    ]


def summarize_speeches(records: Sequence[ActorSpeechRecord]) -> JsonObject:
    per_actor = Counter(len(record.hearings) for record in records)
    return {
        "actors": len(records),
        "hearings": len({hearing_id for record in records for hearing_id in record.hearing_ids}),
        "turns": sum(len(hearing.turns) for record in records for hearing in record.hearings),
        "hearings_per_actor": {str(count): actors for count, actors in sorted(per_actor.items())},
    }


def load_speeches(path: Path) -> list[ActorSpeechRecord]:
    if not path.is_file():
        raise ConfigError(f"{path}: actor speeches file not found")
    try:
        return read_actor_speeches(path)
    except (OSError, ValueError, ValidationError) as error:
        raise ConfigError(f"{path}: cannot read actor speeches: {error}") from error


def build_split_filter(config: SplitFilterConfig) -> JsonObject:
    selection = load_split_selection(config.manifest_path, config.splits, config.lds_sha256)
    records = load_speeches(config.speeches_path)
    filtered = filter_speeches(records, selection.hearing_ids)
    write_actor_speeches(filtered, config.output_path)
    stats: JsonObject = {
        "split_manifest": selection.manifest,
        "input": {
            "path": str(config.speeches_path),
            "sha256": sha256_of_file(config.speeches_path),
            **summarize_speeches(records),
        },
        "output": {
            "path": str(config.output_path),
            "sha256": sha256_of_file(config.output_path),
            **summarize_speeches(filtered),
            "actors_without_split_hearings": len(records) - len(filtered),
        },
    }
    write_json(stats, config.stats_path)
    return stats
