from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from bookworm.actors.schemas import ActorSpeechRecord, read_actor_speeches, write_actor_speeches
from bookworm.data.io import JsonObject, read_json_object, sha256_of_file, write_json
from bookworm.errors import ConfigError
from bookworm.profiles.config import SplitFilterConfig
from bookworm.udv.schemas import UdvRecord, load_udv_jsonl

LINK_DESCRIPTION = (
    "a UDV of an evaluation hearing is linked to the actor who owns its evidence turn"
    " (hearing_id, evidence.speaker_turn) in the unfiltered speech file; UDVs without an"
    " evidence turn, or whose turn was dropped when the speech file was built, stay unlinked"
)


@dataclass(frozen=True)
class SplitSelection:
    hearing_ids: frozenset[int]
    manifest: JsonObject


def split_hearing_ids(manifest: JsonObject, path: Path, splits: Sequence[str]) -> frozenset[int]:
    hearing_ids: set[int] = set()
    for name in splits:
        ids = manifest.get(name)
        if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
            raise ConfigError(f"{path}: split manifest has no hearing list for {name!r}")
        hearing_ids.update(ids)
    return frozenset(hearing_ids)


def read_checked_manifest(path: Path, lds_sha256: str) -> JsonObject:
    manifest = read_json_object(path, "split manifest")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("sha256") != lds_sha256:
        raise ConfigError(f"{path} was built from another LDS file")
    return manifest


def selection_from_manifest(
    manifest: JsonObject, path: Path, splits: Sequence[str]
) -> SplitSelection:
    split_version = manifest.get("split_version")
    if not isinstance(split_version, str):
        raise ConfigError(f"{path}: split manifest has no split_version")
    hearing_ids = split_hearing_ids(manifest, path, splits)
    return SplitSelection(
        hearing_ids=hearing_ids,
        manifest={
            "path": str(path),
            "sha256": sha256_of_file(path),
            "split_version": split_version,
            "splits": list(splits),
            "hearings": len(hearing_ids),
        },
    )


def load_split_selection(path: Path, splits: Sequence[str], lds_sha256: str) -> SplitSelection:
    return selection_from_manifest(read_checked_manifest(path, lds_sha256), path, splits)


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


def load_udvs(path: Path) -> list[UdvRecord]:
    if not path.is_file():
        raise ConfigError(f"{path}: UDV file not found")
    try:
        return load_udv_jsonl(path)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read UDVs: {error}") from error


def evidence_turn_owners(
    records: Iterable[ActorSpeechRecord], hearing_ids: frozenset[int]
) -> dict[tuple[int, int], str]:
    return {
        (hearing.hearing_id, turn.turn_index): record.actor
        for record in records
        for hearing in record.hearings
        if hearing.hearing_id in hearing_ids
        for turn in hearing.turns
    }


def summarize_evaluation(
    records: Sequence[ActorSpeechRecord],
    filtered: Sequence[ActorSpeechRecord],
    udvs: Sequence[UdvRecord],
    eval_hearings: frozenset[int],
) -> JsonObject:
    profiled = {record.actor for record in filtered}
    owners = evidence_turn_owners(records, eval_hearings)
    speaking = {actor for actor in owners.values() if actor in profiled}
    linked: list[tuple[UdvRecord, str]] = []
    for udv in udvs:
        evidence = udv.evidence
        if udv.hearing_id not in eval_hearings or evidence is None:
            continue
        if evidence.speaker_turn is None:
            continue
        actor = owners.get((udv.hearing_id, evidence.speaker_turn))
        if actor is not None and actor in profiled:
            linked.append((udv, actor))
    tiers = Counter(udv.tier for udv, _ in linked)
    return {
        "hearings": len(eval_hearings),
        "profiled_actors_speaking": len(speaking),
        "udvs": sum(udv.hearing_id in eval_hearings for udv in udvs),
        "linked_udvs": len(linked),
        "linked_actors": len({actor for _, actor in linked}),
        "linked_hearings": len({udv.hearing_id for udv, _ in linked}),
        "linked_udvs_by_tier": dict(sorted(tiers.items())),
    }


def build_split_filter(config: SplitFilterConfig) -> JsonObject:
    manifest = read_checked_manifest(config.manifest_path, config.lds_sha256)
    selection = selection_from_manifest(manifest, config.manifest_path, config.splits)
    eval_hearings = split_hearing_ids(manifest, config.manifest_path, config.eval_splits)
    records = load_speeches(config.speeches_path)
    udvs = load_udvs(config.udv_path)
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
        "evaluation": {
            "splits": list(config.eval_splits),
            "udv_path": str(config.udv_path),
            "udv_sha256": sha256_of_file(config.udv_path),
            "link": LINK_DESCRIPTION,
            **summarize_evaluation(records, filtered, udvs, eval_hearings),
        },
    }
    write_json(stats, config.stats_path)
    return stats
