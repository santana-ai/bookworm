"""Collection of person turns into per-actor speeches, with the turn policy and name merges."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from bookworm.actors.config import ActorsConfig
from bookworm.actors.schemas import (
    CHAIR_ROLE,
    SPEAKER_ROLE,
    SPEECH_SEPARATOR,
    ActorHearing,
    ActorSpeechRecord,
    ActorTurn,
    TurnRole,
    write_actor_speeches,
)
from bookworm.data.io import JsonObject, write_json
from bookworm.data.schemas import HearingRecord
from bookworm.errors import BookwormError
from bookworm.transcript.sentences import STAGE_DIRECTION_PATTERN
from bookworm.transcript.speakers import is_party_info, resolve_turn_name
from bookworm.transcript.text import normalize_name, normalize_whitespace
from bookworm.transcript.turns import Turn, split_into_turns

MERGE_DESCRIPTION = (
    "actors merged across hearings by exact normalized name (accents removed, upper case,"
    " collapsed whitespace), plus the reviewed merge groups and hearing-scoped reassignments"
    " from the merges section of the config; remaining subset-name pairs are listed for review"
)
AMBIGUITY_CRITERION = (
    "pair of distinct actor keys whose word sets are equal or in strict subset relation"
)
NON_PERSON_KEY = "non_person_key"
STAGE_DIRECTION = "stage_direction"
EMPTY = "empty"
SHORT_CHAIR = "short_chair"


class ActorSpeechError(BookwormError):
    pass


def person_name(turn: Turn) -> str:
    name = resolve_turn_name(turn)
    if name == turn.raw_name and turn.party_info and not is_party_info(turn.party_info):
        name = turn.party_info
    return normalize_whitespace(name)


def turn_party(turn: Turn) -> str | None:
    if resolve_turn_name(turn) != turn.raw_name:
        return turn.party_info.rsplit(". ", 1)[1].strip()
    if is_party_info(turn.party_info):
        return turn.party_info
    return None


def turn_words(turn: Turn) -> int:
    return len(turn.speech.split())


def turn_role(turn: Turn, config: ActorsConfig) -> TurnRole:
    return CHAIR_ROLE if turn.raw_name in config.chair_names else SPEAKER_ROLE


def drop_reason(turn: Turn, key: str, role: TurnRole, config: ActorsConfig) -> str | None:
    if key in config.non_person_keys:
        return NON_PERSON_KEY
    if STAGE_DIRECTION_PATTERN.match(turn.speech):
        return STAGE_DIRECTION
    if not turn.speech:
        return EMPTY
    if role == CHAIR_ROLE and turn_words(turn) < config.chair_min_words:
        return SHORT_CHAIR
    return None


def display_name(names: Mapping[str, int]) -> str:
    cased = {name: count for name, count in names.items() if any(c.islower() for c in name)}
    pool = cased or dict(names)
    return sorted(pool.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[0][0]


@dataclass
class ActorAccumulator:
    names: Counter[str] = field(default_factory=Counter)
    party_uf: dict[str, None] = field(default_factory=dict)
    hearings: dict[int, list[ActorTurn]] = field(default_factory=dict)

    def record(self) -> ActorSpeechRecord:
        return ActorSpeechRecord(
            actor=display_name(self.names),
            has_party_header=bool(self.party_uf),
            party_uf=list(self.party_uf),
            hearings=[
                ActorHearing(
                    hearing_id=hearing_id,
                    full_speech=SPEECH_SEPARATOR.join(turn.text for turn in turns),
                    turns=turns,
                )
                for hearing_id, turns in sorted(self.hearings.items())
            ],
        )


@dataclass(frozen=True)
class NonPersonDrop:
    key: str
    raw_name: str
    hearing_id: int


@dataclass(frozen=True)
class ActorSpeeches:
    config: ActorsConfig
    records: dict[str, ActorSpeechRecord]
    pairs: list[JsonObject]
    stats: JsonObject

    @property
    def single_hearing(self) -> list[ActorSpeechRecord]:
        return [record for record in self.records.values() if len(record.hearings) == 1]

    @property
    def multi_hearing(self) -> list[ActorSpeechRecord]:
        return [record for record in self.records.values() if len(record.hearings) >= 2]

    def display_names(self) -> dict[str, str]:
        return {key: record.actor for key, record in self.records.items()}


class ActorCollector:
    def __init__(self, config: ActorsConfig) -> None:
        self.config = config
        self.aliases = config.alias_map
        self.reassignments = config.reassignment_map
        self.actors: dict[str, ActorAccumulator] = {}
        self.alias_usage: Counter[str] = Counter()
        self.reassignment_usage: Counter[tuple[str, int]] = Counter()
        self.non_person: list[NonPersonDrop] = []
        self.dropped: Counter[str] = Counter()
        self.total_turns = 0
        self.verified_turns = 0
        self.hearings = 0

    def merged_key(self, key: str, hearing_id: int) -> str:
        target = self.reassignments.get((key, hearing_id))
        if target is not None:
            self.reassignment_usage[(key, hearing_id)] += 1
            key = target
        canonical = self.aliases.get(key)
        if canonical is not None:
            self.alias_usage[key] += 1
            key = canonical
        return key

    def add_hearing(
        self, hearing_id: int, transcript: str, turns: Sequence[Turn]
    ) -> dict[int, str]:
        self.hearings += 1
        kept: dict[int, str] = {}
        for turn in turns:
            key = self.add_turn(hearing_id, transcript, turn)
            if key is not None:
                kept[turn.turn_index] = key
        return kept

    def add_turn(self, hearing_id: int, transcript: str, turn: Turn) -> str | None:
        self.total_turns += 1
        name = person_name(turn)
        key = normalize_name(name)
        role = turn_role(turn, self.config)
        reason = drop_reason(turn, key, role, self.config)
        if reason == NON_PERSON_KEY:
            self.non_person.append(NonPersonDrop(key, turn.raw_name, hearing_id))
            return None
        if reason is not None:
            self.dropped[reason] += 1
            return None
        verify_turn_text(hearing_id, transcript, turn)
        self.verified_turns += 1
        key = self.merged_key(key, hearing_id)
        actor = self.actors.setdefault(key, ActorAccumulator())
        actor.names[name] += 1
        party = turn_party(turn)
        if party is not None:
            actor.party_uf.setdefault(party)
        actor.hearings.setdefault(hearing_id, []).append(
            ActorTurn(
                turn_index=turn.turn_index,
                role=role,
                start_char=turn.start_char,
                end_char=turn.end_char,
                text=turn.speech,
            )
        )
        return key

    def check_merge_usage(self) -> None:
        unused_aliases = sorted(set(self.aliases) - set(self.alias_usage))
        unused_reassignments = sorted(set(self.reassignments) - set(self.reassignment_usage))
        if unused_aliases or unused_reassignments:
            raise ActorSpeechError(
                f"merge config entries matched no kept turn: aliases={unused_aliases},"
                f" reassignments={unused_reassignments}"
            )

    def finish(self) -> ActorSpeeches:
        self.check_merge_usage()
        records = {key: actor.record() for key, actor in sorted(self.actors.items())}
        check_unique_display_names(records)
        pairs = ambiguous_name_pairs(records)
        stats = self.stats(list(records.values()), pairs)
        return ActorSpeeches(config=self.config, records=records, pairs=pairs, stats=stats)

    def stats(
        self, records: Sequence[ActorSpeechRecord], pairs: Sequence[JsonObject]
    ) -> JsonObject:
        config = self.config
        single = [record for record in records if len(record.hearings) == 1]
        multi = [record for record in records if len(record.hearings) >= 2]
        by_hearing_count = Counter(len(record.hearings) for record in records)
        return {
            "dataset": {
                "path": str(config.lds_path),
                "sha256": config.expected_sha256,
                "hearings": self.hearings,
            },
            "policy": {
                "chair_min_words": config.chair_min_words,
                "non_person_keys": list(config.non_person_keys),
                "merge": MERGE_DESCRIPTION,
            },
            "turns": {
                "total": self.total_turns,
                "kept": self.verified_turns,
                "verified_against_transcript": self.verified_turns,
                "dropped_non_person": non_person_summary(self.non_person),
                "dropped_stage_direction": self.dropped[STAGE_DIRECTION],
                "dropped_empty": self.dropped[EMPTY],
                "dropped_short_chair": self.dropped[SHORT_CHAIR],
            },
            "merges": {
                "groups": len(set(self.aliases.values())),
                "alias_keys": len(self.aliases),
                "alias_turns_kept": sum(self.alias_usage.values()),
                "reassignments": len(self.reassignments),
                "reassigned_turns_kept": sum(self.reassignment_usage.values()),
            },
            "hearings_per_actor": {str(k): v for k, v in sorted(by_hearing_count.items())},
            "files": {
                "single_hearing": file_stats(single, str(config.single_hearing_path)),
                "multi_hearing": file_stats(multi, str(config.multi_hearing_path)),
            },
            "ambiguous_name_pairs": {
                "path": str(config.ambiguous_names_path),
                "criterion": AMBIGUITY_CRITERION,
                "pairs": len(pairs),
                "equal_tokens": sum(pair["relation"] == "equal_tokens" for pair in pairs),
                "name_subset": sum(pair["relation"] == "name_subset" for pair in pairs),
            },
        }


def verify_turn_text(hearing_id: int, transcript: str, turn: Turn) -> None:
    if transcript[turn.start_char : turn.end_char] != turn.speech:
        raise ActorSpeechError(f"turn text mismatch: hearing {hearing_id} turn {turn.turn_index}")


def check_unique_display_names(records: Mapping[str, ActorSpeechRecord]) -> None:
    keys_by_name: dict[str, list[str]] = {}
    for key, record in records.items():
        keys_by_name.setdefault(record.actor, []).append(key)
    shared = {name: keys for name, keys in keys_by_name.items() if len(keys) > 1}
    if shared:
        raise ActorSpeechError(f"actor display names shared by several keys: {shared}")


def pair_side(key: str, record: ActorSpeechRecord) -> JsonObject:
    return {
        "key": key,
        "actor": record.actor,
        "has_party_header": record.has_party_header,
        "party_uf": record.party_uf,
        "hearing_ids": record.hearing_ids,
        "turns": sum(len(hearing.turns) for hearing in record.hearings),
    }


def ambiguous_name_pairs(records: Mapping[str, ActorSpeechRecord]) -> list[JsonObject]:
    keys = sorted(records)
    tokens = {key: set(key.split()) for key in keys}
    pairs: list[JsonObject] = []
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            tokens_a, tokens_b = tokens[key_a], tokens[key_b]
            if not (tokens_a <= tokens_b or tokens_b <= tokens_a):
                continue
            record_a, record_b = records[key_a], records[key_b]
            pairs.append(
                {
                    "relation": "equal_tokens" if tokens_a == tokens_b else "name_subset",
                    "a": pair_side(key_a, record_a),
                    "b": pair_side(key_b, record_b),
                    "shared_hearing_ids": sorted(
                        set(record_a.hearing_ids) & set(record_b.hearing_ids)
                    ),
                }
            )
    return pairs


def words_by_role(records: Iterable[ActorSpeechRecord]) -> dict[TurnRole, int]:
    words: dict[TurnRole, int] = {SPEAKER_ROLE: 0, CHAIR_ROLE: 0}
    for record in records:
        for hearing in record.hearings:
            for turn in hearing.turns:
                words[turn.role] += len(turn.text.split())
    return words


def file_stats(records: Sequence[ActorSpeechRecord], path: str) -> JsonObject:
    words = words_by_role(records)
    return {
        "path": path,
        "actors": len(records),
        "with_party_header": sum(record.has_party_header for record in records),
        "without_party_header": sum(not record.has_party_header for record in records),
        "speaker_words": words[SPEAKER_ROLE],
        "chair_words": words[CHAIR_ROLE],
    }


def non_person_summary(drops: Sequence[NonPersonDrop]) -> JsonObject:
    by_key = Counter(drop.key for drop in drops)
    return {
        "turns_dropped": len(drops),
        "by_key": dict(sorted(by_key.items())),
        "turns": [
            {"key": drop.key, "raw_name": drop.raw_name, "hearing_id": drop.hearing_id}
            for drop in drops
        ],
    }


def collect_actor_speeches(
    hearings: Iterable[HearingRecord], config: ActorsConfig
) -> ActorSpeeches:
    """Collect the per-actor speeches of the hearings without building UDVs."""
    collector = ActorCollector(config)
    for hearing in hearings:
        collector.add_hearing(
            hearing.id, hearing.transcricao, split_into_turns(hearing.transcricao)
        )
    return collector.finish()


def write_actor_outputs(speeches: ActorSpeeches) -> None:
    config = speeches.config
    write_actor_speeches(speeches.single_hearing, config.single_hearing_path)
    write_actor_speeches(speeches.multi_hearing, config.multi_hearing_path)
    write_json(
        {"criterion": AMBIGUITY_CRITERION, "pairs": speeches.pairs}, config.ambiguous_names_path
    )
    write_json(speeches.stats, config.stats_path)
