import copy
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES_DIR

from bookworm import ConfigError, HearingRecord, Turn, load_hearings, split_into_turns
from bookworm.actors import (
    ActorHearing,
    ActorSpeechRecord,
    ActorTurn,
    UdvActorLink,
    read_actor_speeches,
    read_udv_actor_links,
    write_actor_speeches,
    write_udv_actor_links,
)
from bookworm.actors.config import ActorsConfig, load_actors_config
from bookworm.actors.speeches import (
    AMBIGUITY_CRITERION,
    ActorCollector,
    ActorSpeechError,
    ActorSpeeches,
    check_unique_display_names,
    collect_actor_speeches,
    display_name,
    drop_reason,
    person_name,
    turn_party,
    turn_role,
    write_actor_outputs,
)

ACTORS_CONFIG = FIXTURES_DIR / "actors_mini.toml"
ACTORS_LDS = FIXTURES_DIR / "lds_actors.jsonl"
EXPECTED_KEYS = [
    "ANA SOUZA",
    "DEP. HELENA PRADO",
    "JOAO SILVA",
    "MARCOS PEREIRA",
    "PEREIRA",
    "SOUZA ANA",
]


def raw_config() -> dict[str, Any]:
    return tomllib.loads(ACTORS_CONFIG.read_text(encoding="utf-8"))


def make_turn(raw_name: str, party_info: str = "", speech: str = "Uma fala qualquer.") -> Turn:
    return Turn(
        turn_index=0,
        raw_name=raw_name,
        party_info=party_info,
        speech=speech,
        start_char=0,
        end_char=len(speech),
    )


@pytest.fixture(scope="module")
def config() -> ActorsConfig:
    return load_actors_config(ACTORS_CONFIG)


@pytest.fixture(scope="module")
def hearings() -> list[HearingRecord]:
    return load_hearings(ACTORS_LDS, load_actors_config(ACTORS_CONFIG).expected_sha256)


@pytest.fixture(scope="module")
def speeches(hearings: list[HearingRecord], config: ActorsConfig) -> ActorSpeeches:
    return collect_actor_speeches(hearings, config)


def test_config_reads_every_table(config: ActorsConfig) -> None:
    assert config.lds_path == Path("lds_actors.jsonl")
    assert config.chair_names == ("PRESIDENTE", "PRESIDENTA")
    assert config.non_person_keys == ("INTERPRETE",)
    assert config.chair_min_words == 10
    assert config.output_paths == (
        Path("actors/single.jsonl"),
        Path("actors/multi.jsonl"),
        Path("actors/ambiguous_names.json"),
        Path("actors/stats.json"),
    )
    assert config.alias_map == {"JOAO": "JOAO SILVA"}
    assert config.reassignment_map == {("MARCOS", 11): "MARCOS PEREIRA"}
    assert config.source["measurement"] == {"output_path": "ignored.json"}


def test_config_without_merges_has_no_aliases() -> None:
    raw = raw_config()
    del raw["merges"]
    config = ActorsConfig.from_mapping(raw)
    assert config.alias_map == {}
    assert config.reassignment_map == {}


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("dataset", "sha256"),
        ("speakers", "chair_min_words"),
        ("speeches", "stats_path"),
    ],
)
def test_config_requires_each_key(table: str, key: str) -> None:
    raw = raw_config()
    del raw[table][key]
    with pytest.raises(ConfigError, match=rf"missing key \[{table}\]\.{key}"):
        ActorsConfig.from_mapping(raw, origin="actors.toml")


def test_config_rejects_an_alias_in_two_groups() -> None:
    raw = raw_config()
    raw["merges"]["groups"].append({"canonical": "OUTRO", "aliases": ["JOAO"]})
    with pytest.raises(ConfigError, match="more than one group"):
        ActorsConfig.from_mapping(raw)


def test_config_rejects_a_repeated_reassignment() -> None:
    raw = raw_config()
    raw["merges"]["reassignments"].append(copy.deepcopy(raw["merges"]["reassignments"][0]))
    with pytest.raises(ConfigError, match="reassigned more than once"):
        ActorsConfig.from_mapping(raw)


def test_config_rejects_a_wrong_type() -> None:
    raw = raw_config()
    raw["speakers"]["chair_min_words"] = "ten"
    with pytest.raises(ConfigError, match="chair_min_words"):
        ActorsConfig.from_mapping(raw)


def test_person_name_prefers_the_name_inside_a_chair_header() -> None:
    assert person_name(make_turn("PRESIDENTE", "Dep. Helena Prado. PSB - PE")) == (
        "Dep. Helena Prado"
    )
    assert person_name(make_turn("PRESIDENTE", "João  Silva")) == "João Silva"
    assert person_name(make_turn("JOÃO SILVA", "Bloco/PT - SP")) == "JOÃO SILVA"
    assert person_name(make_turn("MARCOS PEREIRA")) == "MARCOS PEREIRA"


def test_turn_party_reads_party_and_state() -> None:
    assert turn_party(make_turn("PRESIDENTE", "Dep. Helena Prado. PSB - PE")) == "PSB - PE"
    assert turn_party(make_turn("JOÃO SILVA", "Bloco/PT - SP")) == "Bloco/PT - SP"
    assert turn_party(make_turn("PRESIDENTE", "João Silva")) is None
    assert turn_party(make_turn("MARCOS PEREIRA")) is None


def test_turn_role_uses_the_raw_header_name(config: ActorsConfig) -> None:
    assert turn_role(make_turn("PRESIDENTA", "Ana"), config) == "chair"
    assert turn_role(make_turn("ANA"), config) == "speaker"


@pytest.mark.parametrize(
    ("turn", "key", "reason"),
    [
        (make_turn("INTÉRPRETE"), "INTERPRETE", "non_person_key"),
        (make_turn("ANA", speech="(Palmas.) (Risos.)"), "ANA", "stage_direction"),
        (make_turn("ANA", speech=""), "ANA", "empty"),
        (make_turn("PRESIDENTE", "Ana", speech="Com a palavra."), "ANA", "short_chair"),
        (make_turn("PRESIDENTE", "Ana", speech=" ".join(["palavra"] * 10)), "ANA", None),
        (make_turn("ANA", speech="Curta."), "ANA", None),
    ],
)
def test_drop_reason(turn: Turn, key: str, reason: str | None, config: ActorsConfig) -> None:
    assert drop_reason(turn, key, turn_role(turn, config), config) == reason


def test_display_name_prefers_cased_then_frequent_then_long() -> None:
    assert display_name({"JOÃO SILVA": 5, "João Silva": 1}) == "João Silva"
    assert display_name({"JOÃO SILVA": 2, "JOÃO": 2}) == "JOÃO SILVA"
    assert display_name({"JOÃO": 3, "JOÃO SILVA": 2}) == "JOÃO"
    assert display_name({"ANA B": 1, "ANA A": 1}) == "ANA A"


def test_collection_groups_turns_by_merged_key(speeches: ActorSpeeches) -> None:
    records = speeches.records
    assert list(records) == EXPECTED_KEYS
    assert speeches.display_names()["JOAO SILVA"] == "João Silva"
    joao = records["JOAO SILVA"]
    assert joao.actor == "João Silva"
    assert joao.party_uf == ["Bloco/PT - SP"]
    assert joao.has_party_header
    assert [
        (hearing.hearing_id, [(turn.turn_index, turn.role) for turn in hearing.turns])
        for hearing in joao.hearings
    ] == [(10, [(1, "speaker")]), (11, [(0, "chair"), (2, "speaker")])]
    marcos = records["MARCOS PEREIRA"]
    assert marcos.hearing_ids == [10, 11]
    assert not marcos.has_party_header
    helena = records["DEP. HELENA PRADO"]
    assert [turn.turn_index for turn in helena.hearings[0].turns] == [4]


def test_full_speech_joins_the_turn_texts(
    speeches: ActorSpeeches, hearings: list[HearingRecord]
) -> None:
    transcripts = {hearing.id: hearing.transcricao for hearing in hearings}
    for record in speeches.records.values():
        for hearing in record.hearings:
            assert hearing.full_speech == "\n\n".join(turn.text for turn in hearing.turns)
            for turn in hearing.turns:
                assert transcripts[hearing.hearing_id][turn.start_char : turn.end_char] == (
                    turn.text
                )


def test_single_and_multi_hearing_split(speeches: ActorSpeeches) -> None:
    assert [record.actor for record in speeches.multi_hearing] == [
        "João Silva",
        "MARCOS PEREIRA",
    ]
    assert [record.actor for record in speeches.single_hearing] == [
        "ANA SOUZA",
        "Dep. Helena Prado",
        "PEREIRA",
        "SOUZA ANA",
    ]


def test_stats_count_every_drop_and_merge(speeches: ActorSpeeches) -> None:
    stats = speeches.stats
    assert stats["dataset"] == {
        "path": "lds_actors.jsonl",
        "sha256": "bb3b7cdd7aa8cf9befb7932f190ca955d0f3e43dc6803bc883ae36d5252c5be1",
        "hearings": 2,
    }
    assert stats["turns"] == {
        "total": 13,
        "kept": 9,
        "verified_against_transcript": 9,
        "dropped_non_person": {
            "turns_dropped": 1,
            "by_key": {"INTERPRETE": 1},
            "turns": [{"key": "INTERPRETE", "raw_name": "INTÉRPRETE", "hearing_id": 10}],
        },
        "dropped_stage_direction": 1,
        "dropped_empty": 1,
        "dropped_short_chair": 1,
    }
    assert stats["merges"] == {
        "groups": 1,
        "alias_keys": 1,
        "alias_turns_kept": 1,
        "reassignments": 1,
        "reassigned_turns_kept": 1,
    }
    assert stats["hearings_per_actor"] == {"1": 4, "2": 2}
    assert stats["files"]["multi_hearing"] == {
        "path": "actors/multi.jsonl",
        "actors": 2,
        "with_party_header": 1,
        "without_party_header": 1,
        "speaker_words": 47,
        "chair_words": 16,
    }
    assert stats["ambiguous_name_pairs"]["pairs"] == 2


def test_ambiguous_pairs_list_equal_and_subset_names(speeches: ActorSpeeches) -> None:
    assert [
        (pair["relation"], pair["a"]["key"], pair["b"]["key"], pair["shared_hearing_ids"])
        for pair in speeches.pairs
    ] == [
        ("equal_tokens", "ANA SOUZA", "SOUZA ANA", [11]),
        ("name_subset", "MARCOS PEREIRA", "PEREIRA", [10]),
    ]
    assert speeches.pairs[1]["a"]["turns"] == 2
    assert speeches.pairs[1]["a"]["hearing_ids"] == [10, 11]


def test_unused_merge_entries_are_an_error(
    hearings: list[HearingRecord], config: ActorsConfig
) -> None:
    with pytest.raises(ActorSpeechError, match="matched no kept turn"):
        collect_actor_speeches(hearings[:1], config)


def test_display_names_must_be_unique() -> None:
    record = ActorSpeechRecord(actor="Ana", has_party_header=False, party_uf=[], hearings=[])
    with pytest.raises(ActorSpeechError, match="shared by several keys"):
        check_unique_display_names({"ANA": record, "ANA A": record})


def test_turn_text_must_match_the_transcript(config: ActorsConfig) -> None:
    collector = ActorCollector(config)
    turn = make_turn("ANA", speech="Texto que não está na transcrição.")
    with pytest.raises(ActorSpeechError, match="turn text mismatch: hearing 7 turn 0"):
        collector.add_hearing(7, "outro texto", [turn])


def test_add_hearing_returns_the_key_of_each_kept_turn(
    hearings: list[HearingRecord], config: ActorsConfig
) -> None:
    collector = ActorCollector(config)
    second = hearings[1]
    kept = collector.add_hearing(
        second.id, second.transcricao, split_into_turns(second.transcricao)
    )
    assert kept == {
        0: "JOAO SILVA",
        1: "MARCOS PEREIRA",
        2: "JOAO SILVA",
        3: "ANA SOUZA",
        4: "SOUZA ANA",
    }


def test_outputs_are_written_at_the_configured_paths(
    speeches: ActorSpeeches, config: ActorsConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_actor_outputs(speeches)
    assert read_actor_speeches(tmp_path / "actors" / "multi.jsonl") == speeches.multi_hearing
    assert read_actor_speeches(tmp_path / "actors" / "single.jsonl") == speeches.single_hearing
    ambiguous = json.loads((tmp_path / "actors" / "ambiguous_names.json").read_text("utf-8"))
    assert ambiguous == {"criterion": AMBIGUITY_CRITERION, "pairs": speeches.pairs}
    stats = json.loads((tmp_path / "actors" / "stats.json").read_text("utf-8"))
    assert stats == speeches.stats


def test_speech_records_round_trip_in_the_script_field_order(tmp_path: Path) -> None:
    record = ActorSpeechRecord(
        actor="Ana Lima",
        has_party_header=True,
        party_uf=["PT - BA"],
        hearings=[
            ActorHearing(
                hearing_id=4,
                full_speech="Olá.",
                turns=[
                    ActorTurn(turn_index=2, role="speaker", start_char=5, end_char=9, text="Olá.")
                ],
            )
        ],
    )
    path = tmp_path / "speeches.jsonl"
    write_actor_speeches([record], path)
    line = path.read_text(encoding="utf-8")
    assert line.startswith('{"actor": "Ana Lima", "has_party_header": true, "party_uf"')
    assert "Olá" in line
    assert read_actor_speeches(path) == [record]


def test_links_round_trip(tmp_path: Path) -> None:
    links = [
        UdvActorLink(
            udv_id="udv-1-0-0",
            hearing_id=1,
            actor_key="ANA LIMA",
            actor="Ana Lima",
            matched_turns=2,
            linked_turns=2,
        ),
        UdvActorLink(
            udv_id="udv-1-1-0",
            hearing_id=1,
            actor_key=None,
            actor=None,
            matched_turns=0,
            linked_turns=0,
        ),
    ]
    path = tmp_path / "links.jsonl"
    write_udv_actor_links(links, path)
    assert read_udv_actor_links(path) == links


def test_schemas_reject_unknown_roles_and_fields() -> None:
    with pytest.raises(ValueError, match="role"):
        ActorTurn.model_validate(
            {"turn_index": 0, "role": "guest", "start_char": 0, "end_char": 1, "text": "a"}
        )
    with pytest.raises(ValueError, match="extra"):
        UdvActorLink.model_validate(
            {
                "udv_id": "udv-1-0-0",
                "hearing_id": 1,
                "actor_key": None,
                "actor": None,
                "matched_turns": 0,
                "linked_turns": 0,
                "tier": "quote_found",
            }
        )
