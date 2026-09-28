from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import LDS_SHA256, udv_artifact_path

from bookworm import HearingRecord, UdvRecord, load_udv_jsonl, resolve_hearing_people
from bookworm.actors.config import load_actors_config
from bookworm.actors.schemas import ActorSpeechRecord
from bookworm.actors.speeches import ActorCollector
from bookworm.pipeline import PendingLink, pending_links, resolve_link
from bookworm.profiles.split_filter import (
    filter_speeches,
    load_split_selection,
    summarize_evaluation,
)
from bookworm.transcript.turns import split_into_turns

pytestmark = pytest.mark.dataset

UDV_RUN = "udv_v1.jsonl"
MANIFEST = "temporal_v1.json"
EXTRA_LIBRARY_LINKS = {
    "udv-6-1-0": "Aureo Ribeiro",
    "udv-6-1-1": "Aureo Ribeiro",
    "udv-58-2-0": "Soraya Santos",
    "udv-117-0-0": "Alfredo Gaspar",
    "udv-195-2-2": "KRISZTIAN KATONA",
}
UPSTREAM_NAME_CASES = {
    "udv-40-2-0": "TOINHO DO JUDÔ",
    "udv-40-2-1": "TOINHO DO JUDÔ",
    "udv-40-2-2": "TOINHO DO JUDÔ",
    "udv-146-2-0": "ANGELA ABOIN",
    "udv-206-5-0": "ZARA FIGUEIREDO",
    "udv-206-5-1": "ZARA FIGUEIREDO",
}


@dataclass(frozen=True)
class LinkRules:
    udvs: dict[str, UdvRecord]
    library: dict[str, str]
    evidence_turn: dict[str, str]
    matched_turns: dict[str, frozenset[int]]
    multi_hearing: list[ActorSpeechRecord]
    train: frozenset[int]
    test: frozenset[int]


@pytest.fixture(scope="module")
def rules(
    lds_hearings: list[HearingRecord],
    challenge_dir: Path,
    udv_artifacts_dir: Path,
    split_artifacts_dir: Path,
) -> LinkRules:
    config_path = challenge_dir / "configs" / "hearing_actors.toml"
    if not config_path.is_file():
        pytest.skip(f"actors config not found at {config_path}")
    collector = ActorCollector(load_actors_config(config_path))
    pending: list[PendingLink] = []
    kept_by_hearing: dict[int, dict[int, str]] = {}
    matched_turns: dict[str, frozenset[int]] = {}
    for hearing in lds_hearings:
        turns = split_into_turns(hearing.transcricao)
        people = resolve_hearing_people(hearing, turns)
        kept = collector.add_hearing(hearing.id, hearing.transcricao, turns)
        kept_by_hearing[hearing.id] = dict(kept)
        hearing_pending = pending_links(hearing.id, people, kept)
        pending.extend(hearing_pending)
        for link, person_turns in zip(
            hearing_pending,
            (
                frozenset(turn.turn_index for turn in person.matched_turns)
                for person in people
                for _ in person.participant.opinioes
            ),
            strict=True,
        ):
            matched_turns[link.udv_id] = person_turns
    actors = collector.finish()
    manifest_path = split_artifacts_dir / MANIFEST
    names = actors.display_names()
    udvs = {udv.id: udv for udv in load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, UDV_RUN))}
    library = {
        link.udv_id: link.actor
        for link in (resolve_link(item, names) for item in pending)
        if link.actor is not None
    }
    return LinkRules(
        udvs=udvs,
        library=library,
        evidence_turn=evidence_turn_links(udvs.values(), kept_by_hearing, names),
        matched_turns=matched_turns,
        multi_hearing=actors.multi_hearing,
        train=load_split_selection(manifest_path, ["train"], LDS_SHA256).hearing_ids,
        test=load_split_selection(manifest_path, ["test"], LDS_SHA256).hearing_ids,
    )


def evidence_turn_links(
    udvs: Iterable[UdvRecord],
    kept_by_hearing: Mapping[int, Mapping[int, str]],
    names: Mapping[str, str],
) -> dict[str, str]:
    links: dict[str, str] = {}
    for udv in udvs:
        if udv.evidence is None or udv.evidence.speaker_turn is None:
            continue
        key = kept_by_hearing[udv.hearing_id].get(udv.evidence.speaker_turn)
        if key is not None:
            links[udv.id] = names[key]
    return links


def test_both_rules_cover_the_same_udvs(rules: LinkRules) -> None:
    assert set(rules.matched_turns) == set(rules.udvs)
    assert (len(rules.udvs), len(rules.library), len(rules.evidence_turn)) == (2203, 2104, 2099)


def test_rules_never_disagree_when_both_link(rules: LinkRules) -> None:
    both = set(rules.library) & set(rules.evidence_turn)
    assert both == set(rules.evidence_turn)
    assert [uid for uid in sorted(both) if rules.library[uid] != rules.evidence_turn[uid]] == []


def test_extra_library_links_have_a_dropped_evidence_turn(rules: LinkRules) -> None:
    extra = set(rules.library) - set(rules.evidence_turn)
    assert {uid: rules.library[uid] for uid in extra} == EXTRA_LIBRARY_LINKS
    for uid in extra:
        evidence = rules.udvs[uid].evidence
        assert evidence is not None
        assert evidence.speaker_turn is not None
        assert evidence.speaker_turn in rules.matched_turns[uid]


def test_evidence_turn_is_always_a_matched_turn(rules: LinkRules) -> None:
    outside = [
        uid
        for uid, udv in rules.udvs.items()
        if udv.evidence is not None
        and udv.evidence.speaker_turn is not None
        and udv.evidence.speaker_turn not in rules.matched_turns[uid]
    ]
    assert outside == []


def test_name_resolution_cases_are_shared_by_both_rules(rules: LinkRules) -> None:
    for uid, actor in UPSTREAM_NAME_CASES.items():
        assert (rules.library[uid], rules.evidence_turn[uid]) == (actor, actor), uid


def test_both_rules_give_the_same_test_split_counts(rules: LinkRules) -> None:
    train, test = rules.train, rules.test
    profiled = {record.actor for record in filter_speeches(rules.multi_hearing, train)}
    for links in (rules.library, rules.evidence_turn):
        linked = {
            uid: actor
            for uid, actor in links.items()
            if rules.udvs[uid].hearing_id in test and actor in profiled
        }
        hearings = {rules.udvs[uid].hearing_id for uid in linked}
        assert (len(linked), len(set(linked.values())), len(hearings)) == (106, 48, 27)
    evaluation = summarize_evaluation(
        rules.multi_hearing,
        filter_speeches(rules.multi_hearing, train),
        list(rules.udvs.values()),
        test,
    )
    counts = (
        evaluation["linked_udvs"],
        evaluation["linked_actors"],
        evaluation["linked_hearings"],
    )
    assert counts == (106, 48, 27)
