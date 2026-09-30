import copy
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CROSS_TURN_TEXT,
    JOAO_FIRST_SENTENCE,
    MINI_THRESHOLD,
    TURNS_PHYSICS_SENTENCE,
    TURNS_REFORM_TURN_1,
    TURNS_TRAINING_SENTENCE,
    TURNS_VECTORS,
    StubEncoder,
)

from bookworm import (
    Actor,
    CachedEncoder,
    ConfigError,
    EvidenceSettings,
    HearingRecord,
    Method,
    UdvRecord,
    UdvRun,
    build_udvs,
    compare_with_baseline,
    locate_turn_sentence_span,
    split_into_turns,
    summarize_run,
    verify_udv_run,
)
from bookworm.udv.verify import (
    coverage_hearing_ids,
    coverage_hearings,
    coverage_threshold,
    expected_ids,
    index_people,
)

Records = list[UdvRecord]
CARLOS_SENTENCE = "Começamos agora o debate sobre as cooperativas pequenas."


@dataclass(frozen=True)
class MiniRun:
    run: UdvRun
    source: dict[str, Any]
    hearings: list[HearingRecord]

    def coverage(self, records: Records) -> dict[str, Any]:
        return summarize_run(
            "mini",
            records,
            self.run.people,
            self.run.hearings,
            encoder_runtime={},
            hearing_seconds=self.run.hearing_seconds,
            config_source=self.source,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            environment={},
        )

    def problems(self, records: Records) -> dict[str, list[str]]:
        return verify_udv_run("mini", records, self.coverage(records), self.hearings).problems


def build_mini_run(
    hearings: list[HearingRecord], encoder: StubEncoder, config_path: Path
) -> MiniRun:
    run = build_udvs(hearings, CachedEncoder(encoder), EvidenceSettings(MINI_THRESHOLD))
    source = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return MiniRun(run=run, source=source, hearings=hearings)


@pytest.fixture
def mini(mini_hearing_list: list[HearingRecord], udv_mini_config_path: Path) -> MiniRun:
    return build_mini_run(mini_hearing_list, StubEncoder(), udv_mini_config_path)


@pytest.fixture
def turns(turns_hearing: HearingRecord, udv_mini_config_path: Path) -> MiniRun:
    return build_mini_run([turns_hearing], StubEncoder(TURNS_VECTORS), udv_mini_config_path)


def mutate(
    records: Records,
    record_id: str,
    *,
    evidence_update: dict[str, Any] | None = None,
    **fields: Any,
) -> Records:
    mutated: Records = []
    for record in records:
        if record.id == record_id:
            update = dict(fields)
            if evidence_update is not None:
                assert record.evidence is not None
                update["evidence"] = record.evidence.model_copy(update=evidence_update)
            record = record.model_copy(update=update)
        mutated.append(record)
    return mutated


Mutation = Callable[[Records], Records]

MUTATIONS: dict[str, tuple[Mutation, dict[str, list[str]]]] = {
    "duplicate": (
        lambda records: [*records, records[0]],
        {"duplicate_id": ["udv-1-0-0"]},
    ),
    "missing": (
        lambda records: [record for record in records if record.id != "udv-2-7-0"],
        {"missing_id": ["udv-2-7-0"]},
    ),
    "unexpected": (
        lambda records: mutate(records, "udv-2-7-0", id="udv-9-0-0"),
        {"missing_id": ["udv-2-7-0"], "unexpected_id": ["udv-9-0-0"]},
    ),
    "proposition": (
        lambda records: mutate(records, "udv-1-0-1", proposition="Outra opinião qualquer."),
        {"proposition_mismatch": ["udv-1-0-1"]},
    ),
    "actor": (
        lambda records: mutate(
            records, "udv-1-0-1", actor=Actor(name="João Silva", role="Senador (PT-SP)")
        ),
        {"actor_mismatch": ["udv-1-0-1"]},
    ),
    "method_threshold": (
        lambda records: mutate(
            records,
            "udv-1-0-1",
            method=Method(
                encoder="stub-encoder", revision="stub-revision-1", embedding_threshold=0.5
            ),
        ),
        {"method_threshold_mismatch": ["udv-1-0-1"]},
    ),
    "not_resolved_but_matched": (
        lambda records: mutate(records, "udv-2-5-0", tier="person_not_resolved"),
        {"person_not_resolved_but_matched": ["udv-2-5-0"]},
    ),
    "not_resolved_shape": (
        lambda records: mutate(records, "udv-2-3-0", provenance="model"),
        {"person_not_resolved_shape": ["udv-2-3-0"]},
    ),
    "resolved_tier_but_unmatched": (
        lambda records: mutate(records, "udv-2-3-0", tier="no_evidence"),
        {"resolved_tier_but_unmatched": ["udv-2-3-0"]},
    ),
    "no_evidence_shape": (
        lambda records: mutate(
            records, "udv-1-0-1", tier="no_evidence", evidence=None, provenance=None
        ),
        {"no_evidence_shape": ["udv-1-0-1"]},
    ),
    "evidence_missing": (
        lambda records: mutate(records, "udv-1-0-1", evidence=None),
        {"evidence_missing": ["udv-1-0-1"]},
    ),
    "quote_provenance": (
        lambda records: mutate(records, "udv-1-0-0", provenance="model"),
        {"quote_shape": ["udv-1-0-0"]},
    ),
    "quote_support_type": (
        lambda records: mutate(
            records, "udv-1-0-0", evidence_update={"support_type": "semantic_similarity"}
        ),
        {"quote_shape": ["udv-1-0-0"]},
    ),
    "quote_score": (
        lambda records: mutate(records, "udv-1-0-0", evidence_update={"score": 0.9}),
        {"quote_score_not_null": ["udv-1-0-0"]},
    ),
    "quote_not_trusted": (
        lambda records: mutate(
            records,
            "udv-1-0-1",
            tier="quote_found",
            provenance="weak",
            evidence_update={"support_type": "direct_quote", "score": None},
        ),
        {"quote_not_trusted": ["udv-1-0-1"]},
    ),
    "quote_prefix": (
        lambda records: mutate(
            records, "udv-1-0-0", evidence_update={"quote_prefix": "o texto volte"}
        ),
        {"quote_prefix_mismatch": ["udv-1-0-0"]},
    ),
    "quote_text": (
        lambda records: mutate(
            records,
            "udv-1-0-0",
            evidence_update={"text": JOAO_FIRST_SENTENCE, "start_char": 590, "end_char": 689},
        ),
        {"quote_text_mismatch": ["udv-1-0-0"]},
    ),
    "semantic_provenance": (
        lambda records: mutate(records, "udv-1-0-1", provenance="weak"),
        {"semantic_shape": ["udv-1-0-1"]},
    ),
    "semantic_direct_quote": (
        lambda records: mutate(
            records, "udv-1-0-1", evidence_update={"support_type": "direct_quote"}
        ),
        {"semantic_shape": ["udv-1-0-1"], "short_quote_support_mismatch": ["udv-1-0-1"]},
    ),
    "semantic_but_trusted": (
        lambda records: mutate(
            records,
            "udv-1-0-0",
            tier="semantic_match_high",
            provenance="model",
            evidence_update={
                "support_type": "semantic_similarity",
                "score": 0.9,
                "quote_prefix": None,
            },
        ),
        {"semantic_but_trusted_quote_findable": ["udv-1-0-0"]},
    ),
    "evidence_not_person_sentence": (
        lambda records: mutate(
            records,
            "udv-1-0-1",
            evidence_update={
                "text": CARLOS_SENTENCE,
                "start_char": 59,
                "end_char": 115,
                "speaker_turn": 0,
            },
        ),
        {
            "evidence_not_in_single_actor_turn": ["udv-1-0-1"],
            "evidence_not_person_sentence": ["udv-1-0-1"],
            "speaker_turn_not_actor": ["udv-1-0-1"],
        },
    ),
    "cross_turn_text": (
        lambda records: mutate(
            records,
            "udv-1-1-2",
            evidence_update={
                "text": CROSS_TURN_TEXT,
                "start_char": None,
                "end_char": None,
                "speaker_turn": None,
            },
        ),
        {
            "evidence_not_in_single_actor_turn": ["udv-1-1-2"],
            "evidence_not_person_sentence": ["udv-1-1-2"],
            "evidence_offsets_missing": ["udv-1-1-2"],
        },
    ),
    "unlocated_evidence": (
        lambda records: mutate(
            records,
            "udv-1-0-1",
            evidence_update={"start_char": None, "end_char": None, "speaker_turn": None},
        ),
        {"evidence_offsets_missing": ["udv-1-0-1"]},
    ),
    "short_quote_type": (
        lambda records: mutate(
            records, "udv-1-1-1", evidence_update={"support_type": "semantic_similarity"}
        ),
        {"short_quote_support_mismatch": ["udv-1-1-1"]},
    ),
    "short_quote_prefix": (
        lambda records: mutate(records, "udv-1-1-1", evidence_update={"quote_prefix": None}),
        {"short_quote_prefix_mismatch": ["udv-1-1-1"]},
    ),
    "short_quote_claimed_without_quote": (
        lambda records: mutate(
            records,
            "udv-1-0-1",
            evidence_update={
                "support_type": "semantic_with_short_quote",
                "quote_prefix": "fiscalização",
            },
        ),
        {
            "short_quote_support_mismatch": ["udv-1-0-1"],
            "short_quote_prefix_mismatch": ["udv-1-0-1"],
        },
    ),
    "score_above_range": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"score": 1.5}),
        {"score_out_of_range": ["udv-1-0-1"]},
    ),
    "score_missing": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"score": None}),
        {"score_out_of_range": ["udv-1-0-1"]},
    ),
    "weak_tier_above_threshold": (
        lambda records: mutate(records, "udv-1-0-1", tier="semantic_match_weak"),
        {"tier_inconsistent_with_score": ["udv-1-0-1"]},
    ),
    "high_tier_below_threshold": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"score": 0.5}),
        {"tier_inconsistent_with_score": ["udv-1-0-1"]},
    ),
    "partial_offsets_without_start": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"start_char": None}),
        {"offset_shape": ["udv-1-0-1"], "evidence_offsets_missing": ["udv-1-0-1"]},
    ),
    "partial_offsets_without_end": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"end_char": None}),
        {"offset_shape": ["udv-1-0-1"]},
    ),
    "turn_without_offsets": (
        lambda records: mutate(
            records, "udv-1-0-1", evidence_update={"start_char": None, "end_char": None}
        ),
        {"offset_shape": ["udv-1-0-1"], "evidence_offsets_missing": ["udv-1-0-1"]},
    ),
    "shifted_offsets": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"start_char": 591}),
        {"offset_text_mismatch": ["udv-1-0-1"]},
    ),
    "foreign_turn": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"speaker_turn": 0}),
        {"speaker_turn_not_actor": ["udv-1-0-1"], "semantic_turn_mismatch": ["udv-1-0-1"]},
    ),
    "missing_turn": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"speaker_turn": None}),
        {"speaker_turn_not_actor": ["udv-1-0-1"]},
    ),
    "span_in_other_actor_turn": (
        lambda records: mutate(records, "udv-1-1-1", evidence_update={"speaker_turn": 5}),
        {"span_outside_turn": ["udv-1-1-1"], "semantic_turn_mismatch": ["udv-1-1-1"]},
    ),
    "quote_in_other_actor_turn": (
        lambda records: mutate(records, "udv-1-1-0", evidence_update={"speaker_turn": 5}),
        {"span_outside_turn": ["udv-1-1-0"], "quote_turn_mismatch": ["udv-1-1-0"]},
    ),
    "span_ends_past_the_turn": (
        lambda records: mutate(records, "udv-1-0-1", evidence_update={"end_char": 743}),
        {"offset_text_mismatch": ["udv-1-0-1"], "span_outside_turn": ["udv-1-0-1"]},
    ),
}
DELETE = object()


def test_clean_fixture_run_has_no_problem(mini: MiniRun) -> None:
    verification = verify_udv_run(
        "mini", mini.run.records, mini.coverage(mini.run.records), mini.hearings
    )
    assert verification.problems == {}
    assert verification.ok
    assert (verification.records, verification.expected_records) == (14, 14)
    assert (verification.people_total, verification.people_resolved) == (11, 9)
    assert verification.by_tier == {
        "quote_found": 5,
        "semantic_match_high": 6,
        "semantic_match_weak": 0,
        "no_evidence": 1,
        "person_not_resolved": 2,
    }
    assert verification.by_support_type == {
        "direct_quote": 5,
        "semantic_with_short_quote": 1,
        "semantic_similarity": 5,
    }


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_mutation_yields_its_problem_code(mini: MiniRun, name: str) -> None:
    mutation, expected = MUTATIONS[name]
    assert mini.problems(mutation(mini.run.records)) == expected


def test_clean_turns_fixture_run_has_no_problem(turns: MiniRun) -> None:
    assert turns.problems(turns.run.records) == {}


def turn_span_update(hearing: HearingRecord, text: str, turn_index: int) -> dict[str, Any]:
    turns = split_into_turns(hearing.transcricao)
    span = locate_turn_sentence_span(text, hearing.transcricao, turns, turn_index)
    assert span is not None
    return {
        "text": text,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "speaker_turn": span.speaker_turn,
    }


def substring_span_update(hearing: HearingRecord, text: str, turn_index: int) -> dict[str, Any]:
    turn = split_into_turns(hearing.transcricao)[turn_index]
    start = hearing.transcricao.index(text, turn.start_char, turn.end_char)
    return {
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
        "speaker_turn": turn_index,
    }


def test_quote_offsets_in_another_occurrence_of_the_sentence(turns: MiniRun) -> None:
    update = turn_span_update(turns.hearings[0], TURNS_PHYSICS_SENTENCE, 5)
    records = mutate(turns.run.records, "udv-3-0-3", evidence_update=update)
    assert turns.problems(records) == {"quote_turn_mismatch": ["udv-3-0-3"]}


def test_quote_evidence_from_the_first_occurrence(turns: MiniRun) -> None:
    update = turn_span_update(turns.hearings[0], TURNS_REFORM_TURN_1, 1)
    records = mutate(turns.run.records, "udv-3-0-0", evidence_update=update)
    assert turns.problems(records) == {
        "quote_text_mismatch": ["udv-3-0-0"],
        "quote_turn_mismatch": ["udv-3-0-0"],
    }


def test_semantic_offsets_inside_a_longer_sentence_of_another_turn(turns: MiniRun) -> None:
    update = substring_span_update(turns.hearings[0], TURNS_TRAINING_SENTENCE, 3)
    records = mutate(turns.run.records, "udv-3-0-4", evidence_update=update)
    assert turns.problems(records) == {"semantic_turn_mismatch": ["udv-3-0-4"]}


def test_semantic_evidence_for_an_opinion_whose_later_quote_is_trusted(turns: MiniRun) -> None:
    update = turn_span_update(turns.hearings[0], TURNS_PHYSICS_SENTENCE, 1)
    records = mutate(
        turns.run.records,
        "udv-3-0-1",
        tier="semantic_match_high",
        provenance="model",
        evidence_update={
            **update,
            "support_type": "semantic_with_short_quote",
            "score": 0.9,
            "quote_prefix": "faltam professores de",
        },
    )
    assert turns.problems(records) == {
        "semantic_but_trusted_quote_findable": ["udv-3-0-1"],
        "short_quote_support_mismatch": ["udv-3-0-1"],
        "short_quote_prefix_mismatch": ["udv-3-0-1"],
    }


def test_semantic_evidence_for_a_trusted_single_quote(turns: MiniRun) -> None:
    records = mutate(
        turns.run.records,
        "udv-3-0-2",
        tier="semantic_match_high",
        provenance="model",
        evidence_update={"support_type": "semantic_similarity", "score": 0.9, "quote_prefix": None},
    )
    assert turns.problems(records) == {"semantic_but_trusted_quote_findable": ["udv-3-0-2"]}


def test_short_prefix_support_uses_the_first_occurrence(turns: MiniRun) -> None:
    update = turn_span_update(
        turns.hearings[0], "O transporte escolar é caro para as prefeituras pequenas.", 4
    )
    records = mutate(turns.run.records, "udv-3-1-2", evidence_update=update)
    assert turns.problems(records) == {
        "short_quote_support_mismatch": ["udv-3-1-2"],
        "short_quote_prefix_mismatch": ["udv-3-1-2"],
    }


def replace_path(payload: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    if value is DELETE:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value


def test_coverage_counters_are_recounted(mini: MiniRun) -> None:
    coverage = copy.deepcopy(mini.coverage(mini.run.records))
    coverage["opinions"]["total"] = 15
    coverage["opinions"]["by_tier"]["quote_found"] = 6
    coverage["people"]["total"] = 12
    coverage["people"]["resolved"] = 10
    coverage["evidence_offsets"]["total"] = 12
    coverage["evidence_support_types"]["direct_quote"] = 4
    coverage["evidence_support_types"]["semantic_similarity"] = 6
    coverage["evidence_offsets"]["located"] = 12
    problems = verify_udv_run("mini", mini.run.records, coverage, mini.hearings).problems
    assert problems == {
        "coverage": [
            "coverage.opinions.total: reported 15, recomputed 14",
            "coverage.people.total: reported 12, recomputed 11",
            "coverage.people.resolved: reported 10, recomputed 9",
            "coverage.evidence_offsets.total: reported 12, recomputed 11",
            "coverage.evidence_offsets.located: reported 12, recomputed 11",
            "coverage.by_tier.quote_found: reported 6, recomputed 5",
            "coverage.evidence_support_types.direct_quote: reported 4, recomputed 5",
            "coverage.evidence_support_types.semantic_similarity: reported 6, recomputed 5",
        ]
    }


def test_malformed_coverage_counters_are_reported(mini: MiniRun) -> None:
    coverage = copy.deepcopy(mini.coverage(mini.run.records))
    del coverage["evidence_offsets"]
    coverage["opinions"]["by_tier"]["no_evidence"] = "1"
    coverage["people"]["total"] = True
    coverage["evidence_support_types"] = []
    problems = verify_udv_run("mini", mini.run.records, coverage, mini.hearings).problems
    assert problems == {
        "coverage": [
            "coverage_non_integer:people.total",
            "coverage_missing_key:evidence_offsets.total",
            "coverage_missing_key:evidence_offsets.located",
            "coverage_non_integer:opinions.by_tier.no_evidence",
            "coverage_missing_key:evidence_support_types.direct_quote",
            "coverage_missing_key:evidence_support_types.semantic_with_short_quote",
            "coverage_missing_key:evidence_support_types.semantic_similarity",
        ]
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("config",), DELETE, "coverage config.evidence.embedding_threshold is missing"),
        (("config", "evidence"), [], "coverage config.evidence.embedding_threshold is missing"),
        (("config", "evidence", "embedding_threshold"), "0.6", "not a number"),
        (("config", "evidence", "embedding_threshold"), True, "not a number"),
        (("hearings",), DELETE, "coverage hearings.ids is missing"),
        (("hearings", "ids"), None, "not a list of integers"),
        (("hearings", "ids"), [1, "2"], "not a list of integers"),
        (("hearings", "ids"), [1, False], "not a list of integers"),
    ],
)
def test_coverage_without_its_parameters_raises_config_error(
    mini: MiniRun, path: tuple[str, ...], value: object, message: str
) -> None:
    coverage = copy.deepcopy(mini.coverage(mini.run.records))
    replace_path(coverage, path, value)
    with pytest.raises(ConfigError, match=re.escape(message)):
        verify_udv_run("mini", mini.run.records, coverage, mini.hearings)


def test_coverage_parameters_keep_their_json_types(mini: MiniRun) -> None:
    coverage = copy.deepcopy(mini.coverage(mini.run.records))
    assert coverage_threshold(coverage) == MINI_THRESHOLD
    assert coverage_hearing_ids(coverage) == [1, 2]
    coverage["config"]["evidence"]["embedding_threshold"] = 1
    threshold = coverage_threshold(coverage)
    assert (threshold, type(threshold)) == (1, int)


def test_coverage_threshold_drives_tier_checks(mini: MiniRun) -> None:
    coverage = copy.deepcopy(mini.coverage(mini.run.records))
    coverage["config"]["evidence"]["embedding_threshold"] = 0.7
    problems = verify_udv_run("mini", mini.run.records, coverage, mini.hearings).problems
    assert sorted(problems["method_threshold_mismatch"]) == [
        record.id for record in mini.run.records
    ]
    assert problems["tier_inconsistent_with_score"] == ["udv-1-0-1"]


def test_schema_errors_are_reported_first(mini: MiniRun) -> None:
    records = [record for record in mini.run.records if record.id != "udv-2-7-0"]
    verification = verify_udv_run(
        "mini",
        records,
        mini.coverage(records),
        mini.hearings,
        schema_errors=["line 14 (udv-2-7-0): tier: Input should be ..."],
    )
    assert list(verification.problems)[:2] == ["schema_invalid", "missing_id"]
    assert verification.problems["schema_invalid"] == [
        "line 14 (udv-2-7-0): tier: Input should be ..."
    ]


def test_only_hearings_listed_in_coverage_are_expected(mini: MiniRun) -> None:
    records = [record for record in mini.run.records if record.hearing_id == 2]
    coverage = mini.coverage(records)
    coverage["hearings"]["ids"] = [2]
    coverage["people"]["total"] = 8
    coverage["people"]["resolved"] = 6
    verification = verify_udv_run("mini", records, coverage, mini.hearings)
    assert verification.problems == {}
    assert verification.ok
    assert (verification.records, verification.expected_records) == (8, 8)
    assert [hearing.id for hearing in coverage_hearings(coverage, mini.hearings)] == [2]


def test_report_lists_counts_and_at_most_five_examples(mini: MiniRun) -> None:
    records = [
        record.model_copy(update={"proposition": "Trocada."})
        if record.id.startswith("udv-1-") or record.id.startswith("udv-2-0")
        else record
        for record in mini.run.records
    ]
    report = verify_udv_run("mini", records, mini.coverage(records), mini.hearings).to_report()
    assert list(report) == [
        "run_name",
        "records",
        "expected_records",
        "people_resolved",
        "people_total",
        "by_tier",
        "by_support_type",
        "problems",
    ]
    assert report["problems"]["proposition_mismatch"] == {
        "count": 7,
        "examples": ["udv-1-0-0", "udv-1-0-1", "udv-1-1-0", "udv-1-1-1", "udv-1-1-2"],
    }


def test_index_people_and_expected_ids(mini: MiniRun) -> None:
    people = index_people(mini.hearings)
    assert list(people)[:4] == [(1, 0), (1, 1), (1, 2), (2, 0)]
    expected = expected_ids(people)
    assert list(expected) == [record.id for record in mini.run.records]
    indexed, opinion = expected["udv-2-2-0"]
    assert indexed.hearing.id == 2
    assert indexed.person.participant.nome == "Roberto"
    assert opinion.startswith("Disse que")


def test_compare_with_baseline_without_changes(mini: MiniRun) -> None:
    assert compare_with_baseline(mini.run.records, mini.run.records) == {
        "baseline_records": 14,
        "evidence_changed": 0,
        "tier_moves": {},
        "extra_in_run": 0,
    }


def test_compare_with_baseline_counts_moves_and_evidence_changes(mini: MiniRun) -> None:
    current = mutate(mini.run.records, "udv-1-0-1", tier="semantic_match_weak")
    current = mutate(current, "udv-1-2-0", evidence_update={"score": 0.99})
    current = [record for record in current if record.id != "udv-2-7-0"]
    current = [*current, mini.run.records[0].model_copy(update={"id": "udv-3-0-0"})]
    assert compare_with_baseline(current, mini.run.records) == {
        "baseline_records": 14,
        "evidence_changed": 1,
        "tier_moves": {"semantic_match_high->semantic_match_weak": 1, "missing_in_run": 1},
        "extra_in_run": 1,
    }
