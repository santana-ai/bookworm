import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import udv_artifact_path

from bookworm import (
    HearingRecord,
    Turn,
    is_trusted_quote,
    load_jsonl,
    resolve_person_speech,
    sentences_agree,
    split_into_turns,
    write_jsonl,
)
from bookworm.transcript.offsets import locate_turn_sentence_span
from bookworm.transcript.sentences import TurnSentence, split_turn_sentences
from bookworm.udv.quotes import find_opinion_turn_quote_match

pytestmark = pytest.mark.dataset

SEMANTIC_TIERS = ("semantic_match_high", "semantic_match_weak")
V1_ENCODER_INDEPENDENT = {
    "people": (1065, 1020),
    "opinions": 2203,
    "tier_families": {
        "quote_found": 277,
        "semantic": 1828,
        "no_evidence": 8,
        "person_not_resolved": 90,
    },
    "support_types": {
        "direct_quote": 277,
        "semantic_with_short_quote": 111,
        "semantic_similarity": 1717,
    },
    "offsets": (2105, 2105),
    "unlocated": [],
}
EXPECTED = {"udv_v1": V1_ENCODER_INDEPENDENT, "udv_v1_pre": V1_ENCODER_INDEPENDENT}
EXAMPLES_PER_CHECK = 5


@dataclass
class ParityReport:
    record_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    people_total: int = 0
    people_resolved: int = 0
    tier_families: Counter[str] = field(default_factory=Counter)
    support_types: Counter[str] = field(default_factory=Counter)
    located: int = 0
    evidences: int = 0
    unlocated: list[str] = field(default_factory=list)
    mismatches: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def check(self, name: str, record_id: str, ours: object, artifact: object) -> None:
        if ours != artifact:
            self.mismatches[name] += 1
            examples = self.examples.setdefault(name, [])
            if len(examples) < EXAMPLES_PER_CHECK:
                examples.append(f"{record_id}: bookworm={ours!r} artifact={artifact!r}")


def tier_family(tier: str) -> str:
    return "semantic" if tier in SEMANTIC_TIERS else tier


def artifact_span(evidence: dict[str, Any]) -> dict[str, Any] | None:
    if evidence["start_char"] is None:
        return None
    return {key: evidence[key] for key in ("start_char", "end_char", "speaker_turn")}


def check_evidence(
    report: ParityReport,
    record: dict[str, Any],
    transcript: str,
    person: tuple[list[Turn], list[TurnSentence]],
    opinion: str,
) -> None:
    matched, units = person
    evidence = record["evidence"]
    record_id = record["id"]
    match = find_opinion_turn_quote_match(opinion, matched)
    if is_trusted_quote(match):
        report.check("quote_text", record_id, match.sentence, evidence["text"])
        report.check("quote_prefix", record_id, match.prefix, evidence["quote_prefix"])
        report.check("quote_support_type", record_id, "direct_quote", evidence["support_type"])
        source_turn = match.turn_index
    else:
        text = evidence["text"]
        source_turns = [unit.turn_index for unit in units if unit.text == text]
        report.check("semantic_text_is_person_sentence", record_id, True, bool(source_turns))
        supported = match is not None and sentences_agree(match.sentence, text)
        expected_type = "semantic_with_short_quote" if supported else "semantic_similarity"
        report.check("semantic_support_type", record_id, expected_type, evidence["support_type"])
        expected_prefix = match.prefix if supported and match is not None else None
        report.check("semantic_quote_prefix", record_id, expected_prefix, evidence["quote_prefix"])
        report.check(
            "semantic_source_turn",
            record_id,
            True,
            evidence["speaker_turn"] in source_turns,
        )
        source_turn = evidence["speaker_turn"]
    span = locate_turn_sentence_span(evidence["text"], transcript, matched, source_turn)
    report.check(
        "span",
        record_id,
        None if span is None else dataclasses.asdict(span),
        artifact_span(evidence),
    )


def build_report(hearings: list[HearingRecord], records: list[dict[str, Any]]) -> ParityReport:
    report = ParityReport(artifact_ids=[record["id"] for record in records])
    by_id = {record["id"]: record for record in records}
    for hearing in hearings:
        turns = split_into_turns(hearing.transcricao)
        for person_index, person in enumerate(hearing.metadados.envolvidos):
            matched, _ = resolve_person_speech(person.nome, turns)
            units = split_turn_sentences(matched)
            report.people_total += 1
            report.people_resolved += bool(matched)
            for opinion_index, opinion in enumerate(person.opinioes):
                record_id = f"udv-{hearing.id}-{person_index}-{opinion_index}"
                report.record_ids.append(record_id)
                record = by_id.get(record_id)
                if record is None:
                    report.mismatches["missing_record"] += 1
                    continue
                report.check("hearing_id", record_id, hearing.id, record["hearing_id"])
                report.check(
                    "actor",
                    record_id,
                    {"name": person.nome, "role": person.cargo},
                    record["actor"],
                )
                report.check("proposition", record_id, opinion, record["proposition"])
                if not matched:
                    expected_family = "person_not_resolved"
                elif is_trusted_quote(find_opinion_turn_quote_match(opinion, matched)):
                    expected_family = "quote_found"
                elif not units:
                    expected_family = "no_evidence"
                else:
                    expected_family = "semantic"
                family = tier_family(record["tier"])
                report.check("tier_family", record_id, expected_family, family)
                report.tier_families[family] += 1
                evidence = record["evidence"]
                if evidence is None:
                    continue
                report.evidences += 1
                report.support_types[evidence["support_type"]] += 1
                if evidence["start_char"] is None:
                    report.unlocated.append(record_id)
                else:
                    report.located += 1
                check_evidence(report, record, hearing.transcricao, (matched, units), opinion)
    return report


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def run_name(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


@pytest.fixture(scope="module")
def parity_report(
    run_name: str, udv_artifacts_dir: Path, lds_hearings: list[HearingRecord]
) -> ParityReport:
    records = load_jsonl(udv_artifact_path(udv_artifacts_dir, f"{run_name}.jsonl"))
    coverage = json.loads(
        udv_artifact_path(udv_artifacts_dir, f"{run_name}_coverage.json").read_text("utf-8")
    )
    wanted = set(coverage["hearings"]["ids"])
    hearings = [hearing for hearing in lds_hearings if hearing.id in wanted]
    assert [hearing.id for hearing in hearings] == coverage["hearings"]["ids"]
    return build_report(hearings, records)


def test_record_ids_follow_pipeline_order(parity_report: ParityReport, run_name: str) -> None:
    assert parity_report.record_ids == parity_report.artifact_ids
    assert len(parity_report.record_ids) == EXPECTED[run_name]["opinions"]


def test_people_resolution(parity_report: ParityReport, run_name: str) -> None:
    people = (parity_report.people_total, parity_report.people_resolved)
    assert people == EXPECTED[run_name]["people"]


def counts_with_zeros(counter: Counter[str], expected: object) -> dict[str, int]:
    assert isinstance(expected, dict)
    assert set(counter) <= set(expected)
    return {key: counter[key] for key in expected}


def test_tier_families(parity_report: ParityReport, run_name: str) -> None:
    expected = EXPECTED[run_name]["tier_families"]
    assert counts_with_zeros(parity_report.tier_families, expected) == expected


def test_support_types(parity_report: ParityReport, run_name: str) -> None:
    expected = EXPECTED[run_name]["support_types"]
    assert counts_with_zeros(parity_report.support_types, expected) == expected


def test_evidence_offsets(parity_report: ParityReport, run_name: str) -> None:
    offsets = (parity_report.located, parity_report.evidences)
    assert offsets == EXPECTED[run_name]["offsets"]
    assert parity_report.unlocated == EXPECTED[run_name]["unlocated"]


def test_no_stage_mismatch(parity_report: ParityReport) -> None:
    assert dict(parity_report.mismatches) == {}, parity_report.examples


def test_write_jsonl_reproduces_artifact_bytes(
    run_name: str, udv_artifacts_dir: Path, tmp_path: Path
) -> None:
    artifact = udv_artifact_path(udv_artifacts_dir, f"{run_name}.jsonl")
    rewritten = tmp_path / artifact.name
    write_jsonl(load_jsonl(artifact), rewritten)
    assert rewritten.read_bytes() == artifact.read_bytes()
