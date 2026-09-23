from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bookworm.data.io import is_json_integer, is_json_integer_list, is_json_number, json_path
from bookworm.data.schemas import HearingRecord
from bookworm.errors import ConfigError
from bookworm.transcript.sentences import sentences_agree, turn_text
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.build import PersonSpeech, resolve_hearing_people, udv_id
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    QuotePolicy,
    TurnQuoteMatch,
    find_opinion_turn_quote_match,
    is_trusted_quote,
)
from bookworm.udv.schemas import (
    SUPPORT_TYPES,
    TIERS,
    Evidence,
    Provenance,
    SupportType,
    Tier,
    UdvRecord,
)

JsonObject = dict[str, Any]
PersonKey = tuple[int, int]

EXAMPLES_PER_PROBLEM = 5
SCORE_BOUND = 1.0001
SCHEMA_PROBLEM = "schema_invalid"
COVERAGE_PROBLEM = "coverage"
THRESHOLD_PATH = ("config", "evidence", "embedding_threshold")
HEARING_IDS_PATH = ("hearings", "ids")


@dataclass(frozen=True, slots=True)
class IndexedPerson:
    hearing: HearingRecord
    person: PersonSpeech


@dataclass(frozen=True)
class UdvVerification:
    run_name: str
    records: int
    expected_records: int
    people_resolved: int
    people_total: int
    by_tier: dict[Tier, int]
    by_support_type: dict[SupportType, int]
    problems: dict[str, list[str]]

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_report(self) -> JsonObject:
        return {
            "run_name": self.run_name,
            "records": self.records,
            "expected_records": self.expected_records,
            "people_resolved": self.people_resolved,
            "people_total": self.people_total,
            "by_tier": dict(self.by_tier),
            "by_support_type": dict(self.by_support_type),
            "problems": {
                kind: {"count": len(items), "examples": items[:EXAMPLES_PER_PROBLEM]}
                for kind, items in self.problems.items()
            },
        }


def index_people(hearings: Sequence[HearingRecord]) -> dict[PersonKey, IndexedPerson]:
    people: dict[PersonKey, IndexedPerson] = {}
    for hearing in hearings:
        for person in resolve_hearing_people(hearing):
            people[(hearing.id, person.index)] = IndexedPerson(hearing=hearing, person=person)
    return people


def expected_ids(
    people: Mapping[PersonKey, IndexedPerson],
) -> dict[str, tuple[IndexedPerson, str]]:
    expected: dict[str, tuple[IndexedPerson, str]] = {}
    for (hearing_id, person_index), indexed in people.items():
        for opinion_index, opinion_text in enumerate(indexed.person.participant.opinioes):
            expected[udv_id(hearing_id, person_index, opinion_index)] = (indexed, opinion_text)
    return expected


def check_record(
    record: UdvRecord,
    indexed: IndexedPerson,
    opinion_text: str,
    threshold: float,
    policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> list[str]:
    problems: list[str] = []
    person = indexed.person
    participant = person.participant
    if record.proposition != opinion_text:
        problems.append("proposition_mismatch")
    if (record.actor.name, record.actor.role) != (participant.nome, participant.cargo):
        problems.append("actor_mismatch")
    if record.method.embedding_threshold != threshold:
        problems.append("method_threshold_mismatch")
    tier, evidence, provenance = record.tier, record.evidence, record.provenance
    if tier == "person_not_resolved":
        if person.matched_turns:
            problems.append("person_not_resolved_but_matched")
        if evidence is not None or provenance is not None:
            problems.append("person_not_resolved_shape")
        return problems
    if not person.matched_turns:
        problems.append("resolved_tier_but_unmatched")
    if tier == "no_evidence":
        if evidence is not None or provenance is not None or person.sentences:
            problems.append("no_evidence_shape")
        return problems
    if evidence is None:
        return [*problems, "evidence_missing"]
    problems.extend(check_single_turn(evidence, person))
    quote_match = find_opinion_turn_quote_match(opinion_text, person.matched_turns, policy)
    if tier == "quote_found":
        problems.extend(check_quote_evidence(evidence, quote_match, provenance, policy))
    else:
        if provenance != "model" or evidence.support_type == "direct_quote":
            problems.append("semantic_shape")
        if is_trusted_quote(quote_match, policy):
            problems.append("semantic_but_trusted_quote_findable")
        if evidence.text not in person.sentences:
            problems.append("evidence_not_person_sentence")
        problems.extend(check_short_quote_support(evidence, quote_match, policy))
        score = evidence.score
        if score is None or not -SCORE_BOUND <= score <= SCORE_BOUND:
            problems.append("score_out_of_range")
        elif (tier == "semantic_match_high") != (score >= threshold):
            problems.append("tier_inconsistent_with_score")
    problems.extend(check_offsets(evidence, indexed))
    problems.extend(check_source_turn(evidence, quote_match, person, tier, policy))
    return problems


def check_single_turn(evidence: Evidence, person: PersonSpeech) -> list[str]:
    if any(evidence.text in turn_text(turn) for turn in person.matched_turns):
        return []
    return ["evidence_not_in_single_actor_turn"]


def check_quote_evidence(
    evidence: Evidence,
    quote_match: TurnQuoteMatch | None,
    provenance: Provenance | None,
    policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> list[str]:
    problems: list[str] = []
    if provenance != "weak" or evidence.support_type != "direct_quote":
        problems.append("quote_shape")
    if evidence.score is not None:
        problems.append("quote_score_not_null")
    if not is_trusted_quote(quote_match, policy):
        return [*problems, "quote_not_trusted"]
    if evidence.quote_prefix != quote_match.prefix:
        return [*problems, "quote_prefix_mismatch"]
    if quote_match.sentence != evidence.text:
        problems.append("quote_text_mismatch")
    return problems


def check_short_quote_support(
    evidence: Evidence,
    quote_match: TurnQuoteMatch | None,
    policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> list[str]:
    supported = (
        quote_match is not None
        and not is_trusted_quote(quote_match, policy)
        and sentences_agree(quote_match.sentence, evidence.text)
    )
    expected_support_type = "semantic_with_short_quote" if supported else "semantic_similarity"
    expected_prefix = quote_match.prefix if supported and quote_match is not None else None
    problems: list[str] = []
    if evidence.support_type != expected_support_type:
        problems.append("short_quote_support_mismatch")
    if evidence.quote_prefix != expected_prefix:
        problems.append("short_quote_prefix_mismatch")
    return problems


def check_offsets(evidence: Evidence, indexed: IndexedPerson) -> list[str]:
    person = indexed.person
    if evidence.start_char is None:
        if evidence.end_char is not None or evidence.speaker_turn is not None:
            return ["offset_shape", "evidence_offsets_missing"]
        return ["evidence_offsets_missing"]
    if evidence.end_char is None:
        return ["offset_shape"]
    problems: list[str] = []
    transcript = indexed.hearing.transcricao
    span = transcript[evidence.start_char : evidence.end_char]
    if normalize_whitespace(span) != evidence.text:
        problems.append("offset_text_mismatch")
    turn = next(
        (turn for turn in person.matched_turns if turn.turn_index == evidence.speaker_turn), None
    )
    if turn is None:
        problems.append("speaker_turn_not_actor")
    elif not (turn.start_char <= evidence.start_char and evidence.end_char <= turn.end_char):
        problems.append("span_outside_turn")
    return problems


def check_source_turn(
    evidence: Evidence,
    quote_match: TurnQuoteMatch | None,
    person: PersonSpeech,
    tier: Tier,
    policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> list[str]:
    if evidence.speaker_turn is None:
        return []
    if tier == "quote_found":
        trusted = is_trusted_quote(quote_match, policy)
        if trusted and quote_match is not None and evidence.speaker_turn != quote_match.turn_index:
            return ["quote_turn_mismatch"]
        return []
    source_turns = {
        turn_index
        for sentence, turn_index in zip(person.sentences, person.sentence_turns, strict=True)
        if sentence == evidence.text
    }
    if source_turns and evidence.speaker_turn not in source_turns:
        return ["semantic_turn_mismatch"]
    return []


def coverage_counters(
    records: Sequence[UdvRecord], people: Mapping[PersonKey, IndexedPerson]
) -> list[tuple[str, tuple[str, ...], int]]:
    by_tier = Counter(record.tier for record in records)
    evidences = [record.evidence for record in records if record.evidence is not None]
    return [
        ("opinions.total", ("opinions", "total"), len(records)),
        ("people.total", ("people", "total"), len(people)),
        (
            "people.resolved",
            ("people", "resolved"),
            sum(1 for indexed in people.values() if indexed.person.matched_turns),
        ),
        ("evidence_offsets.total", ("evidence_offsets", "total"), len(evidences)),
        (
            "evidence_offsets.located",
            ("evidence_offsets", "located"),
            sum(1 for evidence in evidences if evidence.start_char is not None),
        ),
        *[(f"by_tier.{tier}", ("opinions", "by_tier", tier), by_tier[tier]) for tier in TIERS],
        *[
            (
                f"evidence_support_types.{support_type}",
                ("evidence_support_types", support_type),
                sum(1 for evidence in evidences if evidence.support_type == support_type),
            )
            for support_type in SUPPORT_TYPES
        ],
    ]


def check_coverage(
    coverage: Mapping[str, Any],
    records: Sequence[UdvRecord],
    people: Mapping[PersonKey, IndexedPerson],
) -> list[str]:
    problems: list[str] = []
    for name, path, recomputed in coverage_counters(records, people):
        found, reported = json_path(coverage, path)
        if not found:
            problems.append(f"coverage_missing_key:{'.'.join(path)}")
        elif not is_json_integer(reported):
            problems.append(f"coverage_non_integer:{'.'.join(path)}")
        elif reported != recomputed:
            problems.append(f"coverage.{name}: reported {reported}, recomputed {recomputed}")
    return problems


def compare_with_baseline(
    records: Sequence[UdvRecord], baseline: Sequence[UdvRecord]
) -> JsonObject:
    current = {record.id: record for record in records}
    moves: Counter[str] = Counter()
    evidence_changed = 0
    for old in baseline:
        new = current.get(old.id)
        if new is None:
            moves["missing_in_run"] += 1
            continue
        if old.tier != new.tier:
            moves[f"{old.tier}->{new.tier}"] += 1
        if old.evidence != new.evidence:
            evidence_changed += 1
    return {
        "baseline_records": len(baseline),
        "evidence_changed": evidence_changed,
        "tier_moves": dict(moves),
        "extra_in_run": len(set(current) - {record.id for record in baseline}),
    }


def coverage_threshold(coverage: Mapping[str, Any]) -> float | int:
    found, threshold = json_path(coverage, THRESHOLD_PATH)
    if not found or not is_json_number(threshold):
        raise ConfigError(f"coverage {'.'.join(THRESHOLD_PATH)} is missing or not a number")
    return threshold


def coverage_hearing_ids(coverage: Mapping[str, Any]) -> list[int]:
    found, hearing_ids = json_path(coverage, HEARING_IDS_PATH)
    if not found or not is_json_integer_list(hearing_ids):
        raise ConfigError(
            f"coverage {'.'.join(HEARING_IDS_PATH)} is missing or not a list of integers"
        )
    return hearing_ids


def coverage_hearings(
    coverage: Mapping[str, Any], hearings: Sequence[HearingRecord]
) -> list[HearingRecord]:
    hearing_ids = set(coverage_hearing_ids(coverage))
    return [hearing for hearing in hearings if hearing.id in hearing_ids]


def verify_udv_run(
    run_name: str,
    records: Sequence[UdvRecord],
    coverage: Mapping[str, Any],
    hearings: Sequence[HearingRecord],
    *,
    schema_errors: Sequence[str] = (),
    quote_policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
) -> UdvVerification:
    threshold = coverage_threshold(coverage)
    people = index_people(coverage_hearings(coverage, hearings))
    expected = expected_ids(people)

    problems: defaultdict[str, list[str]] = defaultdict(list)
    for error in schema_errors:
        problems[SCHEMA_PROBLEM].append(error)
    ids = Counter(record.id for record in records)
    for record_id, count in ids.items():
        if count > 1:
            problems["duplicate_id"].append(record_id)
    for record_id in sorted(set(expected) - set(ids)):
        problems["missing_id"].append(record_id)
    for record in records:
        if record.id not in expected:
            problems["unexpected_id"].append(record.id)
            continue
        indexed, opinion_text = expected[record.id]
        for problem in check_record(record, indexed, opinion_text, threshold, quote_policy):
            problems[problem].append(record.id)
    for problem in check_coverage(coverage, records, people):
        problems[COVERAGE_PROBLEM].append(problem)

    return UdvVerification(
        run_name=run_name,
        records=len(records),
        expected_records=len(expected),
        people_resolved=sum(1 for indexed in people.values() if indexed.person.matched_turns),
        people_total=len(people),
        by_tier={tier: sum(1 for record in records if record.tier == tier) for tier in TIERS},
        by_support_type={
            support_type: sum(
                1
                for record in records
                if record.evidence is not None and record.evidence.support_type == support_type
            )
            for support_type in SUPPORT_TYPES
        },
        problems=dict(problems),
    )
