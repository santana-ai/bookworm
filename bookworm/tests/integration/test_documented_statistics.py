import statistics
from collections import Counter
from pathlib import Path

import pytest
from conftest import udv_artifact_path

from bookworm import HearingRecord, Participant, load_udv_jsonl

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def people(lds_hearings: list[HearingRecord]) -> list[Participant]:
    return [person for hearing in lds_hearings for person in hearing.metadados.envolvidos]


def word_range(texts: list[str]) -> tuple[int, float, int]:
    counts = [len(text.split()) for text in texts]
    return min(counts), statistics.median(counts), max(counts)


def test_ids_run_from_1_to_206_in_file_order(lds_hearings: list[HearingRecord]) -> None:
    assert [hearing.id for hearing in lds_hearings] == list(range(1, 207))


def test_no_text_field_is_blank(
    lds_hearings: list[HearingRecord], people: list[Participant]
) -> None:
    blank = {
        "assunto": sum(not hearing.metadados.assunto.strip() for hearing in lds_hearings),
        "nome": sum(not person.nome.strip() for person in people),
        "cargo": sum(not person.cargo.strip() for person in people),
        "opinioes": sum(not opinion.strip() for person in people for opinion in person.opinioes),
    }
    assert blank == {"assunto": 0, "nome": 0, "cargo": 0, "opinioes": 0}


def test_participant_counts(lds_hearings: list[HearingRecord], people: list[Participant]) -> None:
    per_hearing = [len(hearing.metadados.envolvidos) for hearing in lds_hearings]
    assert (min(per_hearing), max(per_hearing), len(people)) == (2, 12, 1065)
    assert len({person.nome.strip().upper() for person in people}) == 879


def test_opinion_counts(people: list[Participant]) -> None:
    per_person = [len(person.opinioes) for person in people]
    assert (min(per_person), max(per_person), sum(per_person)) == (1, 15, 2203)


def test_word_counts(lds_hearings: list[HearingRecord]) -> None:
    assert word_range([hearing.materia for hearing in lds_hearings]) == (288, 607, 1215)
    assert word_range([hearing.transcricao for hearing in lds_hearings]) == (4437, 16424.5, 147728)


def test_quote_prefix_word_counts(udv_artifacts_dir: Path) -> None:
    records = load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, "udv_v1.jsonl"))
    counts = Counter(
        (record.evidence.support_type, len(record.evidence.quote_prefix.split()))
        for record in records
        if record.evidence is not None and record.evidence.quote_prefix is not None
    )
    assert counts == {
        ("direct_quote", 10): 122,
        ("direct_quote", 6): 152,
        ("direct_quote", 7): 3,
        ("semantic_with_short_quote", 5): 1,
        ("semantic_with_short_quote", 4): 65,
        ("semantic_with_short_quote", 3): 37,
        ("semantic_with_short_quote", 2): 7,
        ("semantic_with_short_quote", 1): 1,
    }
