import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bookworm import (
    SUPPORT_TYPES,
    TIERS,
    Evidence,
    UdvRecord,
    load_udv_jsonl,
    read_udv_jsonl,
    write_udv_jsonl,
)
from bookworm.udv.schemas import parse_udv_lines

LEGACY_RECORD: dict[str, Any] = {
    "id": "udv-7-2-0",
    "hearing_id": 7,
    "actor": {"name": "Joana Prado", "role": "Diretora de Programas"},
    "proposition": 'Afirmou que "o edital atrasou três meses".',
    "evidence": {
        "text": "O edital atrasou três meses por causa da revisão jurídica.",
        "support_type": "semantic_with_short_quote",
        "score": 0.38978198170661926,
        "quote_prefix": "o edital atrasou",
        "start_char": 1520,
        "end_char": 1578,
        "speaker_turn": 12,
    },
    "tier": "semantic_match_weak",
    "provenance": "model",
    "method": {"encoder": "encoder-x", "revision": "abc123", "embedding_threshold": 0.47},
}
UNRESOLVED_RECORD: dict[str, Any] = {
    "id": "udv-7-3-0",
    "hearing_id": 7,
    "actor": {"name": "Rui", "role": "Convidado"},
    "proposition": "Pediu mais prazo.",
    "evidence": None,
    "tier": "person_not_resolved",
    "provenance": None,
    "method": {"encoder": "encoder-x", "revision": "abc123", "embedding_threshold": 0.47},
}


def legacy_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def with_changes(record: dict[str, Any], **changes: Any) -> dict[str, Any]:
    changed: dict[str, Any] = json.loads(json.dumps(record))
    changed.update(changes)
    return changed


def test_tier_and_support_type_order() -> None:
    assert TIERS == (
        "quote_found",
        "semantic_match_high",
        "semantic_match_weak",
        "no_evidence",
        "person_not_resolved",
    )
    assert SUPPORT_TYPES == ("direct_quote", "semantic_with_short_quote", "semantic_similarity")


def test_field_order_is_the_jsonl_key_order() -> None:
    assert list(UdvRecord.model_fields) == list(LEGACY_RECORD)
    assert list(Evidence.model_fields) == list(LEGACY_RECORD["evidence"])


@pytest.mark.parametrize("record", [LEGACY_RECORD, UNRESOLVED_RECORD])
def test_json_line_round_trip_is_byte_identical(record: dict[str, Any]) -> None:
    line = legacy_line(record)
    parsed = UdvRecord.from_json_line(line)
    assert parsed.to_json_line() == line
    assert parsed.to_dict() == record


def test_non_ascii_text_is_written_unescaped() -> None:
    line = UdvRecord.from_json_line(legacy_line(LEGACY_RECORD)).to_json_line()
    assert "três" in line
    assert "\\u" not in line


@pytest.mark.parametrize(
    ("changes", "location"),
    [
        ({"tier": "semantic_match_medium"}, "tier"),
        ({"provenance": "human"}, "provenance"),
        ({"hearing_id": "7"}, "hearing_id"),
        ({"hearing_id": True}, "hearing_id"),
        ({"extra": 1}, "extra"),
        ({"actor": {"name": "Joana Prado"}}, "actor.role"),
        ({"method": {"encoder": "x", "revision": "y", "embedding_threshold": "0.47"}}, "method"),
        ({"method": {"encoder": "x", "revision": "y", "embedding_threshold": True}}, "method"),
    ],
)
def test_schema_rejects_invalid_records(changes: dict[str, Any], location: str) -> None:
    with pytest.raises(ValidationError) as caught:
        UdvRecord.model_validate(with_changes(LEGACY_RECORD, **changes))
    locations = {".".join(str(part) for part in error["loc"]) for error in caught.value.errors()}
    assert any(found.startswith(location) for found in locations)


@pytest.mark.parametrize(
    ("threshold", "written"), [(1, "1"), (1.0, "1.0"), (0.47, "0.47"), (0, "0")]
)
def test_method_threshold_keeps_its_number_type(threshold: float, written: str) -> None:
    record = UdvRecord.model_validate(
        with_changes(
            LEGACY_RECORD,
            method={"encoder": "x", "revision": "y", "embedding_threshold": threshold},
        )
    )
    assert type(record.method.embedding_threshold) is type(threshold)
    assert record.to_json_line().endswith(f'"embedding_threshold": {written}}}}}')
    assert UdvRecord.from_json_line(record.to_json_line()) == record


def test_schema_rejects_unknown_support_type() -> None:
    record = with_changes(LEGACY_RECORD)
    record["evidence"]["support_type"] = "paraphrase"
    with pytest.raises(ValidationError):
        UdvRecord.model_validate(record)


def test_schema_rejects_missing_keys() -> None:
    record = with_changes(LEGACY_RECORD)
    del record["provenance"]
    with pytest.raises(ValidationError):
        UdvRecord.model_validate(record)


def test_records_are_frozen() -> None:
    record = UdvRecord.model_validate(LEGACY_RECORD)
    with pytest.raises(ValidationError):
        record.__setattr__("tier", "quote_found")


def test_parse_udv_lines_reports_invalid_lines_and_keeps_valid_ones() -> None:
    bad_support = with_changes(LEGACY_RECORD, id="udv-7-2-1")
    bad_support["evidence"]["support_type"] = "paraphrase"
    lines = [
        legacy_line(LEGACY_RECORD),
        legacy_line(bad_support),
        "{not json",
        legacy_line(UNRESOLVED_RECORD),
    ]
    parsed = parse_udv_lines(lines)
    assert [record.id for record in parsed.records] == ["udv-7-2-0", "udv-7-3-0"]
    assert len(parsed.errors) == 2
    assert parsed.errors[0].startswith("line 2 (udv-7-2-1): evidence.support_type")
    assert parsed.errors[1].startswith("line 3 (None): ")


def test_write_and_load_udv_jsonl_reproduce_bytes(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    original.write_text(
        legacy_line(LEGACY_RECORD) + "\n" + legacy_line(UNRESOLVED_RECORD) + "\n",
        encoding="utf-8",
    )
    records = load_udv_jsonl(original)
    rewritten = tmp_path / "nested" / "rewritten.jsonl"
    write_udv_jsonl(records, rewritten)
    assert rewritten.read_bytes() == original.read_bytes()
    assert read_udv_jsonl(rewritten).errors == []
