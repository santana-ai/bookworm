import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bookworm import (
    DatasetIntegrityError,
    iter_jsonl,
    load_gated_jsonl,
    load_hearings,
    load_jsonl,
    sha256_of_file,
    verify_sha256,
    write_json,
    write_jsonl,
)
from bookworm.data.io import is_json_integer, is_json_integer_list, is_json_number, json_path

RECORDS: list[dict[str, Any]] = [
    {"id": "udv-1-0-0", "text": "Audiência sobre regulação", "score": 0.5, "tags": ["ação"]},
    {"id": "udv-1-0-1", "text": "Sem acento", "score": None, "nested": {"b": 1, "a": 2}},
]


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    payload = b"linha 1\nlinha 2\n" * 100_000
    path.write_bytes(payload)
    assert sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_verify_sha256_accepts_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    verify_sha256(path, hashlib.sha256(b'{"a": 1}\n').hexdigest())


def test_load_gated_jsonl_raises_integrity_error_instead_of_exiting(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(DatasetIntegrityError) as raised:
        load_gated_jsonl(path, "0" * 64)
    assert raised.value.path == path
    assert raised.value.expected_sha256 == "0" * 64
    assert raised.value.actual_sha256 == sha256_of_file(path)
    assert str(raised.value) == f"{path}: sha256 {sha256_of_file(path)} != expected {'0' * 64}"


def test_load_gated_jsonl_returns_records_when_hash_matches(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(RECORDS, path)
    assert load_gated_jsonl(path, sha256_of_file(path)) == RECORDS


def test_write_jsonl_matches_json_dumps_without_ascii_escaping(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "records.jsonl"
    write_jsonl(RECORDS, path)
    expected = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in RECORDS)
    assert path.read_bytes() == expected.encode("utf-8")
    assert "Audiência" in path.read_text(encoding="utf-8")


def test_jsonl_roundtrip_preserves_key_order(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(RECORDS, path)
    loaded = load_jsonl(path)
    assert loaded == RECORDS
    assert [list(record) for record in loaded] == [list(record) for record in RECORDS]
    assert list(iter_jsonl(path)) == RECORDS


def test_write_json_uses_indent_two_and_no_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "out" / "coverage.json"
    payload = {"run_name": "execução", "counts": {"total": 3}}
    write_json(payload, path)
    assert path.read_text(encoding="utf-8") == json.dumps(payload, ensure_ascii=False, indent=2)


def test_load_hearings_matches_plain_json_parsing(lds_mini_path: Path) -> None:
    hearings = load_hearings(lds_mini_path, sha256_of_file(lds_mini_path))
    raw_records = load_jsonl(lds_mini_path)
    assert [hearing.id for hearing in hearings] == [1, 2]
    for hearing, raw in zip(hearings, raw_records, strict=True):
        assert hearing.transcricao == raw["transcricao"]
        assert hearing.materia == raw["materia"]
        assert hearing.metadados.assunto == raw["metadados"]["assunto"]
        assert [
            (person.nome, person.cargo, list(person.opinioes))
            for person in hearing.metadados.envolvidos
        ] == [
            (person["nome"], person["cargo"], person["opinioes"])
            for person in raw["metadados"]["envolvidos"]
        ]


def test_load_hearings_checks_hash_when_given(lds_mini_path: Path) -> None:
    with pytest.raises(DatasetIntegrityError):
        load_hearings(lds_mini_path, "f" * 64)


@pytest.mark.parametrize(
    ("value", "integer", "number"),
    [
        (3, True, True),
        (0, True, True),
        (2.5, False, True),
        (True, False, False),
        ("3", False, False),
    ],
)
def test_json_scalar_guards_exclude_booleans(value: object, integer: bool, number: bool) -> None:
    assert is_json_integer(value) is integer
    assert is_json_number(value) is number


@pytest.mark.parametrize(
    ("value", "expected"),
    [([], True), ([1, 2], True), ([1, True], False), ([1.0], False), (None, False), ((1,), False)],
)
def test_is_json_integer_list(value: object, expected: bool) -> None:
    assert is_json_integer_list(value) is expected


def test_json_path_walks_nested_objects() -> None:
    payload = {"a": {"b": {"c": None}}, "list": [1]}
    assert json_path(payload, ("a", "b", "c")) == (True, None)
    assert json_path(payload, ("a", "x")) == (False, None)
    assert json_path(payload, ("list", "0")) == (False, None)
    assert json_path([], ("a",)) == (False, None)
    assert json_path(payload, ()) == (True, payload)
