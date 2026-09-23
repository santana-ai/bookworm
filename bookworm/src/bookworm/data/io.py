import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeGuard

from bookworm.data.schemas import HearingRecord
from bookworm.errors import DatasetIntegrityError

JsonObject = dict[str, Any]

READ_CHUNK_BYTES = 1 << 20


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> None:
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise DatasetIntegrityError(path, actual, expected_sha256)


def iter_jsonl(path: Path) -> Iterator[JsonObject]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record: JsonObject = json.loads(line)
            yield record


def load_jsonl(path: Path) -> list[JsonObject]:
    return list(iter_jsonl(path))


def load_gated_jsonl(path: Path, expected_sha256: str) -> list[JsonObject]:
    verify_sha256(path, expected_sha256)
    return load_jsonl(path)


def load_hearings(path: Path, expected_sha256: str | None = None) -> list[HearingRecord]:
    if expected_sha256 is not None:
        verify_sha256(path, expected_sha256)
    with path.open(encoding="utf-8") as handle:
        return [HearingRecord.model_validate_json(line) for line in handle]


def is_json_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def is_json_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def is_json_integer_list(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(is_json_integer(item) for item in value)


def json_path(payload: object, path: Sequence[str]) -> tuple[bool, Any]:
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def json_value(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def json_dict_factory(pairs: list[tuple[str, Any]]) -> JsonObject:
    return {key: json_value(value) for key, value in pairs}


def write_jsonl(records: Iterable[JsonObject], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: JsonObject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
