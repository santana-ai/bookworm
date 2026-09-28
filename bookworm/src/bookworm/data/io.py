"""JSON and JSONL reading and writing, sha256 checks and LDS loading."""

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeGuard

from bookworm.data.schemas import HearingRecord
from bookworm.errors import ConfigError, DatasetIntegrityError

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
    """Read the LDS file, checking its sha256 first when one is given."""
    if expected_sha256 is not None:
        verify_sha256(path, expected_sha256)
    with path.open(encoding="utf-8") as handle:
        return [HearingRecord.model_validate_json(line) for line in handle]


def read_json_object(path: Path, description: str) -> JsonObject:
    if not path.is_file():
        raise ConfigError(f"{path}: {description} not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigError(f"{path}: {description} is not UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: invalid JSON: {error}") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot read {description}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: {description} is not a JSON object")
    return payload


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


def required_field(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> Any:
    """Value at a nested key path; raises ``ConfigError`` naming ``origin`` when it is missing."""
    value: Any = source
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"{origin}: {'.'.join(keys)} is missing")
        value = value[key]
    return value


def required_text(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> str:
    value = required_field(source, keys, origin)
    if not isinstance(value, str):
        raise ConfigError(f"{origin}: {'.'.join(keys)} is not text")
    return value


def required_number(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> float:
    value = required_field(source, keys, origin)
    if not is_json_number(value):
        raise ConfigError(f"{origin}: {'.'.join(keys)} is not a number")
    return float(value)


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


def write_json(payload: JsonObject | list[JsonObject], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
