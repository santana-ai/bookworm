import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pydantic import Field, ValidationError

from bookworm.errors import ConfigError
from bookworm.models import StrictModel


class ProfileRecord(StrictModel):
    actor: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    model: str
    prompt_version: str
    n_statements: int = Field(ge=0)
    n_hearings: int = Field(ge=0)
    hearing_ids: list[int]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    generated_at: str
    duration_seconds: float = Field(ge=0)

    @classmethod
    def from_json_line(cls, line: str) -> "ProfileRecord":
        return cls.model_validate_json(line)

    def to_json_line(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


@dataclass(frozen=True)
class ParsedProfileLines:
    records: list[ProfileRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_profile_lines(lines: Iterable[str]) -> ParsedProfileLines:
    parsed = ParsedProfileLines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed.records.append(ProfileRecord.from_json_line(line))
        except ValidationError as error:
            parsed.errors.append(f"line {line_number}: {error.errors()[0]['msg']}")
    return parsed


def read_profile_lines(path: Path) -> ParsedProfileLines:
    try:
        with path.open(encoding="utf-8") as handle:
            return parse_profile_lines(handle)
    except UnicodeDecodeError as error:
        raise ConfigError(f"{path}: profile file is not UTF-8: {error}") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot read profile file: {error}") from error


def read_profiles(path: Path) -> list[ProfileRecord]:
    parsed = read_profile_lines(path)
    if parsed.errors:
        raise ConfigError(f"{path}: {parsed.errors[0]}")
    counts = Counter(record.actor for record in parsed.records)
    duplicated = sorted(actor for actor, count in counts.items() if count > 1)
    if duplicated:
        raise ConfigError(f"{path}: duplicated actors: {duplicated}")
    return parsed.records


def append_profile(record: ProfileRecord, handle: TextIO) -> None:
    handle.write(record.to_json_line() + "\n")
    handle.flush()


def write_profiles(records: Iterable[ProfileRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json_line() + "\n")
