import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import ValidationError

from bookworm.models import StrictModel

Tier = Literal[
    "quote_found",
    "semantic_match_high",
    "semantic_match_weak",
    "no_evidence",
    "person_not_resolved",
]
SupportType = Literal["direct_quote", "semantic_with_short_quote", "semantic_similarity"]
Provenance = Literal["weak", "model"]

TIERS: tuple[Tier, ...] = get_args(Tier)
SUPPORT_TYPES: tuple[SupportType, ...] = get_args(SupportType)
SEMANTIC_TIERS: tuple[Tier, ...] = ("semantic_match_high", "semantic_match_weak")


class Actor(StrictModel):
    name: str
    role: str


class Evidence(StrictModel):
    text: str
    support_type: SupportType
    score: float | None
    quote_prefix: str | None
    start_char: int | None
    end_char: int | None
    speaker_turn: int | None


class Method(StrictModel):
    encoder: str
    revision: str
    embedding_threshold: float | int


class UdvRecord(StrictModel):
    id: str
    hearing_id: int
    actor: Actor
    proposition: str
    evidence: Evidence | None
    tier: Tier
    provenance: Provenance | None
    method: Method

    @classmethod
    def from_json_line(cls, line: str) -> "UdvRecord":
        return cls.model_validate(json.loads(line))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json_line(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


@dataclass(frozen=True)
class ParsedUdvLines:
    records: list[UdvRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def describe_invalid_line(line_number: int, line: str, error: Exception) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        payload = None
    record_id = payload.get("id") if isinstance(payload, dict) else None
    reason = (
        "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors()
        )
        if isinstance(error, ValidationError)
        else str(error)
    )
    return f"line {line_number} ({record_id}): {reason}"


def parse_udv_lines(lines: Iterable[str]) -> ParsedUdvLines:
    parsed = ParsedUdvLines()
    for line_number, line in enumerate(lines, start=1):
        try:
            parsed.records.append(UdvRecord.from_json_line(line))
        except (ValidationError, json.JSONDecodeError) as error:
            parsed.errors.append(describe_invalid_line(line_number, line, error))
    return parsed


def read_udv_jsonl(path: Path) -> ParsedUdvLines:
    with path.open(encoding="utf-8") as handle:
        return parse_udv_lines(handle)


def load_udv_jsonl(path: Path) -> list[UdvRecord]:
    with path.open(encoding="utf-8") as handle:
        return [UdvRecord.from_json_line(line) for line in handle]


def write_udv_jsonl(records: Iterable[UdvRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json_line() + "\n")
