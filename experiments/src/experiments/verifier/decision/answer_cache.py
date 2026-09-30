"""Append-only JSONL cache of decision answers keyed by model signature."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.common.cache_lock import CacheLockedError, acquire_writer_lock
from experiments.verifier.decision.questions import (
    DecisionModelError,
    DecisionQuestion,
    serialize_state,
    sha256_text,
)
from experiments.verifier.runtime import incomplete_tail

Record = dict[str, Any]


def read_jsonl_lines(path: Path) -> tuple[list[Record], int]:
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    tail = incomplete_tail(data)
    records = []
    for number, line in enumerate(data[: len(data) - tail].decode().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise DecisionModelError(
                f"{path}: line {number} is not valid JSON ({error})"
            ) from error
    return records, tail


def append_jsonl(path: Path, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        f.flush()
        os.fsync(f.fileno())


@dataclass
class AnswerCache:
    path: Path
    signature: str
    entries: dict[str, Record] = field(default_factory=dict)
    other_signature_lines: int = 0
    incomplete_tail_bytes: int = 0

    @classmethod
    def open(cls, path: Path, signature: str) -> "AnswerCache":
        try:
            acquire_writer_lock(path)
        except CacheLockedError as error:
            raise DecisionModelError(str(error)) from error
        cache = cls(path, signature)
        records, cache.incomplete_tail_bytes = read_jsonl_lines(path)
        if cache.incomplete_tail_bytes:
            data = path.read_bytes()
            with open(path, "r+b") as f:
                f.truncate(len(data) - cache.incomplete_tail_bytes)
        for record in records:
            if record.get("signature") != signature:
                cache.other_signature_lines += 1
                continue
            cache.entries.setdefault(record["key"], record)
        return cache

    def key(self, question: DecisionQuestion, state: Record) -> str:
        return sha256_text(
            f"{self.signature}\x1e{question.payload_json()}\x1e{serialize_state(state)}"
        )

    def append(self, records: Sequence[Record]) -> None:
        stamped = [{"signature": self.signature, **record} for record in records]
        append_jsonl(self.path, stamped)
        for record in stamped:
            self.entries.setdefault(record["key"], record)
