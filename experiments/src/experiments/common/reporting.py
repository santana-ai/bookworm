"""Small pieces every report shares: the creation time, file records and rounded numbers."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookworm import sha256_of_file

Record = dict[str, Any]

SCORE_DECIMALS = 4


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_record(path: Path, rows: int | None = None) -> Record:
    """The path and sha256 of a file a report reads or writes, with its row count when given."""
    counted = {} if rows is None else {"rows": rows}
    return {"path": str(path), **counted, "sha256": sha256_of_file(path)}


def rounded(value: float, digits: int = SCORE_DECIMALS) -> float:
    return round(float(value), digits)


def rounded_or_none(value: float | None, digits: int = SCORE_DECIMALS) -> float | None:
    return None if value is None else rounded(value, digits)
