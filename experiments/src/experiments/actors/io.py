"""File helpers shared by the actor modules: TOML and JSONL reading, content hashes and
resumable JSONL runs."""

import hashlib
import json
import logging
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm import article_date, load_gated_jsonl, sha256_of_file, write_jsonl

Record = dict[str, Any]

FINGERPRINT_LENGTH = 16
BR_DATE_FORMAT = "%d/%m/%Y"


def read_toml(path: Path) -> Record:
    with open(path, "rb") as f:
        return tomllib.load(f)


def file_info(path: Path) -> Record:
    return {"path": str(path), "sha256": sha256_of_file(path)}


def canonical_sha256(payload: Any) -> str:
    """sha256 of the JSON text of a payload with sorted keys, so key order never changes it."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(*parts: Any) -> str:
    return canonical_sha256(parts)[:FINGERPRINT_LENGTH]


def read_jsonl_rows(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield (line number, value) per JSON line, skipping blank lines and logging invalid ones."""
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                warn_invalid_line(line_number, path)
                continue
            yield line_number, row


def warn_invalid_line(line_number: int, path: Path) -> None:
    logging.warning("ignoring invalid line %d in %s", line_number, path)


def load_rows(path: Path, key: str) -> dict[str, Record]:
    """Rows of a JSONL file by the value of `key`, or nothing when the file does not exist."""
    if not path.exists():
        return {}
    return {row[key]: row for _, row in read_jsonl_rows(path)}


def read_split_hearings(manifest_path: Path, lds_sha256: str) -> Record:
    """The split manifest, after checking that it was built from this LDS file."""
    with open(manifest_path) as f:
        manifest: Record = json.load(f)
    if manifest["dataset"]["sha256"] != lds_sha256:
        raise SystemExit(f"{manifest_path} was built from another LDS file")
    return manifest


def hearing_metadata(lds_path: Path, lds_sha256: str) -> dict[int, Record]:
    """Article date (as a date and as DD/MM/YYYY) and subject of every hearing."""
    metadata: dict[int, Record] = {}
    for hearing in load_gated_jsonl(lds_path, lds_sha256):
        published = article_date(hearing["materia"])
        if published is None:
            raise ValueError(f"hearing {hearing['id']} has no article date")
        metadata[hearing["id"]] = {
            "date": published,
            "date_br": published.strftime(BR_DATE_FORMAT),
            "assunto": hearing["metadados"]["assunto"],
        }
    return metadata


@dataclass(frozen=True)
class Job:
    """One row of a resumable run: its id, a log label, the fingerprint of everything that
    determines it, and the call that computes it."""

    id: str
    label: str
    fingerprint: str
    compute: Callable[[], Record]


def run_resumable(path: Path, id_field: str, jobs: Iterable[Job], total: int) -> list[Record]:
    """Compute the jobs whose row is missing from `path` or has another fingerprint, appending
    each new row as soon as it exists, then rewrite `path` with the rows in job order."""
    done = load_rows(path, id_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(path, "a") as output:
        for index, job in enumerate(jobs, start=1):
            row = done.get(job.id)
            if row is None or row["fingerprint"] != job.fingerprint:
                started = time.monotonic()
                row = {**job.compute(), "fingerprint": job.fingerprint}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                logging.info(
                    "[%d/%d] %s: %.1fs", index, total, job.label, time.monotonic() - started
                )
            rows.append(row)
    write_jsonl(rows, path)
    return rows
