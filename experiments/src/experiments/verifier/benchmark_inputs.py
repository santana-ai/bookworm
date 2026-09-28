"""Checked reads of the NLI benchmark and of the NLI file it was built from.

The verifier and the translation step read the same inputs and apply the same checks, so both
take any configuration that names these files.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from bookworm import load_gated_jsonl, sha256_of_file

from experiments.data.nli_benchmark import iter_opinions

Record = dict[str, Any]


class BenchmarkInputs(Protocol):
    @property
    def nli_path(self) -> Path: ...

    @property
    def nli_sha256(self) -> str: ...

    @property
    def benchmark_path(self) -> Path: ...

    @property
    def benchmark_report_path(self) -> Path: ...


def split_records(splits: tuple[str, ...], final_test: bool) -> Record:
    return {"splits": list(splits), "final_test_flag": final_test, "test_read": "test" in splits}


def check_benchmark_file(config: BenchmarkInputs) -> Record:
    """The benchmark file record, refusing a file whose sha256 differs from its build report."""
    with open(config.benchmark_report_path) as f:
        recorded = json.load(f)["artifact"]["sha256"]
    actual = sha256_of_file(config.benchmark_path)
    if actual != recorded:
        raise SystemExit(f"{config.benchmark_path}: sha256 {actual} != its report {recorded}")
    return {"path": str(config.benchmark_path), "sha256": actual}


def check_row_split(row: Record, split_of: dict[int, str]) -> None:
    manifest_split = split_of.get(row["hearing_id"])
    if manifest_split != row["split"]:
        raise SystemExit(f"{row['id']}: split {row['split']!r} != manifest {manifest_split!r}")


def load_nli_chunks(config: BenchmarkInputs, rows: list[Record]) -> dict[str, list[str]]:
    """The retrieved chunks of each benchmark row, checked against the row's opinion."""
    wanted = {row["id"]: row for row in rows}
    chunks: dict[str, list[str]] = {}
    for hearing, person_index, _, opinion_index, opinion in iter_opinions(
        load_gated_jsonl(config.nli_path, config.nli_sha256)
    ):
        row_id = f"nli-{hearing['id']}-{person_index}-{opinion_index}"
        if row_id not in wanted:
            continue
        if opinion["opiniao"] != wanted[row_id]["opinion"]:
            raise SystemExit(f"{row_id}: the NLI opinion differs from the benchmark row")
        chunks[row_id] = list(opinion["chunks_proximos"])
    missing = sorted(set(wanted) - set(chunks))
    if missing:
        raise SystemExit(f"benchmark rows missing from the NLI file: {missing[:10]}")
    return chunks
