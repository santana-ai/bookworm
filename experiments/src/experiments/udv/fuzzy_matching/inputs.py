"""The hearings, the reference UDV run and the resolved people the experiment reads."""

import json
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, sha256_of_file

from experiments.common.transcript import split_into_turns
from experiments.common.udv_run import resolve_hearing_people

Record = dict[str, Any]


def select_hearings(
    lds: list[Record], split_of: dict[int, str], splits: tuple[str, ...]
) -> list[Record]:
    missing = [hearing["id"] for hearing in lds if hearing["id"] not in split_of]
    if missing:
        raise SystemExit(f"hearings missing from the split manifest: {missing}")
    return [hearing for hearing in lds if split_of[hearing["id"]] in splits]


def reference_source(
    run: Path, records_path: Path, coverage_path: Path, coverage: Record, position: int
) -> Record:
    config = coverage.get("config", {})
    encoder = config.get("encoder", {})
    return {
        "run_name": run.name,
        "path": str(records_path),
        "sha256": sha256_of_file(records_path),
        "coverage_path": str(coverage_path),
        "coverage_sha256": sha256_of_file(coverage_path),
        "coverage_created_at": coverage.get("created_at"),
        "encoder": encoder.get("name"),
        "encoder_revision": encoder.get("revision"),
        "embedding_threshold": config.get("evidence", {}).get("embedding_threshold"),
        "has_pipeline_section": "pipeline" in coverage,
        "preference_position": position,
        "fallback_used": position > 0,
    }


def select_reference_run(runs: tuple[Path, ...]) -> tuple[dict[str, Record], Record]:
    """The records of the first configured UDV run that exists, and where they came from."""
    for position, run in enumerate(runs):
        records_path = run.with_name(f"{run.name}.jsonl")
        coverage_path = run.with_name(f"{run.name}_coverage.json")
        if not (records_path.exists() and coverage_path.exists()):
            continue
        with open(coverage_path) as f:
            coverage = json.load(f)
        records = {record["id"]: record for record in load_jsonl(records_path)}
        source = {
            **reference_source(run, records_path, coverage_path, coverage, position),
            "runs_skipped": [str(skipped) for skipped in runs[:position]],
            "records": len(records),
        }
        return records, source
    raise SystemExit(f"no reference UDV run found among {[str(run) for run in runs]}")


def hearing_people(hearing: Record) -> tuple[list[Record], list[Record]]:
    return split_into_turns(hearing["transcricao"]), resolve_hearing_people(hearing)


def udv_id(hearing_id: int, person_index: int, opinion_index: int) -> str:
    return f"udv-{hearing_id}-{person_index}-{opinion_index}"
