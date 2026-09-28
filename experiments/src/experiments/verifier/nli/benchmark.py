"""Benchmark rows and premise units, with checks that the NLI chunks match the transcripts."""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_gated_jsonl, load_jsonl, sha256_of_file

from experiments.common.transcript import normalize_whitespace
from experiments.data.nli_benchmark import iter_opinions
from experiments.verifier.nli.config import SPLIT_NAMES, VerifierConfig

Record = dict[str, Any]

WHITESPACE_PATTERN = re.compile(r"\s+")
PAIR_STRING_FIELDS = ("pair_id", "query_id", "proposition")


@dataclass(frozen=True)
class PremiseUnit:
    unit_id: str
    hearing_id: int
    split: str
    hypothesis: str
    items: tuple[str, ...]


def check_benchmark_file(config: VerifierConfig) -> Record:
    with open(config.benchmark_report_path) as f:
        recorded = json.load(f)["artifact"]["sha256"]
    actual = sha256_of_file(config.benchmark_path)
    if actual != recorded:
        raise SystemExit(f"{config.benchmark_path}: sha256 {actual} != its report {recorded}")
    return {"path": str(config.benchmark_path), "sha256": actual}


def load_benchmark_rows(
    config: VerifierConfig, split_of: dict[int, str], splits: tuple[str, ...]
) -> list[Record]:
    rows = []
    for row in load_jsonl(config.benchmark_path):
        manifest_split = split_of.get(row["hearing_id"])
        if manifest_split != row["split"]:
            raise SystemExit(f"{row['id']}: split {row['split']!r} != manifest {manifest_split!r}")
        if row["split"] in splits:
            rows.append(row)
    return rows


def first_rows(rows: list[Record], limit: int) -> list[Record]:
    taken: Counter[str] = Counter()
    kept = []
    for row in rows:
        if taken[row["split"]] < limit:
            taken[row["split"]] += 1
            kept.append(row)
    return kept


def sample_rows(
    rows: list[Record], limit: int | None, seed: int, rule: str = "random"
) -> list[Record]:
    if limit is None:
        return rows
    if rule == "first":
        return first_rows(rows, limit)
    kept: set[str] = set()
    for index, split in enumerate(SPLIT_NAMES):
        members = [row["id"] for row in rows if row["split"] == split]
        if len(members) <= limit:
            kept.update(members)
            continue
        rng = np.random.default_rng([seed, index])
        kept.update(members[pick] for pick in rng.choice(len(members), size=limit, replace=False))
    return [row for row in rows if row["id"] in kept]


def load_nli_chunks(config: VerifierConfig, rows: list[Record]) -> dict[str, list[str]]:
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


def load_transcripts(config: VerifierConfig, hearing_ids: set[int]) -> dict[int, str]:
    return {
        record["id"]: record["transcricao"]
        for record in load_gated_jsonl(config.lds_path, config.lds_sha256)
        if record["id"] in hearing_ids
    }


def rebuild_chunk(chunk: Record, transcript: str) -> str:
    if not chunk["located"]:
        return ""
    return "\n".join(transcript[s["start_char"] : s["end_char"]] for s in chunk["segments"])


def without_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub("", text)


def chunk_source_problem(chunk: Record, nli_text: str, rebuilt: str) -> str | None:
    if not chunk["located"]:
        return None if not nli_text.split() else "not_located_but_text"
    if without_whitespace(rebuilt) != without_whitespace(nli_text):
        return "text_differs"
    return None


def verify_chunk_sources(
    rows: list[Record], nli_chunks: dict[str, list[str]], transcripts: dict[int, str]
) -> Record:
    counts: Counter[str] = Counter()
    problems: list[Record] = []
    for row in rows:
        texts = nli_chunks[row["id"]]
        if len(texts) != len(row["chunks"]):
            problems.append({"id": row["id"], "problem": "chunk_count"})
            continue
        for chunk, text in zip(row["chunks"], texts, strict=True):
            rebuilt = rebuild_chunk(chunk, transcripts[row["hearing_id"]])
            counts["chunks"] += 1
            problem = chunk_source_problem(chunk, text, rebuilt)
            if problem is not None:
                problems.append(
                    {"id": row["id"], "position": chunk["position"], "problem": problem}
                )
            elif not chunk["located"]:
                counts["empty_not_located"] += 1
            elif normalize_whitespace(rebuilt) == normalize_whitespace(text):
                counts["identical_after_whitespace_normalization"] += 1
            else:
                counts["differ_only_in_whitespace"] += 1
    if problems:
        raise SystemExit(f"chunk text sources disagree: {problems[:10]}")
    return {
        "method": (
            "each located chunk is rebuilt from its benchmark offsets in the LDS transcript "
            "(segments joined by a line break) and compared with chunks_proximos after removing "
            "all whitespace; any other difference stops the run"
        ),
        "opinions": len(rows),
        "opinions_without_chunks": sum(1 for row in rows if not row["chunks"]),
        **dict(sorted(counts.items())),
    }


def benchmark_units(
    rows: list[Record],
    nli_chunks: dict[str, list[str]],
    transcripts: dict[int, str],
    text_source: str,
) -> list[PremiseUnit]:
    units = []
    for row in rows:
        if text_source == "nli_chunks":
            texts = nli_chunks[row["id"]]
        else:
            texts = [
                rebuild_chunk(chunk, transcripts[row["hearing_id"]]) for chunk in row["chunks"]
            ]
        units.append(
            PremiseUnit(
                unit_id=row["id"],
                hearing_id=row["hearing_id"],
                split=row["split"],
                hypothesis=normalize_whitespace(row["opinion"]),
                items=tuple(normalize_whitespace(text) for text in texts),
            )
        )
    return units


def pair_evidence(evidence: Any, line_number: int) -> tuple[str, ...]:
    if isinstance(evidence, str):
        return (normalize_whitespace(evidence),)
    if isinstance(evidence, list) and evidence and all(isinstance(e, str) for e in evidence):
        return tuple(normalize_whitespace(text) for text in evidence)
    raise SystemExit(f"line {line_number}: evidence must be a string or a non-empty string list")


def check_pair_row(row: Record, line_number: int) -> None:
    for name in PAIR_STRING_FIELDS:
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise SystemExit(f"line {line_number}: {name} must be a non-empty string")
    if type(row.get("hearing_id")) is not int:
        raise SystemExit(f"line {line_number}: hearing_id must be an integer")
    if "meta" in row and not isinstance(row["meta"], dict):
        raise SystemExit(f"line {line_number}: meta must be an object")


def load_pair_units(
    path: Path, split_of: dict[int, str], allowed: tuple[str, ...]
) -> tuple[list[PremiseUnit], list[Record]]:
    rows = load_jsonl(path)
    units: list[PremiseUnit] = []
    refused: Counter[str] = Counter()
    for line_number, row in enumerate(rows, start=1):
        check_pair_row(row, line_number)
        split = split_of.get(row["hearing_id"])
        if split is None:
            raise SystemExit(
                f"line {line_number}: hearing {row['hearing_id']} is not in the manifest"
            )
        if split not in allowed:
            refused[split] += 1
            continue
        units.append(
            PremiseUnit(
                unit_id=row["pair_id"],
                hearing_id=row["hearing_id"],
                split=split,
                hypothesis=normalize_whitespace(row["proposition"]),
                items=pair_evidence(row["evidence"], line_number),
            )
        )
    if refused:
        raise SystemExit(
            f"pairs from splits outside {list(allowed)} refused: {dict(refused)} "
            "(test pairs need --final-test)"
        )
    duplicated = [key for key, count in Counter(u.unit_id for u in units).items() if count > 1]
    if duplicated:
        raise SystemExit(f"pair_id values are not unique: {duplicated[:10]}")
    return units, rows


def distinct_items(items: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def concatenated_premise(unit: PremiseUnit, separator: str, concatenate_single: bool) -> str | None:
    nonempty = [item for item in unit.items if item]
    if not nonempty or (len(nonempty) == 1 and not concatenate_single):
        return None
    return separator.join(nonempty)


def unit_header(unit: PremiseUnit) -> Record:
    nonempty = sum(1 for item in unit.items if item)
    return {
        "id": unit.unit_id,
        "hearing_id": unit.hearing_id,
        "split": unit.split,
        "item_count": len(unit.items),
        "nonempty_items": nonempty,
        "no_premise": nonempty == 0,
    }
