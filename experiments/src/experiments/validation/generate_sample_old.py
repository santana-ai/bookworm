import argparse
import csv
import hashlib
import io
import json
import platform
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from bookworm import load_gated_jsonl, load_jsonl, sha256_of_file, write_json
from scipy.stats import binomtest

from experiments.common import transcript, udv_run
from experiments.common.provenance import REPOSITORY_DIR, source_hashes
from experiments.common.transcript import (
    TURN_HEADER_PATTERN,
    extract_quotes,
    normalize_whitespace,
    quote_prefix_pattern,
    quote_prefixes,
    split_into_turns,
    strip_accents,
)
from experiments.common.udv_run import SUPPORT_TYPES, TIERS

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
ALWAYS_READABLE_SPLITS = ("train", "validation")
QUESTIONS = ("trecho_sustenta", "pessoa_falou")
NO_EVIDENCE_SUPPORT = "none"
STAGES = ("sample", "repeat")
SEED_STREAMS = ("stratum_draw", "row_order", "repeat_draw", "repeat_order")
DISPLAY_COLUMNS = (
    "hearing_id",
    "pergunta",
    "participante",
    "cargo",
    "afirmacao",
    "trecho",
    "contexto_antes",
    "contexto_depois",
    "link",
)
FILL_COLUMNS = ("julgamento", "existe_trecho_melhor", "observacao")
CSV_COLUMNS = ("item_id", *DISPLAY_COLUMNS, *FILL_COLUMNS)
CHECKED_COLUMNS = ("hearing_id", "pergunta", "participante", "afirmacao", "trecho")
CSV_DELIMITERS = (";", ",", "\t")
FORMULA_PREFIXES = ("=", "+", "-", "@")
HIDDEN_ITEM_FIELDS = (
    "udv_ids",
    "stratum",
    "tier",
    "support_type",
    "score",
    "quote_prefix",
    "quote_cue_in_trecho",
)
ANNOTATION_CSV = "annotation.csv"
ANNOTATION_KEY = "annotation_key.json"
REANNOTATION_CSV = "reannotation.csv"
REANNOTATION_KEY = "reannotation_key.json"
SAMPLE_REPORT = "sample_report.json"
REANNOTATION_REPORT = "reannotation_report.json"
PRECISION_REPORT_PATTERNS = ("precision_report*.json",)
ROUND_DECIMALS = 4


@dataclass(frozen=True)
class Stratum:
    name: str
    question: str
    tiers: tuple[str, ...]
    support_types: tuple[str, ...]
    target: int


@dataclass(frozen=True)
class Question:
    name: str
    judgments: tuple[str, ...]
    better_passage_required: bool
    prompt: str


@dataclass(frozen=True)
class ValidationConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    sample_splits: tuple[str, ...]
    udv_dir: Path
    version: str
    seed: int
    seed_streams: dict[str, int]
    output_dir: Path
    transcripts_dir: Path
    item_id_prefix: str
    context_chars: int
    csv_delimiter: str
    csv_encoding: str
    strata: tuple[Stratum, ...]
    questions: dict[str, Question]
    better_passage_values: tuple[str, ...]
    repeat_items: int
    repeat_item_id_prefix: str
    repeat_questions: tuple[str, ...]
    repeat_min_hours: float
    confidence_level: float
    source: Record = field(default_factory=dict)


@dataclass(frozen=True)
class Locations:
    sample_dir: Path
    transcripts_dir: Path


def check_split_names(splits: tuple[str, ...]) -> None:
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"splits must be distinct names among {SPLIT_NAMES}, got {splits}")


def check_strata(strata: tuple[Stratum, ...]) -> None:
    names = [stratum.name for stratum in strata]
    if len(set(names)) != len(names):
        raise SystemExit(f"stratum names must be unique: {names}")
    claimed: dict[tuple[str, str], str] = {}
    for stratum in strata:
        if stratum.question not in QUESTIONS:
            raise SystemExit(f"{stratum.name}: question must be one of {QUESTIONS}")
        if not set(stratum.tiers) <= set(TIERS):
            raise SystemExit(f"{stratum.name}: tiers must be among {TIERS}")
        if not set(stratum.support_types) <= {*SUPPORT_TYPES, NO_EVIDENCE_SUPPORT}:
            raise SystemExit(f"{stratum.name}: unknown support type in {stratum.support_types}")
        if stratum.target < 1:
            raise SystemExit(f"{stratum.name}: target must be >= 1")
        for cell in ((t, s) for t in stratum.tiers for s in stratum.support_types):
            if cell in claimed:
                raise SystemExit(f"{cell} is claimed by {claimed[cell]} and {stratum.name}")
            claimed[cell] = stratum.name


def load_validation_config(path: Path) -> ValidationConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sample, repeat, annotation = raw["sample"], raw["repeat"], raw["annotation"]
    questions = {
        name: Question(
            name=name,
            judgments=tuple(spec["judgments"]),
            better_passage_required=spec["better_passage_required"],
            prompt=spec["prompt"],
        )
        for name, spec in annotation["questions"].items()
    }
    if set(questions) != set(QUESTIONS):
        raise SystemExit(f"annotation.questions must define exactly {QUESTIONS}")
    strata = tuple(
        Stratum(
            name=spec["name"],
            question=spec["question"],
            tiers=tuple(spec["tiers"]),
            support_types=tuple(spec["support_types"]),
            target=spec["target"],
        )
        for spec in sample["strata"]
    )
    check_strata(strata)
    sample_splits = tuple(raw["splits"]["sample_splits"])
    check_split_names(sample_splits)
    streams = dict(sample["seed_streams"])
    if set(streams) != set(SEED_STREAMS) or len(set(streams.values())) != len(streams):
        raise SystemExit(f"sample.seed_streams must give distinct values to {SEED_STREAMS}")
    repeat_questions = tuple(repeat["eligible_questions"])
    if not repeat_questions or not set(repeat_questions) <= set(QUESTIONS):
        raise SystemExit(f"repeat.eligible_questions must be among {QUESTIONS}")
    if sample["csv_delimiter"] not in CSV_DELIMITERS:
        raise SystemExit(f"sample.csv_delimiter must be one of {CSV_DELIMITERS}")
    if sample["item_id_prefix"] == repeat["item_id_prefix"]:
        raise SystemExit("sample and repeat item id prefixes must differ")
    confidence_level = raw["report"]["confidence_level"]
    if not 0 < confidence_level < 1:
        raise SystemExit("report.confidence_level must be in (0, 1)")
    return ValidationConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        sample_splits=sample_splits,
        udv_dir=Path(raw["source"]["udv_dir"]),
        version=sample["version"],
        seed=sample["seed"],
        seed_streams=streams,
        output_dir=Path(sample["output_dir"]),
        transcripts_dir=Path(sample["transcripts_dir"]),
        item_id_prefix=sample["item_id_prefix"],
        context_chars=sample["context_chars"],
        csv_delimiter=sample["csv_delimiter"],
        csv_encoding=sample["csv_encoding"],
        strata=strata,
        questions=questions,
        better_passage_values=tuple(annotation["better_passage_values"]),
        repeat_items=repeat["items"],
        repeat_item_id_prefix=repeat["item_id_prefix"],
        repeat_questions=repeat_questions,
        repeat_min_hours=float(repeat["min_hours_after_first_pass"]),
        confidence_level=confidence_level,
        source=raw,
    )


def stream_rng(config: ValidationConfig, stream: str) -> np.random.Generator:
    return np.random.default_rng([config.seed, config.seed_streams[stream]])


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), ROUND_DECIMALS)


def wilson_interval(successes: int, trials: int, confidence_level: float) -> Record:
    if trials == 0:
        return {"successes": 0, "trials": 0, "estimate": None, "low": None, "high": None}
    interval = binomtest(successes, trials).proportion_ci(
        confidence_level=confidence_level, method="wilson"
    )
    return {
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials,
        "low": float(interval.low),
        "high": float(interval.high),
    }


def min_successes_to_pass(trials: int, min_lower: float, confidence_level: float) -> int | None:
    for successes in range(trials + 1):
        low = wilson_interval(successes, trials, confidence_level)["low"]
        if low is not None and low >= min_lower:
            return successes
    return None


def criteria_feasibility(
    config: ValidationConfig, trials_by_stratum: dict[str, int]
) -> list[Record]:
    rows = []
    for rule in config.source["criteria"]["rules"]:
        trials = trials_by_stratum.get(rule["stratum"], 0)
        needed = min_successes_to_pass(trials, rule["min_wilson_lower"], config.confidence_level)
        rows.append(
            {
                "rule": rule["name"],
                "stratum": rule["stratum"],
                "trials": trials,
                "min_successes_to_pass": needed,
                "max_failures_to_pass": None if needed is None else trials - needed,
            }
        )
    return rows


def resolve_splits(
    config: ValidationConfig, override: list[str] | None, dry_run: bool, final_test: bool
) -> tuple[str, ...]:
    if override is not None and not dry_run:
        raise SystemExit("--splits overrides the configured splits only together with --dry-run")
    splits = tuple(override) if override is not None else config.sample_splits
    check_split_names(splits)
    if "test" in splits and not final_test:
        raise SystemExit(
            f"the sample is drawn from {splits}, which includes test; pass --final-test to "
            "confirm that test hearings may be read"
        )
    return splits


def readable_splits(splits: tuple[str, ...], final_test: bool) -> set[str]:
    return {*ALWAYS_READABLE_SPLITS, *splits, *(("test",) if final_test else ())}


def require_final_test(key: Record, final_test: bool) -> None:
    if "test" in key["splits_used"] and not final_test:
        raise SystemExit(
            "this sample was drawn from test hearings; pass --final-test to read its sheets"
        )


def resolve_locations(
    config: ValidationConfig,
    sample_name: str,
    dry_run: bool,
    output_dir: Path | None,
    create_temp: bool,
) -> Locations:
    if not dry_run:
        if output_dir is not None:
            raise SystemExit("--output-dir is accepted only together with --dry-run")
        return Locations(
            config.output_dir / sample_name, config.transcripts_dir / sample_name / "transcripts"
        )
    if output_dir is None:
        if not create_temp:
            raise SystemExit("a dry-run repeat stage needs --output-dir of its dry-run sample")
        output_dir = Path(tempfile.mkdtemp(prefix="validation_dry_run_"))
    base = output_dir.resolve()
    if base == REPOSITORY_DIR or REPOSITORY_DIR in base.parents:
        raise SystemExit(f"dry-run output must be outside the repository {REPOSITORY_DIR}")
    return Locations(
        base / "validation" / sample_name, base / "cache" / sample_name / "transcripts"
    )


def load_split_lookup(manifest_path: Path, lds_sha256: str) -> tuple[dict[int, str], Record]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != lds_sha256:
        raise SystemExit(f"{manifest_path} was built from another LDS file")
    lookup: dict[int, str] = {}
    for name in SPLIT_NAMES:
        for hearing_id in manifest[name]:
            if hearing_id in lookup:
                raise SystemExit(f"hearing {hearing_id} is listed in two splits")
            lookup[hearing_id] = name
    source = {
        "path": str(manifest_path),
        "sha256": sha256_of_file(manifest_path),
        "split_version": manifest["split_version"],
    }
    return lookup, source


def calibration_artifact_problems(
    artifact: Record,
    threshold: float,
    splits: tuple[str, ...],
    split_of: dict[int, str],
    manifest_sha256: str,
) -> tuple[Record, list[str]]:
    leak = artifact.get("leak_check") or {}
    calibration_splits = list(leak.get("calibration_splits") or [])
    hearings = list(leak.get("hearing_ids_used") or [])
    rules = artifact.get("rules") or {}
    producing = sorted(
        name for name, rule in rules.items() if rule.get("threshold_rounded") == threshold
    )
    manifest = (artifact.get("sources") or {}).get("splits", {}).get("sha256")
    overlap = sorted(hearing for hearing in hearings if split_of.get(hearing) in splits)
    details = {
        "calibration_version": artifact.get("calibration_version"),
        "calibration_splits": calibration_splits,
        "calibration_hearings": len(hearings),
        "calibration_hearings_in_sampled_splits": overlap,
        "rules_giving_this_threshold": producing,
        "leak_check_passed": leak.get("passed"),
        "split_manifest_matches": manifest == manifest_sha256,
    }
    problems = []
    if not calibration_splits or not hearings:
        problems.append("the calibration artifact records no calibration splits or hearings")
    if "test" in calibration_splits or set(calibration_splits) & set(splits):
        problems.append(
            f"calibrated on {calibration_splits}, which includes test or a sampled split"
        )
    if overlap:
        problems.append(f"{len(overlap)} calibration hearings belong to the sampled splits")
    if not producing:
        problems.append(f"no rule of the calibration artifact gives the threshold {threshold}")
    if leak.get("passed") is not True:
        problems.append("the leak check of the calibration artifact did not pass")
    if manifest != manifest_sha256:
        problems.append("the calibration artifact was built on another split manifest")
    return details, problems


def threshold_calibration_check(
    coverage: Record,
    splits: tuple[str, ...],
    split_of: dict[int, str],
    manifest_sha256: str,
    rule: str,
) -> Record:
    evidence = coverage["config"]["evidence"]
    threshold = evidence["embedding_threshold"]
    source = evidence.get("calibration_source")
    record: Record = {
        "rule": rule,
        "embedding_threshold": threshold,
        "calibration_source": source,
        "calibration_method": evidence.get("calibration_method"),
        "sampled_splits": list(splits),
    }
    path = Path(source) if isinstance(source, str) else None
    if path is None or path.suffix != ".json" or not path.is_file():
        problems = [f"calibration_source {source!r} is not a calibration artifact file"]
    else:
        with open(path) as f:
            artifact = json.load(f)
        details, problems = calibration_artifact_problems(
            artifact, threshold, splits, split_of, manifest_sha256
        )
        record |= {"calibration_artifact_sha256": sha256_of_file(path), **details}
    return {**record, "problems": problems, "passed": not problems}


def load_coverage(config: ValidationConfig, run_name: str) -> Record:
    run_path = config.udv_dir / f"{run_name}.jsonl"
    coverage_path = config.udv_dir / f"{run_name}_coverage.json"
    for path in (run_path, coverage_path):
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
    with open(coverage_path) as f:
        coverage: Record = json.load(f)
    if coverage["config"]["dataset"]["sha256"] != config.lds_sha256:
        raise SystemExit(f"{coverage_path} was built from another LDS file")
    return coverage


def load_run(
    config: ValidationConfig, run_name: str, split_of: dict[int, str], readable: set[str]
) -> tuple[list[Record], Record]:
    run_path = config.udv_dir / f"{run_name}.jsonl"
    coverage_path = config.udv_dir / f"{run_name}_coverage.json"
    coverage = load_coverage(config, run_name)
    all_records = load_jsonl(run_path)
    missing = sorted({r["hearing_id"] for r in all_records if r["hearing_id"] not in split_of})
    if missing:
        raise SystemExit(f"hearings of the run missing from the split manifest: {missing}")
    records = [record for record in all_records if split_of[record["hearing_id"]] in readable]
    source = {
        "run_name": run_name,
        "path": str(run_path),
        "sha256": sha256_of_file(run_path),
        "coverage_path": str(coverage_path),
        "coverage_sha256": sha256_of_file(coverage_path),
        "coverage_created_at": coverage.get("created_at"),
        "records_in_run": len(all_records),
        "records_read": len(records),
        "embedding_threshold": coverage["config"]["evidence"]["embedding_threshold"],
        "encoders": sorted(
            {f"{r['method']['encoder']}@{r['method']['revision']}" for r in records}
        ),
    }
    return records, source


def record_order(record: Record) -> tuple[int, ...]:
    return tuple(int(part) for part in record["id"].split("-")[1:])


def support_of(record: Record) -> str:
    evidence = record["evidence"]
    return NO_EVIDENCE_SUPPORT if evidence is None else evidence["support_type"]


def stratum_of(record: Record, strata: tuple[Stratum, ...]) -> Stratum:
    cell = (record["tier"], support_of(record))
    matches = [s for s in strata if cell[0] in s.tiers and cell[1] in s.support_types]
    if len(matches) != 1:
        raise SystemExit(
            f"{record['id']}: (tier, support) {cell} matches {len(matches)} strata; the strata "
            "must partition every UDV of the run"
        )
    return matches[0]


def population_counts(
    records: list[Record],
    split_of: dict[int, str],
    strata: tuple[Stratum, ...],
    readable: set[str],
) -> dict[str, dict[str, int]]:
    counts = {
        split: {stratum.name: 0 for stratum in strata} for split in SPLIT_NAMES if split in readable
    }
    for record in records:
        counts[split_of[record["hearing_id"]]][stratum_of(record, strata).name] += 1
    return counts


def sum_population(counts: dict[str, dict[str, int]], splits: tuple[str, ...]) -> dict[str, int]:
    names = next(iter(counts.values())).keys()
    return {name: sum(counts[split][name] for split in splits) for name in names}


def draw_strata(
    records: list[Record], strata: tuple[Stratum, ...], rng: np.random.Generator
) -> tuple[dict[str, list[Record]], dict[str, Record]]:
    pools: dict[str, list[Record]] = {stratum.name: [] for stratum in strata}
    for record in records:
        pools[stratum_of(record, strata).name].append(record)
    drawn: dict[str, list[Record]] = {}
    summary: dict[str, Record] = {}
    for stratum in strata:
        pool = sorted(pools[stratum.name], key=record_order)
        size = min(stratum.target, len(pool))
        picks = sorted(int(i) for i in rng.choice(len(pool), size=size, replace=False))
        drawn[stratum.name] = [pool[index] for index in picks]
        summary[stratum.name] = {
            "question": stratum.question,
            "target": stratum.target,
            "population": len(pool),
            "drawn_udvs": size,
            "all_taken": size == len(pool),
            "shortfall": stratum.target - size,
        }
    return drawn, summary


def header_text(match: Any) -> str:
    return normalize_whitespace(match.group(0)).rstrip(" -")


def hearing_view(hearing: Record) -> Record:
    transcript = hearing["transcricao"]
    matches = list(TURN_HEADER_PATTERN.finditer(transcript))
    return {
        "id": hearing["id"],
        "transcript": transcript,
        "turns": split_into_turns(transcript),
        "headers": [header_text(match) for match in matches],
        "opening_end": matches[0].start() if matches else len(transcript),
    }


def format_offset(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def context_window(
    transcript: str, turn: Record, start: int, end: int, chars: int
) -> tuple[str, str]:
    left = max(turn["start_char"], start - chars)
    right = min(turn["end_char"], end + chars)
    before = normalize_whitespace(transcript[left:start])
    after = normalize_whitespace(transcript[end:right])
    if left > turn["start_char"]:
        before = "… " + before.partition(" ")[2]
    else:
        before = ("[início do turno] " + before).strip()
    if right < turn["end_char"]:
        after = after.rpartition(" ")[0] + " …"
    else:
        after = (after + " [fim do turno]").strip()
    return before, after


def quote_cue_in_trecho(statement: str, passage: str) -> bool:
    return any(
        quote_prefix_pattern(prefix).search(passage)
        for quote in extract_quotes(statement)
        for prefix, _ in quote_prefixes(quote)
    )


def evidence_item(record: Record, view: Record, stratum: Stratum, chars: int) -> Record:
    evidence = record["evidence"]
    start, end, turn_index = evidence["start_char"], evidence["end_char"], evidence["speaker_turn"]
    if start is None:
        before, after = "", ""
        link = "sem localização na transcrição; procure o trecho no arquivo da audiência"
    else:
        transcript = view["transcript"]
        if normalize_whitespace(transcript[start:end]) != evidence["text"]:
            raise SystemExit(f"{record['id']}: offsets do not reproduce the evidence text")
        turn = view["turns"][turn_index]
        if not turn["start_char"] <= start < end <= turn["end_char"]:
            raise SystemExit(f"{record['id']}: evidence span is outside turn {turn_index}")
        before, after = context_window(transcript, turn, start, end, chars)
        link = (
            f"caracteres {format_offset(start)}–{format_offset(end)} do turno {turn_index}; "
            f"cabeçalho do turno: {view['headers'][turn_index]}"
        )
    return {
        "question": stratum.question,
        "stratum": stratum.name,
        "udv_ids": [record["id"]],
        "tier": record["tier"],
        "support_type": evidence["support_type"],
        "score": evidence["score"],
        "quote_prefix": evidence["quote_prefix"],
        "quote_cue_in_trecho": quote_cue_in_trecho(record["proposition"], evidence["text"]),
        "speaker_turn": turn_index,
        "start_char": start,
        "end_char": end,
        "display": {
            "hearing_id": str(record["hearing_id"]),
            "pergunta": stratum.question,
            "participante": record["actor"]["name"],
            "cargo": record["actor"]["role"] or "",
            "afirmacao": record["proposition"],
            "trecho": evidence["text"],
            "contexto_antes": before,
            "contexto_depois": after,
            "link": link,
        },
    }


def speaker_directory(view: Record) -> str:
    counts: dict[str, int] = {}
    for header in view["headers"]:
        counts[header] = counts.get(header, 0) + 1
    entries = "; ".join(f"{header} [{count}]" for header, count in counts.items())
    return f"Cabeçalhos de fala detectados na transcrição [número de turnos]: {entries}"


def speaker_items(
    records: list[Record], views: dict[int, Record], stratum: Stratum
) -> list[Record]:
    groups: dict[tuple[int, int], list[Record]] = {}
    for record in records:
        hearing_id, person_index, _ = record_order(record)
        groups.setdefault((hearing_id, person_index), []).append(record)
    items = []
    for (hearing_id, person_index), members in groups.items():
        tiers = {member["tier"] for member in members}
        if len(tiers) != 1 or any(member["evidence"] is not None for member in members):
            raise SystemExit(f"hearing {hearing_id}, person {person_index}: mixed speaker records")
        actor = members[0]["actor"]
        if len(members) == 1:
            statement = members[0]["proposition"]
        else:
            statement = " ".join(
                f"[{number}] {member['proposition']}" for number, member in enumerate(members, 1)
            )
        items.append(
            {
                "question": stratum.question,
                "stratum": stratum.name,
                "udv_ids": [member["id"] for member in members],
                "tier": tiers.pop(),
                "support_type": None,
                "score": None,
                "quote_prefix": None,
                "quote_cue_in_trecho": None,
                "person_index": person_index,
                "display": {
                    "hearing_id": str(hearing_id),
                    "pergunta": stratum.question,
                    "participante": actor["name"],
                    "cargo": actor["role"] or "",
                    "afirmacao": statement,
                    "trecho": "",
                    "contexto_antes": speaker_directory(views[hearing_id]),
                    "contexto_depois": "",
                    "link": "sem trecho; consulte a transcrição inteira da audiência",
                },
            }
        )
    return items


def build_items(
    drawn: dict[str, list[Record]],
    strata: tuple[Stratum, ...],
    views: dict[int, Record],
    chars: int,
) -> list[Record]:
    items: list[Record] = []
    for stratum in strata:
        records = drawn[stratum.name]
        if stratum.question == "pessoa_falou":
            items.extend(speaker_items(records, views, stratum))
        else:
            items.extend(
                evidence_item(record, views[record["hearing_id"]], stratum, chars)
                for record in records
            )
    return items


def assign_item_ids(
    items: list[Record], rng: np.random.Generator, prefix: str
) -> dict[str, Record]:
    order = rng.permutation(len(items))
    width = max(3, len(str(len(items))))
    return {
        f"{prefix}{position:0{width}d}": items[int(index)]
        for position, index in enumerate(order, start=1)
    }


def spreadsheet_value(text: str) -> str:
    return "'" + text if text[:1] in FORMULA_PREFIXES else text


def restore_value(text: str) -> str:
    return text[1:] if text[:1] == "'" and text[1:2] in FORMULA_PREFIXES else text


def sheet_rows(items: dict[str, Record]) -> list[Record]:
    return [
        {"item_id": item_id, **item["display"], **{column: "" for column in FILL_COLUMNS}}
        for item_id, item in items.items()
    ]


def write_annotation_csv(rows: list[Record], path: Path, config: ValidationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=config.csv_encoding, newline="") as f:
        writer = csv.writer(
            f, delimiter=config.csv_delimiter, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([spreadsheet_value(row[column]) for column in CSV_COLUMNS])


def detect_delimiter(header_line: str) -> str | None:
    for delimiter in CSV_DELIMITERS:
        header = [name.strip() for name in next(csv.reader([header_line], delimiter=delimiter))]
        if set(CSV_COLUMNS) <= set(header):
            return delimiter
    return None


def read_annotation_csv(path: Path, config: ValidationConfig) -> list[Record]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    try:
        text = path.read_text(encoding=config.csv_encoding)
    except UnicodeDecodeError as error:
        raise SystemExit(
            f"{path} is not {config.csv_encoding}; save it as CSV UTF-8: {error}"
        ) from error
    lines = text.splitlines()
    delimiter = detect_delimiter(lines[0]) if lines else None
    if delimiter is None:
        raise SystemExit(f"{path}: the header must contain the columns {CSV_COLUMNS}")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    rows = []
    for row in reader:
        cleaned = {
            (name or "").strip(): value or "" for name, value in row.items() if name is not None
        }
        if any(value.strip() for value in cleaned.values()):
            rows.append(cleaned)
    return rows


def normalize_label(value: str) -> str:
    label = strip_accents(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label


def validate_annotation(
    rows: list[Record], key: Record, config: ValidationConfig
) -> tuple[dict[str, Record], list[str]]:
    items = key["items"]
    judged: dict[str, Record] = {}
    problems: list[str] = []
    for line, row in enumerate(rows, start=2):
        item_id = row.get("item_id", "").strip()
        if item_id not in items:
            problems.append(f"line {line}: item_id {item_id!r} is not in the key")
            continue
        if item_id in judged:
            problems.append(f"line {line}: item_id {item_id} appears more than once")
            continue
        item = items[item_id]
        for column in CHECKED_COLUMNS:
            shown = normalize_whitespace(restore_value(row.get(column, "")))
            if shown != normalize_whitespace(item["display"][column]):
                problems.append(f"{item_id}: column {column} differs from the generated sheet")
        question = config.questions[item["question"]]
        judgment = normalize_label(row.get("julgamento", ""))
        if not judgment:
            problems.append(f"{item_id}: julgamento is empty")
        elif judgment not in question.judgments:
            problems.append(
                f"{item_id}: julgamento {row['julgamento']!r} is not one of {question.judgments}"
            )
        better = normalize_label(row.get("existe_trecho_melhor", ""))
        if not better and question.better_passage_required:
            problems.append(f"{item_id}: existe_trecho_melhor is empty")
        elif better and better not in config.better_passage_values:
            problems.append(
                f"{item_id}: existe_trecho_melhor {row['existe_trecho_melhor']!r} is not one of "
                f"{config.better_passage_values}"
            )
        judged[item_id] = {
            "judgment": judgment,
            "better_passage": better or None,
            "note": row.get("observacao", "").strip(),
        }
    for item_id in sorted(set(items) - set(judged)):
        problems.append(f"{item_id}: row missing from the sheet")
    return judged, problems


def forbidden_token_hits(rows: list[Record], strata: tuple[Stratum, ...]) -> list[str]:
    tokens = {*TIERS, *SUPPORT_TYPES, *(stratum.name for stratum in strata)}
    return [
        f"{row['item_id']}.{column}"
        for row in rows
        for column in CSV_COLUMNS
        if any(token in row[column] for token in tokens)
    ]


def render_transcript(view: Record) -> str:
    lines = [
        f"Audiência {view['id']}: transcrição completa.",
        "Cada turno começa com [turno N | caracteres início–fim] seguido do cabeçalho de fala.",
        "",
    ]
    opening = view["transcript"][: view["opening_end"]].strip()
    if opening:
        lines += ["[antes do primeiro cabeçalho de fala]", opening, ""]
    for turn, header in zip(view["turns"], view["headers"], strict=True):
        span = f"{format_offset(turn['start_char'])}–{format_offset(turn['end_char'])}"
        lines += [f"[turno {turn['turn_index']} | caracteres {span}] {header}", turn["speech"], ""]
    return "\n".join(lines)


def write_transcripts(views: dict[int, Record], directory: Path) -> Record:
    directory.mkdir(parents=True, exist_ok=True)
    for hearing_id, view in sorted(views.items()):
        (directory / f"{hearing_id}.txt").write_text(render_transcript(view), encoding="utf-8")
    return {"dir": str(directory), "files": len(views), "hearing_ids": sorted(views)}


def file_entry(path: Path) -> Record:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_of_file(path)}


def ensure_new_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} already exists and is not empty; it is never overwritten")


def code_hashes() -> Record:
    return {
        **source_hashes(transcript, *transcript.SOURCES),
        **source_hashes(udv_run),
        **source_hashes(Path(__file__)),
    }


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def trials_by_stratum(items: dict[str, Record]) -> dict[str, int]:
    trials: dict[str, int] = {}
    for item in items.values():
        trials[item["stratum"]] = trials.get(item["stratum"], 0) + len(item["udv_ids"])
    return trials


def quote_cue_counts(items: dict[str, Record], strata: tuple[Stratum, ...]) -> Record:
    counts: Record = {}
    for stratum in strata:
        if stratum.question != "trecho_sustenta":
            continue
        flags = [
            bool(item["quote_cue_in_trecho"])
            for item in items.values()
            if item["stratum"] == stratum.name
        ]
        counts[stratum.name] = {"rows": len(flags), "with_cue": sum(flags)}
    return counts


def blinding_section(
    rows: list[Record], items: dict[str, Record], config: ValidationConfig
) -> Record:
    hits = forbidden_token_hits(rows, config.strata)
    if hits:
        raise SystemExit(f"blinding check failed, tier or stratum names in the sheet: {hits[:10]}")
    return {
        "csv_columns": list(CSV_COLUMNS),
        "hidden_in_key_only": list(HIDDEN_ITEM_FIELDS),
        "row_order": "random permutation (seed stream row_order); item ids follow the row order",
        "forbidden_token_hits": 0,
        "known_leaks": [
            "a direct_quote item can be recognized when the afirmacao holds a quotation whose "
            "words reappear in trecho",
            "a semantic_with_short_quote item is recognizable in the same way: this support type "
            "is assigned only when a quote prefix of the afirmacao is found in the sentence that "
            "becomes trecho, so the quoted words reappear in trecho; semantic_similarity items "
            "(the semantic_match_high and semantic_match_weak strata) show the cue only by "
            "coincidence, so only high against weak stays blind",
            "pessoa_falou rows are recognizable by their question, which is asked only of the "
            "speaker_check stratum",
        ],
        "quote_cue": {
            "definition": (
                "quote_cue_in_trecho is true when a quotation extracted from afirmacao "
                "(udv_pipeline.extract_quotes) has a prefix of 10, 6, 4 or 3 words "
                "(udv_pipeline.quote_prefixes) that udv_pipeline.quote_prefix_pattern finds in "
                "trecho; it is stored per item in the key only and the precision report splits "
                "each evidence stratum by it"
            ),
            "rows_by_stratum": quote_cue_counts(items, config.strata),
        },
    }


def run_sample_stage(args: argparse.Namespace, config: ValidationConfig) -> None:
    started = time.perf_counter()
    splits = resolve_splits(config, args.splits, args.dry_run, args.final_test)
    readable = readable_splits(splits, args.final_test)
    sample_name = f"{config.version}_{args.run_name}"
    locations = resolve_locations(config, sample_name, args.dry_run, args.output_dir, True)
    ensure_new_dir(locations.sample_dir)
    ensure_new_dir(locations.transcripts_dir)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    calibration = threshold_calibration_check(
        load_coverage(config, args.run_name),
        splits,
        split_of,
        split_source["sha256"],
        config.source["source"]["threshold_calibration_rule"],
    )
    if not calibration["passed"]:
        message = (
            f"the tier boundary of {args.run_name} (embedding_threshold "
            f"{calibration['embedding_threshold']}) has no calibration record that excludes test "
            f"and the sampled splits: {calibration['problems']}"
        )
        if not args.dry_run:
            raise SystemExit(f"{message}; the sample is refused")
        print(f"WARNING {message}; accepted only because this is a dry run, and recorded")
    records, run_source = load_run(config, args.run_name, split_of, readable)
    by_split = population_counts(records, split_of, config.strata, readable)
    sampled_records = [r for r in records if split_of[r["hearing_id"]] in splits]
    drawn, draw_summary = draw_strata(
        sampled_records, config.strata, stream_rng(config, "stratum_draw")
    )
    hearing_ids = {record["hearing_id"] for group in drawn.values() for record in group}
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    views = {h["id"]: hearing_view(h) for h in lds if h["id"] in hearing_ids}
    del lds
    items = assign_item_ids(
        build_items(drawn, config.strata, views, config.context_chars),
        stream_rng(config, "row_order"),
        config.item_id_prefix,
    )
    rows = sheet_rows(items)
    blinding = blinding_section(rows, items, config)
    rows_by_stratum: dict[str, int] = {}
    for item in items.values():
        rows_by_stratum[item["stratum"]] = rows_by_stratum.get(item["stratum"], 0) + 1
    for name, summary in draw_summary.items():
        summary["rows"] = rows_by_stratum.get(name, 0)
    all_available = set(SPLIT_NAMES) <= readable
    population = {
        "unit": "UDV",
        "sampled_splits": sum_population(by_split, splits),
        "by_split": by_split,
        "all_splits": sum_population(by_split, SPLIT_NAMES) if all_available else None,
        "all_splits_note": (
            "all three splits were readable"
            if all_available
            else "test was not readable "
            "(no --final-test), so the population of the whole run is not recorded"
        ),
    }
    criteria = config.source["criteria"]
    csv_path = locations.sample_dir / ANNOTATION_CSV
    write_annotation_csv(rows, csv_path, config)
    transcripts = write_transcripts(views, locations.transcripts_dir)
    key = {
        "role": "annotation",
        "sample_name": sample_name,
        "sample_version": config.version,
        "created_at": now_iso(),
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": list(splits),
        "splits_declared": list(config.sample_splits),
        "run": run_source,
        "threshold_calibration": calibration,
        "population": population,
        "strata": [asdict(stratum) for stratum in config.strata],
        "sample": draw_summary,
        "criteria": criteria,
        "criteria_sha256": canonical_sha256(criteria),
        "csv": {
            "file": ANNOTATION_CSV,
            "delimiter": config.csv_delimiter,
            "encoding": config.csv_encoding,
            "columns": list(CSV_COLUMNS),
            "sha256_at_creation": sha256_of_file(csv_path),
        },
        "transcripts_dir": str(locations.transcripts_dir),
        "items": items,
    }
    key_path = locations.sample_dir / ANNOTATION_KEY
    write_json(key, key_path)
    with_offsets = sum(1 for item in items.values() if item.get("start_char") is not None)
    report = {
        "stage": "sample",
        "sample_name": sample_name,
        "created_at": key["created_at"],
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": list(splits),
        "splits_declared": list(config.sample_splits),
        "splits_reason": config.source["splits"]["sample_splits_reason"],
        "splits_readable": sorted(readable),
        "split_manifest": split_source,
        "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
        "run": run_source,
        "threshold_calibration": calibration,
        "population": population,
        "sample": {
            "by_stratum": draw_summary,
            "rows": len(rows),
            "udvs": sum(len(item["udv_ids"]) for item in items.values()),
            "hearings": len(hearing_ids),
            "evidence_rows_with_offsets": with_offsets,
            "strata_with_shortfall": [n for n, s in draw_summary.items() if s["shortfall"] > 0],
        },
        "criteria": criteria,
        "criteria_sha256": key["criteria_sha256"],
        "criteria_feasibility_at_drawn_size": criteria_feasibility(
            config, trials_by_stratum(items)
        ),
        "blinding": blinding,
        "seeds": {"seed": config.seed, "streams": config.seed_streams, "generator": "numpy PCG64"},
        "outputs": {
            "annotation_csv": file_entry(csv_path),
            "annotation_key": file_entry(key_path),
            "transcripts": transcripts,
        },
        "code": code_hashes(),
        "timing": {"elapsed_seconds": round(time.perf_counter() - started, 1)},
        "environment": environment(),
        "config": config.source,
    }
    write_json(report, locations.sample_dir / SAMPLE_REPORT)
    print(f"sample {sample_name} from {list(splits)} -> {locations.sample_dir}")
    for name, summary in draw_summary.items():
        print(
            f"  {name:26s} population={summary['population']:5d} target={summary['target']:3d} "
            f"drawn={summary['drawn_udvs']:3d} rows={summary['rows']:3d}"
            + (" (all taken)" if summary["all_taken"] else "")
        )
    print(f"  {len(rows)} rows, transcripts in {locations.transcripts_dir}")


def load_key(path: Path, role: str) -> Record:
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    with open(path) as f:
        key: Record = json.load(f)
    if key.get("role") != role:
        raise SystemExit(f"{path} is not a {role} key")
    return key


def existing_precision_reports(sample_dir: Path) -> list[Record]:
    found = sorted(
        {path for pattern in PRECISION_REPORT_PATTERNS for path in sample_dir.glob(pattern)}
    )
    entries = []
    for path in found:
        with open(path) as f:
            payload = json.load(f)
        entries.append(
            {
                "file": path.name,
                "sha256": sha256_of_file(path),
                "created_at": payload.get("created_at"),
                "interim": payload.get("interim"),
            }
        )
    return entries


def repeat_same_relative_order(chosen: list[str], new_items: dict[str, Record]) -> bool:
    return [item["original_item_id"] for item in new_items.values()] == sorted(chosen)


def run_repeat_stage(args: argparse.Namespace, config: ValidationConfig) -> None:
    started = time.perf_counter()
    if args.splits is not None:
        raise SystemExit("--splits is not used by the repeat stage")
    sample_name = f"{config.version}_{args.run_name}"
    locations = resolve_locations(config, sample_name, args.dry_run, args.output_dir, False)
    key_path = locations.sample_dir / ANNOTATION_KEY
    key = load_key(key_path, "annotation")
    if key["sample_name"] != sample_name or key["dry_run"] != args.dry_run:
        raise SystemExit(f"{key_path} belongs to another sample or mode")
    require_final_test(key, args.final_test)
    csv_path = locations.sample_dir / REANNOTATION_CSV
    repeat_key_path = locations.sample_dir / REANNOTATION_KEY
    if csv_path.exists() or repeat_key_path.exists():
        raise SystemExit(f"the repeat sheet of {sample_name} already exists; it is never redrawn")
    first_path = locations.sample_dir / ANNOTATION_CSV
    earlier_reports = existing_precision_reports(locations.sample_dir)
    if earlier_reports:
        print(
            f"WARNING precision numbers were computed before the repeat sheet exists "
            f"({[entry['file'] for entry in earlier_reports]}); the repeat sheet is created and "
            "this is recorded in reannotation_key.json and in the final precision report"
        )
    _, problems = validate_annotation(read_annotation_csv(first_path, config), key, config)
    if problems:
        print("\n".join(problems[:50]))
        raise SystemExit(
            f"{len(problems)} problems in {first_path}; the repeat sheet is created only after "
            "the first pass is complete"
        )
    modified = datetime.fromtimestamp(first_path.stat().st_mtime, UTC)
    hours = (datetime.now(UTC) - modified).total_seconds() / 3600
    if hours < config.repeat_min_hours:
        raise SystemExit(
            f"{first_path} was last modified {hours:.1f} h ago; the repeat sheet needs at least "
            f"{config.repeat_min_hours:g} h"
        )
    eligible = sorted(
        item_id
        for item_id, item in key["items"].items()
        if item["question"] in config.repeat_questions
    )
    size = min(config.repeat_items, len(eligible))
    picks = stream_rng(config, "repeat_draw").choice(len(eligible), size=size, replace=False)
    chosen = sorted(eligible[int(index)] for index in picks)
    originals = [
        {
            "original_item_id": item_id,
            "question": key["items"][item_id]["question"],
            "display": key["items"][item_id]["display"],
        }
        for item_id in chosen
    ]
    new_items = assign_item_ids(
        originals, stream_rng(config, "repeat_order"), config.repeat_item_id_prefix
    )
    rows = sheet_rows(new_items)
    write_annotation_csv(rows, csv_path, config)
    repeat_key = {
        "role": "reannotation",
        "sample_name": sample_name,
        "created_at": now_iso(),
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": key["splits_used"],
        "annotation_key_sha256": sha256_of_file(key_path),
        "first_pass": {
            "file": ANNOTATION_CSV,
            "sha256": sha256_of_file(first_path),
            "modified_at": modified.isoformat(timespec="seconds"),
            "hours_since_modification": round(hours, 2),
        },
        "precision_reports_before_repeat_sheet": earlier_reports,
        "eligible_questions": list(config.repeat_questions),
        "eligible_items": len(eligible),
        "requested_items": config.repeat_items,
        "drawn_items": size,
        "same_relative_order_as_first_pass": repeat_same_relative_order(chosen, new_items),
        "csv": {
            "file": REANNOTATION_CSV,
            "delimiter": config.csv_delimiter,
            "encoding": config.csv_encoding,
            "columns": list(CSV_COLUMNS),
            "sha256_at_creation": sha256_of_file(csv_path),
        },
        "items": new_items,
    }
    write_json(repeat_key, repeat_key_path)
    report = {
        "stage": "repeat",
        "sample_name": sample_name,
        "created_at": repeat_key["created_at"],
        "dry_run": args.dry_run,
        "final_test": args.final_test,
        "splits_used": key["splits_used"],
        "first_pass": repeat_key["first_pass"],
        "precision_reports_before_repeat_sheet": earlier_reports,
        "eligible_questions": list(config.repeat_questions),
        "eligible_reason": config.source["repeat"]["eligible_reason"],
        "eligible_items": len(eligible),
        "drawn_items": size,
        "same_relative_order_as_first_pass": repeat_key["same_relative_order_as_first_pass"],
        "seeds": {"seed": config.seed, "streams": config.seed_streams, "generator": "numpy PCG64"},
        "outputs": {
            "reannotation_csv": file_entry(csv_path),
            "reannotation_key": file_entry(repeat_key_path),
        },
        "code": code_hashes(),
        "timing": {"elapsed_seconds": round(time.perf_counter() - started, 1)},
        "environment": environment(),
        "config": config.source,
    }
    write_json(report, locations.sample_dir / REANNOTATION_REPORT)
    print(f"repeat sheet with {size} of {len(eligible)} eligible items -> {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the stratified human validation sample of a UDV run (stage sample) or, "
        "after the first pass is complete, the blind repeat sheet (stage repeat)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/validation_sample.toml"))
    parser.add_argument("--stage", choices=STAGES, default="sample")
    parser.add_argument("--run-name", required=True, help="UDV run under source.udv_dir")
    parser.add_argument(
        "--final-test", action="store_true", help="allow reading test hearings (final sample)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write outside the repository (a new temp dir unless --output-dir is given)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="dry runs only")
    parser.add_argument(
        "--splits", nargs="+", default=None, help="dry runs only: splits to sample instead"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_validation_config(args.config)
    if args.stage == "sample":
        run_sample_stage(args, config)
    else:
        run_repeat_stage(args, config)


if __name__ == "__main__":
    main()
