"""The blind annotation sheet: its CSV columns, writing and reading it, checking a filled one."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from bookworm import sha256_of_file

from experiments.common.transcript import normalize_whitespace, strip_accents
from experiments.validation.generate_sample.config import CSV_DELIMITERS, ValidationConfig

Record = dict[str, Any]

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
FORMULA_PREFIXES = ("=", "+", "-", "@")
ANNOTATION_CSV = "annotation.csv"
ANNOTATION_KEY = "annotation_key.json"
REANNOTATION_CSV = "reannotation.csv"
REANNOTATION_KEY = "reannotation_key.json"
SAMPLE_REPORT = "sample_report.json"
REANNOTATION_REPORT = "reannotation_report.json"
PRECISION_REPORT_PATTERNS = ("precision_report*.json",)


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


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


def csv_section(file_name: str, path: Path, config: ValidationConfig) -> Record:
    return {
        "file": file_name,
        "delimiter": config.csv_delimiter,
        "encoding": config.csv_encoding,
        "columns": list(CSV_COLUMNS),
        "sha256_at_creation": sha256_of_file(path),
    }


def detect_delimiter(header_line: str) -> str | None:
    for delimiter in CSV_DELIMITERS:
        header = [name.strip() for name in next(csv.reader([header_line], delimiter=delimiter))]
        if set(CSV_COLUMNS) <= set(header):
            return delimiter
    return None


def read_annotation_csv(path: Path, config: ValidationConfig) -> list[Record]:
    """The non-empty rows of a filled sheet, in any delimiter a spreadsheet may save it with."""
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


def shown_column_problems(item_id: str, row: Record, item: Record) -> list[str]:
    problems = []
    for column in CHECKED_COLUMNS:
        shown = normalize_whitespace(restore_value(row.get(column, "")))
        if shown != normalize_whitespace(item["display"][column]):
            problems.append(f"{item_id}: column {column} differs from the generated sheet")
    return problems


def label_problems(item_id: str, row: Record, item: Record, config: ValidationConfig) -> list[str]:
    problems = []
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
    return problems


def judgment_of(row: Record) -> Record:
    return {
        "judgment": normalize_label(row.get("julgamento", "")),
        "better_passage": normalize_label(row.get("existe_trecho_melhor", "")) or None,
        "note": row.get("observacao", "").strip(),
    }


def validate_annotation(
    rows: list[Record], key: Record, config: ValidationConfig
) -> tuple[dict[str, Record], list[str]]:
    """The judgment of each row of a filled sheet and every problem found against its key."""
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
        problems += shown_column_problems(item_id, row, items[item_id])
        problems += label_problems(item_id, row, items[item_id], config)
        judged[item_id] = judgment_of(row)
    for item_id in sorted(set(items) - set(judged)):
        problems.append(f"{item_id}: row missing from the sheet")
    return judged, problems


def load_key(path: Path, role: str) -> Record:
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    with open(path) as f:
        key: Record = json.load(f)
    if key.get("role") != role:
        raise SystemExit(f"{path} is not a {role} key")
    return key


def require_final_test(key: Record, final_test: bool) -> None:
    if "test" in key["splits_used"] and not final_test:
        raise SystemExit(
            "this sample was drawn from test hearings; pass --final-test to read its sheets"
        )


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


def ensure_new_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} already exists and is not empty; it is never overwritten")
