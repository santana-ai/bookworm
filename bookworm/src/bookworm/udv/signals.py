"""Verifier, question and translation signals of each UDV, read from a verifier report."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm.data.io import JsonObject, is_json_number, read_json_object, sha256_of_file
from bookworm.errors import ConfigError
from bookworm.transcript.sentences import SENTENCE_BOUNDARY_PATTERN, is_sentence
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.schemas import UdvRecord

LAYA_SCORERS = ("laya_multi_pt", "laya_en_en")
XNLI_SCORER = "xnli_mdeberta"
SCORERS = (*LAYA_SCORERS, XNLI_SCORER)
TRANSLATED_SCORER = "laya_en_en"
XNLI_LABELS = ("entailment", "neutral", "contradiction")
CONTENT_PATTERN = re.compile(r"\w")
TRANSLATION_KEY_SEPARATOR = "\x1e"
HEAVY_ARTIFACT_MANIFEST = "challenge/artifacts/MANIFEST_heavy.tsv"
UNSCORED: JsonObject = {
    "scored": False,
    "verifier": None,
    "laya": None,
    "xnli": None,
    "translation": None,
}


def has_content(text: str) -> bool:
    return CONTENT_PATTERN.search(text) is not None


def translation_key(signature_sha256: str, text: str) -> str:
    return hashlib.sha256(
        f"{signature_sha256}{TRANSLATION_KEY_SEPARATOR}{text}".encode()
    ).hexdigest()


@dataclass(frozen=True)
class Segmentation:
    join_abbreviations: frozenset[str]
    join_short_parts: bool

    def needs_join(self, text: str) -> bool:
        if text.split()[-1] in self.join_abbreviations:
            return True
        return self.join_short_parts and not is_sentence(text)

    def segments(self, chunk: str) -> list[str]:
        parts = [
            part
            for part in (
                normalize_whitespace(raw)
                for raw in SENTENCE_BOUNDARY_PATTERN.split(normalize_whitespace(chunk))
            )
            if part
        ]
        units: list[str] = []
        pending = ""
        for part in parts:
            text = f"{pending} {part}" if pending else part
            if self.needs_join(text):
                pending = text
                continue
            units.append(text)
            pending = ""
        if pending and units:
            units[-1] = f"{units[-1]} {pending}"
        elif pending:
            units.append(pending)
        return units


@dataclass(frozen=True)
class Translations:
    signature_sha256: str
    segmentation: Segmentation
    entries: Mapping[str, str]
    path: Path

    def lookup(self, text: str) -> str:
        normalized = normalize_whitespace(text)
        if not has_content(normalized):
            return normalized
        found = self.entries.get(translation_key(self.signature_sha256, normalized))
        if found is None:
            raise ConfigError(
                f"{self.path}: no translation of a {len(normalized)}-character text under "
                f"signature {self.signature_sha256[:16]}"
            )
        return found

    def translate_opinion(self, opinion: str) -> str:
        return self.lookup(opinion)

    def translate_chunk(self, chunk: str) -> str:
        translated = (self.lookup(segment).strip() for segment in self.segmentation.segments(chunk))
        return " ".join(text for text in translated if text)


def read_rows(path: Path, description: str) -> list[JsonObject]:
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConfigError(
                    f"{path}: line {number} of the {description} is not JSON"
                ) from error
            if not isinstance(row, dict):
                raise ConfigError(f"{path}: line {number} of the {description} is not an object")
            rows.append(row)
    return rows


def field(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> Any:
    value: Any = source
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"{origin}: {'.'.join(keys)} is missing")
        value = value[key]
    return value


def text_field(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> str:
    value = field(source, keys, origin)
    if not isinstance(value, str):
        raise ConfigError(f"{origin}: {'.'.join(keys)} is not text")
    return value


def number_field(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> float:
    value = field(source, keys, origin)
    if not is_json_number(value):
        raise ConfigError(f"{origin}: {'.'.join(keys)} is not a number")
    return float(value)


def file_entry(entry: object, description: str, origin: str) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{origin}: the {description} entry is not an object")
    return entry


def entry_path(entry: object, description: str, origin: str) -> Path:
    return Path(text_field(file_entry(entry, description, origin), ("path",), origin))


def recorded_files(report: JsonObject, origin: str) -> dict[str, object]:
    entries: dict[str, object] = {"verifier output": field(report, ("outputs",), origin)}
    for name in SCORERS:
        entries[f"{name} scores"] = field(report, ("inputs", "udv_score_files", name), origin)
        entries[f"{name} score report"] = field(report, ("score_runs", name), origin)
    return entries


def missing_recorded_files(report: JsonObject, origin: str) -> list[Path]:
    paths = (
        entry_path(entry, description, origin)
        for description, entry in recorded_files(report, origin).items()
    )
    return [path for path in paths if not path.is_file()]


def missing_signal_files(report_path: Path) -> list[Path]:
    """List the files a verifier report records that are missing from the working directory."""
    report = read_json_object(report_path, "verifier report")
    return missing_recorded_files(report, str(report_path))


def check_recorded_files(report: JsonObject, origin: str) -> None:
    missing = missing_recorded_files(report, origin)
    if missing:
        raise ConfigError(
            f"{origin}: files recorded in the verifier report are missing ({len(missing)}): "
            f"{', '.join(str(path) for path in missing)}. Files left out of the repository "
            f"because of their size are listed, with the command that regenerates each one, "
            f"in {HEAVY_ARTIFACT_MANIFEST}"
        )


def checked_file(entry: object, description: str, origin: str) -> Path:
    checked = file_entry(entry, description, origin)
    path = Path(text_field(checked, ("path",), origin))
    expected = text_field(checked, ("sha256",), origin)
    if not path.is_file():
        raise ConfigError(f"{path}: {description} not found")
    actual = sha256_of_file(path)
    if actual != expected:
        raise ConfigError(f"{path}: sha256 {actual} differs from {expected} recorded in {origin}")
    return path


def rows_by_id(rows: Sequence[JsonObject], key: str, path: Path) -> dict[str, JsonObject]:
    indexed: dict[str, JsonObject] = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str):
            raise ConfigError(f"{path}: a row has no text {key}")
        if identifier in indexed:
            raise ConfigError(f"{path}: {identifier} appears twice")
        indexed[identifier] = row
    return indexed


def check_same_ids(found: Iterable[str], expected: Iterable[str], origin: str) -> None:
    found_set = set(found)
    expected_set = set(expected)
    if found_set == expected_set:
        return
    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)
    raise ConfigError(
        f"{origin}: UDV ids do not match the run ({len(missing)} missing, e.g. {missing[:3]}; "
        f"{len(extra)} unknown, e.g. {extra[:3]})"
    )


def question_entry(question_id: str, spec: object, origin: str) -> JsonObject:
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{origin}: question {question_id} is not an object")
    payload = field(spec, ("payload",), origin)
    kind = text_field(payload, ("type",), origin)
    criteria = payload.get("criteria") if isinstance(payload, Mapping) else None
    options: list[str] | None
    if isinstance(criteria, Mapping):
        options = [str(key) for key in criteria]
    elif isinstance(criteria, list):
        options = [str(item) for item in criteria]
    else:
        options = None
    return {
        "id": question_id,
        "type": kind,
        "instructions": text_field(payload, ("instructions",), origin),
        "options": options,
        "support_option": spec.get("support_option"),
        "reverses": spec.get("reverses"),
    }


def battery(reports: Mapping[str, JsonObject], origins: Mapping[str, str]) -> list[JsonObject]:
    first = LAYA_SCORERS[0]
    questions = field(reports[first], ("questions",), origins[first])
    if not isinstance(questions, Mapping) or not questions:
        raise ConfigError(f"{origins[first]}: questions is empty or not an object")
    for name in LAYA_SCORERS[1:]:
        if field(reports[name], ("questions",), origins[name]) != questions:
            raise ConfigError(f"{origins[name]}: questions differ from those of {first}")
    return [question_entry(str(key), spec, origins[first]) for key, spec in questions.items()]


def single_item(row: JsonObject, path: Path) -> JsonObject:
    items = row.get("items")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise ConfigError(f"{path}: {row.get('id')} does not have exactly one premise item")
    item: JsonObject = items[0]
    return item


def laya_answers(row: JsonObject, question_ids: Sequence[str], path: Path) -> JsonObject:
    signals = single_item(row, path).get("signals")
    if not isinstance(signals, Mapping):
        raise ConfigError(f"{path}: {row.get('id')} has no signals")
    answers: JsonObject = {}
    for question_id in question_ids:
        value = signals.get(question_id)
        if not is_json_number(value):
            raise ConfigError(f"{path}: {row.get('id')} has no number for {question_id}")
        answers[question_id] = float(value)
    return answers


def xnli_answer(row: JsonObject, path: Path) -> JsonObject:
    item = single_item(row, path)
    probabilities = item.get("probabilities")
    if not isinstance(probabilities, Mapping) or not all(
        is_json_number(probabilities.get(label)) for label in XNLI_LABELS
    ):
        raise ConfigError(f"{path}: {row.get('id')} has no {'/'.join(XNLI_LABELS)} probabilities")
    return {
        **{label: float(probabilities[label]) for label in XNLI_LABELS},
        "truncated": item.get("truncated") is True,
    }


def udv_threshold_of(report: JsonObject, origin: str) -> JsonObject | None:
    if "udv_threshold" not in report:
        return None
    block = field(report, ("udv_threshold",), origin)
    interval = field(block, ("interval",), origin)
    return {
        "value": number_field(block, ("exact",), origin),
        "rounded": number_field(block, ("value",), origin),
        "rule": text_field(block, ("rule",), origin),
        "interval": [
            number_field(interval, ("low",), origin),
            number_field(interval, ("high",), origin),
        ],
        "source": text_field(block, ("path",), origin),
    }


def udv_decision(row: JsonObject, record: UdvRecord, probability: float, cut: float) -> bool:
    supported = row.get("supported_at_udv_threshold")
    if not isinstance(supported, bool) or supported != (probability >= cut):
        raise ConfigError(
            f"{record.id}: supported_at_udv_threshold does not follow from the probability "
            f"and the UDV-premise cut {cut}"
        )
    return supported


def verifier_decision(
    row: JsonObject, record: UdvRecord, path: Path, udv_cut: float | None = None
) -> JsonObject | None:
    support_type = None if record.evidence is None else record.evidence.support_type
    if (
        row.get("hearing_id") != record.hearing_id
        or row.get("tier") != record.tier
        or row.get("support_type") != support_type
    ):
        raise ConfigError(
            f"{path}: {record.id} has hearing, tier or support_type different from the run record"
        )
    scored = row.get("scored")
    probability = row.get("primary_probability")
    supported = row.get("supported_at_train_threshold")
    if scored is False and probability is None and supported is None:
        return None
    if scored is not True or not is_json_number(probability) or not isinstance(supported, bool):
        raise ConfigError(f"{path}: {record.id} has an inconsistent verifier decision")
    if record.evidence is None:
        raise ConfigError(f"{path}: {record.id} is scored but the run record has no evidence")
    decision: JsonObject = {"probability": float(probability), "supported": supported}
    if udv_cut is not None:
        decision["supported_at_udv_threshold"] = udv_decision(
            row, record, float(probability), udv_cut
        )
    return decision


def premise_of(record: UdvRecord) -> str:
    if record.evidence is None:
        raise ConfigError(f"{record.id}: no evidence to translate")
    return normalize_whitespace(record.evidence.text)


def load_translations(
    path: Path,
    signature_sha256: str,
    segmentation: Segmentation,
    texts: Iterable[str],
) -> Translations:
    wanted = {
        translation_key(signature_sha256, text): text
        for text in (normalize_whitespace(raw) for raw in texts)
        if has_content(text)
    }
    if not path.is_file():
        raise ConfigError(f"{path}: translation cache not found")
    entries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key") if isinstance(row, dict) else None
            if key not in wanted or row.get("signature_sha256") != signature_sha256:
                continue
            if row.get("source") != wanted[key] or not isinstance(row.get("translation"), str):
                raise ConfigError(f"{path}: entry {key[:16]} does not hold the text of its key")
            entries[key] = row["translation"]
    missing = len(wanted) - len(entries)
    if missing:
        raise ConfigError(
            f"{path}: {missing} of {len(wanted)} texts have no translation under signature "
            f"{signature_sha256[:16]}"
        )
    return Translations(signature_sha256, segmentation, entries, path)


def translation_source(
    report: JsonObject, origin: str
) -> tuple[Path, str, Segmentation, JsonObject]:
    store = field(report, ("translation", "store"), origin)
    abbreviations = field(store, ("segmentation", "join_abbreviations"), origin)
    join_short = field(store, ("segmentation", "join_short_parts"), origin)
    if not isinstance(abbreviations, list) or not isinstance(join_short, bool):
        raise ConfigError(f"{origin}: translation.store.segmentation is malformed")
    model = {
        key: text_field(report, ("translation", "model", key), origin)
        for key in ("name", "revision", "license")
    }
    return (
        Path(text_field(store, ("path",), origin)),
        text_field(store, ("signature_sha256",), origin),
        Segmentation(frozenset(str(item) for item in abbreviations), join_short),
        model,
    )


@dataclass(frozen=True)
class SiteSignals:
    summary: JsonObject
    by_udv: Mapping[str, JsonObject]

    def for_record(self, record: UdvRecord) -> JsonObject:
        found = self.by_udv.get(record.id)
        if found is None:
            raise ConfigError(f"{record.id}: the verifier output has no row for this UDV")
        return found


def scored_signals(
    record: UdvRecord,
    decision: JsonObject,
    rows: Mapping[str, Mapping[str, JsonObject]],
    score_paths: Mapping[str, Path],
    question_ids: Sequence[str],
    translations: Translations,
) -> JsonObject:
    return {
        "scored": True,
        "verifier": decision,
        "laya": {
            name: laya_answers(rows[name][record.id], question_ids, score_paths[name])
            for name in LAYA_SCORERS
        },
        "xnli": xnli_answer(rows[XNLI_SCORER][record.id], score_paths[XNLI_SCORER]),
        "translation": {
            "premise": translations.translate_chunk(premise_of(record)),
            "hypothesis": translations.translate_opinion(record.proposition),
        },
    }


def signals_summary(
    report: JsonObject,
    report_path: Path,
    reports: Mapping[str, JsonObject],
    origins: Mapping[str, str],
    translation_model: JsonObject,
    questions: list[JsonObject],
) -> JsonObject:
    origin = str(report_path)
    primary = field(report, ("primary",), origin)
    threshold = number_field(primary, ("fit", "threshold"), origin)
    if number_field(primary, ("final_test_result", "threshold"), origin) != threshold:
        raise ConfigError(f"{origin}: primary fit and final test thresholds differ")
    verifier: JsonObject = {
        "name": text_field(primary, ("candidate",), origin),
        "threshold": threshold,
        "threshold_fitted_on": text_field(
            primary, ("final_test_result", "threshold_fitted_on"), origin
        ),
        "report_sha256": sha256_of_file(report_path),
    }
    udv_threshold = udv_threshold_of(report, origin)
    if udv_threshold is not None:
        verifier["udv_threshold"] = udv_threshold
    return {
        "verifier": verifier,
        "scorers": {
            name: {
                "model": text_field(reports[name], ("model", "name"), origins[name]),
                "revision": text_field(reports[name], ("model", "revision"), origins[name]),
                "language": text_field(reports[name], ("language",), origins[name]),
            }
            for name in SCORERS
        },
        "translation": translation_model,
        "questions": questions,
    }


def load_site_signals(
    report_path: Path, records: Sequence[UdvRecord], records_sha256: str
) -> SiteSignals:
    """Read the verifier signals of every UDV of a run, checking each file against its sha256."""
    origin = str(report_path)
    report = read_json_object(report_path, "verifier report")
    recorded_udv = text_field(report, ("inputs", "udv", "sha256"), origin)
    if recorded_udv != records_sha256:
        raise ConfigError(
            f"{origin}: the verifier read a UDV file with sha256 {recorded_udv}, not the run "
            f"records ({records_sha256})"
        )
    check_recorded_files(report, origin)
    verifier_path = checked_file(field(report, ("outputs",), origin), "verifier output", origin)
    verifier_rows = rows_by_id(read_rows(verifier_path, "verifier output"), "udv_id", verifier_path)
    check_same_ids(verifier_rows, (record.id for record in records), str(verifier_path))
    udv_threshold = udv_threshold_of(report, origin)
    udv_cut = None if udv_threshold is None else float(udv_threshold["value"])
    decisions = {
        record.id: verifier_decision(verifier_rows[record.id], record, verifier_path, udv_cut)
        for record in records
    }
    scored = [record for record in records if decisions[record.id] is not None]
    score_paths: dict[str, Path] = {}
    report_paths: dict[str, Path] = {}
    for name in SCORERS:
        score_paths[name] = checked_file(
            field(report, ("inputs", "udv_score_files", name), origin), f"{name} scores", origin
        )
        report_paths[name] = checked_file(
            field(report, ("score_runs", name), origin), f"{name} score report", origin
        )
    reports = {
        name: read_json_object(path, f"{name} score report") for name, path in report_paths.items()
    }
    origins = {name: str(path) for name, path in report_paths.items()}
    questions = battery(reports, origins)
    question_ids = [question["id"] for question in questions]
    rows: dict[str, dict[str, JsonObject]] = {}
    for name, path in score_paths.items():
        rows[name] = rows_by_id(read_rows(path, f"{name} scores"), "id", path)
        check_same_ids(rows[name], (record.id for record in scored), str(path))
    cache_path, signature, segmentation, translation_model = translation_source(
        reports[TRANSLATED_SCORER], origins[TRANSLATED_SCORER]
    )
    texts = [normalize_whitespace(record.proposition) for record in scored]
    texts += [segment for record in scored for segment in segmentation.segments(premise_of(record))]
    translations = load_translations(cache_path, signature, segmentation, texts)
    by_udv: dict[str, JsonObject] = {}
    for record in records:
        decision = decisions[record.id]
        by_udv[record.id] = (
            dict(UNSCORED)
            if decision is None
            else scored_signals(record, decision, rows, score_paths, question_ids, translations)
        )
    summary = signals_summary(report, report_path, reports, origins, translation_model, questions)
    return SiteSignals(summary, by_udv)
