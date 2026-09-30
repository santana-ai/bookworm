"""Verifier, question and translation signals of each UDV, read from a verifier report."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm.data.io import (
    JsonObject,
    is_json_number,
    read_json_object,
    required_field,
    required_number,
    required_text,
    sha256_of_file,
)
from bookworm.errors import ConfigError
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.schemas import UdvRecord
from bookworm.udv.translations import (
    Segmentation,
    Translations,
    load_translations,
    translation_key,
    translation_source,
)

__all__ = [
    "HEAVY_ARTIFACT_MANIFEST",
    "Segmentation",
    "SiteSignals",
    "load_site_signals",
    "missing_signal_files",
    "translation_key",
]

LAYA_SCORERS = ("laya_multi_pt", "laya_en_en")
XNLI_SCORER = "xnli_mdeberta"
SCORERS = (*LAYA_SCORERS, XNLI_SCORER)
TRANSLATED_SCORER = "laya_en_en"
XNLI_LABELS = ("entailment", "neutral", "contradiction")
HEAVY_ARTIFACT_MANIFEST = "experiments/artifacts/MANIFEST_heavy.tsv"
UNSCORED: JsonObject = {
    "scored": False,
    "verifier": None,
    "laya": None,
    "xnli": None,
    "translation": None,
}


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


def file_entry(entry: object, description: str, origin: str) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{origin}: the {description} entry is not an object")
    return entry


def entry_path(entry: object, description: str, origin: str) -> Path:
    return Path(required_text(file_entry(entry, description, origin), ("path",), origin))


def score_file_entry(report: JsonObject, name: str, origin: str) -> object:
    return required_field(report, ("inputs", "udv_score_files", name), origin)


def score_report_entry(report: JsonObject, name: str, origin: str) -> object:
    return required_field(report, ("score_runs", name), origin)


def recorded_files(report: JsonObject, origin: str) -> dict[str, object]:
    entries: dict[str, object] = {"verifier output": required_field(report, ("outputs",), origin)}
    for name in SCORERS:
        entries[f"{name} scores"] = score_file_entry(report, name, origin)
        entries[f"{name} score report"] = score_report_entry(report, name, origin)
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
    path = Path(required_text(checked, ("path",), origin))
    expected = required_text(checked, ("sha256",), origin)
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
    payload = required_field(spec, ("payload",), origin)
    kind = required_text(payload, ("type",), origin)
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
        "instructions": required_text(payload, ("instructions",), origin),
        "options": options,
        "support_option": spec.get("support_option"),
        "reverses": spec.get("reverses"),
    }


def battery(reports: Mapping[str, JsonObject], origins: Mapping[str, str]) -> list[JsonObject]:
    first = LAYA_SCORERS[0]
    questions = required_field(reports[first], ("questions",), origins[first])
    if not isinstance(questions, Mapping) or not questions:
        raise ConfigError(f"{origins[first]}: questions is empty or not an object")
    for name in LAYA_SCORERS[1:]:
        if required_field(reports[name], ("questions",), origins[name]) != questions:
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
    block = required_field(report, ("udv_threshold",), origin)
    interval = required_field(block, ("interval",), origin)
    return {
        "value": required_number(block, ("exact",), origin),
        "rounded": required_number(block, ("value",), origin),
        "rule": required_text(block, ("rule",), origin),
        "interval": [
            required_number(interval, ("low",), origin),
            required_number(interval, ("high",), origin),
        ],
        "source": required_text(block, ("path",), origin),
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


@dataclass(frozen=True)
class SiteSignals:
    summary: JsonObject
    by_udv: Mapping[str, JsonObject]

    def for_record(self, record: UdvRecord) -> JsonObject:
        found = self.by_udv.get(record.id)
        if found is None:
            raise ConfigError(f"{record.id}: the verifier output has no row for this UDV")
        return found


@dataclass(frozen=True)
class ScorerRuns:
    """Checked score files and score reports of the Laya and XNLI scorers."""

    score_paths: dict[str, Path]
    reports: dict[str, JsonObject]
    origins: dict[str, str]

    def rows(self, scored: Sequence[UdvRecord]) -> dict[str, dict[str, JsonObject]]:
        rows: dict[str, dict[str, JsonObject]] = {}
        for name, path in self.score_paths.items():
            rows[name] = rows_by_id(read_rows(path, f"{name} scores"), "id", path)
            check_same_ids(rows[name], (record.id for record in scored), str(path))
        return rows


def load_scorer_runs(report: JsonObject, origin: str) -> ScorerRuns:
    score_paths: dict[str, Path] = {}
    report_paths: dict[str, Path] = {}
    for name in SCORERS:
        score_paths[name] = checked_file(
            score_file_entry(report, name, origin), f"{name} scores", origin
        )
        report_paths[name] = checked_file(
            score_report_entry(report, name, origin), f"{name} score report", origin
        )
    reports = {
        name: read_json_object(path, f"{name} score report") for name, path in report_paths.items()
    }
    origins = {name: str(path) for name, path in report_paths.items()}
    return ScorerRuns(score_paths, reports, origins)


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


def verifier_summary(report: JsonObject, report_path: Path) -> JsonObject:
    origin = str(report_path)
    primary = required_field(report, ("primary",), origin)
    threshold = required_number(primary, ("fit", "threshold"), origin)
    if required_number(primary, ("final_test_result", "threshold"), origin) != threshold:
        raise ConfigError(f"{origin}: primary fit and final test thresholds differ")
    verifier: JsonObject = {
        "name": required_text(primary, ("candidate",), origin),
        "threshold": threshold,
        "threshold_fitted_on": required_text(
            primary, ("final_test_result", "threshold_fitted_on"), origin
        ),
        "report_sha256": sha256_of_file(report_path),
    }
    udv_threshold = udv_threshold_of(report, origin)
    if udv_threshold is not None:
        verifier["udv_threshold"] = udv_threshold
    return verifier


def scorer_summary(report: JsonObject, origin: str) -> JsonObject:
    return {
        "model": required_text(report, ("model", "name"), origin),
        "revision": required_text(report, ("model", "revision"), origin),
        "language": required_text(report, ("language",), origin),
    }


def signals_summary(
    report: JsonObject,
    report_path: Path,
    scorers: ScorerRuns,
    translation_model: JsonObject,
    questions: list[JsonObject],
) -> JsonObject:
    return {
        "verifier": verifier_summary(report, report_path),
        "scorers": {
            name: scorer_summary(scorers.reports[name], scorers.origins[name]) for name in SCORERS
        },
        "translation": translation_model,
        "questions": questions,
    }


def check_udv_input(report: JsonObject, records_sha256: str, origin: str) -> None:
    recorded_udv = required_text(report, ("inputs", "udv", "sha256"), origin)
    if recorded_udv != records_sha256:
        raise ConfigError(
            f"{origin}: the verifier read a UDV file with sha256 {recorded_udv}, not the run "
            f"records ({records_sha256})"
        )


def verifier_decisions(
    report: JsonObject, records: Sequence[UdvRecord], origin: str
) -> dict[str, JsonObject | None]:
    verifier_path = checked_file(
        required_field(report, ("outputs",), origin), "verifier output", origin
    )
    verifier_rows = rows_by_id(read_rows(verifier_path, "verifier output"), "udv_id", verifier_path)
    check_same_ids(verifier_rows, (record.id for record in records), str(verifier_path))
    udv_threshold = udv_threshold_of(report, origin)
    udv_cut = None if udv_threshold is None else float(udv_threshold["value"])
    return {
        record.id: verifier_decision(verifier_rows[record.id], record, verifier_path, udv_cut)
        for record in records
    }


def texts_to_translate(scored: Sequence[UdvRecord], segmentation: Segmentation) -> list[str]:
    texts = [normalize_whitespace(record.proposition) for record in scored]
    texts += [segment for record in scored for segment in segmentation.segments(premise_of(record))]
    return texts


def load_site_signals(
    report_path: Path, records: Sequence[UdvRecord], records_sha256: str
) -> SiteSignals:
    """Read the verifier signals of every UDV of a run, checking each file against its sha256."""
    origin = str(report_path)
    report = read_json_object(report_path, "verifier report")
    check_udv_input(report, records_sha256, origin)
    check_recorded_files(report, origin)
    decisions = verifier_decisions(report, records, origin)
    scored = [record for record in records if decisions[record.id] is not None]
    scorers = load_scorer_runs(report, origin)
    questions = battery(scorers.reports, scorers.origins)
    question_ids = [question["id"] for question in questions]
    rows = scorers.rows(scored)
    source = translation_source(
        scorers.reports[TRANSLATED_SCORER], scorers.origins[TRANSLATED_SCORER]
    )
    translations = load_translations(
        source.path,
        source.signature_sha256,
        source.segmentation,
        texts_to_translate(scored, source.segmentation),
    )
    by_udv: dict[str, JsonObject] = {}
    for record in records:
        decision = decisions[record.id]
        by_udv[record.id] = (
            dict(UNSCORED)
            if decision is None
            else scored_signals(
                record, decision, rows, scorers.score_paths, question_ids, translations
            )
        )
    summary = signals_summary(report, report_path, scorers, source.model, questions)
    return SiteSignals(summary, by_udv)
