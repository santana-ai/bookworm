"""Human validation of a run for the demo index, read from a final precision report."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm.data.io import (
    JsonObject,
    read_json_object,
    required_field,
    required_number,
    required_text,
    sha256_of_file,
)
from bookworm.errors import ConfigError
from bookworm.profiles.review import wilson_interval
from bookworm.udv.schemas import TIERS, UdvRecord
from bookworm.udv.signals import SiteSignals

SUPPORT_QUESTION = "trecho_sustenta"
SPEAKER_QUESTION = "pessoa_falou"
QUESTION_LABELS: dict[str, tuple[str, ...]] = {
    SUPPORT_QUESTION: ("correta", "parcial", "incorreta"),
    SPEAKER_QUESTION: ("falou", "nao_falou", "nao_sei"),
}
SUPPORT_LABELS = QUESTION_LABELS[SUPPORT_QUESTION]
SPEAKER_LABELS = QUESTION_LABELS[SPEAKER_QUESTION]
STRICT_LABELS = ("correta",)
TOLERANT_LABELS = ("correta", "parcial")
INTERVAL_KEYS = ("successes", "trials", "estimate", "low", "high")
CRITERION_KEYS = ("name", "stratum", "metric", "min_wilson_lower", "status", "observed")
SINGLE_ANNOTATOR_PREFIX = "one annotator"
BANDS = ("weak", "uncertain", "strong")


@dataclass(frozen=True)
class Judgment:
    question: str
    label: str


def text_list(source: Mapping[str, Any], keys: Sequence[str], origin: str) -> list[str]:
    value = required_field(source, keys, origin)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{origin}: {'.'.join(keys)} is not a list of text")
    return list(value)


def stratum_judgments(
    name: str, stratum: object, known: Mapping[str, UdvRecord], origin: str
) -> dict[str, Judgment]:
    if not isinstance(stratum, Mapping):
        raise ConfigError(f"{origin}: stratum {name} is not an object")
    question = required_text(stratum, ("question",), origin)
    labels = QUESTION_LABELS.get(question)
    if labels is None:
        raise ConfigError(f"{origin}: stratum {name} has the unknown question {question}")
    by_judgment = required_field(stratum, ("udv_ids_by_judgment",), origin)
    if not isinstance(by_judgment, Mapping):
        raise ConfigError(f"{origin}: strata.{name}.udv_ids_by_judgment is not an object")
    judgments: dict[str, Judgment] = {}
    for label in by_judgment:
        if label not in labels:
            raise ConfigError(
                f"{origin}: stratum {name} has the judgment {label!r}, outside {labels}"
            )
        for udv_id in text_list(by_judgment, (label,), origin):
            if udv_id not in known:
                raise ConfigError(f"{origin}: judged UDV {udv_id} is not in the run records")
            judgments[udv_id] = Judgment(question, label)
    return judgments


def collect_judgments(
    section: Mapping[str, Any], known: Mapping[str, UdvRecord], origin: str
) -> dict[str, Judgment]:
    strata = required_field(section, ("strata",), origin)
    if not isinstance(strata, Mapping) or not strata:
        raise ConfigError(f"{origin}: strata is empty or not an object")
    judgments: dict[str, Judgment] = {}
    for name, stratum in strata.items():
        found = stratum_judgments(str(name), stratum, known, origin)
        repeated = sorted(found.keys() & judgments.keys())
        if repeated:
            raise ConfigError(f"{origin}: UDVs judged in two strata: {repeated[:3]}")
        judgments.update(found)
    return judgments


def label_counts(judgments: Sequence[Judgment], labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(judgment.label for judgment in judgments)
    return {label: counts[label] for label in labels}


def precision(counts: Mapping[str, int], kept: Sequence[str], confidence: float) -> JsonObject:
    return wilson_interval(sum(counts[label] for label in kept), sum(counts.values()), confidence)


def tier_block(judgments: Sequence[Judgment], confidence: float) -> JsonObject:
    counts = label_counts(judgments, SUPPORT_LABELS)
    return {
        "udvs": len(judgments),
        **counts,
        "strict": precision(counts, STRICT_LABELS, confidence),
        "tolerant": precision(counts, TOLERANT_LABELS, confidence),
    }


def tier_blocks(
    judgments: Mapping[str, Judgment], records: Mapping[str, UdvRecord], confidence: float
) -> dict[str, JsonObject]:
    by_tier: dict[str, list[Judgment]] = {}
    for udv_id, judgment in judgments.items():
        if judgment.question == SUPPORT_QUESTION:
            by_tier.setdefault(records[udv_id].tier, []).append(judgment)
    return {tier: tier_block(by_tier[tier], confidence) for tier in TIERS if tier in by_tier}


def interval_of(block: object, origin: str, where: str) -> JsonObject:
    if not isinstance(block, Mapping):
        raise ConfigError(f"{origin}: {where} is not an object")
    return {key: required_field(block, (key,), f"{origin}: {where}") for key in INTERVAL_KEYS}


def check_by_tier(section: Mapping[str, Any], tiers: Mapping[str, JsonObject], origin: str) -> None:
    recorded = section.get("by_tier")
    if recorded is None:
        return
    if not isinstance(recorded, Mapping) or set(recorded) != set(tiers):
        raise ConfigError(f"{origin}: by_tier does not list the tiers of the judged UDVs")
    for tier, block in tiers.items():
        for metric in ("strict", "tolerant"):
            where = f"by_tier.{tier}.{metric}_precision"
            expected = interval_of(recorded[tier].get(f"{metric}_precision"), origin, where)
            if expected != block[metric]:
                raise ConfigError(
                    f"{origin}: {where} is {expected}, but the judged UDVs give {block[metric]}"
                )


def criterion_entry(criterion: object, origin: str) -> JsonObject:
    if not isinstance(criterion, Mapping):
        raise ConfigError(f"{origin}: a criterion is not an object")
    entry = {key: required_field(criterion, (key,), origin) for key in CRITERION_KEYS}
    entry["observed"] = interval_of(entry["observed"], origin, f"criterion {entry['name']}")
    return entry


def criteria_of(section: Mapping[str, Any], origin: str) -> list[JsonObject]:
    criteria = required_field(section, ("criteria",), origin)
    if not isinstance(criteria, list):
        raise ConfigError(f"{origin}: criteria is not a list")
    return [criterion_entry(criterion, origin) for criterion in criteria]


def inheritance_of(section: Mapping[str, Any], origin: str) -> JsonObject | None:
    if "inheritance" not in section:
        return None
    return {
        "judged_inherited": int(required_number(section, ("judged_inherited",), origin)),
        "judged_reannotated": int(required_number(section, ("judged_reannotated",), origin)),
        "rule": required_text(section, ("inheritance", "inheritance_rule"), origin),
        "assumption": required_text(section, ("inheritance", "assumption"), origin),
    }


def speaker_check_of(judgments: Mapping[str, Judgment]) -> JsonObject | None:
    spoken = [judgment for judgment in judgments.values() if judgment.question == SPEAKER_QUESTION]
    if not spoken:
        return None
    return {"udvs": len(spoken), **label_counts(spoken, SPEAKER_LABELS)}


def band_of(probability: float, low: float, high: float) -> str:
    if probability < low:
        return BANDS[0]
    return BANDS[1] if probability < high else BANDS[2]


@dataclass(frozen=True)
class SiteValidation:
    path: Path
    sha256: str
    run_name: str
    report: JsonObject
    judgments: dict[str, Judgment]
    tiers: dict[str, JsonObject]
    criteria: list[JsonObject]
    inheritance: JsonObject | None

    @property
    def judged_udvs(self) -> int:
        return len(self.judgments)

    def support_ids(self) -> list[str]:
        return sorted(
            udv_id
            for udv_id, judgment in self.judgments.items()
            if judgment.question == SUPPORT_QUESTION
        )

    def verifier_bands(self, signals: SiteSignals | None) -> JsonObject | None:
        if signals is None:
            return None
        verifier = signals.summary["verifier"]
        udv_threshold = verifier.get("udv_threshold")
        if not isinstance(udv_threshold, Mapping):
            return None
        low = float(udv_threshold["value"])
        high = float(verifier["threshold"])
        if not low < high:
            raise ConfigError(
                f"the UDV-premise cut {low} is not below the verifier training cut {high}"
            )
        grouped: dict[str, list[Judgment]] = {band: [] for band in BANDS}
        unscored = 0
        for udv_id in self.support_ids():
            found = signals.by_udv.get(udv_id)
            if found is None:
                raise ConfigError(f"{udv_id}: the verifier signals have no row for this UDV")
            decision = found["verifier"]
            if decision is None:
                unscored += 1
                continue
            grouped[band_of(float(decision["probability"]), low, high)].append(
                self.judgments[udv_id]
            )
        return {
            "cuts": [low, high],
            "bands": [
                {
                    "band": band,
                    "udvs": len(grouped[band]),
                    **label_counts(grouped[band], SUPPORT_LABELS),
                }
                for band in BANDS
            ],
            "unscored": unscored,
        }

    def index_block(self, signals: SiteSignals | None) -> JsonObject:
        origin = str(self.path)
        semantics = required_text(self.report, ("label_semantics", SUPPORT_QUESTION), origin)
        return {
            "source": {"path": str(self.path), "sha256": self.sha256},
            "run": self.run_name,
            "status": required_text(self.report, ("status",), origin),
            "sample_name": required_text(self.report, ("sample_name",), origin),
            "splits": text_list(self.report, ("splits_used",), origin),
            "confidence_level": required_number(self.report, ("confidence_level",), origin),
            "single_annotator": semantics.startswith(SINGLE_ANNOTATOR_PREFIX),
            "criteria_declared_on": required_text(self.report, ("criteria_declared_on",), origin),
            "judged_items": int(required_number(self.report, ("judged_items",), origin)),
            "tiers": self.tiers,
            "criteria": self.criteria,
            "inheritance": self.inheritance,
            "speaker_check": speaker_check_of(self.judgments),
            "udvs": {
                udv_id: {"question": judgment.question, "judgment": judgment.label}
                for udv_id, judgment in sorted(self.judgments.items())
            },
            "verifier_bands": self.verifier_bands(signals),
        }


def load_site_validation(path: Path, records: Sequence[UdvRecord], run_name: str) -> SiteValidation:
    """Read the human judgments of ``run_name`` and check them against the run records."""
    origin = str(path)
    report = read_json_object(path, "precision report")
    section = required_field(report, (run_name,), origin)
    if not isinstance(section, Mapping):
        raise ConfigError(f"{origin}: {run_name} is not an object")
    section_origin = f"{origin}: {run_name}"
    confidence = required_number(report, ("confidence_level",), origin)
    known = {record.id: record for record in records}
    judgments = collect_judgments(section, known, section_origin)
    tiers = tier_blocks(judgments, known, confidence)
    check_by_tier(section, tiers, section_origin)
    return SiteValidation(
        path=path,
        sha256=sha256_of_file(path),
        run_name=run_name,
        report=report,
        judgments=judgments,
        tiers=tiers,
        criteria=criteria_of(section, section_origin),
        inheritance=inheritance_of(section, section_origin),
    )
