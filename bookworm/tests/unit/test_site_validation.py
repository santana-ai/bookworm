import copy
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES_DIR, MINI_THRESHOLD, StubEncoder

from bookworm import CachedEncoder, ConfigError, EvidenceSettings, UdvRecord, build_udvs
from bookworm.data.io import JsonObject, load_hearings, sha256_of_file, write_json
from bookworm.profiles.review import wilson_interval
from bookworm.udv.signals import SiteSignals
from bookworm.udv.site_validation import band_of, load_site_validation

RUN = "run_v2"
CONFIDENCE = 0.95
LOW = 0.3
HIGH = 0.7


@pytest.fixture(scope="module")
def records() -> list[UdvRecord]:
    hearings = load_hearings(FIXTURES_DIR / "lds_mini.jsonl")
    encoder = CachedEncoder(StubEncoder())
    return build_udvs(hearings, encoder, EvidenceSettings(MINI_THRESHOLD)).records


def interval(successes: int, trials: int) -> JsonObject:
    return wilson_interval(successes, trials, CONFIDENCE)


def precision_pair(correct: int, partial: int, trials: int) -> JsonObject:
    return {
        "strict_precision": interval(correct, trials),
        "tolerant_precision": interval(correct + partial, trials),
    }


def report() -> JsonObject:
    return {
        "status": "final",
        "sample_name": "sample_v1",
        "splits_used": ["test"],
        "confidence_level": CONFIDENCE,
        "criteria_declared_on": "2026-01-01",
        "judged_items": 9,
        "label_semantics": {"trecho_sustenta": "one annotator's reading of the passage"},
        RUN: {
            "strata": {
                "direct_quote": {
                    "question": "trecho_sustenta",
                    "udv_ids_by_judgment": {
                        "correta": ["udv-1-0-0", "udv-1-1-0"],
                        "parcial": ["udv-2-0-0"],
                        "incorreta": [],
                    },
                },
                "semantic_match_high": {
                    "question": "trecho_sustenta",
                    "udv_ids_by_judgment": {
                        "correta": ["udv-1-0-1"],
                        "parcial": [],
                        "incorreta": ["udv-1-1-1", "udv-1-2-0"],
                    },
                },
                "speaker_check": {
                    "question": "pessoa_falou",
                    "udv_ids_by_judgment": {
                        "falou": ["udv-2-3-0"],
                        "nao_falou": ["udv-2-5-0"],
                        "nao_sei": [],
                    },
                },
            },
            "by_tier": {
                "quote_found": {"judged_udvs": 3, **precision_pair(2, 1, 3)},
                "semantic_match_high": {"judged_udvs": 3, **precision_pair(1, 0, 3)},
            },
            "criteria": [
                {
                    "name": "quote_found_strict_precision",
                    "stratum": "direct_quote",
                    "metric": "strict_precision",
                    "min_wilson_lower": 0.9,
                    "status": "FAIL",
                    "valid_for_decision": True,
                    "observed": interval(2, 3),
                }
            ],
            "inheritance": {"inheritance_rule": "keep the label", "assumption": "same turn"},
            "judged_inherited": 5,
            "judged_reannotated": 4,
        },
    }


def write_report(tmp_path: Path, payload: JsonObject) -> Path:
    path = tmp_path / "precision.json"
    write_json(payload, path)
    return path


def signals(records: list[UdvRecord], probabilities: dict[str, float]) -> SiteSignals:
    by_udv: dict[str, JsonObject] = {}
    for record in records:
        value = probabilities.get(record.id)
        verifier = None if value is None else {"probability": value, "supported": value >= HIGH}
        by_udv[record.id] = {"scored": verifier is not None, "verifier": verifier}
    summary = {"verifier": {"threshold": HIGH, "udv_threshold": {"value": LOW}}}
    return SiteSignals(summary, by_udv)


def test_validation_block_reports_tiers_criteria_and_judgments(
    tmp_path: Path, records: list[UdvRecord]
) -> None:
    path = write_report(tmp_path, report())
    validation = load_site_validation(path, records, RUN)
    block = validation.index_block(None)
    assert block["source"] == {"path": str(path), "sha256": sha256_of_file(path)}
    assert block["single_annotator"] is True
    assert block["splits"] == ["test"]
    assert block["tiers"] == {
        "quote_found": {
            "udvs": 3,
            "correta": 2,
            "parcial": 1,
            "incorreta": 0,
            "strict": interval(2, 3),
            "tolerant": interval(3, 3),
        },
        "semantic_match_high": {
            "udvs": 3,
            "correta": 1,
            "parcial": 0,
            "incorreta": 2,
            "strict": interval(1, 3),
            "tolerant": interval(1, 3),
        },
    }
    assert [criterion["status"] for criterion in block["criteria"]] == ["FAIL"]
    assert "valid_for_decision" not in block["criteria"][0]
    assert block["inheritance"] == {
        "judged_inherited": 5,
        "judged_reannotated": 4,
        "rule": "keep the label",
        "assumption": "same turn",
    }
    assert block["speaker_check"] == {"udvs": 2, "falou": 1, "nao_falou": 1, "nao_sei": 0}
    assert block["udvs"]["udv-2-5-0"] == {"question": "pessoa_falou", "judgment": "nao_falou"}
    assert list(block["udvs"]) == sorted(block["udvs"])
    assert block["verifier_bands"] is None
    assert validation.judged_udvs == 8


def test_validation_bands_join_judgments_with_the_verifier(
    tmp_path: Path, records: list[UdvRecord]
) -> None:
    validation = load_site_validation(write_report(tmp_path, report()), records, RUN)
    probabilities = {"udv-1-0-0": 0.1, "udv-1-1-0": LOW, "udv-2-0-0": HIGH, "udv-1-0-1": 0.9}
    bands = validation.index_block(signals(records, probabilities))["verifier_bands"]
    assert bands == {
        "cuts": [LOW, HIGH],
        "bands": [
            {"band": "weak", "udvs": 1, "correta": 1, "parcial": 0, "incorreta": 0},
            {"band": "uncertain", "udvs": 1, "correta": 1, "parcial": 0, "incorreta": 0},
            {"band": "strong", "udvs": 2, "correta": 1, "parcial": 1, "incorreta": 0},
        ],
        "unscored": 2,
    }
    train_only = SiteSignals({"verifier": {"threshold": HIGH}}, {})
    assert validation.index_block(train_only)["verifier_bands"] is None
    swapped = SiteSignals({"verifier": {"threshold": LOW, "udv_threshold": {"value": HIGH}}}, {})
    with pytest.raises(ConfigError, match="is not below"):
        validation.index_block(swapped)


def test_band_edges_belong_to_the_upper_band() -> None:
    assert [band_of(value, LOW, HIGH) for value in (0.29, LOW, 0.69, HIGH)] == [
        "weak",
        "uncertain",
        "uncertain",
        "strong",
    ]


def with_change(change: Any) -> JsonObject:
    payload = copy.deepcopy(report())
    change(payload[RUN])
    return payload


def unknown_id(section: JsonObject) -> None:
    section["strata"]["direct_quote"]["udv_ids_by_judgment"]["correta"].append("udv-9-0-0")


def bad_label(section: JsonObject) -> None:
    section["strata"]["direct_quote"]["udv_ids_by_judgment"]["talvez"] = []


def speaker_label_on_support(section: JsonObject) -> None:
    section["strata"]["direct_quote"]["udv_ids_by_judgment"]["falou"] = []


def by_tier_mismatch(section: JsonObject) -> None:
    section["by_tier"]["quote_found"]["strict_precision"] = interval(3, 3)


def missing_tier(section: JsonObject) -> None:
    del section["by_tier"]["semantic_match_high"]


def twice(section: JsonObject) -> None:
    section["strata"]["semantic_match_high"]["udv_ids_by_judgment"]["correta"].append("udv-1-0-0")


def no_criteria(section: JsonObject) -> None:
    del section["criteria"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (unknown_id, "udv-9-0-0 is not in the run records"),
        (bad_label, "'talvez'"),
        (speaker_label_on_support, "'falou'"),
        (by_tier_mismatch, "by_tier.quote_found.strict_precision"),
        (missing_tier, "by_tier does not list"),
        (twice, "judged in two strata"),
        (no_criteria, "criteria is missing"),
    ],
)
def test_validation_refuses_an_inconsistent_report(
    tmp_path: Path, records: list[UdvRecord], change: Any, message: str
) -> None:
    path = write_report(tmp_path, with_change(change))
    with pytest.raises(ConfigError, match=message):
        load_site_validation(path, records, RUN)


def test_validation_refuses_a_report_without_the_run(
    tmp_path: Path, records: list[UdvRecord]
) -> None:
    path = write_report(tmp_path, report())
    with pytest.raises(ConfigError, match="other_run is missing"):
        load_site_validation(path, records, "other_run")


def test_validation_without_by_tier_or_inheritance(
    tmp_path: Path, records: list[UdvRecord]
) -> None:
    payload = report()
    for key in ("by_tier", "inheritance", "judged_inherited", "judged_reannotated"):
        del payload[RUN][key]
    payload["label_semantics"]["trecho_sustenta"] = "two annotators' reading"
    block = load_site_validation(write_report(tmp_path, payload), records, RUN).index_block(None)
    assert block["inheritance"] is None
    assert block["single_annotator"] is False
    assert block["tiers"]["quote_found"]["udvs"] == 3
