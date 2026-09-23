import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from bookworm.data.schemas import HearingRecord
from bookworm.transcript.sentences import SENTENCE_BOUNDARY_PATTERN
from bookworm.udv.build import PersonSpeech
from bookworm.udv.quotes import DEFAULT_QUOTE_POLICY, QuotePolicy
from bookworm.udv.schemas import SUPPORT_TYPES, TIERS, UdvRecord

JsonObject = dict[str, Any]

SENTENCE_SEGMENTATION = "per matched turn, concatenated in turn order"
QUOTE_SEARCH = "inside each matched turn"
QUOTE_SELECTION = "most prefix words over all quotes, earliest quote on ties"
QUOTE_OCCURRENCE = "max token Jaccard with the opinion for trusted prefixes, first on ties"


def distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def runtime_environment() -> JsonObject:
    return {
        "python": platform.python_version(),
        "torch": distribution_version("torch"),
        "sentence_transformers": distribution_version("sentence-transformers"),
        "platform": platform.platform(),
    }


def pipeline_description(policy: QuotePolicy = DEFAULT_QUOTE_POLICY) -> JsonObject:
    return {
        "sentence_segmentation": SENTENCE_SEGMENTATION,
        "sentence_boundary_pattern": SENTENCE_BOUNDARY_PATTERN.pattern,
        "quote_patterns": [pattern.pattern for pattern in policy.patterns],
        "quote_search": QUOTE_SEARCH,
        "quote_selection": QUOTE_SELECTION,
        "quote_occurrence": QUOTE_OCCURRENCE,
        "trusted_prefix_words": policy.trusted_prefix_words,
    }


def timing_summary(hearing_seconds: Sequence[float]) -> JsonObject:
    if not hearing_seconds:
        return {
            "elapsed_seconds": 0.0,
            "mean_seconds_per_hearing": 0.0,
            "max_seconds_per_hearing": 0.0,
        }
    return {
        "elapsed_seconds": round(sum(hearing_seconds), 1),
        "mean_seconds_per_hearing": round(float(np.mean(hearing_seconds)), 2),
        "max_seconds_per_hearing": round(max(hearing_seconds), 2),
    }


def summarize_run(
    run_name: str,
    records: Sequence[UdvRecord],
    people: Sequence[PersonSpeech],
    hearings: Sequence[HearingRecord],
    *,
    encoder_runtime: Mapping[str, Any],
    hearing_seconds: Sequence[float],
    config_source: Mapping[str, Any],
    quote_policy: QuotePolicy = DEFAULT_QUOTE_POLICY,
    created_at: datetime | None = None,
    environment: Mapping[str, Any] | None = None,
) -> JsonObject:
    evidences = [record.evidence for record in records if record.evidence is not None]
    moment = datetime.now(UTC) if created_at is None else created_at
    return {
        "run_name": run_name,
        "created_at": moment.isoformat(timespec="seconds"),
        "hearings": {"count": len(hearings), "ids": [hearing.id for hearing in hearings]},
        "people": {
            "total": len(people),
            "resolved": sum(1 for person in people if person.resolved),
        },
        "opinions": {
            "total": len(records),
            "by_tier": {
                tier: sum(1 for record in records if record.tier == tier) for tier in TIERS
            },
        },
        "evidence_offsets": {
            "total": len(evidences),
            "located": sum(1 for evidence in evidences if evidence.start_char is not None),
        },
        "evidence_support_types": {
            support_type: sum(1 for evidence in evidences if evidence.support_type == support_type)
            for support_type in SUPPORT_TYPES
        },
        "pipeline": pipeline_description(quote_policy),
        "encoder_runtime": dict(encoder_runtime),
        "timing": timing_summary(hearing_seconds),
        "environment": dict(runtime_environment() if environment is None else environment),
        "config": dict(config_source),
    }
