import json
import platform
from datetime import UTC, datetime, timedelta, timezone

import pytest
from conftest import MINI_THRESHOLD, StubEncoder

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    QuotePolicy,
    UdvRun,
    build_udvs,
    pipeline_description,
    summarize_run,
)
from bookworm.udv.coverage import distribution_version, runtime_environment, timing_summary
from bookworm.udv.quotes import DOUBLE_QUOTE_PATTERNS

SOURCE = {"evidence": {"embedding_threshold": MINI_THRESHOLD}, "run": {"seed": 42}}
PIPELINE = {
    "sentence_segmentation": "per matched turn, concatenated in turn order",
    "sentence_boundary_pattern": r"(?<=[.!?])\s+|(?<=[.!?]\))(?<!\(\.\.\.\))\s+",
    "quote_patterns": ['“([^”]{10,})”|"([^"]{10,})"', r"(?<!\w)'([^']{10,})'(?!\w)"],
    "quote_search": "inside each matched turn",
    "quote_selection": "most prefix words over all quotes, earliest quote on ties",
    "quote_occurrence": "max token Jaccard with the opinion for trusted prefixes, first on ties",
    "trusted_prefix_words": 6,
}


@pytest.fixture
def mini_run(mini_hearing_list: list[HearingRecord]) -> UdvRun:
    return build_udvs(
        mini_hearing_list, CachedEncoder(StubEncoder()), EvidenceSettings(MINI_THRESHOLD)
    )


def test_summary_keys_follow_the_published_coverage_files(mini_run: UdvRun) -> None:
    summary = summarize_run(
        "mini",
        mini_run.records,
        mini_run.people,
        mini_run.hearings,
        encoder_runtime={"device": "cpu", "max_seq_length": 0, "embedding_dimension": 4},
        hearing_seconds=[0.25, 1.5],
        config_source=SOURCE,
        created_at=datetime(2026, 9, 21, 17, 25, 23, 987654, tzinfo=UTC),
        environment={"python": "3.12.0", "torch": None, "sentence_transformers": None},
    )
    assert json.loads(json.dumps(summary)) == {
        "run_name": "mini",
        "created_at": "2026-09-21T17:25:23+00:00",
        "hearings": {"count": 2, "ids": [1, 2]},
        "people": {"total": 11, "resolved": 9},
        "opinions": {
            "total": 14,
            "by_tier": {
                "quote_found": 5,
                "semantic_match_high": 6,
                "semantic_match_weak": 0,
                "no_evidence": 1,
                "person_not_resolved": 2,
            },
        },
        "evidence_offsets": {"total": 11, "located": 11},
        "evidence_support_types": {
            "direct_quote": 5,
            "semantic_with_short_quote": 1,
            "semantic_similarity": 5,
        },
        "pipeline": PIPELINE,
        "encoder_runtime": {"device": "cpu", "max_seq_length": 0, "embedding_dimension": 4},
        "timing": {
            "elapsed_seconds": 1.8,
            "mean_seconds_per_hearing": 0.88,
            "max_seconds_per_hearing": 1.5,
        },
        "environment": {"python": "3.12.0", "torch": None, "sentence_transformers": None},
        "config": SOURCE,
    }
    assert list(summary) == [
        "run_name",
        "created_at",
        "hearings",
        "people",
        "opinions",
        "evidence_offsets",
        "evidence_support_types",
        "pipeline",
        "encoder_runtime",
        "timing",
        "environment",
        "config",
    ]


def test_created_at_defaults_to_now_in_utc(mini_run: UdvRun) -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    summary = summarize_run(
        "mini",
        mini_run.records,
        mini_run.people,
        mini_run.hearings,
        encoder_runtime={},
        hearing_seconds=[0.1],
        config_source=SOURCE,
    )
    created_at = datetime.fromisoformat(summary["created_at"])
    assert created_at.utcoffset() == timedelta(0)
    assert before <= created_at <= datetime.now(UTC)
    assert list(summary["environment"]) == ["python", "torch", "sentence_transformers", "platform"]


def test_created_at_keeps_the_given_timezone(mini_run: UdvRun) -> None:
    moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=-3)))
    summary = summarize_run(
        "mini",
        mini_run.records,
        mini_run.people,
        mini_run.hearings,
        encoder_runtime={},
        hearing_seconds=[0.1],
        config_source=SOURCE,
        created_at=moment,
    )
    assert summary["created_at"] == "2026-01-02T03:04:05-03:00"


def test_timing_summary_rounding_and_empty_runs() -> None:
    assert timing_summary([0.04, 0.123, 2.456]) == {
        "elapsed_seconds": 2.6,
        "mean_seconds_per_hearing": 0.87,
        "max_seconds_per_hearing": 2.46,
    }
    assert timing_summary([]) == {
        "elapsed_seconds": 0.0,
        "mean_seconds_per_hearing": 0.0,
        "max_seconds_per_hearing": 0.0,
    }


def test_runtime_environment_reads_installed_distributions() -> None:
    environment = runtime_environment()
    assert environment["python"] == platform.python_version()
    assert environment["platform"] == platform.platform()
    assert distribution_version("numpy") is not None
    assert distribution_version("a-distribution-that-does-not-exist") is None


def test_pipeline_description_follows_the_quote_policy() -> None:
    assert pipeline_description() == PIPELINE
    custom = pipeline_description(
        QuotePolicy(trusted_prefix_words=8, patterns=DOUBLE_QUOTE_PATTERNS)
    )
    assert custom == {
        **PIPELINE,
        "quote_patterns": ['“([^”]{10,})”|"([^"]{10,})"'],
        "trusted_prefix_words": 8,
    }


def test_summary_records_the_policy_of_the_run(mini_run: UdvRun) -> None:
    policy = QuotePolicy(trusted_prefix_words=10)
    summary = summarize_run(
        "mini",
        mini_run.records,
        mini_run.people,
        mini_run.hearings,
        encoder_runtime={},
        hearing_seconds=[0.1],
        config_source=SOURCE,
        quote_policy=policy,
    )
    assert summary["pipeline"]["trusted_prefix_words"] == 10
