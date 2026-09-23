import functools
import json
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import udv_artifact_path

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    UdvRecord,
    build_udvs,
    load_udv_jsonl,
)
from bookworm.udv.verify import coverage_hearings

pytestmark = pytest.mark.model

RUN_NAME = "udv_v1"
FULL_RUN_VARIABLE = "BOOKWORM_PARITY_FULL"
DEFAULT_HEARINGS = 20
SCORE_TOLERANCE = 1e-5
EXACT_EVIDENCE_FIELDS = (
    "text",
    "support_type",
    "quote_prefix",
    "start_char",
    "end_char",
    "speaker_turn",
)
RecordPair = tuple[UdvRecord, UdvRecord]


@pytest.fixture(scope="module")
def hearing_limit() -> int | None:
    return None if os.environ.get(FULL_RUN_VARIABLE) == "1" else DEFAULT_HEARINGS


@pytest.fixture(scope="module")
def pairs(
    hearing_limit: int | None, udv_artifacts_dir: Path, lds_hearings: list[HearingRecord]
) -> list[RecordPair]:
    sentence_transformer = pytest.importorskip("bookworm.features.sentence_transformer")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    coverage: dict[str, Any] = json.loads(
        udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    config = coverage["config"]
    name, revision = config["encoder"]["name"], config["encoder"]["revision"]
    try:
        huggingface_hub.snapshot_download(name, revision=revision, local_files_only=True)
    except OSError:
        pytest.skip(
            f"{name}@{revision} is not in the local Hugging Face cache; "
            "download it before running the model tests"
        )
    sentence_transformer.seed_torch(config["run"]["seed"])
    encoder = sentence_transformer.SentenceTransformerEncoder(
        name,
        revision,
        config["encoder"]["device"],
        config["encoder"]["batch_size"],
        loader=functools.partial(
            sentence_transformer.load_sentence_transformer, local_files_only=True
        ),
    )
    hearings = coverage_hearings(coverage, lds_hearings)[:hearing_limit]
    run = build_udvs(
        hearings,
        CachedEncoder(encoder),
        EvidenceSettings(config["evidence"]["embedding_threshold"]),
    )
    assert encoder.is_loaded
    wanted = {hearing.id for hearing in hearings}
    artifact = [
        record
        for record in load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}.jsonl"))
        if record.hearing_id in wanted
    ]
    assert [record.id for record in run.records] == [record.id for record in artifact]
    return list(zip(run.records, artifact, strict=True))


def differing(pairs: list[RecordPair], field: str) -> list[str]:
    return [ours.id for ours, theirs in pairs if getattr(ours, field) != getattr(theirs, field)]


def test_ids_actors_and_propositions(pairs: list[RecordPair]) -> None:
    for field in ("hearing_id", "actor", "proposition", "method"):
        assert differing(pairs, field) == [], field


def test_tiers_and_provenance(pairs: list[RecordPair]) -> None:
    assert differing(pairs, "tier") == []
    assert differing(pairs, "provenance") == []


def test_evidence_fields_are_exact(pairs: list[RecordPair]) -> None:
    for ours, theirs in pairs:
        assert (ours.evidence is None) == (theirs.evidence is None), ours.id
        if ours.evidence is None or theirs.evidence is None:
            continue
        for field in EXACT_EVIDENCE_FIELDS:
            assert getattr(ours.evidence, field) == getattr(theirs.evidence, field), (
                ours.id,
                field,
            )


def test_scores_within_tolerance(pairs: list[RecordPair]) -> None:
    for ours, theirs in pairs:
        if ours.evidence is None or theirs.evidence is None:
            continue
        if theirs.evidence.score is None:
            assert ours.evidence.score is None, ours.id
            continue
        assert ours.evidence.score == pytest.approx(theirs.evidence.score, abs=SCORE_TOLERANCE)
