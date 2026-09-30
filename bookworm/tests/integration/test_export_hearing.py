import json
from pathlib import Path

import pytest
from conftest import cache_only_encoder, directory_state, udv_artifact_path

from bookworm import (
    CachedEncoder,
    ConfigError,
    HearingRecord,
    UdvRecord,
    export_hearing,
    extract_quotes,
    load_site_signals,
    load_udv_jsonl,
    pipeline_description,
    sha256_of_file,
)
from bookworm.data.io import JsonObject
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.export import sentence_text, split_of
from bookworm.udv.signals import HEAVY_ARTIFACT_MANIFEST, missing_signal_files

pytestmark = pytest.mark.dataset

RUN_NAME = "udv_v1"
HEARING_ID = 70
SEMANTIC_TIERS = ("semantic_match_high", "semantic_match_weak")
HISTORICAL_RUN = "udv_v0"
VERIFIER_REPORT = Path("artifacts/udv/udv_v1_verifier_report.json")
TRANSLATION_CACHE_DIR = Path("artifacts/cache/translation")
HISTORICAL_MISMATCHES = {
    1: "udv-1-1-2",
    45: "udv-45-0-2",
    169: "udv-169-2-0",
    170: "udv-170-4-0",
}


@pytest.fixture(scope="module")
def coverage(udv_artifacts_dir: Path) -> JsonObject:
    path = udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}_coverage.json")
    payload: JsonObject = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module")
def records(udv_artifacts_dir: Path) -> list[UdvRecord]:
    path = udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}.jsonl")
    return [record for record in load_udv_jsonl(path) if record.hearing_id == HEARING_ID]


@pytest.fixture(scope="module")
def hearing(lds_hearings: list[HearingRecord]) -> HearingRecord:
    return next(hearing for hearing in lds_hearings if hearing.id == HEARING_ID)


@pytest.fixture(scope="module")
def exported(
    hearing: HearingRecord,
    records: list[UdvRecord],
    coverage: JsonObject,
    embedding_cache_dir: Path,
    split_artifacts_dir: Path,
) -> JsonObject:
    manifest = json.loads((split_artifacts_dir / "temporal_v1.json").read_text(encoding="utf-8"))
    before = directory_state(embedding_cache_dir)
    payload = export_hearing(
        hearing,
        records,
        CachedEncoder(cache_only_encoder(coverage), embedding_cache_dir, read_only=True),
        run_name=RUN_NAME,
        pipeline=coverage["pipeline"],
        split=split_of(manifest, HEARING_ID),
    )
    assert directory_state(embedding_cache_dir) == before
    return payload


def span_text(transcript: str, start: int | None, end: int | None) -> str:
    assert start is not None
    assert end is not None
    return normalize_whitespace(transcript[start:end])


def test_hearing_summary_and_run(exported: JsonObject, hearing: HearingRecord) -> None:
    assert exported["hearing"] == {
        "id": HEARING_ID,
        "split": "train",
        "article_date": "2022-08-03",
        "assunto": hearing.metadados.assunto,
        "materia": hearing.materia,
        "transcript_chars": 63834,
        "transcript_words": 10243,
    }
    assert exported["transcript"] == hearing.transcricao
    assert exported["run"] == {
        "name": RUN_NAME,
        "encoder": "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder",
        "revision": "a01887015444f7599669c509447c5bdbce958916",
        "threshold": 0.45,
    }


def test_udvs_are_the_published_records(exported: JsonObject, records: list[UdvRecord]) -> None:
    assert len(exported["udvs"]) == len(records) == 12
    for udv, record in zip(exported["udvs"], records, strict=True):
        assert {key: udv[key] for key in record.to_dict()} == record.to_dict()
        assert udv["quotes"] == extract_quotes(record.proposition)


def test_turn_sentences_slice_the_transcript(exported: JsonObject) -> None:
    transcript = exported["transcript"]
    assert len(exported["turns"]) == 21
    assert len(exported["people"]) == 6
    for turn in exported["turns"]:
        for sentence in turn["sentences"]:
            start, end = sentence[0], sentence[1]
            assert turn["start"] <= start < end <= turn["end"]
            if len(sentence) == 2:
                assert sentence_text(transcript, sentence) == span_text(transcript, start, end)
            else:
                assert sentence[2] != span_text(transcript, start, end)


def test_every_evidence_offset_slices_the_transcript(exported: JsonObject) -> None:
    transcript = exported["transcript"]
    evidences = [udv["evidence"] for udv in exported["udvs"] if udv["evidence"] is not None]
    assert len(evidences) == 10
    for evidence in evidences:
        text = span_text(transcript, evidence["start_char"], evidence["end_char"])
        assert text == evidence["text"]


def test_candidates_come_from_the_actor_turns(exported: JsonObject) -> None:
    transcript = exported["transcript"]
    people = {person["index"]: person for person in exported["people"]}
    sentences_per_turn = {turn["index"]: len(turn["sentences"]) for turn in exported["turns"]}
    for udv in exported["udvs"]:
        person = people[int(udv["id"].split("-")[2])]
        assert udv["n_candidates"] == sum(sentences_per_turn[turn] for turn in person["turns"])
        assert len(udv["candidates"]) == min(8, udv["n_candidates"])
        for candidate in udv["candidates"]:
            assert candidate["turn"] in person["turns"]
            text = span_text(transcript, candidate["start"], candidate["end"])
            assert text == candidate["text"]


def test_top_candidate_is_the_semantic_evidence(exported: JsonObject) -> None:
    semantic = [udv for udv in exported["udvs"] if udv["tier"] in SEMANTIC_TIERS]
    assert len(semantic) == 9
    for udv in semantic:
        top, evidence = udv["candidates"][0], udv["evidence"]
        assert (top["text"], top["turn"], top["start"], top["end"]) == (
            evidence["text"],
            evidence["speaker_turn"],
            evidence["start_char"],
            evidence["end_char"],
        )
        assert top["score"] == round(evidence["score"], 4)


@pytest.fixture(scope="module")
def historical_records(udv_artifacts_dir: Path) -> list[UdvRecord]:
    return load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, f"{HISTORICAL_RUN}.jsonl"))


@pytest.fixture(scope="module")
def historical_coverage(udv_artifacts_dir: Path) -> JsonObject:
    path = udv_artifact_path(udv_artifacts_dir, f"{HISTORICAL_RUN}_coverage.json")
    payload: JsonObject = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.mark.parametrize(("hearing_id", "record_id"), sorted(HISTORICAL_MISMATCHES.items()))
def test_a_run_of_the_previous_pipeline_is_refused(
    hearing_id: int,
    record_id: str,
    historical_records: list[UdvRecord],
    historical_coverage: JsonObject,
    lds_hearings: list[HearingRecord],
    embedding_cache_dir: Path,
) -> None:
    hearing = next(item for item in lds_hearings if item.id == hearing_id)
    records = [record for record in historical_records if record.hearing_id == hearing_id]
    encoder = CachedEncoder(
        cache_only_encoder(historical_coverage), embedding_cache_dir, read_only=True
    )
    assert "pipeline" not in historical_coverage
    with pytest.raises(ConfigError, match="coverage pipeline is missing"):
        export_hearing(hearing, records, encoder, run_name=HISTORICAL_RUN, pipeline=None)
    before = directory_state(embedding_cache_dir)
    with pytest.raises(ConfigError, match=f"^{record_id}: the recorded evidence"):
        export_hearing(
            hearing,
            records,
            encoder,
            run_name=HISTORICAL_RUN,
            pipeline=pipeline_description(),
        )
    assert directory_state(embedding_cache_dir) == before


def test_signals_of_hearing_70_match_the_verifier_output(
    experiments_dir: Path, udv_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(experiments_dir)
    if not TRANSLATION_CACHE_DIR.is_dir():
        pytest.skip(f"translation cache not found at {experiments_dir / TRANSLATION_CACHE_DIR}")
    missing = missing_signal_files(VERIFIER_REPORT)
    if missing:
        pytest.skip(
            f"verifier score files not found ({', '.join(map(str, missing))}); they are "
            f"regenerable heavy artifacts listed in {HEAVY_ARTIFACT_MANIFEST}"
        )
    records_path = udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}.jsonl")
    run_records = load_udv_jsonl(records_path)
    signals = load_site_signals(VERIFIER_REPORT, run_records, sha256_of_file(records_path))
    records = [record for record in run_records if record.hearing_id == HEARING_ID]
    verifier = {
        row["udv_id"]: row
        for row in map(
            json.loads,
            Path("artifacts/udv/udv_v1_verifier.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if row["hearing_id"] == HEARING_ID
    }
    for record in records:
        found = signals.for_record(record)
        row = verifier[record.id]
        assert found["scored"] is row["scored"]
        if row["scored"]:
            assert found["verifier"]["probability"] == row["primary_probability"]
            assert found["laya"]["laya_multi_pt"]["p4_supports"] == row["p4_supports"]
            assert found["translation"]["premise"]
            assert found["translation"]["hypothesis"]
    first = signals.for_record(records[0])
    assert round(first["verifier"]["probability"], 2) == 0.28
    assert first["verifier"]["supported"] is False
    assert [record.tier for record in records].count("person_not_resolved") == 2
    assert round(signals.summary["verifier"]["threshold"], 4) == 0.7478
