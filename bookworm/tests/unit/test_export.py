import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import typer
from conftest import (
    MINI_THRESHOLD,
    THANKS_SENTENCE,
    TURNS_PHYSICS_SENTENCE,
    TURNS_VECTORS,
    StubEncoder,
)
from typer.testing import CliRunner

from bookworm import (
    CachedEncoder,
    ConfigError,
    EvidenceSettings,
    HearingRecord,
    QuotePolicy,
    SentenceEncoder,
    UdvConfig,
    UdvRecord,
    build_udvs,
    export_hearing,
    load_hearings,
    pipeline_description,
)
from bookworm.cli import create_app
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.export import DEFAULT_TOP_K, check_run_pipeline, split_of
from bookworm.udv.quotes import DOUBLE_QUOTE_PATTERNS

JsonObject = dict[str, Any]

runner = CliRunner()
CONFIG = "udv_mini.toml"
TOP_LEVEL_KEYS = ["hearing", "transcript", "turns", "people", "udvs", "run"]
HEARING_KEYS = [
    "id",
    "split",
    "article_date",
    "assunto",
    "materia",
    "transcript_chars",
    "transcript_words",
]
UDV_KEYS = [
    "id",
    "hearing_id",
    "actor",
    "proposition",
    "evidence",
    "tier",
    "provenance",
    "method",
    "candidates",
    "n_candidates",
    "quotes",
]
MANIFEST = {"train": [1], "validation": [2], "test": []}
PIPELINE = pipeline_description()
DOUBLE_ONLY = QuotePolicy(patterns=DOUBLE_QUOTE_PATTERNS)


def stub_factory(config: UdvConfig, hearings: Sequence[HearingRecord]) -> SentenceEncoder:
    return StubEncoder()


def unused_factory(config: UdvConfig, hearings: Sequence[HearingRecord]) -> SentenceEncoder:
    raise AssertionError("the export commands must not build an encoder")


stub_app = create_app(encoder_factory=stub_factory)
export_only_app = create_app(encoder_factory=unused_factory)


def mini_records(hearings: list[HearingRecord], hearing_id: int) -> list[UdvRecord]:
    run = build_udvs(hearings, CachedEncoder(StubEncoder()), EvidenceSettings(MINI_THRESHOLD))
    return [record for record in run.records if record.hearing_id == hearing_id]


@pytest.fixture
def exported(mini_hearings: dict[int, HearingRecord]) -> JsonObject:
    hearing = mini_hearings[1]
    return export_hearing(
        hearing,
        mini_records([hearing], 1),
        CachedEncoder(StubEncoder()),
        run_name="mini",
        pipeline=PIPELINE,
        top_k=3,
        split="train",
    )


def slice_text(transcript: str, start: int | None, end: int | None) -> str:
    assert start is not None and end is not None
    return normalize_whitespace(transcript[start:end])


def test_export_has_the_demo_shape(exported: JsonObject) -> None:
    assert list(exported) == TOP_LEVEL_KEYS
    assert list(exported["hearing"]) == HEARING_KEYS
    hearing = exported["hearing"]
    assert (hearing["id"], hearing["split"], hearing["article_date"]) == (1, "train", "2024-03-12")
    assert hearing["assunto"] == "Regulação das cooperativas de pequeno porte"
    assert hearing["transcript_chars"] == len(exported["transcript"])
    assert hearing["transcript_words"] == len(exported["transcript"].split())
    assert exported["run"] == {
        "name": "mini",
        "encoder": "stub-encoder",
        "revision": "stub-revision-1",
        "threshold": MINI_THRESHOLD,
    }
    assert [list(udv) for udv in exported["udvs"]] == [UDV_KEYS] * 6
    json.dumps(exported)


def test_export_lists_every_turn_with_its_sentences(exported: JsonObject) -> None:
    transcript = exported["transcript"]
    turns = exported["turns"]
    assert [turn["index"] for turn in turns] == list(range(7))
    assert [turn["speaker"] for turn in turns][:2] == ["PRESIDENTE", "MARCOS PEREIRA"]
    assert turns[0]["party"] == "Dep. Carlos Nunes. PL - RJ"
    assert turns[5]["sentences"][0]["text"] == THANKS_SENTENCE
    for turn in turns:
        assert set(turn) == {"index", "speaker", "party", "start", "end", "sentences"}
        for sentence in turn["sentences"]:
            assert slice_text(transcript, sentence["start"], sentence["end"]) == sentence["text"]
            assert turn["start"] <= sentence["start"] < sentence["end"] <= turn["end"]


def test_export_lists_people_with_their_turns(exported: JsonObject) -> None:
    assert exported["people"] == [
        {
            "index": 0,
            "name": "João Silva",
            "role": "Deputado (PT-SP)",
            "turns": [3],
            "resolved": True,
        },
        {
            "index": 1,
            "name": "Marcos Pereira",
            "role": "Presidente da Associação de Cooperativas do Interior",
            "turns": [1, 5],
            "resolved": True,
        },
        {
            "index": 2,
            "name": "Carlos Nunes",
            "role": "Deputado (PL-RJ), presidente da comissão",
            "turns": [0, 2, 4, 6],
            "resolved": True,
        },
    ]


def test_export_ranks_candidates_among_the_actor_sentences(
    exported: JsonObject, mini_hearings: dict[int, HearingRecord]
) -> None:
    records = {record.id: record for record in mini_records([mini_hearings[1]], 1)}
    people = {person["index"]: person for person in exported["people"]}
    for udv in exported["udvs"]:
        record = records[udv["id"]]
        assert {key: udv[key] for key in UDV_KEYS[:8]} == record.to_dict()
        person = people[int(udv["id"].split("-")[2])]
        candidates = udv["candidates"]
        assert len(candidates) == min(3, udv["n_candidates"])
        assert [candidate["score"] for candidate in candidates] == sorted(
            (candidate["score"] for candidate in candidates), reverse=True
        )
        for candidate in candidates:
            assert candidate["turn"] in person["turns"]
            text = slice_text(exported["transcript"], candidate["start"], candidate["end"])
            assert text == candidate["text"]
        if record.tier.startswith("semantic") and record.evidence is not None:
            top = candidates[0]
            assert (top["text"], top["turn"], top["start"], top["end"]) == (
                record.evidence.text,
                record.evidence.speaker_turn,
                record.evidence.start_char,
                record.evidence.end_char,
            )
            assert record.evidence.score is not None
            assert top["score"] == round(record.evidence.score, 4)
    by_id = {udv["id"]: udv for udv in exported["udvs"]}
    assert [udv["n_candidates"] for udv in exported["udvs"]] == [2, 2, 5, 5, 5, 6]
    assert by_id["udv-1-1-1"]["quotes"] == [
        "prazos de transição realmente longos para as cooperativas"
    ]
    assert by_id["udv-1-0-1"]["quotes"] == []


def test_export_keeps_the_first_of_tied_candidates(turns_hearing: HearingRecord) -> None:
    encoder = CachedEncoder(StubEncoder(TURNS_VECTORS))
    run = build_udvs([turns_hearing], encoder, EvidenceSettings(MINI_THRESHOLD))
    exported = export_hearing(
        turns_hearing, run.records, encoder, run_name="turns", pipeline=PIPELINE
    )
    physics = next(udv for udv in exported["udvs"] if udv["id"] == "udv-3-0-3")
    assert physics["n_candidates"] == 9
    assert len(physics["candidates"]) == DEFAULT_TOP_K
    assert {candidate["score"] for candidate in physics["candidates"]} == {1.0}
    assert [candidate["turn"] for candidate in physics["candidates"]] == [1, 1, 1, 3, 3, 3, 5, 5]
    assert physics["candidates"][1]["text"] == TURNS_PHYSICS_SENTENCE
    assert physics["evidence"]["text"] == TURNS_PHYSICS_SENTENCE


def test_export_of_people_without_sentences(mini_hearings: dict[int, HearingRecord]) -> None:
    hearing = mini_hearings[2]
    exported = export_hearing(
        hearing,
        mini_records([hearing], 2),
        CachedEncoder(StubEncoder()),
        run_name="mini",
        pipeline=PIPELINE,
    )
    by_id = {udv["id"]: udv for udv in exported["udvs"]}
    assert (by_id["udv-2-3-0"]["candidates"], by_id["udv-2-3-0"]["n_candidates"]) == ([], 0)
    assert (by_id["udv-2-5-0"]["candidates"], by_id["udv-2-5-0"]["n_candidates"]) == ([], 0)
    assert exported["hearing"]["split"] is None
    assert exported["hearing"]["article_date"] == "2024-05-20"
    assert exported["people"][3] == {
        "index": 3,
        "name": "Ana",
        "role": "Representante dos pontos de cultura",
        "turns": [],
        "resolved": False,
    }
    assert by_id["udv-2-0-0"]["quotes"] == [
        "os mestres da cultura popular precisam de reconhecimento formal do Estado"
    ]


def test_export_rejects_records_of_another_hearing_or_encoder(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    records = mini_records([hearing], 1)
    encoder = CachedEncoder(StubEncoder())
    with pytest.raises(ConfigError, match="do not match the LDS opinions"):
        export_hearing(hearing, records[1:], encoder, run_name="mini", pipeline=PIPELINE)
    with pytest.raises(ConfigError, match="has no records"):
        export_hearing(hearing, [], encoder, run_name="mini", pipeline=PIPELINE)
    with pytest.raises(ConfigError, match="top_k"):
        export_hearing(hearing, records, encoder, run_name="mini", pipeline=PIPELINE, top_k=0)
    with pytest.raises(ConfigError, match="was built with stub-encoder@stub-revision-1"):
        export_hearing(
            hearing,
            records,
            CachedEncoder(StubEncoder(revision="other")),
            run_name="mini",
            pipeline=PIPELINE,
        )


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        (None, "coverage pipeline is missing or not an object"),
        (["per matched turn"], "coverage pipeline is missing or not an object"),
        (
            {**PIPELINE, "quote_patterns": PIPELINE["quote_patterns"][:1]},
            "differs from the current pipeline in quote_patterns$",
        ),
        (
            {**PIPELINE, "sentence_segmentation": "whole speech", "extra": 1},
            "differs from the current pipeline in sentence_segmentation, extra$",
        ),
        (
            {key: value for key, value in PIPELINE.items() if key != "quote_search"},
            "differs from the current pipeline in quote_search$",
        ),
    ],
)
def test_export_rejects_a_run_of_another_pipeline(
    mini_hearings: dict[int, HearingRecord], pipeline: object, message: str
) -> None:
    hearing = mini_hearings[1]
    encoder = StubEncoder()
    with pytest.raises(ConfigError, match=message):
        export_hearing(
            hearing,
            mini_records([hearing], 1),
            CachedEncoder(encoder),
            run_name="mini",
            pipeline=pipeline,
        )
    assert encoder.calls == []


def test_check_run_pipeline_accepts_the_policy_of_the_run() -> None:
    check_run_pipeline(PIPELINE)
    check_run_pipeline(pipeline_description(DOUBLE_ONLY), DOUBLE_ONLY)
    with pytest.raises(ConfigError, match="quote_patterns"):
        check_run_pipeline(pipeline_description(DOUBLE_ONLY))


def test_export_quotes_follow_the_patterns_of_the_run(turns_hearing: HearingRecord) -> None:
    encoder = CachedEncoder(StubEncoder(TURNS_VECTORS))
    settings = EvidenceSettings(MINI_THRESHOLD, quote_policy=DOUBLE_ONLY)
    double_run = build_udvs([turns_hearing], encoder, settings)
    exported = export_hearing(
        turns_hearing,
        double_run.records,
        encoder,
        run_name="double",
        pipeline=pipeline_description(DOUBLE_ONLY),
        quote_policy=DOUBLE_ONLY,
    )
    single = next(udv for udv in exported["udvs"] if udv["id"] == "udv-3-0-2")
    assert (single["tier"], single["quotes"]) == ("semantic_match_high", [])
    default_run = build_udvs([turns_hearing], encoder, EvidenceSettings(MINI_THRESHOLD))
    exported = export_hearing(
        turns_hearing, default_run.records, encoder, run_name="default", pipeline=PIPELINE
    )
    single = next(udv for udv in exported["udvs"] if udv["id"] == "udv-3-0-2")
    assert (single["tier"], single["quotes"]) == (
        "quote_found",
        ["as escolas rurais precisam de internet de qualidade"],
    )
    with pytest.raises(ConfigError, match="quote_patterns"):
        export_hearing(
            turns_hearing,
            double_run.records,
            encoder,
            run_name="double",
            pipeline=pipeline_description(DOUBLE_ONLY),
        )


def semantic_index(records: Sequence[UdvRecord]) -> int:
    return next(
        index for index, record in enumerate(records) if record.tier == "semantic_match_high"
    )


@pytest.mark.parametrize(
    "update",
    [
        {"start_char": 0},
        {"text": "Outra sentença qualquer da fala."},
        {"speaker_turn": 99},
        {"score": 0.123},
        {"score": None},
    ],
)
def test_export_rejects_evidence_that_is_not_the_top_candidate(
    mini_hearings: dict[int, HearingRecord], update: dict[str, Any]
) -> None:
    hearing = mini_hearings[1]
    records = mini_records([hearing], 1)
    index = semantic_index(records)
    evidence = records[index].evidence
    assert evidence is not None
    records[index] = records[index].model_copy(
        update={"evidence": evidence.model_copy(update=update)}
    )
    with pytest.raises(ConfigError, match=f"{records[index].id}: the recorded evidence"):
        export_hearing(
            hearing, records, CachedEncoder(StubEncoder()), run_name="mini", pipeline=PIPELINE
        )


def test_export_rejects_a_semantic_record_without_evidence(
    mini_hearings: dict[int, HearingRecord],
) -> None:
    hearing = mini_hearings[1]
    records = mini_records([hearing], 1)
    index = semantic_index(records)
    records[index] = records[index].model_copy(update={"evidence": None})
    with pytest.raises(ConfigError, match=f"{records[index].id}: the recorded evidence"):
        export_hearing(
            hearing, records, CachedEncoder(StubEncoder()), run_name="mini", pipeline=PIPELINE
        )


def test_split_of_reads_the_manifest() -> None:
    assert split_of(MANIFEST, 1) == "train"
    assert split_of(MANIFEST, 2) == "validation"
    with pytest.raises(ConfigError, match="hearing 3 is in no split"):
        split_of(MANIFEST, 3)
    with pytest.raises(ConfigError, match="test is missing"):
        split_of({"train": [1], "validation": [2]}, 5)


def build(application: typer.Typer, run_name: str) -> Any:
    return runner.invoke(application, ["build-udvs", "--config", CONFIG, "--run-name", run_name])


def export(application: typer.Typer, *options: str) -> Any:
    return runner.invoke(application, ["export-hearing", "--config", CONFIG, *options])


def test_cli_exports_a_hearing(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    manifest = mini_workdir / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
    output = mini_workdir / "demo" / "hearing1.json"
    result = export(
        stub_app,
        "--run-name",
        "mini",
        "--hearing",
        "1",
        "--output",
        str(output),
        "--top-k",
        "3",
        "--split-manifest",
        str(manifest),
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "hearing": 1,
        "split": "train",
        "turns": 7,
        "udvs": 6,
        "output": str(output),
    }
    payload = json.loads(output.read_text(encoding="utf-8"))
    hearings = load_hearings(mini_workdir / "lds_mini.jsonl")
    expected = export_hearing(
        hearings[0],
        mini_records(hearings, 1),
        CachedEncoder(StubEncoder()),
        run_name="mini",
        pipeline=PIPELINE,
        top_k=3,
        split="train",
    )
    assert payload == expected


def test_cli_export_defaults_to_eight_candidates_and_no_split(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    output = mini_workdir / "hearing2.json"
    result = export(stub_app, "--run-name", "mini", "--hearing", "2", "--output", str(output))
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["hearing"]["split"] is None
    assert max(len(udv["candidates"]) for udv in payload["udvs"]) == 3


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--run-name", "never_built", "--hearing", "1"], "run file not found"),
        (["--run-name", "mini", "--hearing", "9"], "hearing 9 is not part of run mini"),
        (
            ["--run-name", "mini", "--hearing", "1", "--split-manifest", "absent.json"],
            "split manifest not found",
        ),
    ],
)
def test_cli_export_input_errors_exit_two(
    mini_workdir: Path, options: list[str], message: str
) -> None:
    assert build(stub_app, "mini").exit_code == 0
    result = export(stub_app, *options, "--output", str(mini_workdir / "x.json"))
    assert result.exit_code == 2
    assert message in result.stderr
    assert not (mini_workdir / "x.json").exists()


def rewrite_config(workdir: Path, old: str, new: str) -> None:
    path = workdir / CONFIG
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_cli_export_rejects_another_encoder(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    rewrite_config(mini_workdir, 'revision = "stub-revision-1"', 'revision = "stub-revision-2"')
    output = mini_workdir / "x.json"
    result = export(stub_app, "--run-name", "mini", "--hearing", "1", "--output", str(output))
    assert result.exit_code == 2
    assert "but the encoder is stub-encoder@stub-revision-2" in result.stderr
    assert not output.exists()


def test_cli_export_reads_only_the_cache_of_the_run(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    cache = mini_workdir / "cache"
    before = {path.name: path.read_bytes() for path in cache.iterdir()}
    output = mini_workdir / "hearing1.json"
    options = ("--run-name", "mini", "--hearing", "1", "--output", str(output))
    result = export(export_only_app, *options)
    assert result.exit_code == 0, result.output
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == before
    expected = mini_workdir / "expected.json"
    options = ("--run-name", "mini", "--hearing", "1", "--output", str(expected))
    assert export(stub_app, *options).exit_code == 0
    assert output.read_bytes() == expected.read_bytes()


def test_cli_export_fails_on_a_cache_miss(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    for path in (mini_workdir / "cache").glob("sentences_1_*.npy"):
        path.unlink()
    output = mini_workdir / "x.json"
    result = export(stub_app, "--run-name", "mini", "--hearing", "1", "--output", str(output))
    assert result.exit_code == 2
    assert (
        "sentences_1: no cached embeddings for 13 texts of stub-encoder@stub-revision-1@cpu "
        "(cache/sentences_1_" in result.stderr
    )
    assert "only reads the embedding cache written by build-udvs" in result.stderr
    assert not output.exists()
    assert sorted(path.name[:11] for path in (mini_workdir / "cache").iterdir()) == [
        "opinions_1_",
        "opinions_2_",
        "sentences_2",
    ]


@pytest.mark.parametrize("runtime", [None, {"device": 1}])
def test_cli_export_needs_the_device_of_the_run(
    mini_workdir: Path, runtime: JsonObject | None
) -> None:
    assert build(stub_app, "mini").exit_code == 0
    path = mini_workdir / "out" / "mini_coverage.json"
    coverage = json.loads(path.read_text(encoding="utf-8"))
    if runtime is None:
        del coverage["encoder_runtime"]
    else:
        coverage["encoder_runtime"] = runtime
    path.write_text(json.dumps(coverage), encoding="utf-8")
    output = mini_workdir / "x.json"
    result = export(stub_app, "--run-name", "mini", "--hearing", "1", "--output", str(output))
    assert result.exit_code == 2
    assert "mini_coverage.json: encoder_runtime.device is missing or not text" in result.stderr


def test_cli_export_rejects_invalid_run_lines(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    path = mini_workdir / "out" / "mini.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"quote_found"', '"quote_maybe"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = export(
        stub_app, "--run-name", "mini", "--hearing", "1", "--output", str(mini_workdir / "x.json")
    )
    assert result.exit_code == 2
    assert "line 1 (udv-1-0-0)" in result.stderr


def test_cli_export_rejects_malformed_coverage(mini_workdir: Path) -> None:
    assert build(stub_app, "mini").exit_code == 0
    (mini_workdir / "out" / "mini_coverage.json").write_text("{}", encoding="utf-8")
    result = export(
        stub_app, "--run-name", "mini", "--hearing", "1", "--output", str(mini_workdir / "x.json")
    )
    assert result.exit_code == 2
    assert "hearings.ids is missing" in result.stderr


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        (None, "coverage pipeline is missing or not an object"),
        (
            pipeline_description(DOUBLE_ONLY),
            "coverage pipeline differs from the current pipeline in quote_patterns",
        ),
    ],
)
def test_cli_export_rejects_a_run_of_another_pipeline(
    mini_workdir: Path, pipeline: JsonObject | None, message: str
) -> None:
    assert build(stub_app, "mini").exit_code == 0
    path = mini_workdir / "out" / "mini_coverage.json"
    coverage = json.loads(path.read_text(encoding="utf-8"))
    if pipeline is None:
        del coverage["pipeline"]
    else:
        coverage["pipeline"] = pipeline
    path.write_text(json.dumps(coverage), encoding="utf-8")
    output = mini_workdir / "x.json"
    result = export(stub_app, "--run-name", "mini", "--hearing", "1", "--output", str(output))
    assert result.exit_code == 2
    assert f"{path.relative_to(mini_workdir)}: {message}" in result.stderr
    assert not output.exists()


def test_cli_export_rejects_a_non_positive_top_k(mini_workdir: Path) -> None:
    result = export(
        stub_app,
        "--run-name",
        "mini",
        "--hearing",
        "1",
        "--output",
        str(mini_workdir / "x.json"),
        "--top-k",
        "0",
    )
    assert result.exit_code == 2
