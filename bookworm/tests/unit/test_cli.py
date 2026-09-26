import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import typer
from conftest import FIXTURE_VECTORS, MINI_THRESHOLD, THANKS_OPINION, StubEncoder
from typer.testing import CliRunner

import bookworm.features
from bookworm import (
    CachedEncoder,
    ConfigError,
    EvidenceSettings,
    HearingRecord,
    SentenceEncoder,
    UdvConfig,
    UdvRecord,
    __version__,
    build_temporal_split,
    build_udvs,
    load_hearings,
    load_split_config,
    load_udv_config,
    sha256_of_file,
    write_udv_jsonl,
)
from bookworm.cli import app, create_app, default_encoder_factory

runner = CliRunner()
CONFIG = "udv_mini.toml"
SPLIT_CONFIG = "splits_mini.toml"
SPLIT_SHA256 = "1036e0413037fac926bea474d6e97cc7baef3e6213a95f9331345f733723dce3"
WRONG_SHA256 = "0" * 64
MINI_SHA256 = "311f0fb9eebcfb9091a1722e6973e0cbb3c6870eb8d8c58023124529e36a854f"
NOT_UTF8 = b"\xff\xfe\n"
RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def stub_factory(config: UdvConfig, hearings: Sequence[HearingRecord]) -> SentenceEncoder:
    return StubEncoder()


def alternative_stub_factory(
    config: UdvConfig, hearings: Sequence[HearingRecord]
) -> SentenceEncoder:
    vectors = {text: vector for text, vector in FIXTURE_VECTORS.items() if text != THANKS_OPINION}
    return StubEncoder(vectors, revision="stub-revision-alternative")


stub_app = create_app(encoder_factory=stub_factory)
alternative_app = create_app(encoder_factory=alternative_stub_factory)


def build(application: typer.Typer, *options: str) -> Any:
    return runner.invoke(application, ["build-udvs", "--config", CONFIG, *options])


def verify(application: typer.Typer, *options: str) -> Any:
    return runner.invoke(application, ["verify-udvs", "--config", CONFIG, *options])


def output_ids(workdir: Path, run_name: str) -> list[str]:
    lines = (workdir / "out" / f"{run_name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["id"] for line in lines]


def rewrite_config(workdir: Path, old: str, new: str) -> None:
    path = workdir / CONFIG
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_help_lists_the_version_option_and_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--version" in result.output
    assert "build-udvs" in result.output
    assert "verify-udvs" in result.output
    assert "build-splits" in result.output
    assert "verify-splits" in result.output
    assert "export-hearing" in result.output
    assert "export-site" in result.output


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_build_writes_records_and_coverage(mini_workdir: Path) -> None:
    result = build(stub_app, "--run-name", "mini")
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["total"] == 14
    assert summary["by_tier"]["quote_found"] == 5
    assert list(summary)[-3:] == [
        "elapsed_seconds",
        "mean_seconds_per_hearing",
        "max_seconds_per_hearing",
    ]
    assert "[2/2] hearing 2: 8 opinions" in result.stderr
    hearings = load_hearings(mini_workdir / "lds_mini.jsonl")
    expected = build_udvs(hearings, CachedEncoder(StubEncoder()), EvidenceSettings(MINI_THRESHOLD))
    written = (mini_workdir / "out" / "mini.jsonl").read_text(encoding="utf-8")
    assert written == "".join(record.to_json_line() + "\n" for record in expected.records)
    coverage = json.loads((mini_workdir / "out" / "mini_coverage.json").read_text("utf-8"))
    assert coverage["run_name"] == "mini"
    assert coverage["hearings"] == {"count": 2, "ids": [1, 2]}
    assert coverage["encoder_runtime"]["embedding_dimension"] == 4
    assert (
        coverage["pipeline"]["sentence_segmentation"]
        == "per matched turn, concatenated in turn order"
    )
    assert coverage["config"] == load_udv_config(mini_workdir / CONFIG).source
    assert len(list((mini_workdir / "cache").iterdir())) == 4


@pytest.mark.parametrize("command", ["build-udvs", "verify-udvs"])
def test_run_name_is_required(mini_workdir: Path, command: str) -> None:
    result = runner.invoke(stub_app, [command, "--config", CONFIG])
    assert result.exit_code == 2
    assert "--run-name" in result.stderr
    assert not (mini_workdir / "out").exists()


def test_build_refuses_to_replace_an_existing_run(mini_workdir: Path) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    run_files = [mini_workdir / "out" / name for name in ("mini.jsonl", "mini_coverage.json")]
    before = [path.read_bytes() for path in run_files]
    encoder_calls: list[str] = []

    def recording_factory(config: UdvConfig, hearings: Sequence[HearingRecord]) -> SentenceEncoder:
        encoder_calls.append(config.encoder.kind)
        return StubEncoder()

    result = build(create_app(encoder_factory=recording_factory), "--run-name", "mini")
    assert result.exit_code == 2
    assert "mini.jsonl: run file already exists; pass --overwrite to replace it" in result.stderr
    assert [path.read_bytes() for path in run_files] == before
    assert encoder_calls == []
    run_files[0].unlink()
    result = build(stub_app, "--run-name", "mini")
    assert result.exit_code == 2
    assert "mini_coverage.json: run file already exists" in result.stderr
    assert build(stub_app, "--run-name", "mini", "--overwrite").exit_code == 0
    assert run_files[0].read_bytes() == before[0]


def test_build_limit_keeps_the_first_hearings(mini_workdir: Path) -> None:
    assert build(stub_app, "--run-name", "first", "--limit", "1").exit_code == 0
    assert {record_id.split("-")[1] for record_id in output_ids(mini_workdir, "first")} == {"1"}


def test_build_repeated_ids_select_hearings_in_file_order(mini_workdir: Path) -> None:
    assert build(stub_app, "--run-name", "second", "--ids", "2").exit_code == 0
    assert output_ids(mini_workdir, "second")[0] == "udv-2-0-0"
    assert len(output_ids(mini_workdir, "second")) == 8
    assert build(stub_app, "--run-name", "both", "--ids", "2", "--ids", "1").exit_code == 0
    assert output_ids(mini_workdir, "both")[0] == "udv-1-0-0"
    assert len(output_ids(mini_workdir, "both")) == 14


def test_build_rejects_an_empty_selection(mini_workdir: Path) -> None:
    result = build(stub_app, "--run-name", "empty", "--ids", "99")
    assert result.exit_code == 2
    assert "selection is empty" in result.stderr


def test_build_rejects_a_non_positive_limit(mini_workdir: Path) -> None:
    assert build(stub_app, "--run-name", "none", "--limit", "0").exit_code == 2


@pytest.mark.parametrize("application", [stub_app, alternative_app])
def test_verify_exits_zero_on_a_clean_run(mini_workdir: Path, application: typer.Typer) -> None:
    assert build(application, "--run-name", "clean").exit_code == 0
    result = verify(application, "--run-name", "clean")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["problems"] == {}
    assert (report["records"], report["expected_records"]) == (14, 14)


def test_verify_exits_one_on_evidence_without_offsets(mini_workdir: Path) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    path = mini_workdir / "out" / "mini.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["id"] == "udv-1-1-2":
            record["evidence"].update(start_char=None, end_char=None, speaker_turn=None)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    result = verify(stub_app, "--run-name", "mini")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["problems"] == {
        "evidence_offsets_missing": {"count": 1, "examples": ["udv-1-1-2"]},
        "coverage": {
            "count": 1,
            "examples": ["coverage.evidence_offsets.located: reported 11, recomputed 10"],
        },
    }


def test_verify_reports_invalid_lines(mini_workdir: Path) -> None:
    assert build(alternative_app, "--run-name", "clean").exit_code == 0
    path = mini_workdir / "out" / "clean.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace('"semantic_match_high"', '"semantic_match_medium"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = verify(alternative_app, "--run-name", "clean")
    assert result.exit_code == 1
    problems = json.loads(result.stdout)["problems"]
    assert list(problems) == ["schema_invalid", "missing_id", "coverage"]
    assert problems["schema_invalid"]["examples"][0].startswith("line 14 (udv-2-7-0): tier")
    assert problems["missing_id"]["examples"] == ["udv-2-7-0"]


def test_verify_diffs_against_a_baseline(mini_workdir: Path) -> None:
    assert build(alternative_app, "--run-name", "clean").exit_code == 0
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    result = verify(
        alternative_app,
        "--run-name",
        "clean",
        "--baseline",
        str(mini_workdir / "out" / "mini.jsonl"),
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["baseline_diff"] == {
        "baseline_records": 14,
        "evidence_changed": 1,
        "tier_moves": {},
        "extra_in_run": 0,
    }


def test_verify_rejects_an_unreadable_baseline(mini_workdir: Path) -> None:
    assert build(alternative_app, "--run-name", "clean").exit_code == 0
    broken = mini_workdir / "broken.jsonl"
    broken.write_text("{}\n", encoding="utf-8")
    result = verify(alternative_app, "--run-name", "clean", "--baseline", str(broken))
    assert result.exit_code == 2
    assert "cannot read baseline run" in result.stderr


@pytest.mark.parametrize("command", ["build-udvs", "verify-udvs"])
def test_integrity_error_exits_two(mini_workdir: Path, command: str) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    rewrite_config(mini_workdir, MINI_SHA256, WRONG_SHA256)
    result = runner.invoke(stub_app, [command, "--config", CONFIG, "--run-name", "mini"])
    assert result.exit_code == 2
    assert "sha256" in result.stderr


@pytest.mark.parametrize("command", ["build-udvs", "verify-udvs"])
def test_config_errors_exit_two(mini_workdir: Path, command: str) -> None:
    result = runner.invoke(stub_app, [command, "--config", "absent.toml", "--run-name", "mini"])
    assert result.exit_code == 2
    assert "config file not found" in result.stderr


@pytest.mark.parametrize("command", ["build-udvs", "verify-udvs"])
def test_missing_lds_exits_two(mini_workdir: Path, command: str) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    (mini_workdir / "lds_mini.jsonl").unlink()
    result = runner.invoke(stub_app, [command, "--config", CONFIG, "--run-name", "mini"])
    assert result.exit_code == 2
    assert "LDS file not found" in result.stderr


def test_verify_without_run_files_exits_two(mini_workdir: Path) -> None:
    result = verify(stub_app, "--run-name", "never_built")
    assert result.exit_code == 2
    assert "run file not found" in result.stderr


def test_verify_rejects_invalid_coverage_json(mini_workdir: Path) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    (mini_workdir / "out" / "mini_coverage.json").write_text("{", encoding="utf-8")
    result = verify(stub_app, "--run-name", "mini")
    assert result.exit_code == 2
    assert "invalid JSON" in result.stderr


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]", "coverage file is not a JSON object"),
        ("null", "coverage file is not a JSON object"),
        ("{}", "coverage config.evidence.embedding_threshold is missing or not a number"),
        (
            '{"config": {"evidence": {"embedding_threshold": 0.6}}, "hearings": {}}',
            "coverage hearings.ids is missing or not a list of integers",
        ),
    ],
)
def test_verify_rejects_malformed_coverage(mini_workdir: Path, content: str, message: str) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    (mini_workdir / "out" / "mini_coverage.json").write_text(content, encoding="utf-8")
    result = verify(stub_app, "--run-name", "mini")
    assert result.exit_code == 2
    assert message in result.stderr
    assert result.stdout == ""


def test_verify_reports_missing_coverage_counters(mini_workdir: Path) -> None:
    assert build(alternative_app, "--run-name", "clean").exit_code == 0
    path = mini_workdir / "out" / "clean_coverage.json"
    coverage = json.loads(path.read_text(encoding="utf-8"))
    del coverage["evidence_support_types"]
    coverage["opinions"]["total"] = "14"
    path.write_text(json.dumps(coverage), encoding="utf-8")
    result = verify(alternative_app, "--run-name", "clean")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["problems"] == {
        "coverage": {
            "count": 4,
            "examples": [
                "coverage_non_integer:opinions.total",
                "coverage_missing_key:evidence_support_types.direct_quote",
                "coverage_missing_key:evidence_support_types.semantic_with_short_quote",
                "coverage_missing_key:evidence_support_types.semantic_similarity",
            ],
        }
    }


@pytest.mark.parametrize(
    ("file_name", "message"),
    [("mini.jsonl", "run file is not UTF-8"), ("mini_coverage.json", "coverage file is not UTF-8")],
)
def test_verify_rejects_run_files_that_are_not_utf8(
    mini_workdir: Path, file_name: str, message: str
) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    with (mini_workdir / "out" / file_name).open("ab") as handle:
        handle.write(NOT_UTF8)
    result = verify(stub_app, "--run-name", "mini")
    assert result.exit_code == 2
    assert message in result.stderr


@pytest.mark.skipif(RUNNING_AS_ROOT, reason="file permissions do not apply to root")
@pytest.mark.parametrize(
    ("file_name", "message"),
    [("mini.jsonl", "cannot read run file"), ("mini_coverage.json", "cannot read coverage file")],
)
def test_verify_rejects_unreadable_run_files(
    mini_workdir: Path, file_name: str, message: str
) -> None:
    assert build(stub_app, "--run-name", "mini").exit_code == 0
    (mini_workdir / "out" / file_name).chmod(0)
    result = verify(stub_app, "--run-name", "mini")
    assert result.exit_code == 2
    assert message in result.stderr


def test_integer_threshold_is_written_as_in_the_toml(mini_workdir: Path) -> None:
    rewrite_config(mini_workdir, "embedding_threshold = 0.6", "embedding_threshold = 1")
    assert build(alternative_app, "--run-name", "integer").exit_code == 0
    lines = (mini_workdir / "out" / "integer.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 14
    assert all(line.endswith('"embedding_threshold": 1}}') for line in lines)
    coverage = read_json(mini_workdir / "out" / "integer_coverage.json")
    assert coverage["config"]["evidence"]["embedding_threshold"] == 1
    result = verify(alternative_app, "--run-name", "integer")
    assert result.exit_code == 0, result.output


def test_default_factory_builds_a_tfidf_run(mini_workdir: Path) -> None:
    rewrite_config(
        mini_workdir,
        'name = "stub-encoder"\nrevision = "stub-revision-1"\nbatch_size = 8\ndevice = "cpu"',
        'kind = "tfidf"',
    )
    assert build(app, "--run-name", "tfidf").exit_code == 0
    records = [
        json.loads(line)
        for line in (mini_workdir / "out" / "tfidf.jsonl").read_text("utf-8").splitlines()
    ]
    assert {record["method"]["encoder"] for record in records} == {"tfidf"}
    assert [record["id"] for record in records if record["tier"] == "semantic_match_weak"] == [
        "udv-1-0-1",
        "udv-1-1-1",
        "udv-1-2-0",
        "udv-2-1-0",
        "udv-2-7-0",
    ]
    result = verify(app, "--run-name", "tfidf")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["problems"] == {}


def test_default_factory_creates_a_lazy_sentence_transformer(
    udv_mini_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentence_transformer = pytest.importorskip("bookworm.features.sentence_transformer")
    seeds: list[int] = []
    monkeypatch.setattr(sentence_transformer, "seed_torch", seeds.append)
    encoder = default_encoder_factory(load_udv_config(udv_mini_config_path), [])
    assert isinstance(encoder, sentence_transformer.SentenceTransformerEncoder)
    assert encoder.cache_identity == "stub-encoder@stub-revision-1@cpu"
    assert encoder.batch_size == 8
    assert not encoder.is_loaded
    assert seeds == [42]


def test_default_factory_without_the_embeddings_extra(
    udv_mini_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "bookworm.features.sentence_transformer", None)
    monkeypatch.delattr(bookworm.features, "sentence_transformer", raising=False)
    with pytest.raises(ConfigError, match="embeddings"):
        default_encoder_factory(load_udv_config(udv_mini_config_path), [])


def build_splits(*options: str) -> Any:
    return runner.invoke(app, ["build-splits", "--config", SPLIT_CONFIG, *options])


def verify_splits(*options: str) -> Any:
    return runner.invoke(app, ["verify-splits", "--config", SPLIT_CONFIG, *options])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_in_file(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_build_splits_writes_manifest_and_report(splits_workdir: Path) -> None:
    result = build_splits()
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["split_version"] == "mini_temporal"
    assert summary["splits"]["train"] == {
        "hearings": 7,
        "share_of_hearings": 0.7,
        "first_date": "2023-03-01",
        "last_date": "2023-03-23",
        "udvs": 0,
    }
    assert summary["boundaries"]["train_end"] == "2023-03-23"
    manifest = read_json(splits_workdir / "splits" / "mini_temporal.json")
    report = read_json(splits_workdir / "splits" / "mini_temporal_report.json")
    config = load_split_config(splits_workdir / SPLIT_CONFIG)
    expected = build_temporal_split(load_hearings(splits_workdir / "splits_mini.jsonl"), config)
    assert {**manifest, "created_at": None} == {**expected.manifest, "created_at": None}
    assert report["udv_source"] is None
    assert report["config"] == config.source
    assert (splits_workdir / "splits" / "mini_temporal.json").read_text("utf-8").endswith("}")


def test_build_splits_reads_the_udv_run(splits_workdir: Path, split_udvs: list[UdvRecord]) -> None:
    write_udv_jsonl(split_udvs, splits_workdir / "udv" / "mini.jsonl")
    assert build_splits().exit_code == 0
    report = read_json(splits_workdir / "splits" / "mini_temporal_report.json")
    assert report["udv_source"] == "udv/mini.jsonl"
    assert [report["splits"][name]["udvs"] for name in ("train", "validation", "test")] == [
        2,
        1,
        1,
    ]
    result = verify_splits()
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["udvs"] == {"train": 2, "validation": 1, "test": 1}


def test_verify_splits_exits_zero_on_a_clean_build(splits_workdir: Path) -> None:
    assert build_splits().exit_code == 0
    result = verify_splits()
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["problems"] == []
    assert report["hearings"] == {"train": 7, "validation": 1, "test": 2}
    assert report["date_extraction"]["max_lag_days"] == 1


def test_verify_splits_exits_one_on_an_altered_manifest(splits_workdir: Path) -> None:
    assert build_splits().exit_code == 0
    path = splits_workdir / "splits" / "mini_temporal.json"
    manifest = read_json(path)
    manifest["article_dates"]["5"] = "2023-03-15"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_splits()
    assert result.exit_code == 1
    assert json.loads(result.stdout)["problems"] == ["article_date_mismatch:5"]


def test_default_split_config_path(splits_workdir: Path) -> None:
    (splits_workdir / "configs").mkdir()
    (splits_workdir / SPLIT_CONFIG).rename(splits_workdir / "configs" / "splits.toml")
    assert runner.invoke(app, ["build-splits"]).exit_code == 0
    assert runner.invoke(app, ["verify-splits"]).exit_code == 0


def test_verify_splits_without_manifest_exits_two(splits_workdir: Path) -> None:
    result = verify_splits()
    assert result.exit_code == 2
    assert "split manifest not found" in result.stderr


def test_verify_splits_reports_a_null_split(splits_workdir: Path) -> None:
    assert build_splits().exit_code == 0
    path = splits_workdir / "splits" / "mini_temporal.json"
    manifest = read_json(path)
    manifest["validation"] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_splits()
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["problems"] == ["non_integer_ids:validation"]
    assert report["hearings"] == {"train": 7, "validation": None, "test": 2}


def test_verify_splits_reports_a_non_integer_report_counter(
    splits_workdir: Path, split_udvs: list[UdvRecord]
) -> None:
    write_udv_jsonl(split_udvs, splits_workdir / "udv" / "mini.jsonl")
    assert build_splits().exit_code == 0
    path = splits_workdir / "splits" / "mini_temporal_report.json"
    report = read_json(path)
    report["splits"]["train"]["udvs"] = "2"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = verify_splits()
    assert result.exit_code == 1
    assert json.loads(result.stdout)["problems"] == ["report_non_integer:splits.train.udvs"]


@pytest.mark.parametrize(
    ("file_name", "description"),
    [("mini_temporal.json", "split manifest"), ("mini_temporal_report.json", "split report")],
)
def test_verify_splits_rejects_files_that_are_not_utf8(
    splits_workdir: Path, file_name: str, description: str
) -> None:
    assert build_splits().exit_code == 0
    with (splits_workdir / "splits" / file_name).open("ab") as handle:
        handle.write(NOT_UTF8)
    result = verify_splits()
    assert result.exit_code == 2
    assert f"{description} is not UTF-8" in result.stderr


@pytest.mark.skipif(RUNNING_AS_ROOT, reason="file permissions do not apply to root")
def test_verify_splits_rejects_an_unreadable_report(splits_workdir: Path) -> None:
    assert build_splits().exit_code == 0
    (splits_workdir / "splits" / "mini_temporal_report.json").chmod(0)
    result = verify_splits()
    assert result.exit_code == 2
    assert "cannot read split report" in result.stderr


@pytest.mark.parametrize(
    ("content", "message"),
    [("{", "invalid JSON"), ("[]", "is not a JSON object")],
)
def test_verify_splits_rejects_unreadable_files(
    splits_workdir: Path, content: str, message: str
) -> None:
    assert build_splits().exit_code == 0
    (splits_workdir / "splits" / "mini_temporal_report.json").write_text(content, "utf-8")
    result = verify_splits()
    assert result.exit_code == 2
    assert message in result.stderr


@pytest.mark.parametrize("command", ["build-splits", "verify-splits"])
def test_split_commands_exit_two_on_integrity_errors(splits_workdir: Path, command: str) -> None:
    assert build_splits().exit_code == 0
    replace_in_file(splits_workdir / SPLIT_CONFIG, SPLIT_SHA256, WRONG_SHA256)
    result = runner.invoke(app, [command, "--config", SPLIT_CONFIG])
    assert result.exit_code == 2
    assert "sha256" in result.stderr


@pytest.mark.parametrize("command", ["build-splits", "verify-splits"])
def test_split_commands_exit_two_without_config(splits_workdir: Path, command: str) -> None:
    result = runner.invoke(app, [command, "--config", "absent.toml"])
    assert result.exit_code == 2
    assert "config file not found" in result.stderr


@pytest.mark.parametrize("command", ["build-splits", "verify-splits"])
def test_split_commands_exit_two_without_lds(splits_workdir: Path, command: str) -> None:
    assert build_splits().exit_code == 0
    (splits_workdir / "splits_mini.jsonl").unlink()
    result = runner.invoke(app, [command, "--config", SPLIT_CONFIG])
    assert result.exit_code == 2
    assert "LDS file not found" in result.stderr


def test_build_splits_exits_two_on_an_undated_article(splits_workdir: Path) -> None:
    lds = splits_workdir / "splits_mini.jsonl"
    replace_in_file(lds, "23/03/2023 - 15:10", "23/03/2023")
    replace_in_file(splits_workdir / SPLIT_CONFIG, SPLIT_SHA256, sha256_of_file(lds))
    result = build_splits()
    assert result.exit_code == 2
    assert "hearing 7: no publication timestamp found in materia" in result.stderr


def test_build_splits_exits_two_on_an_unreadable_udv_run(splits_workdir: Path) -> None:
    (splits_workdir / "udv").mkdir()
    (splits_workdir / "udv" / "mini.jsonl").write_text("{}\n", encoding="utf-8")
    result = build_splits()
    assert result.exit_code == 2
    assert "cannot read UDV run" in result.stderr
