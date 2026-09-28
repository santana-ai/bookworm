import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import typer
from conftest import (
    MINI_CONFIG,
    MINI_THRESHOLD,
    REPOSITORY_ROOT,
    StubEncoder,
    stub_factory,
    unused_factory,
)
from typer.testing import CliRunner

import bookworm.cli
from bookworm import (
    CachedEncoder,
    ConfigError,
    EvidenceSettings,
    HearingRecord,
    Method,
    SplitName,
    UdvRecord,
    build_udvs,
    display_title,
    export_hearing,
    export_site,
    pipeline_description,
    site_index_entry,
)
from bookworm.cli import app, create_app, default_site_dir
from bookworm.data.io import JsonObject
from bookworm.udv.site import TITLE_ELLIPSIS, TITLE_MAX_CHARS

runner = CliRunner()
PIPELINE = pipeline_description()
MANIFEST = {"train": [1], "validation": [2], "test": []}
MINI_RUN = {
    "name": "mini",
    "encoder": "stub-encoder",
    "revision": "stub-revision-1",
    "threshold": 0.6,
}
INDEX_ENTRY_KEYS = [
    "id",
    "split",
    "article_date",
    "assunto",
    "title",
    "n_udvs",
    "n_people",
    "n_people_resolved",
    "tiers",
    "support_types",
    "transcript_words",
    "actors",
]


stub_app = create_app(encoder_factory=stub_factory)
export_only_app = create_app(encoder_factory=unused_factory)


def mini_records(hearings: Sequence[HearingRecord]) -> list[UdvRecord]:
    return build_udvs(
        hearings, CachedEncoder(StubEncoder()), EvidenceSettings(MINI_THRESHOLD)
    ).records


def files_of(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_display_title_is_the_first_line_of_the_article() -> None:
    materia = "  Deputados   cobram\tprazos \n\nSubtítulo da matéria\n"
    assert display_title(materia, "Assunto da audiência") == "Deputados cobram prazos"


def test_display_title_falls_back_to_the_assunto() -> None:
    assert display_title(" \n\t\n", "  Cumprimento  dos direitos ") == "Cumprimento dos direitos"


def test_display_title_keeps_a_title_at_the_limit() -> None:
    title = ("palavra " * 20)[:TITLE_MAX_CHARS].strip()
    assert len(title) == TITLE_MAX_CHARS - 1
    assert display_title(f"{title}x", "") == f"{title}x"


@pytest.mark.parametrize(
    ("title", "max_chars", "expected"),
    [
        ("aaa bbb ccc", 8, "aaa bbb…"),
        ("aaa bbbb ccc", 8, "aaa…"),
        (
            "Regulamentação precisa de cautela, afirmam especialistas",
            36,
            "Regulamentação precisa de cautela…",
        ),
        ("aaaaaaaaaa", 5, "aaaa…"),
    ],
)
def test_display_title_cuts_long_titles_at_a_word_boundary(
    title: str, max_chars: int, expected: str
) -> None:
    result = display_title(title, "", max_chars)
    assert result == expected
    assert len(result) <= max_chars
    assert result.endswith(TITLE_ELLIPSIS)


def test_default_title_limit_applies_to_long_headlines() -> None:
    headline = " ".join(f"palavra{number}" for number in range(30))
    title = display_title(f"{headline}\nSubtítulo", "Assunto")
    assert len(title) <= TITLE_MAX_CHARS
    assert title.endswith(TITLE_ELLIPSIS)
    assert headline.startswith(title.removesuffix(TITLE_ELLIPSIS))


@pytest.fixture
def payloads(mini_hearing_list: list[HearingRecord]) -> dict[int, JsonObject]:
    records = mini_records(mini_hearing_list)
    return {
        hearing.id: export_hearing(
            hearing,
            [record for record in records if record.hearing_id == hearing.id],
            CachedEncoder(StubEncoder()),
            run_name="mini",
            pipeline=PIPELINE,
            split="validation",
        )
        for hearing in mini_hearing_list
    }


def test_index_entry_summarizes_the_hearing_payload(payloads: dict[int, JsonObject]) -> None:
    entry = site_index_entry(payloads[2])
    assert list(entry) == INDEX_ENTRY_KEYS
    hearing = payloads[2]["hearing"]
    assert entry == {
        "id": 2,
        "split": "validation",
        "article_date": "2024-05-20",
        "assunto": hearing["assunto"],
        "title": "Mestres da cultura popular pedem reconhecimento formal",
        "n_udvs": 8,
        "n_people": 8,
        "n_people_resolved": 6,
        "tiers": {
            "quote_found": 3,
            "semantic_match_high": 2,
            "semantic_match_weak": 0,
            "no_evidence": 1,
            "person_not_resolved": 2,
        },
        "support_types": {
            "direct_quote": 3,
            "semantic_with_short_quote": 0,
            "semantic_similarity": 2,
        },
        "transcript_words": 165,
        "actors": [person["name"] for person in payloads[2]["people"]],
    }
    assert len(entry["actors"]) == 8


def test_index_entry_counts_support_types_of_evidence_only(
    payloads: dict[int, JsonObject],
) -> None:
    entry = site_index_entry(payloads[1])
    with_evidence = [udv for udv in payloads[1]["udvs"] if udv["evidence"] is not None]
    assert sum(entry["tiers"].values()) == entry["n_udvs"] == 6
    assert sum(entry["support_types"].values()) == len(with_evidence) == 6
    assert entry["support_types"] == {
        "direct_quote": 2,
        "semantic_with_short_quote": 1,
        "semantic_similarity": 3,
    }


def test_export_site_writes_every_hearing_and_the_index(
    tmp_path: Path, mini_hearing_list: list[HearingRecord]
) -> None:
    records = mini_records(mini_hearing_list)
    progress: list[tuple[int, int, int]] = []
    site = export_site(
        list(reversed(mini_hearing_list)),
        records,
        CachedEncoder(StubEncoder()),
        tmp_path / "site",
        run_name="mini",
        pipeline=PIPELINE,
        top_k=3,
        split_manifest=MANIFEST,
        on_hearing=lambda number, entry, size: progress.append((number, entry["id"], size)),
    )
    assert sorted(files_of(tmp_path / "site")) == [
        "hearings/1.json",
        "hearings/2.json",
        "index.json",
    ]
    splits: tuple[SplitName, ...] = ("train", "validation")
    for hearing, split in zip(mini_hearing_list, splits, strict=True):
        written = json.loads((tmp_path / "site" / "hearings" / f"{hearing.id}.json").read_text())
        assert written == export_hearing(
            hearing,
            [record for record in records if record.hearing_id == hearing.id],
            CachedEncoder(StubEncoder()),
            run_name="mini",
            pipeline=PIPELINE,
            top_k=3,
            split=split,
        )
    index = json.loads((tmp_path / "site" / "index.json").read_text(encoding="utf-8"))
    assert index == site.index
    assert list(index) == ["run", "hearings"]
    assert index["run"] == MINI_RUN
    assert [entry["id"] for entry in index["hearings"]] == [1, 2]
    assert [entry["split"] for entry in index["hearings"]] == ["train", "validation"]
    sizes = {
        hearing_id: (tmp_path / "site" / "hearings" / f"{hearing_id}.json").stat().st_size
        for hearing_id in (1, 2)
    }
    assert site.hearing_bytes == sizes
    assert site.index_bytes == (tmp_path / "site" / "index.json").stat().st_size
    assert site.total_bytes == sum(len(data) for data in files_of(tmp_path / "site").values())
    assert progress == [(1, 1, sizes[1]), (2, 2, sizes[2])]


def test_export_site_is_deterministic(
    tmp_path: Path, mini_hearing_list: list[HearingRecord]
) -> None:
    records = mini_records(mini_hearing_list)
    for name in ("first", "second"):
        export_site(
            mini_hearing_list,
            records,
            CachedEncoder(StubEncoder()),
            tmp_path / name,
            run_name="mini",
            pipeline=PIPELINE,
        )
    assert files_of(tmp_path / "first") == files_of(tmp_path / "second")
    index = json.loads((tmp_path / "first" / "index.json").read_text(encoding="utf-8"))
    assert [entry["split"] for entry in index["hearings"]] == [None, None]


def test_export_site_checks_every_split_before_writing(
    tmp_path: Path, mini_hearing_list: list[HearingRecord]
) -> None:
    with pytest.raises(ConfigError, match="hearing 2 is in no split"):
        export_site(
            mini_hearing_list,
            mini_records(mini_hearing_list),
            CachedEncoder(StubEncoder()),
            tmp_path / "site",
            run_name="mini",
            pipeline=PIPELINE,
            split_manifest={"train": [1], "validation": [], "test": []},
        )
    assert not (tmp_path / "site").exists()


def test_export_site_rejects_records_outside_the_run(
    tmp_path: Path, mini_hearing_list: list[HearingRecord]
) -> None:
    with pytest.raises(ConfigError, match=r"outside its coverage: \[2\]"):
        export_site(
            mini_hearing_list[:1],
            mini_records(mini_hearing_list),
            CachedEncoder(StubEncoder()),
            tmp_path / "site",
            run_name="mini",
            pipeline=PIPELINE,
        )
    assert not (tmp_path / "site").exists()


def test_export_site_rejects_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="run mini has no hearings to export"):
        export_site(
            [], [], CachedEncoder(StubEncoder()), tmp_path, run_name="mini", pipeline=PIPELINE
        )


def test_export_site_rejects_hearings_with_different_run_blocks(
    tmp_path: Path, mini_hearing_list: list[HearingRecord]
) -> None:
    method = Method(encoder="stub-encoder", revision="stub-revision-1", embedding_threshold=0.7)
    records = [
        record.model_copy(update={"method": method}) if record.hearing_id == 2 else record
        for record in mini_records(mini_hearing_list)
    ]
    with pytest.raises(ConfigError, match=r"hearing 2: run block .* differs from .* of hearing 1"):
        export_site(
            mini_hearing_list,
            records,
            CachedEncoder(StubEncoder()),
            tmp_path / "site",
            run_name="mini",
            pipeline=PIPELINE,
        )
    assert not (tmp_path / "site" / "index.json").exists()


def build(application: typer.Typer) -> Any:
    return runner.invoke(application, ["build-udvs", "--config", MINI_CONFIG, "--run-name", "mini"])


def site(application: typer.Typer, *options: str) -> Any:
    return runner.invoke(application, ["export-site", "--config", MINI_CONFIG, *options])


def hearing_export(workdir: Path, hearing_id: int, *options: str) -> bytes:
    output = workdir / f"expected_{hearing_id}.json"
    result = runner.invoke(
        stub_app,
        [
            "export-hearing",
            "--config",
            MINI_CONFIG,
            "--run-name",
            "mini",
            "--hearing",
            str(hearing_id),
            "--output",
            str(output),
            *options,
        ],
    )
    assert result.exit_code == 0, result.output
    return output.read_bytes()


def test_cli_export_site_writes_the_export_hearing_files(mini_workdir: Path) -> None:
    assert build(stub_app).exit_code == 0
    (mini_workdir / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    cache_before = files_of(mini_workdir / "cache")
    result = site(
        export_only_app,
        "--run-name",
        "mini",
        "--output",
        "site",
        "--split-manifest",
        "manifest.json",
    )
    assert result.exit_code == 0, result.output
    written = files_of(mini_workdir / "site")
    assert json.loads(result.stdout) == {
        "run": "mini",
        "hearings": 2,
        "udvs": 14,
        "bytes": sum(len(data) for data in written.values()),
        "output": "site",
    }
    assert "[1/2] hearing 1: 6 UDVs, " in result.stderr
    assert "[2/2] hearing 2: 8 UDVs, " in result.stderr
    assert files_of(mini_workdir / "cache") == cache_before
    for hearing_id in (1, 2):
        expected = hearing_export(mini_workdir, hearing_id, "--split-manifest", "manifest.json")
        assert written[f"hearings/{hearing_id}.json"] == expected
    index = json.loads(written["index.json"])
    assert index["run"] == MINI_RUN
    assert [entry["n_udvs"] for entry in index["hearings"]] == [6, 8]


def test_cli_export_site_refuses_to_replace_an_export(mini_workdir: Path) -> None:
    assert build(stub_app).exit_code == 0
    options = ("--run-name", "mini", "--output", "site")
    assert site(stub_app, *options).exit_code == 0
    first = files_of(mini_workdir / "site")
    result = site(stub_app, *options)
    assert result.exit_code == 2
    assert "index.json: site export already exists; pass --overwrite" in result.stderr
    (mini_workdir / "site" / "index.json").unlink()
    result = site(stub_app, *options)
    assert result.exit_code == 2
    assert "hearings: site export already exists" in result.stderr
    result = site(stub_app, *options, "--overwrite")
    assert result.exit_code == 0, result.output
    assert files_of(mini_workdir / "site") == first


def test_cli_export_site_default_output(
    mini_workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build(stub_app).exit_code == 0
    monkeypatch.setattr(bookworm.cli, "default_site_dir", lambda: mini_workdir / "web_data")
    result = site(stub_app, "--run-name", "mini")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["output"] == str(mini_workdir / "web_data")
    assert sorted(files_of(mini_workdir / "web_data")) == [
        "hearings/1.json",
        "hearings/2.json",
        "index.json",
    ]


def test_default_site_dir_is_the_web_app_data_of_the_source_tree() -> None:
    assert default_site_dir() == REPOSITORY_ROOT / "bookworm" / "web" / "app" / "data"


def test_default_site_dir_outside_the_source_tree(tmp_path: Path) -> None:
    package_dir = tmp_path / "lib" / "site-packages" / "bookworm"
    with pytest.raises(ConfigError, match="not running from its source tree"):
        default_site_dir(package_dir)


def test_cli_export_site_fails_on_a_cache_miss(mini_workdir: Path) -> None:
    assert build(stub_app).exit_code == 0
    for path in (mini_workdir / "cache").glob("sentences_2_*.npy"):
        path.unlink()
    result = site(stub_app, "--run-name", "mini", "--output", "site")
    assert result.exit_code == 2
    assert "sentences_2: no cached embeddings for" in result.stderr
    assert sorted(files_of(mini_workdir / "site")) == ["hearings/1.json"]


def test_cli_export_site_interrupted_overwrite_leaves_no_index(mini_workdir: Path) -> None:
    assert build(stub_app).exit_code == 0
    options = ("--run-name", "mini", "--output", "site")
    assert site(stub_app, *options).exit_code == 0
    for path in (mini_workdir / "cache").glob("sentences_2_*.npy"):
        path.unlink()
    result = site(stub_app, *options, "--overwrite")
    assert result.exit_code == 2
    assert "sentences_2: no cached embeddings for" in result.stderr
    assert sorted(files_of(mini_workdir / "site")) == ["hearings/1.json", "hearings/2.json"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--run-name", "never_built"], "run file not found"),
        (["--run-name", "mini", "--split-manifest", "absent.json"], "split manifest not found"),
        (["--run-name", "mini", "--top-k", "0"], "--top-k"),
    ],
)
def test_cli_export_site_input_errors_exit_two(
    mini_workdir: Path, options: list[str], message: str
) -> None:
    assert build(stub_app).exit_code == 0
    result = site(stub_app, *options, "--output", "site")
    assert result.exit_code == 2
    assert message in result.stderr
    assert not (mini_workdir / "site").exists()


def test_cli_export_site_of_a_tfidf_run(mini_workdir: Path) -> None:
    config = mini_workdir / MINI_CONFIG
    text = config.read_text(encoding="utf-8")
    block = 'name = "stub-encoder"\nrevision = "stub-revision-1"\nbatch_size = 8\ndevice = "cpu"'
    assert block in text
    config.write_text(text.replace(block, 'kind = "tfidf"'), encoding="utf-8")
    assert build(app).exit_code == 0
    cache_before = files_of(mini_workdir / "cache")
    result = site(app, "--run-name", "mini", "--output", "site")
    assert result.exit_code == 0, result.output
    assert files_of(mini_workdir / "cache") == cache_before
    index = json.loads((mini_workdir / "site" / "index.json").read_text(encoding="utf-8"))
    assert index["run"]["encoder"] == "tfidf"
    assert [entry["n_udvs"] for entry in index["hearings"]] == [6, 8]


def test_cli_export_site_never_imports_the_model_stack(mini_workdir: Path) -> None:
    assert build(stub_app).exit_code == 0
    command = ["export-site", "--config", MINI_CONFIG, "--run-name", "mini", "--output", "site"]
    script = (
        "import sys\n"
        "from bookworm.cli import app\n"
        f"app({command!r}, standalone_mode=False)\n"
        "print([name for name in ('torch', 'sentence_transformers') if name in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=mini_workdir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[-1] == "[]"
    assert (mini_workdir / "site" / "index.json").is_file()
