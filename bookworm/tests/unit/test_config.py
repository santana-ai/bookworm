import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bookworm import (
    ConfigError,
    SentenceTransformerSettings,
    SplitConfig,
    TfidfSettings,
    UdvConfig,
    load_split_config,
    load_udv_config,
    sha256_of_file,
)


def mini_raw(udv_mini_config_path: Path) -> dict[str, Any]:
    return tomllib.loads(udv_mini_config_path.read_text(encoding="utf-8"))


def test_load_udv_config_reads_every_field(udv_mini_config_path: Path) -> None:
    config = load_udv_config(udv_mini_config_path)
    assert config.lds_path == Path("lds_mini.jsonl")
    assert config.expected_sha256 == (
        "311f0fb9eebcfb9091a1722e6973e0cbb3c6870eb8d8c58023124529e36a854f"
    )
    assert config.encoder == SentenceTransformerSettings(
        name="stub-encoder", revision="stub-revision-1", batch_size=8, device="cpu"
    )
    assert config.embedding_threshold == 0.6
    assert config.seed == 42
    assert config.output_dir == Path("out")
    assert config.cache_dir == Path("cache")


def test_source_keeps_the_raw_toml(udv_mini_config_path: Path) -> None:
    raw = mini_raw(udv_mini_config_path)
    config = UdvConfig.from_mapping(raw)
    assert config.source == raw
    assert "kind" not in config.source["encoder"]
    assert config.source["evidence"]["calibration_note"] == "hand-picked for the synthetic fixture"
    raw["evidence"]["embedding_threshold"] = 0.9
    assert config.source["evidence"]["embedding_threshold"] == 0.6


def test_fixture_sha256_matches_the_fixture(
    udv_mini_config_path: Path, lds_mini_path: Path
) -> None:
    assert load_udv_config(udv_mini_config_path).expected_sha256 == sha256_of_file(lds_mini_path)


def test_tfidf_encoder_kind(udv_mini_config_path: Path) -> None:
    raw = mini_raw(udv_mini_config_path)
    raw["encoder"] = {"kind": "tfidf", "max_features": 5000}
    assert UdvConfig.from_mapping(raw).encoder == TfidfSettings(kind="tfidf", max_features=5000)
    raw["encoder"] = {"kind": "tfidf"}
    assert UdvConfig.from_mapping(raw).encoder == TfidfSettings(kind="tfidf")


def test_config_is_frozen(udv_mini_config_path: Path) -> None:
    config = load_udv_config(udv_mini_config_path)
    with pytest.raises(ValidationError):
        config.__setattr__("seed", 1)


@pytest.mark.parametrize(
    ("table", "key", "message"),
    [
        ("dataset", None, "missing table [dataset]"),
        ("encoder", None, "missing table [encoder]"),
        ("dataset", "sha256", "missing key [dataset].sha256"),
        ("evidence", "embedding_threshold", "missing key [evidence].embedding_threshold"),
        ("run", "cache_dir", "missing key [run].cache_dir"),
    ],
)
def test_missing_entries_raise_config_error(
    udv_mini_config_path: Path, table: str, key: str | None, message: str
) -> None:
    raw = mini_raw(udv_mini_config_path)
    if key is None:
        del raw[table]
    else:
        del raw[table][key]
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        UdvConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("table", "key", "value"),
    [
        ("encoder", "batch_size", "8"),
        ("encoder", "batch_size", 0),
        ("encoder", "kind", "word2vec"),
        ("evidence", "embedding_threshold", "0.6"),
        ("evidence", "embedding_threshold", True),
        ("run", "seed", 4.2),
        ("dataset", "lds_path", 7),
    ],
)
def test_wrong_types_raise_config_error(
    udv_mini_config_path: Path, table: str, key: str, value: object
) -> None:
    raw = copy.deepcopy(mini_raw(udv_mini_config_path))
    raw[table][key] = value
    with pytest.raises(ConfigError):
        UdvConfig.from_mapping(raw, origin="broken.toml")


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_udv_config(tmp_path / "absent.toml")


def test_invalid_toml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[dataset\nlds_path = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_udv_config(path)


def test_toml_that_is_not_utf8_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.toml"
    path.write_bytes('[dataset]\nlds_path = "ação"\n'.encode("latin-1"))
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_split_config(path)


def test_unreadable_config_path_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_udv_config(tmp_path)


def test_integer_thresholds_keep_their_type(
    udv_mini_config_path: Path, splits_mini_config_path: Path
) -> None:
    raw = mini_raw(udv_mini_config_path)
    raw["evidence"]["embedding_threshold"] = 1
    threshold = UdvConfig.from_mapping(raw).embedding_threshold
    assert (threshold, type(threshold)) == (1, int)
    split = split_raw(splits_mini_config_path)
    split["leakage"]["near_duplicate_threshold"] = 1
    near = SplitConfig.from_mapping(split).near_duplicate_threshold
    assert (near, type(near)) == (1, int)


def test_challenge_config_still_loads(challenge_dir: Path) -> None:
    path = challenge_dir / "configs" / "udv.toml"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    config = load_udv_config(path)
    assert isinstance(config.encoder, SentenceTransformerSettings)
    assert config.encoder.name == "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder"
    assert config.encoder.revision == "a01887015444f7599669c509447c5bdbce958916"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert config.embedding_threshold == raw["evidence"]["embedding_threshold"]
    assert 0 < config.embedding_threshold < 1
    assert config.source == raw


def split_raw(splits_mini_config_path: Path) -> dict[str, Any]:
    return tomllib.loads(splits_mini_config_path.read_text(encoding="utf-8"))


def test_load_split_config_reads_every_field(splits_mini_config_path: Path) -> None:
    config = load_split_config(splits_mini_config_path)
    assert config.lds_path == Path("splits_mini.jsonl")
    assert config.split_version == "mini_temporal"
    assert (config.train_fraction, config.validation_fraction) == (0.7, 0.1)
    assert config.min_boundary_gap_days == 2
    assert (config.near_duplicate_threshold, config.near_duplicate_pairs) == (0.5, 3)
    assert config.seed == 7
    assert config.output_dir == Path("splits")
    assert config.udv_path == Path("udv/mini.jsonl")
    assert config.manifest_path == Path("splits/mini_temporal.json")
    assert config.report_path == Path("splits/mini_temporal_report.json")
    assert config.source == split_raw(splits_mini_config_path)


def test_split_fixture_sha256_matches_the_fixture(
    splits_mini_config_path: Path, splits_mini_path: Path
) -> None:
    config = load_split_config(splits_mini_config_path)
    assert config.expected_sha256 == sha256_of_file(splits_mini_path)


def test_split_config_source_is_a_copy(splits_mini_config_path: Path) -> None:
    raw = split_raw(splits_mini_config_path)
    config = SplitConfig.from_mapping(raw)
    raw["temporal"]["train_fraction"] = 0.5
    assert config.source["temporal"]["train_fraction"] == 0.7


@pytest.mark.parametrize(
    ("table", "key", "message"),
    [
        ("temporal", None, "missing table [temporal]"),
        ("leakage", "near_duplicate_pairs", "missing key [leakage].near_duplicate_pairs"),
        ("run", "udv_path", "missing key [run].udv_path"),
    ],
)
def test_split_config_missing_entries(
    splits_mini_config_path: Path, table: str, key: str | None, message: str
) -> None:
    raw = split_raw(splits_mini_config_path)
    if key is None:
        del raw[table]
    else:
        del raw[table][key]
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        SplitConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("table", "key", "value"),
    [
        ("temporal", "train_fraction", 0.0),
        ("temporal", "train_fraction", 1.0),
        ("temporal", "validation_fraction", "0.1"),
        ("temporal", "validation_fraction", 0.3),
        ("temporal", "min_boundary_gap_days", 0),
        ("temporal", "split_version", "../escape"),
        ("temporal", "split_version", ""),
        ("leakage", "near_duplicate_pairs", -1),
        ("leakage", "near_duplicate_threshold", True),
        ("leakage", "near_duplicate_threshold", "0.5"),
        ("run", "seed", "7"),
        ("run", "udv_path", 3),
    ],
)
def test_split_config_rejects_invalid_values(
    splits_mini_config_path: Path, table: str, key: str, value: object
) -> None:
    raw = split_raw(splits_mini_config_path)
    raw[table][key] = value
    with pytest.raises(ConfigError, match=r"broken\.toml"):
        SplitConfig.from_mapping(raw, origin="broken.toml")


def test_split_config_is_frozen(splits_mini_config_path: Path) -> None:
    config = load_split_config(splits_mini_config_path)
    with pytest.raises(ValidationError):
        config.__setattr__("seed", 1)


def test_challenge_split_config_still_loads(challenge_split_config_path: Path) -> None:
    config = load_split_config(challenge_split_config_path)
    assert config.split_version == "temporal_v1"
    assert (config.train_fraction, config.validation_fraction) == (0.7, 0.15)
    assert config.min_boundary_gap_days == 2
    assert config.lds_path == Path("dataset/PublicHearingBR_LDS.jsonl")
    assert config.udv_path == Path("artifacts/udv/udv_v0.jsonl")
    assert config.source == tomllib.loads(challenge_split_config_path.read_text(encoding="utf-8"))
