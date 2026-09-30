import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import StubEncoder
from profile_validation_data import (
    ANA_PROFILE,
    CONFIG_NAME,
    FIXTURES_DIR,
    SPLIT_VERSION,
    default_profiles,
    profile_record,
    udv_record,
    write_workdir,
)
from typer.testing import CliRunner

from bookworm.cli import app
from bookworm.config import EncoderSettings, read_toml
from bookworm.errors import ConfigError
from bookworm.features.encoders import SentenceEncoder
from bookworm.profiles.validate import (
    SKIP_REASONS,
    ProfileValidationConfig,
    SplitManifest,
    chance_mrr,
    check_held_out_pair,
    check_profile,
    identification_rank,
    load_profile_validation_config,
    profile_sentences,
    profiles_by_actor,
    read_pairs,
    score_profiles,
    validate_profiles,
)

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_workdir(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def config(workdir: Path) -> ProfileValidationConfig:
    return load_profile_validation_config(workdir / CONFIG_NAME)


def fixture_source() -> dict[str, object]:
    return read_toml(FIXTURES_DIR / CONFIG_NAME)


def test_fixture_config_loads() -> None:
    config = load_profile_validation_config(FIXTURES_DIR / CONFIG_NAME)
    assert config.tiers == ["quote_found", "semantic_match_high"]
    assert config.generation_splits == ["train"]
    assert config.held_out_splits == ["test"]
    assert config.encoder.kind == "tfidf"
    assert config.pairs_path == Path("out/profile_validation_mini_pairs.jsonl")
    assert config.report_path == Path("out/profile_validation_mini_report.json")
    assert config.review_sample_path == Path("review/profile_review_sample.json")
    assert config.review.sizes == {"in_prompt": 2, "held_out": 1}


def test_config_defaults_tiers_and_splits() -> None:
    raw = fixture_source()
    validation = dict(raw["validation"])  # type: ignore[call-overload]
    for key in ("tiers", "generation_splits", "held_out_splits"):
        del validation[key]
    config = ProfileValidationConfig.from_mapping({**raw, "validation": validation})
    assert config.tiers == ["quote_found", "semantic_match_high"]
    assert config.generation_splits == ["train"]
    assert config.held_out_splits == ["test"]


@pytest.mark.parametrize(
    ("table", "key", "value", "message"),
    [
        ("validation", "held_out_splits", ["train"], "both generation and held-out"),
        ("validation", "tiers", ["quote_found", "quote_found"], "repeated"),
        ("validation", "tiers", ["exact"], "tiers"),
        ("validation", "bootstrap_samples", 0, "bootstrap_samples"),
        ("review", "score_bands", [0.5, 0.2], "strictly increasing"),
        ("review", "sizes", {"in_prompt": -1}, ">= 0"),
        ("review", "sizes", {"everything": 1}, "sizes"),
    ],
)
def test_config_rejects_invalid_values(table: str, key: str, value: object, message: str) -> None:
    raw = fixture_source()
    changed = {**raw[table], key: value}  # type: ignore[dict-item]
    with pytest.raises(ConfigError, match=message):
        ProfileValidationConfig.from_mapping({**raw, table: changed})


def test_config_requires_inputs() -> None:
    raw = fixture_source()
    inputs = dict(raw["inputs"])  # type: ignore[call-overload]
    del inputs["links_path"]
    with pytest.raises(ConfigError, match=r"\[inputs\].links_path"):
        ProfileValidationConfig.from_mapping({**raw, "inputs": inputs})


def test_profile_sentences_split_lines_and_sentences() -> None:
    assert profile_sentences(ANA_PROFILE) == [
        "Ana Souza defende a universalização do saneamento básico nas cidades pequenas.",
        "Ela cobra da companhia estadual investimentos em tratamento de esgoto.",
        "Ana critica o atraso nas obras de drenagem urbana.",
    ]
    assert profile_sentences("Linha sem ponto final aqui\nOutra linha também sem ponto") == [
        "Linha sem ponto final aqui",
        "Outra linha também sem ponto",
    ]


def test_identification_rank_counts_ties_against_the_true_profile() -> None:
    assert identification_rank(np.array([0.9, 0.2, 0.1]), 0) == 1
    assert identification_rank(np.array([0.3, 0.8, 0.1]), 0) == 2
    assert identification_rank(np.array([0.5, 0.5, 0.1]), 0) == 2
    assert identification_rank(np.array([0.0, 0.0, 0.0]), 1) == 3


def test_chance_mrr_is_the_mean_reciprocal_rank_of_a_uniform_draw() -> None:
    assert chance_mrr(1) == 1.0
    assert chance_mrr(3) == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def test_score_profiles_takes_the_best_sentence_of_each_profile() -> None:
    propositions = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    sentences = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    scores = score_profiles(propositions, sentences, [2, 1])
    np.testing.assert_allclose(scores.scores, [[1.0, 0.0], [np.sqrt(0.5), 1.0], [0.0, 0.0]])
    assert scores.best_sentence.tolist() == [[1, 0], [0, 0], [0, 0]]


def manifest() -> SplitManifest:
    return SplitManifest(
        SPLIT_VERSION, {1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}
    )


def test_check_profile_refuses_hearings_outside_the_generation_splits() -> None:
    check_profile(profile_record("ana", ANA_PROFILE, [1, 2]), manifest(), ["train"])
    with pytest.raises(ConfigError, match="hearing 5 of split 'test' entered the prompt"):
        check_profile(profile_record("ana", ANA_PROFILE, [1, 5]), manifest(), ["train"])
    with pytest.raises(ConfigError, match="hearing 9 is in no split"):
        check_profile(profile_record("ana", ANA_PROFILE, [9]), manifest(), ["train"])


def test_duplicated_actor_profiles_are_refused() -> None:
    profiles = [profile_record("ana", ANA_PROFILE, [1]), profile_record("ana", ANA_PROFILE, [2])]
    with pytest.raises(ConfigError, match=r"more than one profile: \['Ana'\]"):
        profiles_by_actor(profiles)


def test_check_held_out_pair_refuses_a_hearing_seen_by_the_prompt() -> None:
    udv = udv_record("u99", 5, "quote_found", "Opinião sintética de teste.")
    check_held_out_pair(udv, profile_record("ana", ANA_PROFILE, [1, 2]))
    with pytest.raises(ConfigError, match="held out but entered the prompt"):
        check_held_out_pair(udv, profile_record("ana", ANA_PROFILE, [1, 5]))


def test_validate_profiles_refuses_a_leaking_profile(tmp_path: Path) -> None:
    profiles = [*default_profiles()[:2], profile_record("carla", ANA_PROFILE, [3, 6])]
    write_workdir(tmp_path, profiles=profiles)
    config = ProfileValidationConfig.from_mapping(
        rebase(read_toml(tmp_path / CONFIG_NAME), tmp_path)
    )
    with pytest.raises(ConfigError, match="profile 'Carla': hearing 6"):
        validate_profiles(config)
    assert not config.pairs_path.exists()


def rebase(raw: dict[str, object], base: Path) -> dict[str, object]:
    inputs = {key: str(base / value) for key, value in raw["inputs"].items()}  # type: ignore[attr-defined]
    validation = {
        **raw["validation"],  # type: ignore[dict-item]
        "output_dir": str(base / "out"),
        "cache_dir": str(base / "cache"),
    }
    return {**raw, "inputs": inputs, "validation": validation}


def test_validate_profiles_counts_skips_and_groups(config: ProfileValidationConfig) -> None:
    report = validate_profiles(config)
    assert report["counts"]["udvs"] == 12
    assert report["counts"]["skipped"] == dict.fromkeys(SKIP_REASONS, 1)
    assert report["counts"]["pairs"] == {"in_prompt": 3, "held_out": 2}
    pairs = read_pairs(config.pairs_path)
    assert [(pair.udv_id, pair.group, pair.rank) for pair in pairs] == [
        ("u01", "in_prompt", 1),
        ("u02", "in_prompt", 1),
        ("u03", "in_prompt", 1),
        ("u04", "held_out", 1),
        ("u05", "held_out", 3),
    ]
    assert pairs[3].profile_sentence == (
        "Ela cobra da companhia estadual investimentos em tratamento de esgoto."
    )
    assert pairs[4].best_other_actor == "Ana"
    assert pairs[4].score == 0.0


def test_validate_profiles_reports_metrics_per_group(config: ProfileValidationConfig) -> None:
    validate_profiles(config)
    report = json.loads(config.report_path.read_text(encoding="utf-8"))
    held_out = report["groups"]["held_out"]
    assert held_out["identification"] == {"acc_at_1": 0.5, "mrr": 0.6667}
    assert held_out["chance"] == {"acc_at_1": 0.3333, "mrr": 0.6111}
    assert held_out["splits"] == ["test"]
    assert report["groups"]["in_prompt"]["identification"] == {"acc_at_1": 1.0, "mrr": 1.0}
    bootstrap = held_out["bootstrap"]
    assert bootstrap["unit"] == "hearing"
    assert bootstrap["samples"] == 200
    assert bootstrap["acc_at_1"][0] <= 0.5 <= bootstrap["acc_at_1"][1]
    assert set(report["inputs"]) == {"udvs", "links", "profiles", "split_manifest"}
    assert all(len(entry["sha256"]) == 64 for entry in report["inputs"].values())
    assert report["config"]["validation"]["name"] == "profile_validation_mini"
    assert report["encoder"]["name"] == "tfidf"


def test_validate_profiles_is_deterministic(config: ProfileValidationConfig) -> None:
    first = validate_profiles(config)
    pairs = config.pairs_path.read_bytes()
    second = validate_profiles(config, overwrite=True)
    assert config.pairs_path.read_bytes() == pairs
    assert first["groups"] == second["groups"]


def test_validate_profiles_refuses_to_overwrite(config: ProfileValidationConfig) -> None:
    validate_profiles(config)
    with pytest.raises(ConfigError, match="already exists"):
        validate_profiles(config)


def test_validate_profiles_uses_the_encoder_factory(config: ProfileValidationConfig) -> None:
    corpora: list[list[str]] = []

    def factory(settings: EncoderSettings, corpus: Sequence[str], seed: int) -> SentenceEncoder:
        corpora.append(list(corpus))
        return StubEncoder({}, default=(1.0, 0.0))

    report = validate_profiles(config, encoder_factory=factory)
    assert len(corpora[0]) == report["counts"]["profile_sentences"] == 7
    assert "Cobrou investimentos em tratamento de esgoto." not in corpora[0]
    assert report["groups"]["in_prompt"]["identification"]["acc_at_1"] == 0.0
    assert report["encoder"]["name"] == "stub-encoder"


def test_validate_profiles_refuses_a_profile_without_sentences(tmp_path: Path) -> None:
    profiles = [*default_profiles()[:2], profile_record("carla", "Curto demais.", [3])]
    write_workdir(tmp_path, profiles=profiles)
    config = ProfileValidationConfig.from_mapping(
        rebase(read_toml(tmp_path / CONFIG_NAME), tmp_path)
    )
    with pytest.raises(ConfigError, match=r"without any sentence to compare: \['Carla'\]"):
        validate_profiles(config)


def test_cli_validate_profiles(workdir: Path) -> None:
    result = runner.invoke(app, ["validate-profiles", "--config", CONFIG_NAME])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["groups"]["held_out"]["pairs"] == 2
    again = runner.invoke(app, ["validate-profiles", "--config", CONFIG_NAME])
    assert again.exit_code == 2
    assert "already exists" in again.stderr


def test_cli_validate_profiles_reports_missing_inputs(workdir: Path) -> None:
    (workdir / "profiles.jsonl").unlink()
    result = runner.invoke(app, ["validate-profiles", "--config", CONFIG_NAME])
    assert result.exit_code == 2
    assert "profiles file not found" in result.stderr
