import csv
import json
from pathlib import Path

import pytest
from profile_validation_data import CONFIG_NAME, write_workdir
from typer.testing import CliRunner

from bookworm.cli import app
from bookworm.errors import ConfigError
from bookworm.profiles.review import (
    CSV_COLUMNS,
    band_labels,
    normalize_label,
    sample_profile_review,
    score_band,
    score_profile_review,
    wilson_interval,
)
from bookworm.profiles.validate import (
    ProfileValidationConfig,
    load_profile_validation_config,
    validate_profiles,
)

runner = CliRunner()
FILLED_JUDGMENTS = {
    "u01": "sustentada",
    "u03": "Compatível",
    "u04": "sustentada",
    "u05": "sem relação",
}


def prepared_config(directory: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileValidationConfig:
    write_workdir(directory)
    monkeypatch.chdir(directory)
    config = load_profile_validation_config(directory / CONFIG_NAME)
    validate_profiles(config)
    return config


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileValidationConfig:
    return prepared_config(tmp_path, monkeypatch)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def filled_csv(config: ProfileValidationConfig, judgments: dict[str, str]) -> Path:
    rows = read_rows(config.review.output)
    for row in rows:
        row["julgamento"] = judgments.get(row["udv_id"], "")
    path = config.review.output.with_name("filled.csv")
    write_rows(rows, path)
    return path


def test_band_labels_and_score_band() -> None:
    assert band_labels([]) == ["all"]
    assert band_labels([0.2, 0.5]) == ["[-inf, 0.20)", "[0.20, 0.50)", "[0.50, inf)"]
    assert score_band(0.1, [0.2, 0.5]) == "[-inf, 0.20)"
    assert score_band(0.2, [0.2, 0.5]) == "[0.20, 0.50)"
    assert score_band(0.9, [0.2, 0.5]) == "[0.50, inf)"
    assert score_band(-0.3, []) == "all"


def test_normalize_label() -> None:
    assert normalize_label("  Compatível ") == "compativel"
    assert normalize_label("Sem relação") == "sem_relacao"
    assert normalize_label("sem-relacao") == "sem_relacao"


def test_wilson_interval_matches_known_values() -> None:
    assert wilson_interval(1, 2, 0.95) == {
        "successes": 1,
        "trials": 2,
        "estimate": 0.5,
        "low": 0.0945,
        "high": 0.9055,
    }
    assert wilson_interval(2, 2, 0.95)["low"] == 0.3424
    assert wilson_interval(0, 0, 0.95)["estimate"] is None


def test_sample_writes_a_skeleton_with_empty_judgments(config: ProfileValidationConfig) -> None:
    summary = sample_profile_review(config)
    rows = read_rows(config.review.output)
    assert summary["items"] == len(rows) == 4
    assert all(row["julgamento"] == "" and row["observacao"] == "" for row in rows)
    assert [row["item_id"] for row in rows] == ["P001", "P002", "P003", "P004"]
    assert sorted(row["udv_id"] for row in rows) == ["u01", "u03", "u04", "u05"]
    assert all(row["profiles_file"] == "profiles.jsonl" for row in rows)
    shortfall = {(s["group"], s["band"]): s["shortfall"] for s in summary["strata"]}
    assert shortfall[("in_prompt", "[-inf, 0.20)")] == 2
    sample = json.loads(config.review_sample_path.read_text(encoding="utf-8"))
    assert {item: key["udv_id"] for item, key in sample["items"].items()} == {
        row["item_id"]: row["udv_id"] for row in rows
    }


def test_sample_sheet_is_blind(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    rows = read_rows(config.review.output)
    hidden = {"group", "split", "hearing_id", "score", "score_band"}
    assert not hidden & set(rows[0])
    assert list(rows[0]) == list(CSV_COLUMNS)
    sample = json.loads(config.review_sample_path.read_text(encoding="utf-8"))
    keys = [sample["items"][row["item_id"]] for row in rows]
    assert all(hidden <= set(key) for key in keys)
    assert [key["group"] for key in keys] != sorted(key["group"] for key in keys)
    by_udv = {key["udv_id"]: key for key in keys}
    assert by_udv["u05"]["score_band"] == "[-inf, 0.20)"
    assert by_udv["u04"]["group"] == "held_out"


def test_sample_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: ProfileValidationConfig
) -> None:
    sample_profile_review(config)
    first = config.review.output.read_bytes()
    other = prepared_config(tmp_path / "again", monkeypatch)
    sample_profile_review(other)
    assert (tmp_path / "again" / other.review.output).read_bytes() == first


def test_sample_refuses_to_overwrite(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    with pytest.raises(ConfigError, match="already exists"):
        sample_profile_review(config)


def test_score_report_from_a_filled_sheet(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    report = score_profile_review(config, filled_csv(config, FILLED_JUDGMENTS))
    in_prompt = report["groups"]["in_prompt"]
    assert in_prompt["sample"]["items"] == 2
    assert in_prompt["sample"]["outcomes"]["strict_support"] == wilson_interval(1, 2, 0.95)
    assert in_prompt["sample"]["outcomes"]["tolerant_support"] == wilson_interval(2, 2, 0.95)
    assert in_prompt["sample"]["labels"]["compativel"]["successes"] == 1
    assert in_prompt["population_weighted"]["strict_support"] == {
        "estimate": 0.5,
        "low": 0.0,
        "high": 1.0,
        "uncovered_bands": [],
    }
    held_out = report["groups"]["held_out"]
    assert held_out["sample"]["labels"]["sem_relacao"]["successes"] == 1
    assert held_out["population_weighted"]["tolerant_support"] == {
        "estimate": 0.5,
        "low": 0.5,
        "high": 0.5,
        "uncovered_bands": [],
    }
    assert config.review_report_path.is_file()


def test_score_refuses_invalid_labels(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    path = filled_csv(config, {**FILLED_JUDGMENTS, "u04": "talvez"})
    with pytest.raises(ConfigError, match="invalid judgments"):
        score_profile_review(config, path)


def test_score_refuses_missing_judgments(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    path = filled_csv(config, {"u01": "sustentada"})
    with pytest.raises(ConfigError, match="3 items have no judgment"):
        score_profile_review(config, path)


def test_score_refuses_edited_columns(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    rows = read_rows(config.review.output)
    for row in rows:
        row["julgamento"] = FILLED_JUDGMENTS[row["udv_id"]]
    rows[0]["actor"] = "Outra Pessoa"
    path = config.review.output.with_name("edited.csv")
    write_rows(rows, path)
    with pytest.raises(ConfigError, match=r"edited columns \['actor'\]"):
        score_profile_review(config, path)


def test_score_refuses_an_edited_key_file(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    path = filled_csv(config, FILLED_JUDGMENTS)
    sample = json.loads(config.review_sample_path.read_text(encoding="utf-8"))
    item = next(iter(sample["items"]))
    group = sample["items"][item]["group"]
    sample["items"][item]["group"] = "held_out" if group == "in_prompt" else "in_prompt"
    config.review_sample_path.write_text(json.dumps(sample), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not match the pairs file"):
        score_profile_review(config, path)


def test_score_refuses_dropped_rows(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    rows = read_rows(config.review.output)[1:]
    path = config.review.output.with_name("short.csv")
    write_rows(rows, path)
    with pytest.raises(ConfigError, match="items differ from the sample"):
        score_profile_review(config, path)


def test_score_refuses_pairs_changed_after_sampling(config: ProfileValidationConfig) -> None:
    sample_profile_review(config)
    path = filled_csv(config, FILLED_JUDGMENTS)
    validate_profiles(config.model_copy(update={"seed": 8}), overwrite=True)
    with config.pairs_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ConfigError, match="changed after the review sample"):
        score_profile_review(config, path)


def test_cli_sample_and_score(config: ProfileValidationConfig) -> None:
    sampled = runner.invoke(app, ["sample-profile-review", "--config", CONFIG_NAME])
    assert sampled.exit_code == 0, sampled.output
    assert json.loads(sampled.stdout)["items"] == 4
    path = filled_csv(config, FILLED_JUDGMENTS)
    scored = runner.invoke(
        app, ["score-profile-review", "--config", CONFIG_NAME, "--annotations", str(path)]
    )
    assert scored.exit_code == 0, scored.output
    groups = json.loads(scored.stdout)["groups"]
    assert groups["held_out"]["strict_support"]["successes"] == 1
    invalid = filled_csv(config, {**FILLED_JUDGMENTS, "u01": "ok"})
    refused = runner.invoke(
        app, ["score-profile-review", "--config", CONFIG_NAME, "--annotations", str(invalid)]
    )
    assert refused.exit_code == 2
    assert "invalid judgments" in refused.stderr
