import shutil
from pathlib import Path

import pytest
from conftest import LDS_SHA256, import_challenge_module

from bookworm.actors.config import load_actors_config
from bookworm.actors.schemas import ActorSpeechRecord, write_actor_speeches
from bookworm.actors.speeches import collect_actor_speeches
from bookworm.data.schemas import HearingRecord
from bookworm.profiles.config import PACKAGED_PROMPTS_DIR, load_split_filter_config
from bookworm.profiles.generate import hearing_metadata, render_prompt
from bookworm.profiles.prompts import load_prompts
from bookworm.profiles.split_filter import build_split_filter, filter_speeches, load_split_selection

pytestmark = pytest.mark.dataset

TRAIN_SPLITS = ["train"]
STATS_ARTIFACT = Path("actor_profiles") / "train_speeches_stats.json"


@pytest.fixture(scope="module")
def multi_hearing_speeches(
    lds_hearings: list[HearingRecord], challenge_dir: Path
) -> list[ActorSpeechRecord]:
    config_path = challenge_dir / "configs" / "hearing_actors.toml"
    if not config_path.is_file():
        pytest.skip(f"actors config not found at {config_path}")
    return collect_actor_speeches(lds_hearings, load_actors_config(config_path)).multi_hearing


@pytest.fixture(scope="module")
def expected_stats_path(artifacts_dir: Path) -> Path:
    path = artifacts_dir / STATS_ARTIFACT
    if not path.is_file():
        pytest.skip(f"split-filter stats not found at {path}")
    return path


def test_split_filter_reproduces_the_challenge_stats_byte_for_byte(
    multi_hearing_speeches: list[ActorSpeechRecord],
    expected_stats_path: Path,
    split_artifacts_dir: Path,
    challenge_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_split_filter_config(challenge_dir / "configs" / "actor_profiles.toml")
    config.manifest_path.parent.mkdir(parents=True)
    shutil.copy(split_artifacts_dir / "temporal_v1.json", config.manifest_path)
    write_actor_speeches(multi_hearing_speeches, config.speeches_path)
    stats = build_split_filter(config)
    assert config.stats_path.read_bytes() == expected_stats_path.read_bytes()
    output = stats["output"]
    assert (output["actors"], output["hearings"], output["turns"]) == (264, 139, 4598)
    assert (stats["input"]["actors"], stats["input"]["turns"]) == (301, 6323)


def test_rendered_prompts_match_the_challenge_script(
    multi_hearing_speeches: list[ActorSpeechRecord],
    lds_hearings: list[HearingRecord],
    challenge_dir: Path,
    split_artifacts_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    reference = import_challenge_module(challenge_dir, "utils.generate_actor_profiles")
    monkeypatch.chdir(challenge_dir)
    config = reference.load_config(Path("configs/actor_profiles.toml"))
    runner = reference.ProfileRunner(
        config=config, prompts=reference.load_prompts(config), client=None
    )
    prompts = load_prompts(PACKAGED_PROMPTS_DIR, "system_profile.md", "user_profile.md.j2")
    assert prompts.version == runner.prompts.version
    metadata = hearing_metadata(lds_hearings)
    reference_metadata = {
        hearing_id: {"date": info.date, "date_br": info.date_br, "assunto": info.assunto}
        for hearing_id, info in metadata.items()
    }
    selection = load_split_selection(
        split_artifacts_dir / "temporal_v1.json", TRAIN_SPLITS, LDS_SHA256
    )
    for record in filter_speeches(multi_hearing_speeches, selection.hearing_ids):
        prompt = render_prompt(record, prompts, metadata)
        expected = runner.profile_prompt(record.to_dict(), reference_metadata)
        assert (prompt.system, prompt.user) == expected, record.actor
