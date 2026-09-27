from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bookworm.errors import ConfigError
from bookworm.profiles.config import (
    PACKAGED_PROMPTS_DIR,
    ModelSettings,
    ProfilesConfig,
    SplitFilterConfig,
    load_profiles_config,
)
from bookworm.profiles.llm import GenerationError, finish_generation, load_transformers_client
from bookworm.profiles.schemas import ProfileRecord, read_profile_lines, read_profiles

SETTINGS = ModelSettings(
    name="toy-model", device_map="cpu", temperature=0.2, top_p=0.9, max_output_tokens=5, seed=7
)
CHALLENGE_STYLE: dict[str, dict[str, Any]] = {
    "input": {"speeches_path": "s.jsonl", "lds_path": "lds.jsonl", "lds_sha256": "a" * 64},
    "output": {"profiles_path": "p.jsonl"},
    "split_filter": {
        "manifest_path": "m.json",
        "splits": ["train"],
        "eval_splits": ["test"],
        "udv_path": "u.jsonl",
        "speeches_path": "f.jsonl",
        "stats_path": "st.json",
    },
    "prompts": {"dir": "prompts/actor_profile"},
    "model": {
        "name": "",
        "device_map": "auto",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 3000,
        "seed": 42,
    },
}


def test_finish_generation_strips_a_leading_think_block() -> None:
    result = finish_generation("<think>plano</think>\n  Perfil final. ", 10, 4, 5)
    assert (result.text, result.input_tokens, result.output_tokens) == ("Perfil final.", 10, 4)


def test_finish_generation_rejects_max_tokens() -> None:
    with pytest.raises(GenerationError, match="max_output_tokens=5"):
        finish_generation("texto cortado", 10, 5, 5)


def test_finish_generation_rejects_unterminated_think() -> None:
    with pytest.raises(GenerationError, match="unterminated think"):
        finish_generation("<think>sem fim", 10, 3, 5)


def test_transformers_client_needs_a_model_name() -> None:
    with pytest.raises(ConfigError, match="--model"):
        load_transformers_client(SETTINGS.model_copy(update={"name": ""}))


def test_challenge_style_config_loads() -> None:
    config = ProfilesConfig.from_mapping(CHALLENGE_STYLE)
    assert config.prompts_dir == Path("prompts/actor_profile")
    assert config.system_profile_file == "system_profile.md"
    assert config.model.seed == 42
    overridden = config.with_overrides(Path("x.jsonl"), Path("y.jsonl"), "org/model")
    assert (overridden.speeches_path, overridden.profiles_path) == (
        Path("x.jsonl"),
        Path("y.jsonl"),
    )
    assert overridden.model.name == "org/model"
    assert overridden.model.max_output_tokens == 3000


def test_split_filter_config_reads_the_split_filter_table() -> None:
    config = SplitFilterConfig.from_mapping(CHALLENGE_STYLE)
    assert (config.speeches_path, config.output_path) == (Path("s.jsonl"), Path("f.jsonl"))
    assert config.splits == ["train"]
    assert (config.eval_splits, config.udv_path) == (["test"], Path("u.jsonl"))
    assert config.with_output(Path("o.jsonl")).output_path == Path("o.jsonl")
    assert config.with_output(None) == config


def test_config_without_prompt_dir_uses_packaged_prompts() -> None:
    raw = {key: value for key, value in CHALLENGE_STYLE.items() if key != "prompts"}
    assert ProfilesConfig.from_mapping(raw).prompts_dir == PACKAGED_PROMPTS_DIR


@pytest.mark.parametrize("splits", [[], ["train", "train"], ["dev"], "train"])
def test_invalid_splits_are_config_errors(splits: Any) -> None:
    raw = {**CHALLENGE_STYLE, "split_filter": {**CHALLENGE_STYLE["split_filter"], "splits": splits}}
    with pytest.raises(ConfigError):
        SplitFilterConfig.from_mapping(raw)


@pytest.mark.parametrize(
    "eval_splits", [[], ["test", "test"], ["dev"], ["train"], ["test", "train"]]
)
def test_invalid_eval_splits_are_config_errors(eval_splits: Any) -> None:
    table = {**CHALLENGE_STYLE["split_filter"], "eval_splits": eval_splits}
    with pytest.raises(ConfigError):
        SplitFilterConfig.from_mapping({**CHALLENGE_STYLE, "split_filter": table})


@pytest.mark.parametrize("key", ["eval_splits", "udv_path"])
def test_split_filter_evaluation_keys_are_required(key: str) -> None:
    table = {name: value for name, value in CHALLENGE_STYLE["split_filter"].items() if name != key}
    with pytest.raises(ConfigError, match=f"split_filter].{key}"):
        SplitFilterConfig.from_mapping({**CHALLENGE_STYLE, "split_filter": table})


def test_missing_table_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text('[input]\nspeeches_path = "s.jsonl"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="lds_path"):
        load_profiles_config(path)


def test_challenge_prompts_are_byte_copies(challenge_dir: Path) -> None:
    source = challenge_dir / "prompts" / "actor_profile"
    for name in ("system_profile.md", "user_profile.md.j2"):
        assert (PACKAGED_PROMPTS_DIR / name).read_bytes() == (source / name).read_bytes()


def profile_line(actor: str) -> str:
    return ProfileRecord(
        actor=actor,
        profile="Texto.",
        model="m",
        prompt_version="abc",
        n_statements=1,
        n_hearings=1,
        hearing_ids=[1],
        input_tokens=1,
        output_tokens=1,
        generated_at="2026-09-26T00:00:00+00:00",
        duration_seconds=0.0,
    ).to_json_line()


def test_read_profiles_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    path.write_text(f"{profile_line('a')}\n\n{profile_line('b')}\n", encoding="utf-8")
    assert [record.actor for record in read_profiles(path)] == ["a", "b"]
    path.write_text(f"{profile_line('a')}\n{profile_line('a')}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicated actors"):
        read_profiles(path)
    path.write_text(f"{profile_line('a')}\nnot json\n", encoding="utf-8")
    assert read_profile_lines(path).errors[0].startswith("line 2:")
    with pytest.raises(ConfigError, match="line 2"):
        read_profiles(path)


class FakeBatch(dict[str, Any]):
    def to(self, device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **options: Any) -> FakeBatch:
        self.messages = messages
        return FakeBatch(input_ids=np.zeros((1, 6), dtype=np.int64))

    def decode(self, tokens: Any, skip_special_tokens: bool) -> str:
        return self.reply


class FakeModel:
    device = "cpu"

    def __init__(self, new_tokens: int) -> None:
        self.new_tokens = new_tokens
        self.options: dict[str, Any] = {}

    def generate(self, **options: Any) -> Any:
        self.options = options
        return np.zeros((1, 6 + self.new_tokens), dtype=np.int64)


def test_transformers_client_uses_the_generation_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers_client = pytest.importorskip("bookworm.profiles.transformers_client")
    seeds: list[int] = []
    monkeypatch.setattr(transformers_client, "set_seed", seeds.append)
    tokenizer, model = FakeTokenizer("<think>x</think>Perfil."), FakeModel(3)
    client = transformers_client.TransformersChatClient(
        SETTINGS, loader=lambda settings: (tokenizer, model)
    )
    result = client.chat("sistema", "usuário")
    assert (result.text, result.input_tokens, result.output_tokens) == ("Perfil.", 6, 3)
    assert [message["role"] for message in tokenizer.messages] == ["system", "user"]
    assert seeds == [7]
    assert model.options["max_new_tokens"] == 5
    assert model.options["do_sample"] is True
    truncated = transformers_client.TransformersChatClient(
        SETTINGS, loader=lambda settings: (tokenizer, FakeModel(5))
    )
    with pytest.raises(GenerationError):
        truncated.chat("sistema", "usuário")
