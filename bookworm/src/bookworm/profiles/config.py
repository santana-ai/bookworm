import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import Field, ValidationError, model_validator

from bookworm.config import read_toml, required, section
from bookworm.data.splits import SPLIT_NAMES
from bookworm.errors import ConfigError
from bookworm.models import ConfigModel

PACKAGED_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROFILE = "system_profile.md"
DEFAULT_USER_PROFILE = "user_profile.md.j2"


class ModelSettings(ConfigModel):
    name: str
    device_map: str
    temperature: float | int = Field(ge=0)
    top_p: float | int = Field(gt=0, le=1)
    max_output_tokens: int = Field(gt=0)
    seed: int | None = None


def check_split_names(splits: list[str]) -> None:
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLIT_NAMES):
        raise ValueError(f"splits must be distinct names among {SPLIT_NAMES}, got {splits}")


class SplitFilterConfig(ConfigModel):
    speeches_path: Path
    lds_sha256: str
    manifest_path: Path
    splits: list[str]
    eval_splits: list[str]
    udv_path: Path
    output_path: Path
    stats_path: Path
    source: dict[str, Any]

    @model_validator(mode="after")
    def check_splits(self) -> Self:
        check_split_names(self.splits)
        check_split_names(self.eval_splits)
        if set(self.splits) & set(self.eval_splits):
            raise ValueError(
                f"eval_splits {self.eval_splits} overlap the profile splits {self.splits}"
            )
        return self

    def with_output(self, output_path: Path | None) -> "SplitFilterConfig":
        if output_path is None:
            return self
        return self.model_copy(update={"output_path": output_path})

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], origin: str = "<mapping>") -> "SplitFilterConfig":
        source = copy.deepcopy(dict(raw))
        try:
            return cls.model_validate(
                {
                    "speeches_path": Path(required(source, "input", "speeches_path", origin)),
                    "lds_sha256": required(source, "input", "lds_sha256", origin),
                    "manifest_path": Path(
                        required(source, "split_filter", "manifest_path", origin)
                    ),
                    "splits": required(source, "split_filter", "splits", origin),
                    "eval_splits": required(source, "split_filter", "eval_splits", origin),
                    "udv_path": Path(required(source, "split_filter", "udv_path", origin)),
                    "output_path": Path(required(source, "split_filter", "speeches_path", origin)),
                    "stats_path": Path(required(source, "split_filter", "stats_path", origin)),
                    "source": source,
                }
            )
        except (ValidationError, TypeError) as error:
            raise ConfigError(f"{origin}: {error}") from error


class ProfilesConfig(ConfigModel):
    speeches_path: Path
    lds_path: Path
    lds_sha256: str
    profiles_path: Path
    prompts_dir: Path
    system_profile_file: str
    user_profile_file: str
    model: ModelSettings
    source: dict[str, Any]

    def with_overrides(
        self,
        speeches_path: Path | None = None,
        profiles_path: Path | None = None,
        model_name: str | None = None,
    ) -> "ProfilesConfig":
        update: dict[str, Any] = {}
        if speeches_path is not None:
            update["speeches_path"] = speeches_path
        if profiles_path is not None:
            update["profiles_path"] = profiles_path
        if model_name is not None:
            update["model"] = self.model.model_copy(update={"name": model_name})
        return self.model_copy(update=update)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], origin: str = "<mapping>") -> "ProfilesConfig":
        source = copy.deepcopy(dict(raw))
        prompts = source.get("prompts", {})
        if not isinstance(prompts, Mapping):
            raise ConfigError(f"{origin}: [prompts] must be a table")
        prompts_dir = prompts.get("dir") or None
        try:
            return cls.model_validate(
                {
                    "speeches_path": Path(required(source, "input", "speeches_path", origin)),
                    "lds_path": Path(required(source, "input", "lds_path", origin)),
                    "lds_sha256": required(source, "input", "lds_sha256", origin),
                    "profiles_path": Path(required(source, "output", "profiles_path", origin)),
                    "prompts_dir": (
                        PACKAGED_PROMPTS_DIR if prompts_dir is None else Path(prompts_dir)
                    ),
                    "system_profile_file": prompts.get("system_profile", DEFAULT_SYSTEM_PROFILE),
                    "user_profile_file": prompts.get("user_profile", DEFAULT_USER_PROFILE),
                    "model": dict(section(source, "model", origin)),
                    "source": source,
                }
            )
        except (ValidationError, TypeError) as error:
            raise ConfigError(f"{origin}: {error}") from error


def load_split_filter_config(path: Path) -> SplitFilterConfig:
    return SplitFilterConfig.from_mapping(read_toml(path), origin=str(path))


def load_profiles_config(path: Path) -> ProfilesConfig:
    return ProfilesConfig.from_mapping(read_toml(path), origin=str(path))
