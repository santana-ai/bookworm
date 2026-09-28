"""Configuration of the actor speeches split filter, of profile generation and validation."""

import copy
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

from pydantic import Field, ValidationError, model_validator

from bookworm.config import (
    DEFAULT_ENCODER_KIND,
    SPLIT_VERSION_PATTERN,
    EncoderSettings,
    read_toml,
    required,
    section,
)
from bookworm.data.splits import SPLIT_NAMES, SplitName
from bookworm.errors import ConfigError
from bookworm.models import ConfigModel, StrictModel
from bookworm.profiles.schemas import Group
from bookworm.udv.schemas import Tier

PACKAGED_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROFILE = "system_profile.md"
DEFAULT_USER_PROFILE = "user_profile.md.j2"
DEFAULT_TIERS: tuple[Tier, ...] = ("quote_found", "semantic_match_high")
DEFAULT_GENERATION_SPLITS: tuple[SplitName, ...] = ("train",)
DEFAULT_HELD_OUT_SPLITS: tuple[SplitName, ...] = ("test",)
DEFAULT_CONFIDENCE_LEVEL = 0.95


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


class ReviewSettings(StrictModel):
    seed: int
    sizes: dict[Group, int]
    score_bands: list[float]
    output: Path

    @model_validator(mode="after")
    def check_review(self) -> Self:
        if any(size < 0 for size in self.sizes.values()):
            raise ValueError("review sizes must be >= 0")
        if any(high <= low for low, high in pairwise(self.score_bands)):
            raise ValueError("score_bands must be strictly increasing")
        return self


class ProfileValidationConfig(StrictModel):
    udv_path: Path
    links_path: Path
    profiles_path: Path
    split_manifest: Path
    tiers: list[Tier] = Field(min_length=1)
    generation_splits: list[SplitName] = Field(min_length=1)
    held_out_splits: list[SplitName] = Field(min_length=1)
    encoder: EncoderSettings
    cache_dir: Path
    output_dir: Path
    name: str = Field(pattern=SPLIT_VERSION_PATTERN)
    seed: int
    bootstrap_samples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    review: ReviewSettings
    source: dict[str, Any]

    @model_validator(mode="after")
    def check_splits(self) -> Self:
        for name, values in (
            ("tiers", self.tiers),
            ("generation_splits", self.generation_splits),
            ("held_out_splits", self.held_out_splits),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} has repeated values: {values}")
        shared = sorted(set(self.generation_splits) & set(self.held_out_splits))
        if shared:
            raise ValueError(f"splits {shared} are both generation and held-out splits")
        return self

    @property
    def pairs_path(self) -> Path:
        return self.output_dir / f"{self.name}_pairs.jsonl"

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{self.name}_report.json"

    @property
    def review_sample_path(self) -> Path:
        output = self.review.output
        return output.with_name(f"{output.stem}_sample.json")

    @property
    def review_report_path(self) -> Path:
        output = self.review.output
        return output.with_name(f"{output.stem}_report.json")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], origin: str = "<mapping>"
    ) -> "ProfileValidationConfig":
        source = copy.deepcopy(dict(raw))
        validation = section(source, "validation", origin)
        encoder = dict(section(validation, "encoder", f"{origin} [validation]"))
        encoder.setdefault("kind", DEFAULT_ENCODER_KIND)
        try:
            review = section(source, "review", origin)
            return cls.model_validate(
                {
                    "udv_path": Path(required(source, "inputs", "udv_path", origin)),
                    "links_path": Path(required(source, "inputs", "links_path", origin)),
                    "profiles_path": Path(required(source, "inputs", "profiles_path", origin)),
                    "split_manifest": Path(required(source, "inputs", "split_manifest", origin)),
                    "tiers": list(validation.get("tiers", DEFAULT_TIERS)),
                    "generation_splits": list(
                        validation.get("generation_splits", DEFAULT_GENERATION_SPLITS)
                    ),
                    "held_out_splits": list(
                        validation.get("held_out_splits", DEFAULT_HELD_OUT_SPLITS)
                    ),
                    "encoder": encoder,
                    "cache_dir": Path(required(source, "validation", "cache_dir", origin)),
                    "output_dir": Path(required(source, "validation", "output_dir", origin)),
                    "name": required(source, "validation", "name", origin),
                    "seed": required(source, "validation", "seed", origin),
                    "bootstrap_samples": required(
                        source, "validation", "bootstrap_samples", origin
                    ),
                    "confidence_level": float(
                        validation.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)
                    ),
                    "review": {
                        "seed": required(source, "review", "seed", origin),
                        "sizes": dict(required(source, "review", "sizes", origin)),
                        "score_bands": [float(edge) for edge in review.get("score_bands", [])],
                        "output": Path(required(source, "review", "output", origin)),
                    },
                    "source": source,
                }
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise ConfigError(f"{origin}: {error}") from error


def load_profile_validation_config(path: Path) -> ProfileValidationConfig:
    return ProfileValidationConfig.from_mapping(read_toml(path), origin=str(path))
