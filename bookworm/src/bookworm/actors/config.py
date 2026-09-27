import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bookworm.config import read_toml, required
from bookworm.errors import ConfigError


class _ActorsConfigModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class MergeGroup(_ActorsConfigModel):
    canonical: str
    aliases: tuple[str, ...] = Field(min_length=1)


class MergeReassignment(_ActorsConfigModel):
    key: str
    hearing_id: int
    canonical: str


class ActorsConfig(_ActorsConfigModel):
    lds_path: Path
    expected_sha256: str
    chair_names: tuple[str, ...]
    non_person_keys: tuple[str, ...]
    chair_min_words: int = Field(ge=0)
    single_hearing_path: Path
    multi_hearing_path: Path
    ambiguous_names_path: Path
    stats_path: Path
    merge_groups: tuple[MergeGroup, ...] = ()
    merge_reassignments: tuple[MergeReassignment, ...] = ()
    source: dict[str, Any]

    @model_validator(mode="after")
    def check_merges(self) -> Self:
        aliases = [alias for group in self.merge_groups for alias in group.aliases]
        repeated = sorted({alias for alias in aliases if aliases.count(alias) > 1})
        if repeated:
            raise ValueError(f"merge aliases listed in more than one group: {repeated}")
        scopes = [(entry.key, entry.hearing_id) for entry in self.merge_reassignments]
        if len(set(scopes)) != len(scopes):
            raise ValueError("a (key, hearing_id) pair is reassigned more than once")
        return self

    @property
    def alias_map(self) -> dict[str, str]:
        return {alias: group.canonical for group in self.merge_groups for alias in group.aliases}

    @property
    def reassignment_map(self) -> dict[tuple[str, int], str]:
        return {
            (entry.key, entry.hearing_id): entry.canonical for entry in self.merge_reassignments
        }

    @property
    def output_paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.single_hearing_path,
            self.multi_hearing_path,
            self.ambiguous_names_path,
            self.stats_path,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], origin: str = "<mapping>") -> "ActorsConfig":
        source = copy.deepcopy(dict(raw))
        merges = source.get("merges", {})
        if not isinstance(merges, Mapping):
            raise ConfigError(f"{origin}: [merges] is not a table")
        try:
            return cls.model_validate(
                {
                    "lds_path": Path(required(source, "dataset", "lds_path", origin)),
                    "expected_sha256": required(source, "dataset", "sha256", origin),
                    "chair_names": tuple(required(source, "speakers", "chair_names", origin)),
                    "non_person_keys": tuple(
                        required(source, "speakers", "non_person_keys", origin)
                    ),
                    "chair_min_words": required(source, "speakers", "chair_min_words", origin),
                    "single_hearing_path": speeches_path(source, "single_hearing_path", origin),
                    "multi_hearing_path": speeches_path(source, "multi_hearing_path", origin),
                    "ambiguous_names_path": speeches_path(source, "ambiguous_names_path", origin),
                    "stats_path": speeches_path(source, "stats_path", origin),
                    "merge_groups": tuple(
                        {**group, "aliases": tuple(group.get("aliases", ()))}
                        for group in merges.get("groups", [])
                    ),
                    "merge_reassignments": tuple(merges.get("reassignments", [])),
                    "source": source,
                }
            )
        except (ValidationError, TypeError, AttributeError) as error:
            raise ConfigError(f"{origin}: {error}") from error


def speeches_path(source: Mapping[str, Any], key: str, origin: str) -> Path:
    return Path(required(source, "speeches", key, origin))


def load_actors_config(path: Path) -> ActorsConfig:
    return ActorsConfig.from_mapping(read_toml(path), origin=str(path))
