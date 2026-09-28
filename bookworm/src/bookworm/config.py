import copy
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bookworm.errors import ConfigError
from bookworm.udv.quotes import QuoteExtentMode
from bookworm.udv.windows import SemanticUnit

DEFAULT_ENCODER_KIND = "sentence_transformers"
SPLIT_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class SentenceTransformerSettings(_ConfigModel):
    kind: Literal["sentence_transformers"] = "sentence_transformers"
    name: str
    revision: str
    batch_size: int = Field(gt=0)
    device: str


class TfidfSettings(_ConfigModel):
    kind: Literal["tfidf"]
    max_features: int | None = Field(default=None, gt=0)


EncoderSettings = Annotated[
    SentenceTransformerSettings | TfidfSettings, Field(discriminator="kind")
]


class UdvConfig(_ConfigModel):
    lds_path: Path
    expected_sha256: str
    encoder: EncoderSettings
    embedding_threshold: float | int
    semantic_unit: SemanticUnit = "sentence"
    quote_extent: QuoteExtentMode = "prefix_sentence"
    seed: int
    output_dir: Path
    cache_dir: Path
    source: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], origin: str = "<mapping>") -> "UdvConfig":
        source = copy.deepcopy(dict(raw))
        encoder = dict(section(source, "encoder", origin))
        encoder.setdefault("kind", DEFAULT_ENCODER_KIND)
        try:
            return cls.model_validate(
                {
                    "lds_path": Path(required(source, "dataset", "lds_path", origin)),
                    "expected_sha256": required(source, "dataset", "sha256", origin),
                    "encoder": encoder,
                    "embedding_threshold": required(
                        source, "evidence", "embedding_threshold", origin
                    ),
                    "semantic_unit": section(source, "evidence", origin).get(
                        "semantic_unit", "sentence"
                    ),
                    "quote_extent": section(source, "evidence", origin).get(
                        "quote_extent", "prefix_sentence"
                    ),
                    "seed": required(source, "run", "seed", origin),
                    "output_dir": Path(required(source, "run", "output_dir", origin)),
                    "cache_dir": Path(required(source, "run", "cache_dir", origin)),
                    "source": source,
                }
            )
        except (ValidationError, TypeError) as error:
            raise ConfigError(f"{origin}: {error}") from error


class SplitConfig(_ConfigModel):
    lds_path: Path
    expected_sha256: str
    split_version: str = Field(pattern=SPLIT_VERSION_PATTERN)
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    min_boundary_gap_days: int = Field(ge=1)
    near_duplicate_threshold: float | int
    near_duplicate_pairs: int = Field(ge=0)
    seed: int
    output_dir: Path
    udv_path: Path
    source: dict[str, Any]

    @model_validator(mode="after")
    def check_fractions(self) -> Self:
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must leave room for test")
        return self

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / f"{self.split_version}.json"

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{self.split_version}_report.json"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], origin: str = "<mapping>") -> "SplitConfig":
        source = copy.deepcopy(dict(raw))
        try:
            return cls.model_validate(
                {
                    "lds_path": Path(required(source, "dataset", "lds_path", origin)),
                    "expected_sha256": required(source, "dataset", "sha256", origin),
                    "split_version": required(source, "temporal", "split_version", origin),
                    "train_fraction": required(source, "temporal", "train_fraction", origin),
                    "validation_fraction": required(
                        source, "temporal", "validation_fraction", origin
                    ),
                    "min_boundary_gap_days": required(
                        source, "temporal", "min_boundary_gap_days", origin
                    ),
                    "near_duplicate_threshold": required(
                        source, "leakage", "near_duplicate_threshold", origin
                    ),
                    "near_duplicate_pairs": required(
                        source, "leakage", "near_duplicate_pairs", origin
                    ),
                    "seed": required(source, "run", "seed", origin),
                    "output_dir": Path(required(source, "run", "output_dir", origin)),
                    "udv_path": Path(required(source, "run", "udv_path", origin)),
                    "source": source,
                }
            )
        except (ValidationError, TypeError) as error:
            raise ConfigError(f"{origin}: {error}") from error


def section(raw: Mapping[str, Any], name: str, origin: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{origin}: missing table [{name}]")
    return value


def required(raw: Mapping[str, Any], table: str, key: str, origin: str) -> Any:
    values = section(raw, table, origin)
    if key not in values:
        raise ConfigError(f"{origin}: missing key [{table}].{key}")
    return values[key]


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: config file not found") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot read config file: {error}") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error


def load_udv_config(path: Path) -> UdvConfig:
    return UdvConfig.from_mapping(read_toml(path), origin=str(path))


def load_split_config(path: Path) -> SplitConfig:
    return SplitConfig.from_mapping(read_toml(path), origin=str(path))
