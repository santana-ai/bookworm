"""Shared pydantic base models."""

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Frozen, strict model read from a TOML file; unknown keys are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class StrictModel(BaseModel):
    """Frozen, strict model of a record or config section; unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
