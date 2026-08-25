"""Validate configuration for the independently executable scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from configs.configs_val.DB_configs import BASE_DIR

SCRIPT_CONFIG_PATH = BASE_DIR / "configs" / "scripts.yaml"


def _validate_ym(value: str) -> str:
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f"Invalid YYYYMM value: {value!r}")
    try:
        datetime.strptime(value, "%Y%m")
    except ValueError as error:
        raise ValueError(f"Invalid YYYYMM value: {value!r}") from error
    return value


YearMonth = Annotated[str, AfterValidator(_validate_ym)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RefreshFeaturesConfig(_StrictModel):
    """Feature refresh inputs."""

    ym: YearMonth
    groups: tuple[str, ...] | None
    dry_run: bool


class RefreshPopulationConfig(_StrictModel):
    """Full-population refresh inputs."""

    as_of_ym: YearMonth
    product: str = Field(min_length=1)


class RefreshTrainingPopulationConfig(_StrictModel):
    """Append-only training-population snapshot inputs."""

    ym: YearMonth
    product: tuple[str, ...] | None


class RunPredictionConfig(_StrictModel):
    """Prediction pipeline inputs."""

    product: str = Field(min_length=1)
    population: str = Field(min_length=1)
    ym: YearMonth
    edition: str = Field(min_length=1)
    write_db: bool


class RunPretrainConfig(_StrictModel):
    """Pretraining pipeline inputs."""

    as_of_ym: YearMonth
    product: str = Field(min_length=1)
    population: str = Field(min_length=1)
    rows_per_month: int | None = Field(ge=1)
    output_dir: Path


class RunRetrainConfig(_StrictModel):
    """Formal retraining pipeline inputs."""

    product: str | tuple[str, ...] | None
    population: str | tuple[str, ...] | None
    train_yms: tuple[YearMonth, ...] | None
    backtest_ym: YearMonth | None
    as_of_ym: YearMonth | None
    edition: str = Field(min_length=1)
    rows_per_month: int | None = Field(default=None, ge=1)
    write_db: bool

    @field_validator("product", "population")
    @classmethod
    def validate_selection(
        cls,
        value: str | tuple[str, ...] | None,
    ) -> str | tuple[str, ...] | None:
        """Reject empty or duplicate explicit batch selections."""
        if value is None:
            return value
        names = (value,) if isinstance(value, str) else value
        if not names or any(not name.strip() for name in names):
            raise ValueError("selection must be null, a non-empty name, or a non-empty name list")
        if len(set(names)) != len(names):
            raise ValueError("selection contains duplicate names")
        return value

    @field_validator("train_yms")
    @classmethod
    def validate_train_yms(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Reject an empty explicit training-month sequence."""
        if value is not None and not value:
            raise ValueError("train_yms must be null or contain at least one YYYYMM value")
        return value


class ValidatePretrainReadinessConfig(_StrictModel):
    """Pretraining readiness inspection inputs."""

    product: str = Field(min_length=1)
    population: str = Field(min_length=1)
    ym: YearMonth
    sample_size: int = Field(ge=1)
    output_dir: Path
    sample_rows: int = Field(ge=1)


class ScriptSettings(_StrictModel):
    """Validated inputs for every executable script."""

    refresh_features: RefreshFeaturesConfig
    refresh_population: RefreshPopulationConfig
    refresh_training_population: RefreshTrainingPopulationConfig
    run_prediction: RunPredictionConfig
    run_pretrain: RunPretrainConfig
    run_retrain: RunRetrainConfig
    validate_pretrain_readiness: ValidatePretrainReadinessConfig


def load_script_settings(path: Path = SCRIPT_CONFIG_PATH) -> ScriptSettings:
    """Read and validate all script settings from YAML."""
    with path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Script configuration must be a YAML mapping: {path}")
    return ScriptSettings.model_validate(values)


def resolve_project_path(path: Path) -> Path:
    """Resolve a configured relative path from the project root."""
    return path if path.is_absolute() else BASE_DIR / path
