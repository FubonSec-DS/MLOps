"""Read application settings from YAML and environment variables."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
ENV_PATH = BASE_DIR / "configs" / ".env"


class DatabaseCredentials(BaseModel):
    """Oracle login information."""

    user: str
    password: str
    dsn: str


class OracleSettings(BaseModel):
    """The shared Oracle connection and schema names."""

    query: DatabaseCredentials
    instant_client_path: Path
    feature_schema: str
    population_sampling_schema: str
    population_sampling_table: str
    population_source_schema: str
    model_log_schema: str


class PathSettings(BaseModel):
    """Local data and model paths."""

    fin_table: Path
    model_base: Path


class ModelSettings(BaseModel):
    """Shared model settings."""

    device: str
    random_state: int
    pretrain_size: int
    top_n_features: int


class FeatureSettings(BaseModel):
    """Feature tables and categorical columns."""

    categorical_columns: dict[str, tuple[str, ...]]

    @property
    def categorical_features(self) -> set[str]:
        """Add the source table name to each categorical column name."""
        return {
            f"{table.lower()}__{column}" for table, columns in self.categorical_columns.items() for column in columns
        }


class Settings(BaseModel):
    """Settings used by the application."""

    oracle: OracleSettings
    paths: PathSettings
    models: ModelSettings
    features: FeatureSettings


def project_path(value: str) -> Path:
    """Convert a relative config path to a project path."""
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


@cache
def get_settings() -> Settings:
    """Read config.yaml and apply environment-specific values."""
    load_dotenv(ENV_PATH, override=False)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    oracle = config["oracle"]
    paths = config["paths"]

    return Settings(
        oracle=OracleSettings(
            query=DatabaseCredentials(
                user=os.getenv("DB_USER", ""),
                password=os.getenv("DB_PASSWORD", ""),
                dsn=os.getenv("DB_DSN", ""),
            ),
            instant_client_path=project_path(oracle["instant_client_dir"]),
            feature_schema=oracle["feature_schema"],
            population_sampling_schema=oracle["population_sampling_schema"],
            population_sampling_table=oracle["population_sampling_table"],
            population_source_schema=oracle["population_source_schema"],
            model_log_schema=oracle["model_log_schema"],
        ),
        paths=PathSettings(
            fin_table=project_path(paths["fin_table"]),
            model_base=project_path(paths["model_base"]),
        ),
        models=ModelSettings(**config["models"]),
        features=FeatureSettings(**config["features"]),
    )
