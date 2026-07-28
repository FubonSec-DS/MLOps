import os
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

BASE_DIR = Path(__file__).resolve().parents[2]
YAML_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
ENV_PATH = BASE_DIR / "configs" / ".env"


class OracleSettings(BaseModel):
    """Oracle database settings."""

    user: str = Field(min_length=1)
    password: SecretStr = Field(min_length=1)
    dsn: str = Field(min_length=1)
    instant_client_path: Path = Field()


class DataPipelineSettings(BaseModel):
    """Data pipeline settings."""

    test_1: int
    test_2: int


class ModelsSettings(BaseModel):
    """Models settings."""

    test_1: int
    test_2: int


class Settings(BaseModel):
    """MLOps settings."""

    oracle: OracleSettings
    data_pipeline: DataPipelineSettings
    models: ModelsSettings


@cache
def get_settings() -> Settings:
    """Load settings from the .env file and the config.yaml file.

    Returns:
        Settings: The settings object containing all the configuration values.

    """
    load_dotenv(ENV_PATH, override=False)
    with YAML_CONFIG_PATH.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    return Settings(
        oracle=OracleSettings(
            user=os.getenv("DB_USER", ""),
            password=SecretStr(os.getenv("DB_PASSWORD", "")),
            dsn=os.getenv("DB_DSN", ""),
            instant_client_path=BASE_DIR / cfg["oracle"]["instant_client_dir"],
        ),
        data_pipeline=DataPipelineSettings(**cfg["data_pipeline"]),
        models=ModelsSettings(**cfg["models"]),
    )
