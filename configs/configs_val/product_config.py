"""Validate product YAML and provide immutable runtime configuration objects."""

# 延後解析型別註記
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import yaml
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, ConfigDict, Field, model_validator

from configs.configs_val.DB_configs import BASE_DIR

PRODUCT_CONFIG_PATH = BASE_DIR / "configs" / "products.yaml"

PROSPECT_SEGMENT = "潛客"
NON_PROSPECT_SEGMENT = "非潛客"
SOURCE_SEGMENTS = (PROSPECT_SEGMENT, NON_PROSPECT_SEGMENT)


class _StrictModel(BaseModel):
    # YAML 出現沒有定義的欄位時，直接報錯。
    # 建立後不允許重新指定欄位
    model_config = ConfigDict(extra="forbid", frozen=True)


class XGBoostParams(_StrictModel):
    """Validated XGBoost hyperparameters."""

    # ge=1: 整數，而且大於或等於 1
    max_depth: int = Field(ge=1)
    scale_pos_weight: float = Field(gt=0)
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bylevel: float = Field(gt=0, le=1)
    reg_lambda: float = Field(ge=0)
    min_child_weight: float = Field(ge=0)
    early_stopping_rounds: int | None = Field(default=None, gt=0)


class TrainingSchedule(_StrictModel):
    """Relative periods used to construct a training schedule."""

    train_months_count: int = Field(gt=0)
    label_offset_months: int = Field(ge=0)
    backtest_gap_months: int = Field(ge=0)
    train_interval_months: int = Field(gt=0)


class _ProductSpec(_StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["standard", "explicit"]
    business_target: str | None = None

    # 資格條件欄位
    eligibility_suffix: str = "近一年舊戶"
    eligibility_column: str | None = None
    prediction_eligibility_column: str | None = None
    eligibility_values: tuple[int, ...] = (0,)
    eligibility_null_value: int | None = 0

    # 目標欄位
    target_column: str | None = None
    target_null_value: int | None = 0

    # 客群設定，未指定時，一個產品會建立兩套客群 ("非潛客", "潛客")
    populations: tuple[str, ...] = (NON_PROSPECT_SEGMENT, PROSPECT_SEGMENT)

    # profile 名稱
    bin_profile: str = "default"
    hit_rate_profile: str | None = None
    rank_bins: tuple[int, ...] | None = None

    """
        query_type 會影響 SQL 查詢的選擇或條件：
        standard：一般產品。
        structure：結構型產品。
        special_3m：特殊三個月規則。
        churn：流失預警。
    """
    query_type: Literal["standard", "structure", "special_3m", "churn"] = "standard"
    drop_keywords: tuple[str, ...] = ()
    training_schedule: str = "standard"
    pretrain_profile: str = "pretrain"
    training_profile: str = "training"

    @model_validator(mode="after")
    def validate_product_shape(self) -> _ProductSpec:
        """Require explicit products to provide their non-derived columns."""
        if self.kind == "explicit":
            required = {
                "business_target": self.business_target,
                "eligibility_column": self.eligibility_column,
                "target_column": self.target_column,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"Explicit product {self.name!r} is missing: {', '.join(missing)}")
        if not self.eligibility_values:
            raise ValueError(f"Product {self.name!r} must define at least one eligibility value")
        if not self.populations:
            raise ValueError(f"Product {self.name!r} must define at least one population")
        return self


@dataclass(frozen=True)
class PopulationConfig:
    """Sampling configuration for one model population."""

    population: str
    sample_size: int
    source_segments: tuple[str, ...]
    hit_rate_thresholds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Reject empty, duplicate, or unknown source-segment mappings."""
        if not self.source_segments:
            raise ValueError(f"Population {self.population!r} must contain at least one source segment")
        if len(set(self.source_segments)) != len(self.source_segments):
            raise ValueError(f"Population {self.population!r} contains duplicate source segments")
        unknown = set(self.source_segments) - set(SOURCE_SEGMENTS)
        if unknown:
            raise ValueError(f"Population {self.population!r} contains unknown source segments: {sorted(unknown)}")


@dataclass(frozen=True)
class ProductConfig:
    """Configuration required to sample and train one product."""

    product: str
    business_target: str
    eligibility_column: str
    eligibility_values: tuple[int, ...]
    eligibility_null_value: int | None
    target_column: str
    target_null_value: int | None
    populations: tuple[PopulationConfig, ...]
    # 用來把模型分數切成不同級距。
    bins: tuple[float, ...]
    query_type: Literal["standard", "structure", "special_3m", "churn"] = "standard"
    prediction_eligibility_column: str | None = None
    rank_bins: tuple[int, ...] | None = None
    # 表示建立 ProductConfig 時，
    # 如果沒有傳入 pretrain_params，才呼叫 _default_xgboost("pretrain")
    pretrain_params: XGBoostParams = field(default_factory=lambda: _XGBOOST_PROFILES["pretrain"])
    training_params: XGBoostParams = field(default_factory=lambda: _XGBOOST_PROFILES["training"])
    drop_keywords: tuple[str, ...] = ()
    training_schedule: TrainingSchedule = field(default_factory=lambda: _TRAINING_SCHEDULES["standard"])

    def get_population(self, population: str) -> PopulationConfig:
        """Return a configured population or raise a useful error."""
        for config in self.populations:
            if config.population == population:
                return config
        configured = ", ".join(item.population for item in self.populations)
        raise KeyError(f"Unknown population {population!r}; configured values: {configured}")


with PRODUCT_CONFIG_PATH.open(encoding="utf-8") as file:
    _CONFIG = yaml.safe_load(file)

# 用途是根據 profile 名稱取得 XGBoost 參數。
_XGBOOST_PROFILES = {name: XGBoostParams.model_validate(values) for name, values in _CONFIG["xgboost_profiles"].items()}

# 用途是根據 profile 名稱取得 SCHEDULES 設定。
_TRAINING_SCHEDULES = {
    name: TrainingSchedule.model_validate(values) for name, values in _CONFIG["training_schedules"].items()
}
_PRODUCT_SPECS = tuple(_ProductSpec.model_validate(values) for values in _CONFIG["products"])


def _build_product(spec: _ProductSpec) -> ProductConfig:
    hit_rates = tuple(_CONFIG["hit_rate_profiles"][spec.hit_rate_profile]) if spec.hit_rate_profile else None
    if spec.kind == "standard":
        business_target = "新戶開發與靜止戶活化"
        eligibility_column = f"{spec.name}{spec.eligibility_suffix}"
        target_column = f"{spec.name}Y"
    else:
        business_target = spec.business_target or ""
        eligibility_column = spec.eligibility_column or ""
        target_column = spec.target_column or ""

    return ProductConfig(
        product=spec.name,
        business_target=business_target,
        eligibility_column=eligibility_column,
        prediction_eligibility_column=spec.prediction_eligibility_column,
        eligibility_values=spec.eligibility_values,
        eligibility_null_value=spec.eligibility_null_value,
        target_column=target_column,
        target_null_value=spec.target_null_value,
        populations=tuple(
            PopulationConfig(
                population=name,
                sample_size=_CONFIG["population_profiles"][name]["sample_size"],
                source_segments=tuple(_CONFIG["population_profiles"][name]["source_segments"]),
                hit_rate_thresholds=hit_rates,
            )
            for name in spec.populations
        ),
        bins=tuple(_CONFIG["bin_profiles"][spec.bin_profile]),
        rank_bins=spec.rank_bins,
        query_type=spec.query_type,
        pretrain_params=_XGBOOST_PROFILES[spec.pretrain_profile],
        training_params=_XGBOOST_PROFILES[spec.training_profile],
        drop_keywords=spec.drop_keywords,
        training_schedule=_TRAINING_SCHEDULES[spec.training_schedule],
    )


PRODUCT_CONFIGS: dict[str, ProductConfig] = {spec.name: _build_product(spec) for spec in _PRODUCT_SPECS}


def get_product_config(product_name: str = "流失預警") -> ProductConfig:
    """Return the validated configuration for one product."""
    return PRODUCT_CONFIGS[product_name]


def calculate_training_periods(
    config: ProductConfig,
    reference_date: date | None = None,
) -> tuple[list[str], str, str]:
    """Calculate model-building, OOT backtest, and latest anchor months."""
    if reference_date is None:
        reference_date = date.today() - relativedelta(months=1)

    schedule = config.training_schedule
    latest_date = reference_date.replace(day=1)
    backtest_date = latest_date - relativedelta(months=schedule.label_offset_months)
    train_end_date = backtest_date - relativedelta(months=schedule.backtest_gap_months)
    train_yms = [
        (train_end_date - relativedelta(months=offset * schedule.train_interval_months)).strftime("%Y%m")
        for offset in range(schedule.train_months_count - 1, -1, -1)
    ]
    return train_yms, backtest_date.strftime("%Y%m"), latest_date.strftime("%Y%m")
