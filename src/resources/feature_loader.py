"""Load and combine SQL results into model-ready data frames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pandas as pd
import polars as pl

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import ProductConfig
from src.common.database import OracleDB, get_query_database
from src.resources.queries.feature_queries import (
    FEATURE_TABLE_GROUPS,
    FeatureColumn,
    build_feature_query,
    build_metadata_query,
    build_prediction_feature_query,
    configured_feature_tables,
)

KEY_COLUMNS = ["customer_id", "yyyymm"]
MODEL_COLUMNS = [*KEY_COLUMNS, "y"]


@dataclass
class LoadedDataset:
    """A model-ready frame and its categorical features."""

    frame: pd.DataFrame
    categorical_features: list[str]


def _to_pandas(result: pl.DataFrame) -> pd.DataFrame:
    data = result.to_dict(as_series=False)
    return pd.DataFrame({str(name).lower(): values for name, values in data.items()})


def discover_feature_catalog(database: OracleDB | None = None) -> list[FeatureColumn]:
    """Read feature columns from Oracle metadata."""
    db = database or get_query_database()
    sql, params = build_metadata_query(configured_feature_tables())
    return [
        FeatureColumn(
            table=str(row["TABLE_NAME"]),
            column=str(row["COLUMN_NAME"]),
            data_type=str(row["DATA_TYPE"]),
        )
        for row in db.query(sql, params).to_dicts()
    ]


def _load_fin_features(ym: str, required: Iterable[str] = ()) -> pd.DataFrame | None:
    path = get_settings().paths.fin_table
    required_names = set(required)
    if not path.exists():
        if required_names:
            raise FileNotFoundError(path)
        return None

    frame = pd.read_csv(path, dtype={"ym": str})
    original_names = frame.columns.str.replace(r"\.\d+$", "", regex=True)
    frame = frame.loc[:, ~original_names.duplicated()]
    frame = cast("pd.DataFrame", frame.loc[frame["ym"] == ym].copy())
    if frame.empty:
        if required_names:
            raise ValueError(f"FIN feature file has no row for {ym}")
        return None

    rename_map = {str(column): f"fin__{str(column).lower()}" for column in frame.columns if column != "ym"}
    frame = frame.rename(columns={"ym": "yyyymm", **rename_map})
    missing = required_names - set(frame.columns)
    if missing:
        raise ValueError(f"FIN features not found: {sorted(missing)}")
    return cast("pd.DataFrame", frame.loc[:, ["yyyymm", *sorted(required_names)]]) if required_names else frame


def _append_fin_features(frame: pd.DataFrame, ym: str, required: Iterable[str] = ()) -> pd.DataFrame:
    fin = _load_fin_features(ym, required)
    return frame if fin is None else frame.merge(fin, on="yyyymm", how="left", validate="many_to_one")


def _categorical_features(columns: Iterable[FeatureColumn]) -> list[str]:
    configured = get_settings().features.categorical_features
    text_types = {"CHAR", "CLOB", "NCHAR", "NCLOB", "NVARCHAR2", "VARCHAR2"}
    return [
        column.feature_name
        for column in columns
        if column.feature_name in configured or column.data_type.upper() in text_types
    ]


def load_pretrain_month(
    config: ProductConfig,
    population: str,
    ym: str,
    *,
    catalog: list[FeatureColumn] | None = None,
    database: OracleDB | None = None,
    row_limit: int | None = None,
) -> LoadedDataset:
    """Load every candidate feature for one pretraining month."""
    db = database or get_query_database()
    feature_catalog = catalog or discover_feature_catalog(db)
    frames = []

    for tables in FEATURE_TABLE_GROUPS.values():
        columns = [column for column in feature_catalog if column.table.upper() in tables]
        if columns:
            sql, params = build_feature_query(
                config,
                ym,
                population,
                columns,
                pretrain=True,
                row_limit=row_limit,
            )
            frames.append(_to_pandas(db.query(sql, params)))

    if not frames:
        raise ValueError("No feature groups were loaded")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame.drop(columns="y"),
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    merged = _append_fin_features(merged, ym)
    categorical = [name for name in _categorical_features(feature_catalog) if name in merged.columns]
    return LoadedDataset(merged, categorical)


def _resolve_selected_columns(
    selected_features: Iterable[str],
    catalog: list[FeatureColumn],
) -> tuple[list[FeatureColumn], list[str]]:
    by_name = {column.feature_name: column for column in catalog}
    oracle_columns = []
    fin_columns = []
    for feature in selected_features:
        if feature.startswith("fin__"):
            fin_columns.append(feature)
        elif feature in by_name:
            oracle_columns.append(by_name[feature])
        else:
            raise ValueError(f"Feature source not found: {feature}")
    return oracle_columns, fin_columns


def load_training_month(
    config: ProductConfig,
    population: str,
    ym: str,
    selected_features: list[str],
    *,
    catalog: list[FeatureColumn] | None = None,
    database: OracleDB | None = None,
    row_limit: int | None = None,
) -> LoadedDataset:
    """Load selected features for one training month."""
    db = database or get_query_database()
    feature_catalog = catalog or discover_feature_catalog(db)
    oracle_columns, fin_columns = _resolve_selected_columns(selected_features, feature_catalog)
    sql, params = build_feature_query(
        config,
        ym,
        population,
        oracle_columns,
        pretrain=False,
        row_limit=row_limit,
    )
    frame = _append_fin_features(_to_pandas(db.query(sql, params)), ym, fin_columns)
    selected_frame = cast("pd.DataFrame", frame.loc[:, [*MODEL_COLUMNS, *selected_features]])
    return LoadedDataset(selected_frame, _categorical_features(oracle_columns))


def load_prediction_month(
    config: ProductConfig,
    population: str,
    ym: str,
    selected_features: list[str],
    *,
    catalog: list[FeatureColumn] | None = None,
    database: OracleDB | None = None,
) -> LoadedDataset:
    """Load selected features for one prediction month."""
    db = database or get_query_database()
    feature_catalog = catalog or discover_feature_catalog(db)
    oracle_columns, fin_columns = _resolve_selected_columns(selected_features, feature_catalog)
    sql, params = build_prediction_feature_query(config, ym, population, oracle_columns)
    frame = _append_fin_features(_to_pandas(db.query(sql, params)), ym, fin_columns)
    selected_frame = cast("pd.DataFrame", frame.loc[:, [*KEY_COLUMNS, *selected_features]])
    return LoadedDataset(selected_frame, _categorical_features(oracle_columns))


def combine_months(datasets: Iterable[LoadedDataset]) -> LoadedDataset:
    """Append monthly datasets."""
    items = list(datasets)
    if not items:
        raise ValueError("At least one monthly dataset is required")
    frame = pd.concat([item.frame for item in items], ignore_index=True)
    categorical = sorted({feature for item in items for feature in item.categorical_features})
    return LoadedDataset(frame, categorical)
