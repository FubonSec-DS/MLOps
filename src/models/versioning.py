"""Configured writes to existing model-log and prediction-output tables."""

from __future__ import annotations

import re

import oracledb
import pandas as pd

from configs.configs_val.DB_configs import get_settings
from src.common.database import OracleDB, get_query_database


def _quote_identifier(identifier: str) -> str:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]*", identifier):
        return identifier
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _table(name: str) -> str:
    schema = get_settings().oracle.model_log_schema
    return f"{_quote_identifier(schema)}.{_quote_identifier(name)}"


def _insert_dataframe(
    table_name: str,
    frame: pd.DataFrame,
    database: OracleDB,
    connection: oracledb.Connection,
) -> int:
    """Insert a small output/log DataFrame using named binds."""
    if frame.empty:
        return 0
    columns = list(frame.columns)
    bind_names = [f"value_{index}" for index in range(len(columns))]
    sql = (
        f"INSERT INTO {table_name} ({', '.join(_quote_identifier(c) for c in columns)}) "
        f"VALUES ({', '.join(':' + name for name in bind_names)})"
    )
    records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    rows = [{bind: record[column] for bind, column in zip(bind_names, columns, strict=True)} for record in records]
    return database.insert(sql, rows, connection=connection)


def save_prediction_outputs(ref_info: pd.DataFrame, id_list: pd.DataFrame, log: pd.DataFrame) -> None:
    """Write prediction outputs to existing configured-schema tables."""
    database = get_query_database()
    with database.transaction() as connection:
        _insert_dataframe(_table("MLOPS_REF_INFO_DOUBLE"), ref_info, database, connection)
        _insert_dataframe(_table("MLOPS_ID_LIST_DOUBLE"), id_list, database, connection)
        _insert_dataframe(_table("MLOPS_PREDICT_LOG_DOUBLE"), log, database, connection)


def save_retrain_outputs(retrain_log: pd.DataFrame, model_log: pd.DataFrame) -> None:
    """Write retrain and model metadata to the existing configured-schema tables."""
    database = get_query_database()
    with database.transaction() as connection:
        _insert_dataframe(_table("MLOPS_RETRAIN_LOG_DOUBLE"), retrain_log, database, connection)
        _insert_dataframe(_table("MLOPS_MODEL_LOG_DOUBLE"), model_log, database, connection)
