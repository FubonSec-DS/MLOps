"""Build feature SQL and bind values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import ProductConfig
from src.resources.queries.population_queries import (
    eligibility_predicate,
    population_scope_predicate,
    quote_identifier,
    tables_source_name,
)

FEATURE_TABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "asset": (
        "CF_ASSET1",
        "CF_ASSET2",
        "CF_ASSET3",
        "CF_ASSET4",
        "CF_ACTUBNF",
        "CF_ACTUBNFROI",
        "CF_INVBNF",
        "CF_INVBNFROI",
    ),
    "transaction_1": (
        "CF_TXN_AP",
        "CF_TXN_FD",
        "CF_TXN_FB",
        "CF_TXN_FS",
        "CF_TXN_FU",
        "CF_TXN_STMT",
        "CF_TXN_SN",
        "CF_TXN_STSS",
        "CF_TXN_STST",
    ),
    "transaction_2": (
        "CF_TXN_BSBL",
        "CF_TXN_INSURANCE",
        "CF_TXN_SIP",
        "CF_TXN_STDT",
        "CF_TXN_LOAN",
        "CF_TXN_STYPE",
        "CF_TXN_MAEC",
        "CF_TXN_CURRENCY",
        "CF_TXN_FPBR",
        "CF_AUM",
    ),
    "profile": (
        "CF_PROFILE",
        "CF_PROFOLIO1",
        "CF_PROFOLIO2",
        "CF_PROFOLIO3",
        "CF_PROFOLIO4",
        "CF_PROFOLIO5",
        "CF_PROFOLIO6",
        "CF_PROFOLIO7",
    ),
    "interaction": (
        "CF_INTERACT_ECDAY",
        "CF_INTERACT_ECPROD",
        "CF_INTERACT_EDM",
        "CF_INTERACT_LINE",
        "CF_CONTRACT_DUE",
        "CF_CONTRACT",
        "CF_EVENT",
        "CF_KYCQA",
        "CF_JCI",
        "CF_DGT1",
        "CF_DGT2",
    ),
}


@dataclass(frozen=True)
class FeatureColumn:
    """One Oracle feature and its stable model-facing name."""

    table: str
    column: str
    data_type: str

    @property
    def feature_name(self) -> str:
        """Return ``{table}__{column}`` in a stable lowercase form."""
        name = f"{self.table.lower()}__{self.column.lower()}"
        if len(name) > 128:
            raise ValueError(f"Feature alias exceeds Oracle's 128-character limit: {name}")
        return name


def configured_feature_tables() -> tuple[str, ...]:
    """Flatten configured table groups into a unique table list."""
    return tuple(dict.fromkeys(table for tables in FEATURE_TABLE_GROUPS.values() for table in tables))


def build_metadata_query(tables: Iterable[str]) -> tuple[str, dict[str, Any]]:
    """Build the Oracle column-metadata query for configured tables."""
    settings = get_settings()
    table_list = tuple(tables)
    params: dict[str, Any] = {"owner": settings.oracle.feature_schema.upper()}
    placeholders = []
    for index, table in enumerate(table_list):
        name = f"table_{index}"
        params[name] = table.upper()
        placeholders.append(f":{name}")
    if not placeholders:
        raise ValueError("At least one feature table is required")
    return (
        f"""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_ID
        FROM ALL_TAB_COLUMNS
        WHERE OWNER = :owner
          AND TABLE_NAME IN ({", ".join(placeholders)})
          AND COLUMN_NAME NOT IN ('CUSTOMER_ID', 'YYYYMM')
        ORDER BY TABLE_NAME, COLUMN_ID
        """,
        params,
    )


def build_feature_query(
    config: ProductConfig,
    ym: str,
    population: str,
    columns: Iterable[FeatureColumn],
    *,
    pretrain: bool,
    row_limit: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a cohort-driven, source-qualified feature SELECT."""
    selected = tuple(columns)

    settings = get_settings()
    _, sampling_table = tables_source_name()
    segments = (
        (population,)
        if config.query_type in {"special_3m", "churn"}
        else config.get_population(population).source_segments
    )
    params: dict[str, Any] = {
        "ym": ym,
        "product": config.product,
        **{f"source_segment_{index}": segment for index, segment in enumerate(segments)},
    }
    segment_binds = ", ".join(f":source_segment_{index}" for index in range(len(segments)))
    training_cohort = f"""
    TRAINING_COHORT AS (
        SELECT CUSTOMER_ID, YYYYMM, Y
        FROM {sampling_table}
        WHERE YYYYMM = :ym
          AND PRODUCT = :product
          AND SOURCE_SEGMENT IN ({segment_binds})
    )
    """
    if pretrain:
        params.update(
            {
                "hash_seed": settings.models.random_state,
                "pretrain_size": settings.models.pretrain_size,
            }
        )
        cte = f"""
        WITH {training_cohort},
        NEGATIVE_RANKED AS (
            SELECT CUSTOMER_ID,
                   YYYYMM,
                   Y,
                   ROW_NUMBER() OVER (
                       ORDER BY ORA_HASH(
                           CUSTOMER_ID || '|' || YYYYMM || '|' || :product || '|pretrain',
                           4294967295,
                           :hash_seed
                       ), CUSTOMER_ID
                   ) AS SAMPLE_RN
            FROM TRAINING_COHORT
            WHERE Y = 0
        ),
        POSITIVE_COUNT AS (
            SELECT COUNT(*) AS CNT FROM TRAINING_COHORT WHERE Y = 1
        ),
        PRETRAIN_COHORT AS (
            SELECT CUSTOMER_ID, YYYYMM, Y
            FROM TRAINING_COHORT
            WHERE Y = 1
            UNION ALL
            SELECT CUSTOMER_ID, YYYYMM, Y
            FROM NEGATIVE_RANKED
            WHERE SAMPLE_RN <= GREATEST(:pretrain_size - (SELECT CNT FROM POSITIVE_COUNT), 0)
        )
        """
        cohort = "PRETRAIN_COHORT"
        if row_limit is not None:
            if row_limit <= 0:
                raise ValueError("row_limit must be greater than zero")
            params["row_limit"] = row_limit
            cte += """
            , LIMITED_PRETRAIN_COHORT AS (
                SELECT CUSTOMER_ID, YYYYMM, Y
                FROM (
                    SELECT P.*,
                           ROW_NUMBER() OVER (
                               ORDER BY ORA_HASH(
                                   CUSTOMER_ID || '|' || YYYYMM || '|' || :product || '|readiness',
                                   4294967295,
                                   :hash_seed
                               ), CUSTOMER_ID
                           ) AS READINESS_RN
                    FROM PRETRAIN_COHORT P
                )
                WHERE READINESS_RN <= :row_limit
            )
            """
            cohort = "LIMITED_PRETRAIN_COHORT"
    else:
        cte = f"WITH {training_cohort}"
        cohort = "TRAINING_COHORT"
        if row_limit is not None:
            if row_limit <= 0:
                raise ValueError("row_limit must be greater than zero")
            params.update({"hash_seed": settings.models.random_state, "row_limit": row_limit})
            cte += """
            , LIMITED_TRAINING_COHORT AS (
                SELECT CUSTOMER_ID, YYYYMM, Y
                FROM (
                    SELECT C.*,
                           ROW_NUMBER() OVER (
                               ORDER BY ORA_HASH(
                                   CUSTOMER_ID || '|' || YYYYMM || '|' || :product || '|training',
                                   4294967295,
                                   :hash_seed
                               ), CUSTOMER_ID
                           ) AS TRAINING_RN
                    FROM TRAINING_COHORT C
                )
                WHERE TRAINING_RN <= :row_limit
            )
            """
            cohort = "LIMITED_TRAINING_COHORT"
    tables = tuple(dict.fromkeys(column.table.upper() for column in selected))
    aliases = {table: f"T{index}" for index, table in enumerate(tables)}

    feature_selects = []
    for column in selected:
        alias = aliases[column.table.upper()]
        source_column = quote_identifier(column.column)
        output_column = quote_identifier(column.feature_name)
        feature_selects.append(f"{alias}.{source_column} AS {output_column}")

    joins = []
    for table in tables:
        source = f"{settings.oracle.feature_schema}.{table}"
        alias = aliases[table]
        joins.append(f"LEFT JOIN {source} {alias} ON {alias}.CUSTOMER_ID = C.CUSTOMER_ID AND {alias}.YYYYMM = C.YYYYMM")

    feature_projection = f",{chr(10)}           {', '.join(feature_selects)}" if feature_selects else ""
    sql = f"""
    {cte}
    SELECT C.CUSTOMER_ID,
           C.YYYYMM,
           C.Y{feature_projection}
    FROM {cohort} C
    {chr(10).join(joins)}
    """
    return sql, params


def build_prediction_feature_query(
    config: ProductConfig,
    ym: str,
    population: str,
    columns: Iterable[FeatureColumn],
) -> tuple[str, dict[str, Any]]:
    """Load only model features for the full eligible prediction population."""
    selected = tuple(columns)
    settings = get_settings()
    population_table, _ = tables_source_name()
    eligibility, params = eligibility_predicate(config, prediction=True)
    segment_filter, segment_params = population_scope_predicate(config, population)
    params.update(segment_params)
    params["ym"] = ym
    cohort = f"""
    SELECT CUSTOMER_ID, YYYYMM
    FROM {population_table}
    WHERE YYYYMM = :ym
      AND {segment_filter}
      AND {eligibility}
    """
    tables = tuple(dict.fromkeys(column.table.upper() for column in selected))
    aliases = {table: f"T{index}" for index, table in enumerate(tables)}
    selects = [
        f"{aliases[column.table.upper()]}.{quote_identifier(column.column)} AS {quote_identifier(column.feature_name)}"
        for column in selected
    ]
    joins = [
        f"LEFT JOIN {settings.oracle.feature_schema}.{table} "
        f"{aliases[table]} ON {aliases[table]}.CUSTOMER_ID = C.CUSTOMER_ID "
        f"AND {aliases[table]}.YYYYMM = C.YYYYMM"
        for table in tables
    ]
    feature_projection = f", {', '.join(selects)}" if selects else ""
    sql = f"""
    WITH PREDICTION_COHORT AS ({cohort})
    SELECT C.CUSTOMER_ID, C.YYYYMM{feature_projection}
    FROM PREDICTION_COHORT C
    {chr(10).join(joins)}
    """
    return sql, params
