"""Build product-specific population SQL."""

from __future__ import annotations

import re
from typing import Any

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import ProductConfig

_SIMPLE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")


def quote_identifier(identifier: str) -> str:
    """Quote nonstandard Oracle identifiers such as Chinese column names."""
    return identifier if _SIMPLE_IDENTIFIER.fullmatch(identifier) else f'"{identifier}"'


def tables_source_name() -> tuple[str, str]:
    """Return the configured population source."""
    oracle = get_settings().oracle
    population_table = f"{oracle.population_source_schema}.MLOPS_POPULATION"
    sampling_table = f"{oracle.population_sampling_schema}.{oracle.population_sampling_table}"
    return population_table, sampling_table


def eligibility_predicate(
    config: ProductConfig,
    *,
    prediction: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build the product eligibility predicate and its bind values."""
    expression, params = _eligibility_expression(config, prediction=prediction)
    values = {f"eligibility_{index}": value for index, value in enumerate(config.eligibility_values)}
    params.update(values)
    placeholders = ", ".join(f":{name}" for name in values)
    return f"{expression} IN ({placeholders})", params


def _eligibility_expression(
    config: ProductConfig,
    *,
    prediction: bool = False,
) -> tuple[str, dict[str, Any]]:
    column = config.prediction_eligibility_column if prediction else None
    expression = quote_identifier(column or config.eligibility_column)
    params: dict[str, Any] = {}
    if config.eligibility_null_value is not None:
        params["eligibility_null_value"] = config.eligibility_null_value
        expression = f"NVL({expression}, :eligibility_null_value)"
    return expression, params


def population_scope_predicate(
    config: ProductConfig,
    population: str,
    *,
    alias: str = "",
) -> tuple[str, dict[str, Any]]:
    """Build the source-segment predicate for a model population."""
    if config.query_type in {"special_3m", "churn"}:
        return "1 = 1", {}

    segments = config.get_population(population).source_segments
    params = {f"source_segment_{index}": segment for index, segment in enumerate(segments)}
    placeholders = ", ".join(f":{name}" for name in params)
    condition = f"{alias}SEGMENT IN ({placeholders})"
    if config.query_type == "structure":
        condition += f' AND {alias}"98戶" IS NULL'
    return condition, params


def build_training_sampling_insert_query(
    config: ProductConfig,
    population: str,
    ym: str,
) -> tuple[str, dict[str, Any]]:
    """Build the deterministic training-population INSERT."""
    settings = get_settings()
    eligibility_expression, _ = _eligibility_expression(config)
    eligibility, params = eligibility_predicate(config)
    population_scope, scope_params = population_scope_predicate(config, population)
    params.update(scope_params)
    combined_population = config.query_type in {"special_3m", "churn"}
    source_segment_expression = ":population" if combined_population else "SEGMENT"
    source_segment_group = "" if combined_population else ", SEGMENT"
    if combined_population:
        params["population"] = population
    target_column = quote_identifier(config.target_column)
    if config.target_null_value is None:
        target_expression = target_column
        target_filter = f"AND {target_column} IS NOT NULL"
    else:
        target_expression = f"NVL({target_column}, :target_null_value)"
        target_filter = ""
        params["target_null_value"] = config.target_null_value
    params.update(
        {
            "ym": ym,
            "product": config.product,
            "sample_size": config.get_population(population).sample_size,
            "hash_seed": settings.models.random_state,
        }
    )

    population_table, sampling_table = tables_source_name()

    sql = f"""
    INSERT INTO {sampling_table}
        (CUSTOMER_ID, YYYYMM, PRODUCT, SOURCE_SEGMENT, Y, ELIGIBILITY_TAG)
    SELECT CUSTOMER_ID,
           YYYYMM,
           :product,
           SOURCE_SEGMENT,
           Y,
           ELIGIBILITY_TAG
    FROM (
        SELECT CUSTOMER_ID,
               YYYYMM,
               SOURCE_SEGMENT,
               Y,
               ELIGIBILITY_TAG,
               ROW_NUMBER() OVER (
                    ORDER BY ORA_HASH(
                        CUSTOMER_ID || '|' || YYYYMM || '|' || :product || '|' || SOURCE_SEGMENT,
                        4294967295,
                        :hash_seed
                    ), CUSTOMER_ID
               ) AS SAMPLE_RN
        FROM (
            SELECT /*+ PARALLEL(P 4) */
                   CUSTOMER_ID,
                   YYYYMM,
                   {source_segment_expression} AS SOURCE_SEGMENT,
                   MAX({target_expression}) AS Y,
                   MAX({eligibility_expression}) AS ELIGIBILITY_TAG
            FROM {population_table} P
            WHERE YYYYMM = :ym
              AND {population_scope}
              AND {eligibility}
              {target_filter}
            GROUP BY CUSTOMER_ID, YYYYMM{source_segment_group}
        )
    )
    WHERE SAMPLE_RN <= :sample_size
    """
    return sql, params
