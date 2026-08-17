"""Build the persisted training population."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from configs.configs_val.product_config import PRODUCT_CONFIGS, ProductConfig, calculate_training_periods
from src.common.database import OracleDB, get_query_database
from src.resources.queries.population_queries import build_training_sampling_insert_query, tables_source_name


@dataclass(frozen=True)
class SamplingResult:
    """Training-population SQL result."""

    product: str
    population: str
    yyyymm: str
    sampled_count: int
    replaced_count: int


@dataclass(frozen=True)
class SamplingBatchResult:
    """Result of rebuilding every training-population snapshot for one prediction month."""

    prediction_ym: str
    replaced_count: int
    sampled_count: int
    snapshots: tuple[SamplingResult, ...]


def _prediction_reference_date(prediction_ym: str) -> datetime:
    """Parse the prediction month as the schedule anchor."""
    try:
        return datetime.strptime(prediction_ym, "%Y%m")
    except ValueError as error:
        raise ValueError(f"Invalid prediction YYYYMM value: {prediction_ym!r}") from error


def required_sampling_months(config: ProductConfig, prediction_ym: str) -> tuple[str, ...]:
    """Return model-building and OOT backtest months; exclude the latest anchor."""
    train_yms, backtest_ym, _latest_ym = calculate_training_periods(
        config,
        reference_date=_prediction_reference_date(prediction_ym),
    )
    return tuple(dict.fromkeys([*train_yms, backtest_ym]))


def rebuild_training_populations(
    prediction_ym: str,
    *,
    configs: Iterable[ProductConfig] | None = None,
    database: OracleDB | None = None,
) -> SamplingBatchResult:
    """Replace the table with every snapshot needed to predict one month."""
    selected_configs = tuple(configs) if configs is not None else tuple(PRODUCT_CONFIGS.values())
    if not selected_configs:
        raise ValueError("At least one product configuration is required")

    sampling_plan = tuple(
        (config, ym, population.population)
        for config in selected_configs
        for ym in required_sampling_months(config, prediction_ym)
        for population in config.populations
    )

    db = database or get_query_database()
    population_table, sampling_table = tables_source_name()
    population_schema, population_table_name = population_table.split(".", maxsplit=1)
    sampling_schema, sampling_table_name = sampling_table.split(".", maxsplit=1)
    db.require_table_privileges(population_schema, population_table_name, ("SELECT",))
    db.require_table_privileges(
        sampling_schema,
        sampling_table_name,
        ("SELECT", "INSERT", "DELETE"),
    )
    snapshots: list[SamplingResult] = []
    with db.transaction() as connection:
        replaced_count = db.execute(
            f"DELETE FROM {sampling_table}",
            connection=connection,
        )
        total_snapshots = len(sampling_plan)
        for index, (config, ym, population) in enumerate(sampling_plan, start=1):
            print(
                f"[{index}/{total_snapshots}] sampling product={config.product} population={population} yyyymm={ym}",
                flush=True,
            )
            insert_sql, insert_params = build_training_sampling_insert_query(
                config,
                population,
                ym,
            )
            try:
                sampled_count = db.execute(insert_sql, insert_params, connection=connection)
            except Exception as error:
                raise RuntimeError(
                    f"Sampling insert failed for product={config.product!r}, population={population!r}, yyyymm={ym!r}"
                ) from error
            print(f"[{index}/{total_snapshots}] inserted={sampled_count:,}", flush=True)
            snapshots.append(
                SamplingResult(
                    config.product,
                    population,
                    ym,
                    sampled_count,
                    0,
                )
            )

    return SamplingBatchResult(
        prediction_ym=prediction_ym,
        replaced_count=replaced_count,
        sampled_count=sum(item.sampled_count for item in snapshots),
        snapshots=tuple(snapshots),
    )


def refresh_training_population(
    config: ProductConfig,
    population: str,
    ym: str,
    *,
    database: OracleDB | None = None,
) -> SamplingResult:
    """Replace one monthly training population."""
    db = database or get_query_database()
    population_table, sampling_table = tables_source_name()
    population_schema, population_table_name = population_table.split(".", maxsplit=1)
    sampling_schema, sampling_table_name = sampling_table.split(".", maxsplit=1)
    db.require_table_privileges(population_schema, population_table_name, ("SELECT",))
    db.require_table_privileges(
        sampling_schema,
        sampling_table_name,
        ("SELECT", "INSERT", "DELETE"),
    )
    segments = (
        (population,)
        if config.query_type in {"special_3m", "churn"}
        else config.get_population(population).source_segments
    )
    scope = {
        "ym": ym,
        "product": config.product,
        **{f"source_segment_{index}": segment for index, segment in enumerate(segments)},
    }
    segment_binds = ", ".join(f":source_segment_{index}" for index in range(len(segments)))
    insert_sql, insert_params = build_training_sampling_insert_query(config, population, ym)

    with db.transaction() as connection:
        replaced_count = db.execute(
            f"""
            DELETE FROM {sampling_table}
            WHERE YYYYMM = :ym
              AND PRODUCT = :product
              AND SOURCE_SEGMENT IN ({segment_binds})
            """,
            scope,
            connection=connection,
        )
        sampled_count = db.execute(insert_sql, insert_params, connection=connection)

    return SamplingResult(config.product, population, ym, sampled_count, replaced_count)
