"""Build the persisted training population."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import oracledb

from configs.configs_val.product_config import PRODUCT_CONFIGS, ProductConfig, calculate_training_periods
from src.common.database import OracleDB, get_query_database
from src.resources.queries.population_queries import build_training_sampling_insert_query, tables_source_name


@dataclass(frozen=True)
class SamplingResult:
    """Append-only result for one required training snapshot."""

    product: str
    population: str
    yyyymm: str
    sampled_count: int
    existing_count: int
    action: str


@dataclass(frozen=True)
class SamplingBatchResult:
    """Result of ensuring every required snapshot exists without changing old rows."""

    prediction_ym: str
    inserted_count: int
    existing_count: int
    snapshots: tuple[SamplingResult, ...]

    @property
    def inserted_snapshots(self) -> int:
        """Number of snapshots created by this run."""
        return sum(item.action == "inserted" for item in self.snapshots)

    @property
    def reused_snapshots(self) -> int:
        """Number of existing snapshots left unchanged."""
        return sum(item.action == "reused" for item in self.snapshots)


@dataclass(frozen=True)
class _SnapshotStatus:
    row_count: int
    y0_count: int
    y1_count: int

    @property
    def usable(self) -> bool:
        return self.row_count > 0 and self.y0_count > 0 and self.y1_count > 0


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


def _stored_segments(config: ProductConfig, population: str) -> tuple[str, ...]:
    """Return the SOURCE_SEGMENT values persisted for one model population."""
    if config.query_type in {"special_3m", "churn"}:
        return (population,)
    return config.get_population(population).source_segments


def _snapshot_status(
    database: OracleDB,
    sampling_table: str,
    config: ProductConfig,
    population: str,
    ym: str,
    *,
    connection: oracledb.Connection | None = None,
) -> _SnapshotStatus:
    segments = _stored_segments(config, population)
    params: dict[str, object] = {
        "ym": ym,
        "product": config.product,
        **{f"source_segment_{index}": segment for index, segment in enumerate(segments)},
    }
    segment_binds = ", ".join(f":source_segment_{index}" for index in range(len(segments)))
    result = database.query(
        f"""
        SELECT COUNT(*) AS ROW_COUNT,
               NVL(SUM(CASE WHEN Y = 0 THEN 1 ELSE 0 END), 0) AS Y0_COUNT,
               NVL(SUM(CASE WHEN Y = 1 THEN 1 ELSE 0 END), 0) AS Y1_COUNT
        FROM {sampling_table}
        WHERE YYYYMM = :ym
          AND PRODUCT = :product
          AND SOURCE_SEGMENT IN ({segment_binds})
        """,
        params,
        connection=connection,
    )
    row = result.to_dicts()[0]
    return _SnapshotStatus(int(row["ROW_COUNT"]), int(row["Y0_COUNT"]), int(row["Y1_COUNT"]))


def _require_usable_snapshot(
    status: _SnapshotStatus,
    config: ProductConfig,
    population: str,
    ym: str,
) -> None:
    if status.usable:
        return
    raise ValueError(
        f"Training snapshot cannot support binary training for product={config.product!r}, "
        f"population={population!r}, yyyymm={ym!r}: rows={status.row_count:,}, "
        f"y0={status.y0_count:,}, y1={status.y1_count:,}. "
        "Append-only policy left existing rows unchanged."
    )


def ensure_training_populations(
    prediction_ym: str,
    *,
    configs: Iterable[ProductConfig] | None = None,
    database: OracleDB | None = None,
) -> SamplingBatchResult:
    """Reuse usable snapshots and atomically append only missing snapshots."""
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
        ("SELECT", "INSERT"),
    )
    planned = []
    for config, ym, population in sampling_plan:
        status = _snapshot_status(db, sampling_table, config, population, ym)
        if status.row_count:
            _require_usable_snapshot(status, config, population, ym)
        planned.append((config, ym, population, status))

    snapshots: list[SamplingResult] = []
    with db.transaction() as connection:
        total_snapshots = len(planned)
        for index, (config, ym, population, status) in enumerate(planned, start=1):
            if status.row_count:
                print(
                    f"[{index}/{total_snapshots}] reusing product={config.product} "
                    f"population={population} yyyymm={ym} rows={status.row_count:,}",
                    flush=True,
                )
                snapshots.append(
                    SamplingResult(config.product, population, ym, 0, status.row_count, "reused")
                )
                continue
            print(
                f"[{index}/{total_snapshots}] inserting product={config.product} population={population} yyyymm={ym}",
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
            inserted_status = _snapshot_status(
                db,
                sampling_table,
                config,
                population,
                ym,
                connection=connection,
            )
            _require_usable_snapshot(inserted_status, config, population, ym)
            print(f"[{index}/{total_snapshots}] inserted={sampled_count:,}", flush=True)
            snapshots.append(
                SamplingResult(config.product, population, ym, sampled_count, 0, "inserted")
            )

    return SamplingBatchResult(
        prediction_ym=prediction_ym,
        inserted_count=sum(item.sampled_count for item in snapshots),
        existing_count=sum(item.existing_count for item in snapshots),
        snapshots=tuple(snapshots),
    )


def ensure_training_population(
    config: ProductConfig,
    population: str,
    ym: str,
    *,
    database: OracleDB | None = None,
) -> SamplingResult:
    """Reuse one usable snapshot or append it when it is missing."""
    db = database or get_query_database()
    population_table, sampling_table = tables_source_name()
    population_schema, population_table_name = population_table.split(".", maxsplit=1)
    sampling_schema, sampling_table_name = sampling_table.split(".", maxsplit=1)
    db.require_table_privileges(population_schema, population_table_name, ("SELECT",))
    db.require_table_privileges(
        sampling_schema,
        sampling_table_name,
        ("SELECT", "INSERT"),
    )
    status = _snapshot_status(db, sampling_table, config, population, ym)
    if status.row_count:
        _require_usable_snapshot(status, config, population, ym)
        return SamplingResult(config.product, population, ym, 0, status.row_count, "reused")

    insert_sql, insert_params = build_training_sampling_insert_query(config, population, ym)

    with db.transaction() as connection:
        sampled_count = db.execute(insert_sql, insert_params, connection=connection)
        inserted_status = _snapshot_status(
            db,
            sampling_table,
            config,
            population,
            ym,
            connection=connection,
        )
        _require_usable_snapshot(inserted_status, config, population, ym)

    return SamplingResult(config.product, population, ym, sampled_count, 0, "inserted")


def rebuild_training_populations(
    prediction_ym: str,
    *,
    configs: Iterable[ProductConfig] | None = None,
    database: OracleDB | None = None,
) -> SamplingBatchResult:
    """Backward-compatible alias for the append-only ensure workflow."""
    return ensure_training_populations(prediction_ym, configs=configs, database=database)


def refresh_training_population(
    config: ProductConfig,
    population: str,
    ym: str,
    *,
    database: OracleDB | None = None,
) -> SamplingResult:
    """Backward-compatible alias for ensuring one append-only snapshot."""
    return ensure_training_population(config, population, ym, database=database)
