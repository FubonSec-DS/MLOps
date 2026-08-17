"""Run the population SQL files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from dateutil.relativedelta import relativedelta

from configs.configs_val.product_config import ProductConfig
from src.common.database import OracleDB, get_query_database, read_sql
from src.resources.queries.population_queries import tables_source_name


@dataclass(frozen=True)
class PopulationRefreshResult:
    """Population SQL execution result."""

    yyyymm: str
    inserted_count: int
    replaced_count: int


def population_snapshot_ym(config: ProductConfig, as_of_ym: str) -> str:
    """Derive a mature snapshot month from the product's label horizon."""
    try:
        as_of_date = datetime.strptime(as_of_ym, "%Y%m")
    except ValueError as error:
        raise ValueError(f"Invalid as-of YYYYMM value: {as_of_ym!r}") from error
    return (as_of_date - relativedelta(months=config.training_schedule.label_offset_months)).strftime("%Y%m")


def refresh_population(
    ym: str,
    *,
    replace: bool = False,
    database: OracleDB | None = None,
) -> PopulationRefreshResult:
    """Run the population insert SQL for one month."""
    db = database or get_query_database()
    population_table, _ = tables_source_name()
    population_schema, population_table_name = population_table.split(".", maxsplit=1)
    required_privileges = ("INSERT", "DELETE") if replace else ("INSERT",)
    db.require_table_privileges(population_schema, population_table_name, required_privileges)
    insert_sql = (
        read_sql("refresh_population_table.sql")
        .replace("{{POPULATION_TABLE}}", population_table)
        .rstrip()
        .removesuffix(";")
    )
    replaced_count = 0

    with db.transaction() as connection:
        if replace:
            replaced_count = db.execute(
                f"DELETE FROM {population_table} WHERE YYYYMM = :ym",
                {"ym": ym},
                connection=connection,
            )
        inserted_count = db.execute(insert_sql, {"ym": ym}, connection=connection)

    return PopulationRefreshResult(ym, inserted_count, replaced_count)
