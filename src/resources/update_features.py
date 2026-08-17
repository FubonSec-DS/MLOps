"""Run the feature SQL files."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from src.common.database import OracleDB, get_query_database, read_sql

FEATURE_SQL_FILES = {
    "txn_1": "insert_txn.sql",
    "txn_2": "insert_txn2.sql",
    "profolio": "insert_profolio.sql",
    "bnf": "insert_bnf.sql",
    "dgt": "insert_dgt.sql",
    "interact": "insert_interact.sql",
}


@dataclass(frozen=True)
class FeatureETLResult:
    """Feature SQL execution result."""

    group: str
    statement_count: int
    deleted_count: int
    elapsed_seconds: float = 0.0


def _statement_label(statement: str) -> str:
    """Return the primary SQL operation and object for progress output."""
    match = re.search(
        r"\b(DELETE\s+FROM|INSERT\s+INTO|CREATE\s+TABLE|DROP\s+TABLE|MERGE\s+INTO)\s+([^\s(]+)",
        statement,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"
    first_line = next((line.strip() for line in statement.splitlines() if line.strip()), "UNKNOWN")
    return first_line[:120]


def read_sql_statements(filename: str) -> list[str]:
    """Read the statements in one SQL file."""
    statements = [statement.strip() for statement in read_sql(filename).split(";") if statement.strip()]
    return [statement for statement in statements if ":ym" in statement or not statement.startswith(("--", "/"))]


def run_feature_group(
    group: str,
    ym: str,
    *,
    dry_run: bool = False,
    database: OracleDB | None = None,
    progress: Callable[[str], None] | None = None,
) -> FeatureETLResult:
    """Run one feature SQL file."""
    statements = read_sql_statements(FEATURE_SQL_FILES[group])
    if dry_run:
        return FeatureETLResult(group, len(statements), 0)

    db = database or get_query_database()
    deleted_count = 0
    group_started = perf_counter()

    try:
        with db.transaction() as connection:
            for index, statement in enumerate(statements, start=1):
                label = _statement_label(statement)
                if progress:
                    progress(f"[{group} {index}/{len(statements)}] start {label}")
                statement_started = perf_counter()
                params = {"ym": ym} if ":ym" in statement else None
                row_count = db.execute(statement, params or None, connection=connection)
                elapsed = perf_counter() - statement_started
                if statement.lstrip().upper().startswith("DELETE "):
                    deleted_count += row_count
                if progress:
                    progress(f"[{group} {index}/{len(statements)}] done rows={row_count:,} elapsed={elapsed:.1f}s")
    except Exception:
        if progress:
            progress(f"[{group}] failed after {perf_counter() - group_started:.1f}s; transaction rolled back")
        raise

    return FeatureETLResult(group, len(statements), deleted_count, round(perf_counter() - group_started, 1))


def run_feature_etl(
    ym: str,
    *,
    dry_run: bool = False,
    groups: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[FeatureETLResult]:
    """Run the selected feature SQL files."""
    selected = groups or list(FEATURE_SQL_FILES)
    return [run_feature_group(group, ym, dry_run=dry_run, progress=progress) for group in selected]
