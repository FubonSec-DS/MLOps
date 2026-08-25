"""Shared Oracle client used by the DS_MASK runtime role."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import oracledb
import polars as pl

from configs.configs_val.DB_configs import DatabaseCredentials, get_settings


def read_sql(filename: str) -> str:
    """Read one SQL file from the application SQL directory."""
    return files("src.resources.queries").joinpath("sql", filename).read_text(encoding="utf-8").strip()


class OracleDB:
    """Small Oracle wrapper with explicit transaction support."""

    def __init__(self, credentials: DatabaseCredentials, instant_client_path: Path) -> None:
        """Store one role's credentials and initialize the Oracle thick client."""
        self.connection_configs = {
            "user": credentials.user,
            "password": credentials.password,
            "dsn": credentials.dsn,
        }
        if oracledb.is_thin_mode():
            oracledb.init_oracle_client(lib_dir=str(instant_client_path))

    def connect(self) -> oracledb.Connection:
        """Open a connection owned by the caller."""
        return oracledb.connect(**self.connection_configs)

    @contextmanager
    def connection(self) -> Iterator[oracledb.Connection]:
        """Yield a connection and always close it."""
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[oracledb.Connection]:
        """Commit all DML together or roll it all back."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        connection: oracledb.Connection | None = None,
    ) -> pl.DataFrame:
        """Execute a SELECT and return a Polars DataFrame."""
        if connection is None:
            with self.connection() as owned_connection:
                return self.query(query, params, connection=owned_connection)
        return pl.read_database(
            query,
            connection,
            infer_schema_length=None,
            execute_options={"parameters": params} if params else None,
        )

    def require_table_privileges(
        self,
        table_schema: str,
        table_name: str,
        privileges: Sequence[str],
    ) -> None:
        """Fail before mutation when the current identity lacks table privileges."""
        required = {privilege.upper() for privilege in privileges}
        result = self.query(
            """
            SELECT DISTINCT PRIVILEGE
            FROM ALL_TAB_PRIVS
            WHERE TABLE_SCHEMA = :table_schema
              AND TABLE_NAME = :table_name
              AND GRANTEE IN (
                  SELECT USER FROM DUAL
                  UNION ALL
                  SELECT ROLE FROM SESSION_ROLES
              )
            """,
            {
                "table_schema": table_schema.upper(),
                "table_name": table_name.upper(),
            },
        )
        granted = {str(value).upper() for value in result.get_column("PRIVILEGE").to_list()}
        missing = sorted(required - granted)
        if missing:
            privilege_list = ", ".join(missing)
            qualified_table = f"{table_schema.upper()}.{table_name.upper()}"
            raise PermissionError(
                f"Database identity lacks {privilege_list} on {qualified_table}; "
                f"ask the table owner/DBA to GRANT {privilege_list} ON {qualified_table}"
            )

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        connection: oracledb.Connection | None = None,
    ) -> int:
        """Execute one DML/DDL statement and return its affected row count."""
        if connection is None:
            with self.transaction() as connection:
                return self.execute(query, params, connection=connection)
        with connection.cursor() as cursor:
            cursor.execute(query, params or {})
            return cursor.rowcount

    def insert(
        self,
        query: str,
        rows: Sequence[dict[str, Any]],
        *,
        connection: oracledb.Connection | None = None,
    ) -> int:
        """Execute a named-bind INSERT in batches."""
        if not rows:
            return 0
        if connection is None:
            with self.transaction() as connection:
                return self.insert(query, rows, connection=connection)
        with connection.cursor() as cursor:
            for start in range(0, len(rows), 500_000):
                cursor.executemany(query, rows[start : start + 500_000])
            return len(rows)


@cache
def get_query_database() -> OracleDB:
    """Return the shared DS_MASK client used for all runtime reads and writes."""
    settings = get_settings()
    return OracleDB(settings.oracle.query, settings.oracle.instant_client_path)
