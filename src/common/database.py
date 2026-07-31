from pathlib import Path

import oracledb
import pandas as pd

from src.common.config import get_settings


class OracleDB:
    """A class to handle Oracle database operations using oracledb and pandas."""

    def __init__(self, user: str, password: str, dsn: str, instant_client_path: Path) -> None:
        """Initialize OracleDB.

        Args:
            user (str): The username for the Oracle database.
            password (str): The password for the Oracle database.
            dsn (str): The Data Source Name for the Oracle database, including host, port, and service name.
            instant_client_path (Path): Path to the Oracle Instant Client directory.

        """
        # Connection parameters
        self.connection_configs = {"user": user, "password": password, "dsn": dsn}
        self.batch_size = 500000

        # Only initialize the Oracle client if in thin mode
        if oracledb.is_thin_mode():
            oracledb.init_oracle_client(lib_dir=str(instant_client_path))

    def query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a query and return the result as a Pandas DataFrame.

        Args:
            query (str): The SQL query to fetch data from the database.
            params (dict | None): Optional parameters for the query, e.g., {"param1": value1, "param2": value2}.

        Returns:
            pd.DataFrame: The result of the query as a Pandas DataFrame.

        """
        conn = oracledb.connect(**self.connection_configs)
        cursor = conn.cursor()

        try:
            cursor.execute(query, params or {})
            description = cursor.description

            if description is None:
                return pd.DataFrame()

            else:
                columns = [column[0] for column in description]
                rows = cursor.fetchall()

                return pd.DataFrame.from_records(rows, columns=columns)

        finally:
            cursor.close()
            conn.close()

    def execute(self, query: str, params: dict | None = None) -> None:
        """Execute a query without returning any result, like creating or dropping tables.

        Args:
            query (str): The SQL query to execute.
            params (dict | None): Optional parameters for the query, e.g., {"param1": value1, "param2": value2}.

        """
        conn = oracledb.connect(**self.connection_configs)
        cursor = conn.cursor()

        try:
            cursor.execute(query, params or {})
            conn.commit()

        finally:
            cursor.close()
            conn.close()

    def insert(self, query: str, rows: list[dict]) -> int:
        """Insert rows into database using ``executemany`` in batches.

        Args:
            query (str): SQL insert statement with named bind variables.
            rows (list[dict]): Data to insert, e.g., [{"col1": val1, "col2": val2}, ...].

        Returns:
            int: Total number of inserted rows.

        Raises:
            Exception: If any error occurs during the insertion, the transaction is rolled back and the exception

        """
        conn = oracledb.connect(**self.connection_configs)
        cursor = conn.cursor()

        try:
            for i in range(0, total_num := len(rows), self.batch_size):
                cursor.executemany(query, rows[i : i + self.batch_size])

            conn.commit()

            return total_num

        except Exception:
            conn.rollback()
            raise

        finally:
            cursor.close()
            conn.close()


def get_oracle_db() -> OracleDB:
    """Get an instance of OracleDB with configuration from environment variables.

    Returns:
        OracleDB: An instance of the OracleDB class.

    """
    settings = get_settings()

    return OracleDB(
        user=settings.oracle.user,
        password=settings.oracle.password.get_secret_value(),
        dsn=settings.oracle.dsn,
        instant_client_path=settings.oracle.instant_client_path,
    )
