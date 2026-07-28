from src.common.config import Settings, get_settings
from src.common.database import OracleDB


def main() -> None:  # noqa: D103
    settings: Settings = get_settings()

    db = OracleDB(
        user=settings.oracle.user,
        password=settings.oracle.password.get_secret_value(),
        dsn=settings.oracle.dsn,
        instant_client_path=settings.oracle.instant_client_path,
    )

    print("OracleDB instance created successfully.")
    print("\n" + "===" * 10 + "\n")

    # Create table
    db.execute(
        """
        CREATE TABLE test_table (
            id NUMBER PRIMARY KEY,
            name VARCHAR2(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    print("Table created successfully.")
    print("\n" + "===" * 10 + "\n")

    # Insert data (named bind variables)
    insert_sql = "INSERT INTO test_table (id, name) VALUES (:id, :name)"
    rows = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
    total_inserted = db.insert(insert_sql, rows)

    print(f"Data inserted successfully. Total rows inserted: {total_inserted}")
    print("\n" + "===" * 10 + "\n")

    # Query data without filter (returns polars.DataFrame)
    df = db.query("SELECT id, name, created_at FROM test_table ORDER BY id")

    print("Query executed successfully without filter (polars.DataFrame).")
    print(df)
    print("\n" + "===" * 10 + "\n")

    # Query data with filter (returns polars.DataFrame)
    df = db.query(
        "SELECT id, name, created_at FROM test_table WHERE id > :min_id ORDER BY id",
        params={"min_id": 1},
    )

    print("Query executed successfully with filter (polars.DataFrame).")
    print(df)
    print("\n" + "===" * 10 + "\n")

    # Drop table
    db.execute("DROP TABLE test_table")

    print("Table dropped successfully.")
    print("\n" + "===" * 10 + "\n")


if __name__ == "__main__":
    main()
