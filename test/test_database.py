import oracledb
import pytest

from src.common.database import get_oracle_db

db = get_oracle_db()


def test_01_create_table() -> None:
    """Create the test table."""
    db.execute("CREATE TABLE TEST_ (id NUMBER PRIMARY KEY, name VARCHAR2(32))")


def test_02_insert_rows() -> None:
    """Insert rows into the test table."""
    output = db.insert(
        "INSERT INTO TEST_ (id, name) VALUES (:id, :name)",
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
    )

    assert output == 2


def test_03_select_inserted_rows() -> None:
    """Select rows inserted into the test table."""
    result = db.query("SELECT id, name FROM TEST_ ORDER BY id")

    assert result.shape == (2, 2)


def test_04_drop_table() -> None:
    """Drop the test table."""
    db.execute("DROP TABLE TEST_ PURGE")

    with pytest.raises(oracledb.DatabaseError):
        db.query("SELECT * FROM TEST_")


def test_05_select_large_table() -> None:
    """Select rows from a large table to test performance and memory usage."""
    result = db.query("SELECT * FROM DM_S_VIEW.M_AC_ACCOUNT WHERE ROWNUM <= 10000")

    assert result.shape[0] == 10000
