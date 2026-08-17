"""Tests for SQL-first feature and population workflows."""

import unittest

from src.common.database import read_sql
from src.resources.update_features import FEATURE_SQL_FILES, read_sql_statements


class SQLWorkflowTests(unittest.TestCase):
    """Ensure runtime SQL uses Oracle bind variables instead of Python rendering."""

    def test_feature_sql_uses_ym_bind(self) -> None:
        for filename in FEATURE_SQL_FILES.values():
            statements = read_sql_statements(filename)
            self.assertTrue(statements)
            for statement in statements:
                if not statement.upper().startswith("DROP TABLE"):
                    self.assertIn(":ym", statement)
                self.assertNotIn("DEFINE ", statement.upper())
                self.assertNotRegex(statement, r"&[A-Za-z0-9_]+")

    def test_population_sql_uses_ym_bind(self) -> None:
        sql = read_sql("refresh_population_table.sql")
        self.assertIn("INSERT INTO {{POPULATION_TABLE}}", sql)
        self.assertIn(":ym", sql)
        self.assertNotIn("DEFINE ", sql.upper())
        self.assertNotRegex(sql, r"&[A-Za-z0-9_]+")

    def test_schema_setup_is_stored_in_sql(self) -> None:
        sql = read_sql("setup_population_table.sql")
        self.assertIn("Schema-definition reference only", sql)
        self.assertIn("CREATE TABLE MLOPS_POPULATION", sql)
        self.assertIn("CREATE TABLE MLOPS_POPULATION_SAMPLING", sql)
        self.assertIn("SOURCE_SEGMENT VARCHAR2(32) NOT NULL", sql)
        self.assertIn("Y NUMBER(1) NOT NULL", sql)
        self.assertIn("ELIGIBILITY_TAG NUMBER(1) NOT NULL", sql)
        self.assertNotIn("POPULATION VARCHAR2(32) NOT NULL", sql)
        self.assertIn("GRANT SELECT, INSERT, DELETE ON MLOPS_POPULATION", sql)

    def test_isolated_sampling_test_table_uses_snapshot_contract(self) -> None:
        sql = read_sql("create_population_sampling_test.sql")
        self.assertIn("CREATE TABLE MLOPS_POPULATION_SAMPLING_TEST", sql)
        self.assertIn("SOURCE_SEGMENT VARCHAR2(32) NOT NULL", sql)
        self.assertIn("ELIGIBILITY_TAG NUMBER(1) NOT NULL", sql)
        self.assertIn("'不分潛客'", sql)
        self.assertIn("USING INDEX COMPRESS 3", sql)
        self.assertIn("GRANT SELECT, INSERT, DELETE", sql)
