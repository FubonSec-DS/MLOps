"""Tests for schema-qualified full-population refreshes."""

from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, patch

from configs.configs_val.product_config import get_product_config
from scripts.refresh_population import main as refresh_population_main
from src.common.database import OracleDB
from src.resources.population import population_snapshot_ym, refresh_population


class _FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None, object | None]] = []
        self.connection_token = object()

    def require_table_privileges(
        self,
        table_schema: str,
        table_name: str,
        privileges: tuple[str, ...],
    ) -> None:
        pass

    @contextmanager
    def transaction(self) -> Iterator[object]:
        yield self.connection_token

    def execute(
        self,
        sql: str,
        params: dict[str, object] | None = None,
        *,
        connection: object | None = None,
    ) -> int:
        self.calls.append((sql, params, connection))
        return 2


class PopulationRefreshTests(TestCase):
    @patch("scripts.refresh_population.refresh_population")
    def test_yaml_config_derives_churn_snapshot_and_always_replaces(self, refresh: MagicMock) -> None:
        with redirect_stdout(StringIO()):
            refresh_population_main()

        refresh.assert_called_once_with("202506", replace=True)

    def test_snapshot_month_uses_each_products_label_horizon(self) -> None:
        self.assertEqual(population_snapshot_ym(get_product_config("不限用途"), "202606"), "202603")
        self.assertEqual(population_snapshot_ym(get_product_config("流失預警"), "202606"), "202506")
        self.assertEqual(population_snapshot_ym(get_product_config("海外股流失預警"), "202606"), "202506")

    def test_snapshot_month_rejects_invalid_as_of_month(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid as-of YYYYMM"):
            population_snapshot_ym(get_product_config("流失預警"), "202613")

    def test_refresh_qualifies_insert_and_delete_with_population_schema(self) -> None:
        database = _FakeDatabase()

        result = refresh_population("202503", replace=True, database=cast("OracleDB", database))

        self.assertTrue(database.calls[0][0].startswith("DELETE FROM S_IANLEONG.MLOPS_POPULATION"))
        self.assertTrue(database.calls[1][0].startswith("INSERT INTO S_IANLEONG.MLOPS_POPULATION"))
        self.assertFalse(database.calls[1][0].rstrip().endswith(";"))
        self.assertTrue(all(call[2] is database.connection_token for call in database.calls))
        self.assertEqual(result.replaced_count, 2)
        self.assertEqual(result.inserted_count, 2)
