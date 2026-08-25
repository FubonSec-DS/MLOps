"""Tests for append-only training-population snapshot preparation."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, patch

import polars as pl

from configs.configs_val.product_config import get_product_config
from src.common.database import OracleDB
from src.resources.sampling import (
    ensure_training_population,
    ensure_training_populations,
    required_sampling_months,
)


class _FakeDatabase:
    def __init__(self, statuses: list[tuple[int, int, int]]) -> None:
        self.statuses = statuses
        self.query_calls: list[tuple[str, dict | None, object]] = []
        self.execute_calls: list[tuple[str, dict | None, object]] = []
        self.privilege_calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.connection_token = object()

    def require_table_privileges(
        self,
        table_schema: str,
        table_name: str,
        privileges: tuple[str, ...],
    ) -> None:
        self.privilege_calls.append((table_schema, table_name, privileges))

    @contextmanager
    def transaction(self) -> Iterator[object]:
        yield self.connection_token

    def query(
        self,
        sql: str,
        params: dict[str, object] | None = None,
        *,
        connection: object | None = None,
    ) -> pl.DataFrame:
        self.query_calls.append((sql, params, connection))
        row_count, y0_count, y1_count = self.statuses.pop(0)
        return pl.DataFrame(
            {"ROW_COUNT": [row_count], "Y0_COUNT": [y0_count], "Y1_COUNT": [y1_count]}
        )

    def execute(
        self,
        sql: str,
        params: dict[str, object] | None = None,
        *,
        connection: object | None = None,
    ) -> int:
        self.execute_calls.append((sql, params, connection))
        return 3


class SamplingBatchTests(TestCase):
    """Protect append-only snapshot reuse and insertion behavior."""

    def test_existing_combined_population_is_reused_without_dml(self) -> None:
        config = get_product_config("流失預警")
        database = _FakeDatabase([(10, 8, 2)])

        result = ensure_training_population(
            config,
            "不分潛客",
            "202506",
            database=cast("OracleDB", database),
        )

        self.assertEqual(result.action, "reused")
        self.assertEqual(result.existing_count, 10)
        self.assertEqual(database.execute_calls, [])
        _sql, params, _connection = database.query_calls[0]
        self.assertIsNotNone(params)
        self.assertEqual(cast("dict[str, object]", params)["source_segment_0"], "不分潛客")
        self.assertNotIn("DELETE", {item for call in database.privilege_calls for item in call[2]})

    @patch("src.resources.sampling.build_training_sampling_insert_query")
    def test_missing_snapshot_is_inserted_and_validated_in_same_transaction(self, build_query: MagicMock) -> None:
        config = get_product_config("流失預警")
        database = _FakeDatabase([(0, 0, 0), (3, 2, 1)])
        build_query.return_value = ("INSERT churn", {"ym": "202506"})

        result = ensure_training_population(
            config,
            "不分潛客",
            "202506",
            database=cast("OracleDB", database),
        )

        self.assertEqual(result.action, "inserted")
        self.assertEqual(result.sampled_count, 3)
        self.assertEqual(len(database.execute_calls), 1)
        self.assertIs(database.execute_calls[0][2], database.connection_token)
        self.assertIs(database.query_calls[-1][2], database.connection_token)
        self.assertNotIn("DELETE", database.execute_calls[0][0].upper())

    def test_existing_single_class_snapshot_is_left_unchanged_and_rejected(self) -> None:
        config = get_product_config("流失預警")
        database = _FakeDatabase([(10, 10, 0)])

        with self.assertRaisesRegex(ValueError, "Append-only policy left existing rows unchanged"):
            ensure_training_population(
                config,
                "不分潛客",
                "202506",
                database=cast("OracleDB", database),
            )

        self.assertEqual(database.execute_calls, [])

    def test_latest_month_drives_each_product_schedule(self) -> None:
        expected = {
            "不限用途": (
                "202412",
                "202501",
                "202502",
                "202503",
                "202504",
                "202505",
                "202506",
                "202507",
                "202508",
                "202509",
                "202512",
            ),
            "客群上送": ("202412", "202503", "202506", "202509", "202512"),
            "流失預警": ("202403", "202406", "202409", "202412", "202503"),
        }

        for product, months in expected.items():
            with self.subTest(product=product):
                self.assertEqual(
                    required_sampling_months(get_product_config(product), "202603"),
                    months,
                )

    @patch("src.resources.sampling.build_training_sampling_insert_query")
    def test_batch_appends_missing_snapshots_without_delete(self, build_query: MagicMock) -> None:
        config = get_product_config("不限用途")
        expected_snapshots = len(required_sampling_months(config, "202606")) * len(config.populations)
        database = _FakeDatabase(
            [(0, 0, 0)] * expected_snapshots + [(3, 2, 1)] * expected_snapshots
        )
        build_query.side_effect = lambda product, population, ym: (
            f"INSERT {product.product} {population} {ym}",
            {"ym": ym},
        )

        result = ensure_training_populations(
            "202606",
            configs=[config],
            database=cast("OracleDB", database),
        )

        self.assertEqual(len(database.execute_calls), expected_snapshots)
        self.assertTrue(all(call[2] is database.connection_token for call in database.execute_calls))
        self.assertTrue(all("DELETE" not in call[0].upper() for call in database.execute_calls))
        self.assertEqual(result.inserted_snapshots, expected_snapshots)
        self.assertEqual(result.reused_snapshots, 0)
        self.assertEqual(result.inserted_count, expected_snapshots * 3)
        self.assertEqual(result.existing_count, 0)

    def test_batch_reuses_complete_snapshots_without_dml(self) -> None:
        config = get_product_config("流失預警")
        expected_snapshots = len(required_sampling_months(config, "202603"))
        database = _FakeDatabase([(100, 80, 20)] * expected_snapshots)

        result = ensure_training_populations(
            "202603",
            configs=[config],
            database=cast("OracleDB", database),
        )

        self.assertEqual(database.execute_calls, [])
        self.assertEqual(result.inserted_snapshots, 0)
        self.assertEqual(result.reused_snapshots, expected_snapshots)
        self.assertEqual(result.existing_count, expected_snapshots * 100)

    def test_invalid_prediction_month_is_rejected_before_database_access(self) -> None:
        database = _FakeDatabase([])

        with self.assertRaisesRegex(ValueError, "Invalid prediction YYYYMM"):
            ensure_training_populations(
                "202613",
                configs=[get_product_config("不限用途")],
                database=cast("OracleDB", database),
            )

        self.assertEqual(database.query_calls, [])
        self.assertEqual(database.execute_calls, [])
