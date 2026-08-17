"""Tests for full training-population batch replacement."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, patch

from configs.configs_val.product_config import get_product_config
from src.common.database import OracleDB
from src.resources.sampling import (
    rebuild_training_populations,
    refresh_training_population,
    required_sampling_months,
)


class _FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None, object]] = []
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
        return 7 if sql.startswith("DELETE") else 3


class SamplingBatchTests(TestCase):
    @patch("src.resources.sampling.build_training_sampling_insert_query")
    def test_combined_population_refresh_deletes_its_stored_segment(self, build_query: MagicMock) -> None:
        config = get_product_config("流失預警")
        database = _FakeDatabase()
        build_query.return_value = ("INSERT churn", {"ym": "202506"})

        refresh_training_population(
            config,
            "不分潛客",
            "202506",
            database=cast("OracleDB", database),
        )

        delete_sql, delete_params, _connection = database.calls[0]
        self.assertIn("SOURCE_SEGMENT IN (:source_segment_0)", delete_sql)
        self.assertIsNotNone(delete_params)
        typed_params = cast("dict[str, object]", delete_params)
        self.assertEqual(typed_params["source_segment_0"], "不分潛客")

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
    def test_rebuild_deletes_once_then_appends_every_month_and_population(self, build_query: MagicMock) -> None:
        config = get_product_config("不限用途")
        database = _FakeDatabase()
        build_query.side_effect = lambda product, population, ym: (
            f"INSERT {product.product} {population} {ym}",
            {"ym": ym},
        )

        result = rebuild_training_populations(
            "202606",
            configs=[config],
            database=cast("OracleDB", database),
        )

        expected_snapshots = len(required_sampling_months(config, "202606")) * len(config.populations)
        self.assertEqual(database.calls[0][0], "DELETE FROM S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST")
        self.assertEqual(len(database.calls), expected_snapshots + 1)
        self.assertTrue(all(call[2] is database.connection_token for call in database.calls))
        self.assertEqual(len(result.snapshots), expected_snapshots)
        self.assertEqual(result.replaced_count, 7)
        self.assertEqual(result.sampled_count, expected_snapshots * 3)

    def test_invalid_prediction_month_is_rejected_before_delete(self) -> None:
        database = _FakeDatabase()

        with self.assertRaisesRegex(ValueError, "Invalid prediction YYYYMM"):
            rebuild_training_populations(
                "202613",
                configs=[get_product_config("不限用途")],
                database=cast("OracleDB", database),
            )

        self.assertEqual(database.calls, [])
