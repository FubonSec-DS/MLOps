"""Tests for deterministic, bind-based Oracle SQL generation."""

import os
import unittest
from dataclasses import replace

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import PRODUCT_CONFIGS, PopulationConfig, get_product_config
from src.common.database import read_sql
from src.resources.queries.feature_queries import (
    FeatureColumn,
    build_feature_query,
    build_prediction_feature_query,
)
from src.resources.queries.population_queries import (
    build_training_sampling_insert_query,
)


class QueryBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.update(
            {
                "DB_USER": "DS_MASK",
                "DB_PASSWORD": "query-secret",
                "DB_DSN": "example/service",
            }
        )
        get_settings.cache_clear()

    def test_training_sampling_is_deterministic_and_bounded(self) -> None:
        config = get_product_config("不限用途")
        sql, params = build_training_sampling_insert_query(config, "非潛客", "202503")
        self.assertIn("ORA_HASH", sql)
        self.assertIn("ROW_NUMBER() OVER", sql)
        self.assertIn("GROUP BY CUSTOMER_ID, YYYYMM, SEGMENT", sql)
        self.assertIn("/*+ PARALLEL(P 4) */", sql)
        self.assertIn('MAX(NVL("不限用途Y", :target_null_value)) AS Y', sql)
        self.assertIn("SAMPLE_RN <= :sample_size", sql)
        self.assertIn("SEGMENT IN (:source_segment_0)", sql)
        self.assertNotIn("SEGMENT = :population", sql)
        self.assertIn(":ym", sql)
        self.assertNotIn("202503", sql)
        self.assertEqual(params["product"], "不限用途")
        self.assertNotIn("population", params)
        self.assertEqual(params["source_segment_0"], "非潛客")
        self.assertIn("SOURCE_SEGMENT, Y, ELIGIBILITY_TAG", sql)
        self.assertIn('MAX(NVL("不限用途Y", :target_null_value)) AS Y', sql)
        self.assertIn('NVL("不限用途近一年舊戶", :eligibility_null_value)', sql)

    def test_all_legacy_products_are_configured(self) -> None:
        expected = {
            "不限用途",
            "保險商品",
            "債券型基金",
            "儲蓄型保險商品",
            "台股信用交易",
            "台股定期定額",
            "海外股票定期定額",
            "基金",
            "基金定期定額",
            "境內結構型",
            "境外結構型",
            "平衡型基金",
            "投資型保險商品",
            "期貨",
            "海外債",
            "海外股票",
            "結構型商品",
            "股票型基金",
            "雙向借券",
            "財管商品",
            "客群上送",
            "潛在高價值客戶",
            "流失預警",
            "海外股流失預警",
        }
        self.assertEqual(set(PRODUCT_CONFIGS), expected)

    def test_two_churn_products_keep_distinct_names_and_source_columns(self) -> None:
        domestic = get_product_config("流失預警")
        overseas = get_product_config("海外股流失預警")
        self.assertNotEqual(domestic.product, overseas.product)
        self.assertEqual(overseas.eligibility_column, "海外股流失預警近一年實動")
        self.assertEqual(overseas.target_column, "海外股流失預警Y")
        self.assertNotEqual(domestic.eligibility_column, overseas.eligibility_column)
        self.assertNotEqual(domestic.target_column, overseas.target_column)

    def test_structure_sampling_excludes_98_households(self) -> None:
        config = get_product_config("境內結構型")
        sql, _ = build_training_sampling_insert_query(config, "非潛客", "202503")
        self.assertIn('"98戶" IS NULL', sql)

    def test_special_3m_uses_r_for_sampling_and_p_for_prediction(self) -> None:
        config = get_product_config("客群上送")
        sampling_sql, _ = build_training_sampling_insert_query(config, "不分潛客", "202503")
        prediction_sql, _ = build_prediction_feature_query(
            config,
            "202604",
            "不分潛客",
            [FeatureColumn("CF_PROFILE", "GENDER_CODE", "VARCHAR2")],
        )
        self.assertIn("客群上送前季高交易量客戶R", sampling_sql)
        self.assertIn("客群上送前季高交易量客戶P", prediction_sql)
        self.assertNotIn("SEGMENT IN", sampling_sql)

    def test_special_products_store_combined_population_segment(self) -> None:
        for product in ("客群上送", "潛在高價值客戶", "流失預警"):
            with self.subTest(product=product):
                config = get_product_config(product)
                sampling_sql, sampling_params = build_training_sampling_insert_query(config, "不分潛客", "202503")
                feature_sql, feature_params = build_feature_query(
                    config,
                    "202503",
                    "不分潛客",
                    [FeatureColumn("CF_PROFILE", "GENDER_CODE", "VARCHAR2")],
                    pretrain=False,
                )

                self.assertIn(":population AS SOURCE_SEGMENT", sampling_sql)
                self.assertIn("GROUP BY CUSTOMER_ID, YYYYMM\n", sampling_sql)
                self.assertNotIn("GROUP BY CUSTOMER_ID, YYYYMM, SEGMENT", sampling_sql)
                self.assertEqual(sampling_params["population"], "不分潛客")
                self.assertIn("SOURCE_SEGMENT IN (:source_segment_0)", feature_sql)
                self.assertEqual(feature_params["source_segment_0"], "不分潛客")

    def test_subpopulation_maps_to_one_source_segment(self) -> None:
        config = get_product_config("不限用途")
        prospect_config = replace(
            config,
            populations=(
                PopulationConfig(
                    population="潛客",
                    sample_size=150_000,
                    source_segments=("潛客",),
                ),
            ),
        )
        sql, params = build_training_sampling_insert_query(prospect_config, "潛客", "202503")
        self.assertIn("SEGMENT IN (:source_segment_0)", sql)
        self.assertEqual(params["source_segment_0"], "潛客")

    def test_prediction_uses_the_same_population_segment_mapping(self) -> None:
        config = get_product_config("不限用途")
        sql, params = build_prediction_feature_query(
            config,
            "202603",
            "非潛客",
            [FeatureColumn("CF_PROFILE", "GENDER_CODE", "VARCHAR2")],
        )
        self.assertIn("SEGMENT IN (:source_segment_0)", sql)
        self.assertNotIn("SEGMENT = :population", sql)
        self.assertEqual(params["source_segment_0"], "非潛客")

    def test_population_read_source_uses_the_configured_schema(self) -> None:
        config = get_product_config("流失預警")
        sql, _ = build_prediction_feature_query(
            config,
            "202604",
            "不分潛客",
            [FeatureColumn("CF_PROFILE", "GENDER_CODE", "VARCHAR2")],
        )
        self.assertIn("FROM S_IANLEONG.MLOPS_POPULATION", sql)

    def test_pretrain_keeps_all_positives_then_fills_with_negatives(self) -> None:
        config = get_product_config("流失預警")
        sql, _ = build_feature_query(
            config,
            "202503",
            "不分潛客",
            [FeatureColumn("CF_PROFILE", "GENDER_CODE", "VARCHAR2")],
            pretrain=True,
        )
        self.assertIn("FROM S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST", sql)
        self.assertNotIn("INNER JOIN S_IANLEONG.MLOPS_POPULATION", sql)
        self.assertIn("WHERE Y = 1", sql)
        self.assertIn("WHERE Y = 0", sql)
        self.assertIn("GREATEST(:pretrain_size", sql)
        self.assertIn("UNION ALL", sql)
        self.assertIn("SOURCE_SEGMENT IN", sql)

    def test_feature_alias_encodes_source_table(self) -> None:
        config = get_product_config("流失預警")
        columns = [
            FeatureColumn("CF_ASSET1", "BALANCE", "NUMBER"),
            FeatureColumn("CF_PROFILE", "BALANCE", "NUMBER"),
        ]
        sql, _ = build_feature_query(config, "202503", "不分潛客", columns, pretrain=True)
        self.assertIn("AS cf_asset1__balance", sql)
        self.assertIn("AS cf_profile__balance", sql)
        self.assertIn("LEFT JOIN DS_SEC.CF_ASSET1", sql)
        self.assertNotIn("PARTITION", sql)
        self.assertNotEqual(columns[0].feature_name, columns[1].feature_name)

    def test_readiness_limit_is_deterministic_and_bound(self) -> None:
        config = get_product_config("流失預警")
        sql, params = build_feature_query(
            config,
            "202506",
            "不分潛客",
            [FeatureColumn("CF_PROFILE", "AGE", "NUMBER")],
            pretrain=True,
            row_limit=1_000,
        )
        self.assertIn("LIMITED_PRETRAIN_COHORT", sql)
        self.assertIn("READINESS_RN <= :row_limit", sql)
        self.assertIn("|readiness", sql)
        self.assertEqual(params["row_limit"], 1_000)

    def test_training_limit_is_deterministic_and_bound(self) -> None:
        config = get_product_config("海外股流失預警")
        sql, params = build_feature_query(
            config,
            "202504",
            "不分潛客",
            [FeatureColumn("CF_PROFILE", "AGE", "NUMBER")],
            pretrain=False,
            row_limit=10_000,
        )
        self.assertIn("LIMITED_TRAINING_COHORT", sql)
        self.assertIn("TRAINING_RN <= :row_limit", sql)
        self.assertIn("|training", sql)
        self.assertEqual(params["row_limit"], 10_000)

    def test_population_sql_uses_the_configured_real_columns(self) -> None:
        """Protect against the old nonexistent eligibility-column suffix."""
        sql = read_sql("refresh_population_table.sql")
        config = get_product_config("流失預警")
        self.assertIn(config.eligibility_column, sql)
        self.assertIn(config.target_column, sql)
        self.assertEqual(config.eligibility_column, "流失預警近一年實動")
        self.assertEqual(config.target_column, "流失預警Y")
        self.assertIn(":ym", sql)
        self.assertNotIn("&pre_", sql)


if __name__ == "__main__":
    unittest.main()
