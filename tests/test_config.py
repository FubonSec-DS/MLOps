"""Tests for YAML-backed, validated application configuration."""

import os
import unittest
from datetime import date
from unittest.mock import patch

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import PRODUCT_CONFIGS, calculate_training_periods, get_product_config
from configs.configs_val.script_config import load_script_settings, resolve_project_path


class ConfigTests(unittest.TestCase):
    """Protect the boundary between YAML values, secrets, and Python logic."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = patch.dict(
            os.environ,
            {
                "DB_USER": "query-user",
                "DB_PASSWORD": "query-secret",
                "DB_DSN": "example/service",
            },
        )
        cls.environment.start()
        get_settings.cache_clear()

    @classmethod
    def tearDownClass(cls) -> None:
        get_settings.cache_clear()
        cls.environment.stop()

    def test_yaml_paths_are_resolved_from_project_root(self) -> None:
        settings = get_settings()
        self.assertTrue(settings.paths.fin_table.is_absolute())
        self.assertEqual(settings.paths.fin_table.name, "df_fin_feats.csv")
        self.assertEqual(settings.paths.model_base.name, "model_store")

    def test_categorical_names_are_derived_in_python(self) -> None:
        features = get_settings().features.categorical_features
        self.assertIn("cf_profile__gender_code", features)
        self.assertIn("cf_kycqa__kycqa_q98", features)

    def test_product_tuning_is_loaded_from_yaml(self) -> None:
        config = get_product_config("流失預警")
        self.assertIsNone(config.training_params.early_stopping_rounds)
        self.assertEqual(config.training_schedule.label_offset_months, 12)
        self.assertEqual(config.training_schedule.backtest_gap_months, 3)
        self.assertEqual(config.training_schedule.train_interval_months, 3)
        self.assertEqual(config.bins, (1.0, 0.9, 0.75, 0.6, 0.4, 0.0))
        self.assertEqual(len(PRODUCT_CONFIGS), 24)

    def test_script_inputs_are_loaded_from_yaml(self) -> None:
        scripts = load_script_settings()

        self.assertEqual(scripts.refresh_population.as_of_ym, "202606")
        self.assertEqual(scripts.refresh_population.product, "流失預警")
        self.assertEqual(scripts.refresh_training_population.ym, "202607")
        self.assertIsNotNone(scripts.refresh_training_population.product)
        self.assertIn("海外股流失預警", scripts.refresh_training_population.product or ())
        self.assertFalse(scripts.run_retrain.write_db)
        self.assertEqual(scripts.run_retrain.product, "海外股流失預警")
        self.assertEqual(scripts.run_retrain.population, "不分潛客")
        self.assertIsNone(scripts.run_retrain.train_yms)
        self.assertEqual(scripts.run_retrain.rows_per_month, 2_500)
        self.assertEqual(resolve_project_path(scripts.run_pretrain.output_dir).name, "pretrain_reports")

    def test_standard_two_stage_periods_use_latest_complete_month(self) -> None:
        train_yms, backtest_ym, as_of_ym = calculate_training_periods(
            get_product_config("不限用途"),
            date(2026, 7, 1),
        )
        self.assertEqual(train_yms, [f"2025{month:02d}" for month in range(4, 13)] + ["202601"])
        self.assertEqual(backtest_ym, "202604")
        self.assertEqual(as_of_ym, "202607")

    def test_runtime_uses_one_database_identity(self) -> None:
        settings = get_settings()

        self.assertEqual(settings.oracle.query.user, "query-user")
        self.assertFalse(hasattr(settings.oracle, "population"))
        self.assertEqual(settings.oracle.population_sampling_schema, "S_IANLEONG")
        self.assertEqual(settings.oracle.population_sampling_table, "MLOPS_POPULATION_SAMPLING_TEST")
