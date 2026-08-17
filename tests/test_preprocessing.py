"""Tests for persisted categorical and missing-value semantics."""

import unittest

import pandas as pd

from src.models.preprocessing import (
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY,
    apply_preprocessing,
    fit_preprocessing,
)
from src.resources.feature_loader import _categorical_features
from src.resources.queries.feature_queries import FeatureColumn


class PreprocessingTests(unittest.TestCase):
    def test_missing_and_unknown_categories_are_distinct(self) -> None:
        train = pd.DataFrame({"cf_profile__segment": ["A", None], "cf_asset1__value": [1.0, None]})
        transformed, schema = fit_preprocessing(
            train,
            ["cf_profile__segment", "cf_asset1__value"],
            ["cf_profile__segment"],
        )
        self.assertIn(MISSING_CATEGORY, schema.categories["cf_profile__segment"])
        self.assertTrue(pd.isna(transformed.loc[1, "cf_asset1__value"]))

        prediction = pd.DataFrame({"cf_profile__segment": ["NEW", None], "cf_asset1__value": [2.0, 3.0]})
        output = apply_preprocessing(
            prediction,
            ["cf_profile__segment", "cf_asset1__value"],
            schema,
        )
        self.assertEqual(str(output.loc[0, "cf_profile__segment"]), UNKNOWN_CATEGORY)
        self.assertEqual(str(output.loc[1, "cf_profile__segment"]), MISSING_CATEGORY)

    def test_oracle_text_columns_are_always_categorical(self) -> None:
        columns = [FeatureColumn("CF_DGT2", "DGT_STCQT_FQTPS_M", "VARCHAR2")]
        self.assertEqual(_categorical_features(columns), ["cf_dgt2__dgt_stcqt_fqtps_m"])


if __name__ == "__main__":
    unittest.main()
