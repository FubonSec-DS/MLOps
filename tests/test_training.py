"""Small real-XGBoost smoke test for the supported model path."""

import unittest

import pandas as pd

from configs.configs_val.product_config import get_product_config
from src.pipelines.retrain import _require_binary_target, _validate_periods
from src.models.evaluation import check_retrain_passed
from src.models.preprocessing import fit_preprocessing
from src.models.training import fit_xgboost, select_top_features


class TrainingTests(unittest.TestCase):
    """Exercise categorical CPU training and gain importance."""

    def test_xgboost_categorical_training_and_importance(self) -> None:
        """Fit the exact estimator mode used by pretraining."""
        frame = pd.DataFrame(
            {
                "cf_profile__segment": ["A", "B"] * 20,
                "cf_asset1__value": list(range(40)),
                "y": [0, 1] * 20,
            }
        )
        features = ["cf_profile__segment", "cf_asset1__value"]
        model_frame, _ = fit_preprocessing(frame, features, ["cf_profile__segment"])
        target = pd.Series(frame["y"])
        model = fit_xgboost(
            model_frame,
            target,
            get_product_config("流失預警").pretrain_params.model_copy(update={"n_estimators": 5}),
            device="cpu",
            random_state=42,
        )
        selected, importance = select_top_features(model, features, top_n=2)
        self.assertEqual(set(selected), set(features))
        self.assertEqual(len(importance), 2)
        self.assertEqual(len(model.predict_proba(model_frame)), len(frame))

    def test_xgboost_input_requires_both_target_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain y=0 and y=1"):
            _require_binary_target(pd.Series([1, 1, 1]), "Pretrain cohort")

    def test_two_stage_periods_require_backtest_after_training(self) -> None:
        _validate_periods(["202504", "202505", "202601"], "202604")
        with self.assertRaisesRegex(ValueError, "chronological order"):
            _validate_periods(["202504", "202601"], "202512")

    def test_retrain_acceptance_uses_oot_backtest_only(self) -> None:
        self.assertTrue(check_retrain_passed(0.6))
        self.assertFalse(check_retrain_passed(0.599))


if __name__ == "__main__":
    unittest.main()
