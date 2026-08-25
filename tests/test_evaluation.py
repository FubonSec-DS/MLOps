"""Tests for common presentation-ready model metrics."""

from pathlib import Path
from typing import Any, cast
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.evaluation import evaluate
from src.models.reporting import _model_card, performance_summary, write_performance_report


class _FixedModel:
    def __init__(self, prediction: np.ndarray) -> None:
        self.prediction = prediction

    def predict_proba(self, _frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack((1 - self.prediction, self.prediction))


class EvaluationTests(TestCase):
    """Protect AUC, KS, hit-rate, recall, lift, and decile outputs."""

    def test_common_business_metrics(self) -> None:
        prediction = np.array([0.99, 0.95, 0.90, 0.85, *np.linspace(0.80, 0.05, 16)])
        target = pd.Series([1, 1, 1, 1, *([0] * 16)])
        frame = pd.DataFrame({"feature": range(20)})

        result = evaluate(
            cast("XGBClassifier", _FixedModel(prediction)),
            frame,
            target,
            bins=(0.0, 0.5, 1.0),
        )

        self.assertEqual(result.auc, 1.0)
        self.assertEqual(result.ks, 1.0)
        self.assertEqual(result.base_rate, 0.2)
        self.assertEqual([row["top_percent"] for row in result.top_rate_metrics], [5.0, 10.0, 20.0])
        self.assertEqual(result.top_rate_metrics[-1]["recall"], 1.0)
        self.assertEqual(result.top_rate_metrics[-1]["lift"], 5.0)
        self.assertEqual(len(result.decile_metrics), 10)

    def test_report_files_are_presentation_ready(self) -> None:
        prediction = np.array([0.9, 0.8, 0.2, 0.1])
        target = pd.Series([1, 1, 0, 0])
        frame = pd.DataFrame({"feature": range(4)})
        result = evaluate(
            cast("XGBClassifier", _FixedModel(prediction)),
            frame,
            target,
            bins=(0.0, 0.5, 1.0),
        )
        summary: dict[str, Any] = {
            "product": "測試商品",
            "population": "不分客群",
            "edition": "v1",
            "periods": {"train_yms": ["202501"], "backtest_ym": "202504"},
            "passed": True,
            "development_sample": {"limited": True, "rows_per_month": 100},
            "metrics": performance_summary(result, result),
        }
        importance = pd.DataFrame({"feature": ["feature_a"], "importance": [1.0]})

        card = _model_card(summary, importance)
        self.assertIn("模型成效摘要", card)
        self.assertIn("Top 10 重要特徵", card)
        self.assertIn("流程測試模型", card)

        with patch("pandas.DataFrame.to_csv") as to_csv, patch("pathlib.Path.write_text") as write_text:
            write_performance_report(Path("report"), summary, importance)
        self.assertEqual(to_csv.call_count, 2)
        write_text.assert_called_once()
