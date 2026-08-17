import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.pipelines.retrain import validate_pretrain_readiness
from src.resources.feature_loader import LoadedDataset


class PretrainInspectionTests(unittest.TestCase):
    def test_readiness_writes_runtime_and_dataframe_samples(self) -> None:
        frame = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "yyyymm": ["202503", "202503"],
                "y": [0, 1],
                "cf_asset1__amount": [10.5, None],
                "cf_profile__segment": ["A", None],
            }
        )
        dataset = LoadedDataset(frame, ["cf_profile__segment"])

        report_dir = Path.cwd() / "tests" / "pretrain_reports" / "_test"
        with (
            patch("src.pipelines.retrain.discover_feature_catalog", return_value=[]),
            patch("src.pipelines.retrain.load_pretrain_month", return_value=dataset),
        ):
            result = validate_pretrain_readiness(
                product="不限用途",
                population="非潛客",
                ym="202503",
                sample_size=2,
                report_dir=report_dir,
                sample_rows=1,
            )

            self.assertTrue(result.ready)
            self.assertTrue((report_dir / "README.md").exists())
            self.assertEqual(len(pd.read_csv(report_dir / "joined_frame_sample.csv")), 1)
            self.assertEqual(len(pd.read_csv(report_dir / "preprocessed_matrix_sample.csv")), 1)
            columns = pd.read_csv(report_dir / "columns.csv")
            self.assertEqual(set(columns["role"]), {"key", "target", "feature"})
            runtime = json.loads((report_dir / "runtime_objects.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime[0]["python_type"], "configs.configs_val.product_config.ProductConfig")
            self.assertNotIn("password", json.dumps(runtime).lower())


if __name__ == "__main__":
    unittest.main()
