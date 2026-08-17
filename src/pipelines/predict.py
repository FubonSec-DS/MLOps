"""Artifact-driven prediction pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from typing import cast

import pandas as pd

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import get_product_config
from src.models.prediction import load_model, predict_output
from src.models.preprocessing import PreprocessingSchema, apply_preprocessing
from src.models.versioning import save_prediction_outputs
from src.resources.feature_loader import load_prediction_month


def run_prediction(
    *,
    product: str,
    population: str,
    ym: str,
    edition: str,
    write_db: bool = False,
) -> pd.DataFrame:
    """Load the model contract, query only required features, and score customers."""
    artifact_dir = get_settings().paths.model_base / product / population / edition
    summary = json.loads((artifact_dir / "training_summary.json").read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise RuntimeError(f"Model {product}/{population}/{edition} did not pass retrain checks")
    features = json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))
    preprocessing = PreprocessingSchema.from_dict(
        json.loads((artifact_dir / "preprocessing.json").read_text(encoding="utf-8"))
    )
    model = load_model(artifact_dir / "model.pickle")
    config = get_product_config(product)
    dataset = load_prediction_month(config, population, ym, features)
    X = apply_preprocessing(dataset.frame, features, preprocessing)
    customer_ids = cast("pd.Series", dataset.frame["customer_id"])
    output = predict_output(model, X, customer_ids, bins=config.bins, rank_bins=config.rank_bins)
    output["product"] = f"{product}模型"
    output["population"] = population
    output["snap_date"] = ym
    output["edition"] = edition

    if write_db:
        ref_info = pd.DataFrame(
            [
                {
                    "product": f"{product}模型",
                    "population": population,
                    "snap_date": ym,
                    "edition": edition,
                    "total_count": len(output),
                }
            ]
        )
        log = pd.DataFrame(
            [
                {
                    "模型名稱": f"{product}模型",
                    "母體": population,
                    "snap_date": ym,
                    "版本": edition,
                    "預測人數": len(output),
                    "狀態": "SUCCESS",
                    "執行時間": datetime.now().strftime("%Y%m%d %H:%M"),
                }
            ]
        )
        save_prediction_outputs(ref_info, output, log)
    return output
