"""Prediction output construction for a fitted XGBoost model."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


def load_model(path: Path) -> XGBClassifier:
    """Load a trusted local model artifact."""
    with path.open("rb") as file:
        return pickle.load(file)


def predict_output(
    model: XGBClassifier,
    X: pd.DataFrame,
    customer_ids: pd.Series,
    *,
    bins: Iterable[float],
    rank_bins: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Return customer probabilities, deterministic ranks, and risk bins."""
    prediction = model.predict_proba(X)[:, 1]
    output = pd.DataFrame({"customer_id": customer_ids.to_numpy(), "pred": prediction})
    output["rank"] = output["pred"].rank(ascending=False, method="first").astype(int)
    if rank_bins is not None:
        edges = sorted(set(int(value) for value in rank_bins))
        values = output["rank"]
    else:
        edges = sorted(set(float(value) for value in bins))
        values = output["pred"]
    output["level"] = pd.Series(pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")).astype(str)
    return output
