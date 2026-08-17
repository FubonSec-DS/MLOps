"""Temporal evaluation for binary churn models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


@dataclass(frozen=True)
class EvaluationResult:
    """AUC and risk-bin counts for one temporal split."""

    auc: float
    level_metrics: list[dict[str, int | float]]


def evaluate(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    bins: Iterable[float],
    rank_bins: Iterable[int] | None = None,
) -> EvaluationResult:
    """Evaluate without changing or fitting the model."""
    if y.nunique() < 2:
        raise ValueError("Evaluation target must contain both y=0 and y=1")
    prediction = model.predict_proba(X)[:, 1]
    score = float(roc_auc_score(y, prediction))
    if rank_bins is not None:
        edges = sorted(set(int(value) for value in rank_bins))
        ranks = pd.Series(prediction).rank(ascending=False, method="first").astype(int)
        levels = pd.cut(ranks, bins=edges, include_lowest=True, duplicates="drop")
    else:
        edges = sorted(set(float(value) for value in bins))
        if edges[0] > 0.0:
            edges.insert(0, 0.0)
        if edges[-1] < 1.0:
            edges.append(1.0)
        levels = pd.cut(prediction, bins=edges, include_lowest=True, duplicates="drop")
    frame = pd.DataFrame({"level": levels, "y": y.to_numpy()})
    grouped = frame.groupby("level", observed=True)["y"].agg(["count", "sum", "mean"]).reset_index()
    grouped["level"] = grouped["level"].astype(str)
    grouped = grouped.replace({np.nan: None})
    return EvaluationResult(score, grouped.to_dict(orient="records"))


def check_retrain_passed(
    backtest_auc: float,
    *,
    min_auc: float = 0.6,
) -> bool:
    """Apply the acceptance threshold to the untouched OOT backtest."""
    return backtest_auc >= min_auc
