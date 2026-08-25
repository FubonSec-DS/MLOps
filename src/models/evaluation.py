"""Temporal evaluation for binary churn models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

MetricRow = dict[str, int | float | str | None]


@dataclass(frozen=True)
class EvaluationResult:
    """Common model and business metrics for one temporal split."""

    auc: float
    ks: float
    row_count: int
    positive_count: int
    base_rate: float
    level_metrics: list[MetricRow]
    top_rate_metrics: list[MetricRow]
    decile_metrics: list[MetricRow]
    predictions: np.ndarray = field(repr=False, compare=False)


def _top_rate_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    rates: Iterable[float],
) -> list[MetricRow]:
    """Calculate cumulative hit rate, recall, and lift at common contact rates."""
    order = np.argsort(-prediction, kind="stable")
    positives = int(y.sum())
    base_rate = positives / len(y)
    rows: list[MetricRow] = []
    for rate in sorted(set(float(value) for value in rates)):
        if not 0 < rate <= 1:
            raise ValueError(f"Top-rate values must be in (0, 1], found {rate}")
        selected_count = min(len(y), max(1, ceil(len(y) * rate)))
        selected = order[:selected_count]
        selected_positives = int(y[selected].sum())
        hit_rate = selected_positives / selected_count
        rows.append(
            {
                "top_percent": rate * 100,
                "selected_count": selected_count,
                "positive_count": selected_positives,
                "hit_rate": hit_rate,
                "recall": selected_positives / positives,
                "lift": hit_rate / base_rate,
                "cutoff_score": float(prediction[selected[-1]]),
            }
        )
    return rows


def _decile_metrics(y: np.ndarray, prediction: np.ndarray) -> list[MetricRow]:
    """Split ranked scores into ten groups from highest to lowest risk."""
    order = np.argsort(-prediction, kind="stable")
    positives = int(y.sum())
    base_rate = positives / len(y)
    cumulative_positives = 0
    rows: list[MetricRow] = []
    for decile, selected in enumerate(np.array_split(order, min(10, len(order))), start=1):
        selected_positives = int(y[selected].sum())
        cumulative_positives += selected_positives
        hit_rate = selected_positives / len(selected)
        rows.append(
            {
                "decile": decile,
                "count": len(selected),
                "positive_count": selected_positives,
                "hit_rate": hit_rate,
                "lift": hit_rate / base_rate,
                "cumulative_recall": cumulative_positives / positives,
                "minimum_score": float(prediction[selected].min()),
                "maximum_score": float(prediction[selected].max()),
            }
        )
    return rows


def evaluate(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    bins: Iterable[float],
    rank_bins: Iterable[int] | None = None,
    *,
    top_rates: Iterable[float] = (0.05, 0.10, 0.20),
) -> EvaluationResult:
    """Evaluate without changing or fitting the model."""
    if y.nunique() < 2:
        raise ValueError("Evaluation target must contain both y=0 and y=1")
    target = y.to_numpy(dtype=int)
    prediction = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(target, prediction))
    false_positive_rate, true_positive_rate, _thresholds = roc_curve(target, prediction)
    ks = float(np.max(true_positive_rate - false_positive_rate))
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
    frame = pd.DataFrame({"level": levels, "y": target})
    grouped = frame.groupby("level", observed=True)["y"].agg(["count", "sum", "mean"]).reset_index()
    grouped["level"] = grouped["level"].astype(str)
    grouped = grouped.replace({np.nan: None})
    positives = int(target.sum())
    return EvaluationResult(
        auc=auc,
        ks=ks,
        row_count=len(target),
        positive_count=positives,
        base_rate=positives / len(target),
        level_metrics=grouped.to_dict(orient="records"),
        top_rate_metrics=_top_rate_metrics(target, prediction, top_rates),
        decile_metrics=_decile_metrics(target, prediction),
        predictions=prediction,
    )


def check_retrain_passed(
    backtest_auc: float,
    *,
    min_auc: float = 0.6,
) -> bool:
    """Apply the acceptance threshold to the untouched OOT backtest."""
    return backtest_auc >= min_auc
