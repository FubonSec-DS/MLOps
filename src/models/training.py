"""XGBoost-only feature selection and formal model training."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from configs.configs_val.product_config import XGBoostParams


def build_xgboost(
    params: XGBoostParams,
    *,
    device: str,
    random_state: int,
) -> XGBClassifier:
    """Create the one supported estimator with explicit deterministic settings."""
    kwargs = {
        "n_estimators": params.n_estimators,
        "max_depth": params.max_depth,
        "learning_rate": params.learning_rate,
        "scale_pos_weight": params.scale_pos_weight,
        "colsample_bylevel": params.colsample_bylevel,
        "subsample": params.subsample,
        "reg_lambda": params.reg_lambda,
        "min_child_weight": params.min_child_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": device,
        "enable_categorical": True,
        "importance_type": "gain",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params.early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = params.early_stopping_rounds
    return XGBClassifier(**kwargs)


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: XGBoostParams,
    *,
    device: str,
    random_state: int,
    validation: tuple[pd.DataFrame, pd.Series] | None = None,
) -> XGBClassifier:
    """Fit XGBoost, optionally using a temporal validation month."""
    if y_train.nunique() < 2:
        raise ValueError("Training target must contain both y=0 and y=1")
    model = build_xgboost(params, device=device, random_state=random_state)
    if validation is not None:
        X_valid, y_valid = validation
        if y_valid.nunique() < 2:
            raise ValueError("Validation target must contain both y=0 and y=1")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def feature_importance(model: XGBClassifier, features: Iterable[str]) -> pd.DataFrame:
    """Return explicit gain importance with deterministic tie-breaking."""
    scores = model.get_booster().get_score(importance_type="gain")
    rows = []
    for feature in features:
        raw_score = scores.get(feature, 0.0)
        score = float(raw_score) if isinstance(raw_score, (int, float)) else float(sum(raw_score))
        rows.append((feature, score))
    return pd.DataFrame(rows, columns=["feature", "importance"]).sort_values(
        ["importance", "feature"], ascending=[False, True], ignore_index=True
    )


def select_top_features(
    model: XGBClassifier,
    features: Iterable[str],
    *,
    top_n: int,
) -> tuple[list[str], pd.DataFrame]:
    """Select up to ``top_n`` features without padding."""
    importance = feature_importance(model, features)
    selected = importance.head(top_n)["feature"].tolist()
    return selected, importance


def auc(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """Calculate ROC AUC for one temporal split."""
    if y.nunique() < 2:
        raise ValueError("AUC requires both y=0 and y=1")
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def save_model(model: XGBClassifier, path: Path) -> None:
    """Persist a fitted estimator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
