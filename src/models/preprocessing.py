"""Persistable preprocessing for stable training/prediction semantics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import cast

import pandas as pd

MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "__UNKNOWN__"


@dataclass(frozen=True)
class PreprocessingSchema:
    """Categories learned from training data for each categorical feature."""

    categorical_features: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> PreprocessingSchema:
        """Restore a schema loaded from JSON."""
        return cls(
            categorical_features=tuple(value["categorical_features"]),
            categories={name: tuple(items) for name, items in value["categories"].items()},
        )


def fit_preprocessing(
    frame: pd.DataFrame,
    features: Iterable[str],
    categorical_features: Iterable[str],
) -> tuple[pd.DataFrame, PreprocessingSchema]:
    """Learn categorical vocabularies and transform model features."""
    feature_list = list(features)
    categorical = tuple(sorted(set(categorical_features) & set(feature_list)))
    categories: dict[str, tuple[str, ...]] = {}
    output = cast("pd.DataFrame", frame.loc[:, feature_list].copy())

    for feature in feature_list:
        if feature in categorical:
            series = cast("pd.Series", output[feature])
            values = series.astype("string").fillna(MISSING_CATEGORY)
            observed = sorted(set(values.astype(str)))
            vocabulary = tuple(dict.fromkeys([*observed, MISSING_CATEGORY, UNKNOWN_CATEGORY]))
            categories[feature] = vocabulary
            output[feature] = pd.Categorical(values.astype(str), categories=vocabulary)
        else:
            try:
                output[feature] = pd.to_numeric(cast("pd.Series", output[feature]), errors="raise")
            except ValueError as error:
                raise ValueError(f"Feature {feature!r} was classified as numeric but contains text") from error

    return output, PreprocessingSchema(categorical, categories)


def apply_preprocessing(
    frame: pd.DataFrame,
    features: Iterable[str],
    schema: PreprocessingSchema,
) -> pd.DataFrame:
    """Apply training vocabularies, mapping unseen categories explicitly."""
    feature_list = list(features)
    missing_features = set(feature_list) - set(frame.columns)
    if missing_features:
        raise ValueError(f"Missing model features: {sorted(missing_features)}")
    output = cast("pd.DataFrame", frame.loc[:, feature_list].copy())

    for feature in feature_list:
        if feature in schema.categories:
            vocabulary = schema.categories[feature]
            allowed = set(vocabulary)
            series = cast("pd.Series", output[feature])
            values = series.astype("string").fillna(MISSING_CATEGORY).astype(str)
            values = values.where(values.isin(list(allowed)), UNKNOWN_CATEGORY)
            output[feature] = pd.Categorical(values, categories=vocabulary)
        else:
            output[feature] = pd.to_numeric(cast("pd.Series", output[feature]), errors="raise")
    return output
