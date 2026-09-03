"""Mandatory baselines. Every results table shows the model next to these,
computed on the same test rows -- a number without its baseline is not a
result (see CLAUDE.md principle 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.models.base import CLASS_ORDER


class MajorityClass:
    """Always predicts the most common label seen in training."""

    def __init__(self) -> None:
        self.classes_: list[str] = list(CLASS_ORDER)
        self._majority: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityClass":
        self._majority = y.value_counts().idxmax()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._majority is None:
            raise RuntimeError("MajorityClass.fit() must be called before predict_proba()")
        out = np.zeros((len(X), len(CLASS_ORDER)))
        out[:, CLASS_ORDER.index(self._majority)] = 1.0
        return out


class AlwaysUp:
    """Always predicts 'up', regardless of training data. Ignores fit()."""

    def __init__(self) -> None:
        self.classes_: list[str] = list(CLASS_ORDER)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlwaysUp":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), len(CLASS_ORDER)))
        out[:, CLASS_ORDER.index("up")] = 1.0
        return out
