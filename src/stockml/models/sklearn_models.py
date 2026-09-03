"""Boring sklearn model wrappers. No tuning, no hyperparameter search --
default params with a fixed `random_state` only (CLAUDE.md principle 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stockml.models.base import CLASS_ORDER, reorder_proba
from stockml.models.baselines import AlwaysUp, MajorityClass


class LogReg:
    """Multinomial logistic regression, with scaling fit inside the pipeline
    (never outside it) so the scaler only ever sees training rows.
    """

    def __init__(self, random_state: int = 0) -> None:
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )
        self.classes_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogReg":
        self._pipeline.fit(X, y)
        self.classes_ = list(self._pipeline.named_steps["clf"].classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)
        return reorder_proba(proba, self.classes_)


class HGB:
    """HistGradientBoostingClassifier with default hyperparameters."""

    def __init__(self, random_state: int = 0) -> None:
        self._clf = HistGradientBoostingClassifier(random_state=random_state)
        self.classes_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HGB":
        self._clf.fit(X, y)
        self.classes_ = list(self._clf.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._clf.predict_proba(X)
        return reorder_proba(proba, self.classes_)


MODEL_REGISTRY = {
    "majority_class": lambda seed: MajorityClass(),
    "always_up": lambda seed: AlwaysUp(),
    "logreg": lambda seed: LogReg(random_state=seed),
    "hgb": lambda seed: HGB(random_state=seed),
}
