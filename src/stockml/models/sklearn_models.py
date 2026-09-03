"""Boring sklearn model wrappers. Step 1 uses default params with a fixed
`random_state` only, no tuning (CLAUDE.md principle 5) -- but the
constructors accept the hyperparameters the evolution layer's genomes
choose between (CLAUDE.md's "evolutionary search" section), all defaulted
to today's step-1 values so nothing about step 1 changes.
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


def _sklearn_class_weight(class_weight: str) -> str | None:
    """Map a genome's `class_weight` gene ("none"/"balanced") to what
    sklearn expects (`None`/"balanced") -- "none" isn't a valid sklearn
    class_weight value, so this can't be passed through as-is.
    """
    return None if class_weight == "none" else class_weight


class LogReg:
    """Multinomial logistic regression, with scaling fit inside the pipeline
    (never outside it) so the scaler only ever sees training rows.
    """

    def __init__(
        self,
        random_state: int = 0,
        C: float = 1.0,
        class_weight: str = "none",
    ) -> None:
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                        C=C,
                        class_weight=_sklearn_class_weight(class_weight),
                    ),
                ),
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
    """HistGradientBoostingClassifier. Defaults are sklearn's own defaults."""

    def __init__(
        self,
        random_state: int = 0,
        max_depth: int | None = None,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        class_weight: str = "none",
    ) -> None:
        self._clf = HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            class_weight=_sklearn_class_weight(class_weight),
        )
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
