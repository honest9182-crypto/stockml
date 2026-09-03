"""The `Model` protocol every predictor (baseline or sklearn-backed) implements.

`predict_proba` always returns columns in `CLASS_ORDER`, regardless of what
order the underlying estimator's `classes_` happen to be in -- callers never
need to re-sort.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

CLASS_ORDER = ["down", "stagnant", "up"]


@runtime_checkable
class Model(Protocol):
    classes_: list[str]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n_samples, 3) array with columns in CLASS_ORDER."""
        ...


def predict_labels(model: Model, X: pd.DataFrame) -> np.ndarray:
    """Argmax over `predict_proba`, mapped back to class names in CLASS_ORDER."""
    proba = model.predict_proba(X)
    idx = np.argmax(proba, axis=1)
    return np.array(CLASS_ORDER)[idx]


def reorder_proba(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    """Reorder a (n, k) proba array from `classes` order into CLASS_ORDER.

    A class present in CLASS_ORDER but absent from `classes` (the estimator
    never saw it in training) gets probability 0 -- honest, not imputed.
    """
    out = np.zeros((proba.shape[0], len(CLASS_ORDER)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for j, cls in enumerate(CLASS_ORDER):
        if cls in class_to_idx:
            out[:, j] = proba[:, class_to_idx[cls]]
    return out


class PerTickerModel:
    """Wraps one fitted `Model` per ticker behind the single-`Model` interface,
    for `per_ticker: true` runs. `X` passed to `predict_proba` must carry a
    `ticker` column (stripped before delegating to the per-ticker sub-model)
    so leak diagnostics (`evaluate.shift_test` etc.) work identically whether
    the run is pooled or per-ticker: `shift_test` groups its shift by that
    same `ticker` column, and shifting a value that's constant within each
    group is a no-op wherever it isn't dropped as the group's leading NaN row.

    A ticker with no fitted sub-model (unseen at train time) gets all-zero
    probabilities -- honest, not imputed.
    """

    def __init__(self, models: dict[str, "Model"]) -> None:
        self.models = models
        self.classes_: list[str] = list(CLASS_ORDER)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "ticker" not in X.columns:
            raise ValueError("PerTickerModel.predict_proba requires a 'ticker' column")
        tickers = X["ticker"]
        feat = X.drop(columns=["ticker"])
        out = np.zeros((len(X), len(CLASS_ORDER)))
        for t, idx in tickers.groupby(tickers).groups.items():
            sub_model = self.models.get(t)
            if sub_model is None:
                continue
            positions = X.index.get_indexer(idx)
            out[positions] = sub_model.predict_proba(feat.loc[idx])
        return out
