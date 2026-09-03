"""Builds a fittable `Model` from a `Genome`. The only place that decides
which genes actually matter for a given `model_family` -- everything else
treats a genome's "recessive" genes (e.g. `hgb_*` on a `logreg` genome) as
opaque, inherited values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.evolution.genome import Genome
from stockml.models.base import CLASS_ORDER
from stockml.models.sklearn_models import HGB, LogReg


class BiasedModel:
    """Wraps a fitted `Model`, adding `genome.stagnant_bias` to the
    'stagnant' probability column before anything downstream (argmax,
    reporting) sees it. Keeps `stagnant_bias` entirely out of the
    walk-forward loop and every model class -- it's purely a
    decision-boundary shift applied at prediction time.
    """

    def __init__(self, base_model, stagnant_bias: float) -> None:
        self._base = base_model
        self._stagnant_bias = stagnant_bias
        self.classes_: list[str] = list(CLASS_ORDER)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BiasedModel":
        self._base.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._base.predict_proba(X).copy()
        proba[:, CLASS_ORDER.index("stagnant")] += self._stagnant_bias
        return proba


def build_model(genome: Genome, seed: int):
    """Build the (unfitted) base model `genome.model_family` selects, with
    that family's hyperparameter genes, wrapped for `stagnant_bias`.
    """
    if genome.model_family == "logreg":
        base = LogReg(random_state=seed, C=genome.lr_C, class_weight=genome.class_weight)
    elif genome.model_family == "hgb":
        base = HGB(
            random_state=seed,
            max_depth=genome.hgb_max_depth,
            learning_rate=genome.hgb_learning_rate,
            max_iter=genome.hgb_max_iter,
            min_samples_leaf=genome.hgb_min_samples_leaf,
            l2_regularization=genome.hgb_l2,
            class_weight=genome.class_weight,
        )
    else:
        raise ValueError(f"unknown model_family {genome.model_family!r}")
    return BiasedModel(base, genome.stagnant_bias)
