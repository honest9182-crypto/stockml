"""XGBoost wrapper for the evolution layer's "xgb" model_family
(`evolution/model_builder.py`) -- NOT part of step 1's `MODEL_REGISTRY`
(CLAUDE.md: step 1 stays boring/scikit-learn only; the two seeded
individuals, SEED_LOGREG and SEED_HGB, stay scikit-learn too).

Genes are shared with "hgb" (`evolution/genome.py`'s `hgb_*` genes) rather
than adding a parallel `xgb_*` gene set: `max_depth`, `learning_rate`,
`n_estimators` (<- `hgb_max_iter`), `min_child_weight` (<- `hgb_min_samples_leaf`),
and `reg_lambda` (<- `hgb_l2`) map straight onto XGBoost's own parameter
names. Two caveats worth knowing, both accepted rather than compensated
for -- the point is genomes sharing one grid, not the two families behaving
identically:
- `max_depth=None` means "unlimited" to `HistGradientBoostingClassifier`
  but "use XGBoost's own default (6)" to XGBoost, which has no unlimited
  setting for `tree_method="hist"`.
- `min_child_weight` is a hessian-weight threshold, not a literal minimum
  sample count like `min_samples_leaf` -- the larger grid values (500,
  2000) can prune far more aggressively here than they do for HGB.

`class_weight` (which HGB does support) has no direct multiclass
equivalent in XGBoost's sklearn API, so it's left inactive for xgb, same as
any other gene that doesn't apply to a family -- see genome.py's
"recessive genes" note.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.models.base import CLASS_ORDER, reorder_proba


class XGB:
    """XGBoost gradient boosting, `tree_method="hist"`. `device` is decided
    by the caller (`evolution/device.py`'s `resolve_device`) -- "cuda" when
    a GPU is usable, "cpu" otherwise -- and the caller is what actually
    records which device ran into `FitnessResult`/the run's config copy;
    this class just does what it's told.
    """

    def __init__(
        self,
        random_state: int = 0,
        max_depth: int | None = None,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_child_weight: float = 20,
        reg_lambda: float = 0.0,
        device: str = "cpu",
    ) -> None:
        import xgboost as xgb

        self._clf = xgb.XGBClassifier(
            random_state=random_state,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            tree_method="hist",
            device=device,
            # Explicit, not left to threadpoolctl.threadpool_limits(1) (which
            # evolution/loop.py already wraps every parallel dispatch in):
            # XGBoost's own OpenMP thread pool isn't guaranteed to be one of
            # the libraries threadpoolctl's registry actually controls, and
            # this is the exact oversubscription bug CLAUDE.md documents for
            # HGB/BLAS -- N outer joblib threads each also spawning XGBoost's
            # full-core thread pool would thrash the same way. GPU runs don't
            # need this, but it's harmless there too.
            n_jobs=1,
        )
        self.classes_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGB":
        # Unlike sklearn's own estimators (LogReg/HGB), XGBoost's sklearn API
        # rejects arbitrary string labels -- it requires labels to be
        # contiguous integers starting at 0 for however many classes are
        # actually present, not just "valid CLASS_ORDER values". Encoding
        # against the *full* CLASS_ORDER (fixed {down:0, stagnant:1, up:2})
        # breaks the moment fewer than three classes are present -- e.g.
        # picker.scores.BinaryUp fits on {stagnant, up} only, which maps to
        # {1, 2}, not the {0, 1} XGBoost's binary path demands (observed
        # directly: "Invalid classes inferred... Expected: [0 1], got [1
        # 2]"). Encode only the classes present in *this* y instead, kept in
        # CLASS_ORDER's relative order (not y's encounter order) so the
        # mapping is still fixed and reproducible regardless of row order --
        # for the normal all-three-classes case this is identical to the
        # old fixed mapping, so nothing changes there.
        present = [c for c in CLASS_ORDER if c in set(y)]
        label_to_idx = {c: i for i, c in enumerate(present)}
        y_idx = y.map(label_to_idx)
        self._clf.fit(X, y_idx)
        self.classes_ = [present[i] for i in self._clf.classes_]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._clf.predict_proba(X)
        return reorder_proba(proba, self.classes_)
