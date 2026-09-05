"""Turns a fittable model into a per-(date, ticker) `p_up` score.

Two `score_source`s, selectable in config (CLAUDE.md's "Up-only picker"
section):

- `three_class`: the existing `logreg`/`hgb`/`xgb` model classes, completely
  unchanged, fit on the normal three-class label. `p_up` is just the `up`
  column `walk_forward_single_model` already produces.
- `binary`: the same model classes, wrapped in `BinaryUp`, fit on the label
  collapsed to `{up, stagnant}` instead. A model asked only "is this one of
  the ups?" may rank better than one asked to also separate down from
  stagnant.

Either way, fitting and predicting go through `walk_forward_single_model`
unchanged -- one batched frozen predict over the eval zone, exactly as
step 1 and the evolution layer already do (CLAUDE.md principle 5 /
"Same walk-forward, same zones").
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from stockml.evolution.device import resolve_device
from stockml.models.base import CLASS_ORDER
from stockml.models.sklearn_models import HGB, LogReg
from stockml.models.xgb_model import XGB
from stockml.update import Frozen
from stockml.walk_forward import walk_forward_single_model

# Real, fittable models a picker can be built from -- deliberately just
# logreg/hgb/xgb at their step-1 boring defaults (no tuning), not the
# majority_class/always_up baselines from sklearn_models.MODEL_REGISTRY:
# those aren't models that produce a meaningful p_up ranking, they're the
# step-1 sanity floor, a different thing from picker/baselines.py's roster.
PICKER_MODEL_REGISTRY: dict[str, Callable[[int], Any]] = {
    "logreg": lambda seed: LogReg(random_state=seed),
    "hgb": lambda seed: HGB(random_state=seed),
    "xgb": lambda seed: XGB(random_state=seed, device=resolve_device("xgb")),
}

_EXTRA_COLS = ["date", "ticker", "r_next", "sigma"]


class BinaryUp:
    """Wraps an unfitted base `Model`, fits it on the label collapsed to
    `{up, stagnant}` (down folded into stagnant), and hands back whatever
    the base model's own `predict_proba` produces unchanged.

    That's the whole implementation, on purpose: every model wrapper here
    already reorders its raw output through `models.base.reorder_proba`
    into `[down, stagnant, up]` columns keyed on whichever classes it saw
    in training. Since this label never contains "down", `reorder_proba`
    already leaves that column at 0, and since the classifier's own two
    columns for {stagnant, up} sum to 1, `p_stagnant` naturally comes out
    to exactly `1 - p_up` -- "the not-up mass is never claimed as down"
    falls out of the existing plumbing rather than needing new math.

    Folding "down" into "stagnant" (not some new label) is also what makes
    `XGB` work here with no special-casing: its sklearn API only accepts
    `CLASS_ORDER` values, and "stagnant" already is one.
    """

    def __init__(self, base_model: Any) -> None:
        self._base = base_model
        self.classes_: list[str] = list(CLASS_ORDER)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinaryUp":
        y_binary = y.where(y == "up", "stagnant")
        self._base.fit(X, y_binary)
        self.classes_ = list(self._base.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._base.predict_proba(X)


def compute_scores(
    model: Any,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    sanity_df: pd.DataFrame,
    feature_cols: list[str],
    timing: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Fits `model` once on `train_df`, predicts `eval_df` and `sanity_df`
    with the same frozen fit (one batched call each, via
    `walk_forward_single_model`), and joins back `r_next`/`sigma` from the
    source frames -- `walk_forward_single_model`'s own output doesn't carry
    them, but `select_top_k`/the return-check/`top_vol` all need them.

    Returns (eval_scores, sanity_scores, fitted_model). Each scores frame
    has columns: date, ticker, y_true, p_down, p_stagnant, p_up, r_next, sigma.
    """
    eval_preds, sanity_preds, fitted_model, _ = walk_forward_single_model(
        train_df, eval_df, sanity_df, model, Frozen(), feature_cols, timing=timing
    )
    eval_scores = eval_preds.merge(eval_df[_EXTRA_COLS], on=["date", "ticker"], how="left")
    if len(sanity_preds):
        sanity_scores = sanity_preds.merge(sanity_df[_EXTRA_COLS], on=["date", "ticker"], how="left")
    else:
        sanity_scores = sanity_preds.assign(r_next=pd.Series(dtype=float), sigma=pd.Series(dtype=float))
    return eval_scores, sanity_scores, fitted_model


def build_score_model(model_name: str, seed: int, score_source: str) -> Any:
    """`PICKER_MODEL_REGISTRY[model_name]`'s model, wrapped in `BinaryUp` if
    `score_source == "binary"`, unwrapped (fit on the normal three-class
    label) if `score_source == "three_class"`.
    """
    base = PICKER_MODEL_REGISTRY[model_name](seed)
    if score_source == "binary":
        return BinaryUp(base)
    if score_source == "three_class":
        return base
    raise ValueError(f"score_source must be 'three_class' or 'binary', got {score_source!r}")
