"""The walk-forward fit/predict loop, shared by `run.py` (step 1) and
`evolution/fitness.py` (the GA layer). Moved out of `run.py` so a genome's
evaluation reuses exactly the same code path a step-1 model does -- no
second implementation to keep in sync or accidentally diverge from.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from stockml.models.base import predict_labels
from stockml.update import UpdatePolicy


class _LazyHistory:
    """Accumulates day-chunks without concatenating until `to_frame()` is
    actually called. Eagerly concatenating every test day would be O(n^2)
    in the number of test days; a policy that never inspects history (like
    `Frozen`) should pay nothing for it.
    """

    def __init__(self, initial: pd.DataFrame) -> None:
        self._chunks = [initial]

    def append(self, chunk: pd.DataFrame) -> None:
        self._chunks.append(chunk)

    def to_frame(self) -> pd.DataFrame:
        return pd.concat(self._chunks, ignore_index=True)

    def __len__(self) -> int:
        return sum(len(c) for c in self._chunks)


def _predict_block(model: Any, block: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    y_pred = predict_labels(model, block[feature_cols])
    proba = model.predict_proba(block[feature_cols])
    out = block[["date", "ticker", "label"]].rename(columns={"label": "y_true"}).copy()
    out["y_pred"] = y_pred
    out["p_down"] = proba[:, 0]
    out["p_stagnant"] = proba[:, 1]
    out["p_up"] = proba[:, 2]
    return out.reset_index(drop=True)


def walk_forward_single_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sanity_df: pd.DataFrame,
    model: Any,
    update_policy: UpdatePolicy,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Any, int]:
    """Fit on `train_df`, predict `test_df` one day at a time (so a real
    update policy can slot in without restructuring this loop), then predict
    `sanity_df` once with whatever model resulted -- no further updates.

    `test_df` may be any single scored zone (step 1's test slice, or a
    genome's arena/vault zone) -- the function itself has no notion of
    "test" beyond "the frame it predicts day by day and reports". Pass an
    empty `sanity_df` (e.g. `test_df.iloc[0:0]`) when there's no sanity
    slice to predict, as evolution's fitness evaluation does.
    """
    model.fit(train_df[feature_cols], train_df["label"])
    history = _LazyHistory(train_df)
    n_updates = 0

    test_preds = []
    for day, day_df in test_df.groupby("date", sort=True):
        pred_block = _predict_block(model, day_df, feature_cols)
        test_preds.append(pred_block)

        history.append(day_df)
        y_pred = pred_block["y_pred"].to_numpy()
        y_true = pred_block["y_true"].to_numpy()
        if update_policy.should_update(day, y_pred, y_true, history):
            n_updates += 1
            hist_df = history.to_frame()
            model = update_policy.update(model, hist_df[feature_cols], hist_df["label"])

    test_preds_df = (
        pd.concat(test_preds, ignore_index=True)
        if test_preds
        else pd.DataFrame(columns=["date", "ticker", "y_true", "y_pred", "p_down", "p_stagnant", "p_up"])
    )
    sanity_preds_df = (
        _predict_block(model, sanity_df, feature_cols)
        if len(sanity_df)
        else pd.DataFrame(columns=test_preds_df.columns)
    )
    return test_preds_df, sanity_preds_df, model, n_updates
