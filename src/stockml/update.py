"""Step-2 hook: the online update policy interface.

Only `Frozen` is implemented in step 1 -- it always declines to update. The
walk-forward loop (`run.py`) calls `should_update` after each test day's
outcome is known, and `update` when it returns true, so later policies
(daily refit, refit-on-big-miss, recency-weighted) drop in without touching
the loop itself.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from stockml.models.base import Model


@runtime_checkable
class UpdatePolicy(Protocol):
    def should_update(
        self, day: pd.Timestamp, y_pred: Any, y_true: Any, history: pd.DataFrame
    ) -> bool:
        """Called once per test day, after that day's true label is known."""
        ...

    def update(self, model: Model, X_hist: pd.DataFrame, y_hist: pd.Series) -> Model:
        """Called only when `should_update` returned True for that day."""
        ...


class Frozen:
    """Step-1 default: never updates. The model trained once on the initial
    training window is used, unchanged, for every test and sanity day.
    """

    def should_update(
        self, day: pd.Timestamp, y_pred: Any, y_true: Any, history: pd.DataFrame
    ) -> bool:
        return False

    def update(self, model: Model, X_hist: pd.DataFrame, y_hist: pd.Series) -> Model:
        # Never called since should_update always returns False, but
        # implemented for protocol completeness / testability.
        return model
