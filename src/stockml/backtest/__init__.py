"""Step-3 hook: position sizing / P&L simulation protocol stubs only.

Not implemented in step 1 -- see CLAUDE.md "Explicitly out of scope". These
protocols exist so step 3 (position sizing, transaction costs, path-dependent
reward) can be built against a stable interface without touching step 1/2 code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Policy(Protocol):
    """Turns model predictions into positions. Not implemented in step 1."""

    def position(
        self, day: pd.Timestamp, proba: Any, history: pd.DataFrame
    ) -> float:
        """Return a target position (e.g. -1..1) for `day` given predicted
        class probabilities and any prior state in `history`.
        """
        ...


@runtime_checkable
class Reward(Protocol):
    """Scores a sequence of positions against realized returns. Not
    implemented in step 1.
    """

    def evaluate(self, positions: pd.Series, returns: pd.Series) -> dict[str, float]:
        """Return reward/P&L metrics (e.g. after transaction costs) for a
        path of positions against realized next-day returns.
        """
        ...
