"""The walk-forward loop must call `should_update` after every test day's
outcome is known, and `update` exactly when it returns True -- proven with a
dummy policy that just counts calls, per CLAUDE.md / the update.py docstring.
"""

from __future__ import annotations

import pandas as pd

from stockml.features import build_features_panel, feature_names
from stockml.labels import build_labels
from stockml.models.baselines import MajorityClass
from stockml.run import _walk_forward_single_model
from stockml.update import Frozen, UpdatePolicy
from tests.conftest import make_synthetic_panel


class CountingPolicy:
    """Always requests an update, after checking should_update every day."""

    def __init__(self) -> None:
        self.should_update_calls = 0
        self.update_calls = 0

    def should_update(self, day, y_pred, y_true, history) -> bool:
        self.should_update_calls += 1
        return True

    def update(self, model, X_hist, y_hist):
        self.update_calls += 1
        model.fit(X_hist, y_hist)
        return model


def _build_small_dataset() -> pd.DataFrame:
    panel = make_synthetic_panel(n_tickers=2, n_days=150)
    labeled = build_labels(panel, k=0.5, vol_window=20)
    featured = build_features_panel(labeled, k=0.5, vol_window=20)
    feat_cols = feature_names()
    return featured.dropna(subset=feat_cols).reset_index(drop=True)


def test_frozen_never_updates():
    ds = _build_small_dataset()
    unique_dates = sorted(ds["date"].unique())
    train_end = unique_dates[60]
    train_df = ds[ds["date"] <= train_end]
    test_df = ds[ds["date"] > train_end]
    sanity_df = test_df.iloc[0:0]

    policy = Frozen()
    assert isinstance(policy, UpdatePolicy)
    test_preds, _, _, n_updates = _walk_forward_single_model(
        train_df, test_df, sanity_df, MajorityClass(), policy, feature_names()
    )
    assert n_updates == 0
    assert len(test_preds) == len(test_df)


def test_counting_policy_is_called_once_per_test_day_and_updates_every_time():
    ds = _build_small_dataset()
    unique_dates = sorted(ds["date"].unique())
    train_end = unique_dates[60]
    train_df = ds[ds["date"] <= train_end]
    test_df = ds[ds["date"] > train_end]
    sanity_df = test_df.iloc[0:0]
    n_test_days = test_df["date"].nunique()

    policy = CountingPolicy()
    assert isinstance(policy, UpdatePolicy)
    test_preds, _, _, n_updates = _walk_forward_single_model(
        train_df, test_df, sanity_df, MajorityClass(), policy, feature_names()
    )

    assert policy.should_update_calls == n_test_days
    assert policy.update_calls == n_test_days
    assert n_updates == n_test_days
    assert len(test_preds) == len(test_df)
