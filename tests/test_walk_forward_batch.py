"""walk_forward_single_model's Frozen fast path: one batched predict_proba
call over the whole zone must produce exactly the same predictions -- same
values, same row order -- as the day-by-day path it replaces for Frozen.
"""

from __future__ import annotations

import pandas as pd

from stockml import run as run_mod
from stockml.features import feature_names
from stockml.models.sklearn_models import HGB
from stockml.update import Frozen
from stockml.walk_forward import walk_forward_single_model

CONFIG = "configs/smoke.yaml"


class _NeverUpdatesButNotFrozen:
    """Behaves exactly like Frozen (should_update always False) but isn't
    `isinstance(_, Frozen)`, so walk_forward_single_model takes its
    day-by-day branch instead of the batched one -- used only to get a
    day-by-day reference for this test.
    """

    def should_update(self, day, y_pred, y_true, history) -> bool:
        return False

    def update(self, model, X_hist, y_hist):
        return model


def _train_test_split():
    cfg = run_mod.load_config(CONFIG)
    _raw_panel, dataset = run_mod.build_dataset(cfg)
    unique_dates = sorted(dataset["date"].unique())
    train_end = unique_dates[int(len(unique_dates) * 0.7)]
    train_df = dataset[dataset["date"] <= train_end]
    test_df = dataset[dataset["date"] > train_end]
    return train_df, test_df


def test_frozen_batched_predict_matches_day_by_day_on_smoke_dataset():
    train_df, test_df = _train_test_split()
    feat_cols = feature_names()
    sanity_df = test_df.iloc[0:0]
    assert len(train_df) and len(test_df)

    batched_preds, _, _, n_updates_batched = walk_forward_single_model(
        train_df, test_df, sanity_df, HGB(random_state=0), Frozen(), feat_cols
    )
    day_by_day_preds, _, _, n_updates_day_by_day = walk_forward_single_model(
        train_df, test_df, sanity_df, HGB(random_state=0), _NeverUpdatesButNotFrozen(), feat_cols
    )

    pd.testing.assert_frame_equal(batched_preds, day_by_day_preds)
    assert n_updates_batched == n_updates_day_by_day == 0


def test_frozen_batched_predict_empty_test_zone_matches_day_by_day():
    train_df, test_df = _train_test_split()
    feat_cols = feature_names()
    empty_test_df = test_df.iloc[0:0]
    sanity_df = test_df.iloc[0:0]

    batched_preds, _, _, _ = walk_forward_single_model(
        train_df, empty_test_df, sanity_df, HGB(random_state=0), Frozen(), feat_cols
    )
    day_by_day_preds, _, _, _ = walk_forward_single_model(
        train_df, empty_test_df, sanity_df, HGB(random_state=0), _NeverUpdatesButNotFrozen(), feat_cols
    )

    pd.testing.assert_frame_equal(batched_preds, day_by_day_preds)
    assert len(batched_preds) == 0
