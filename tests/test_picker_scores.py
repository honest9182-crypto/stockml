"""picker/scores.py: `BinaryUp`'s proba semantics (sums to 1, `p_up`
matches the wrapped model's own, `p_down` always 0, `y_pred` is "up" iff
`p_up > 0.5`) across every model family -- including `xgb`, whose sklearn
API rejects non-contiguous-from-0 class labels and broke the first version
of this wrapper (see `models/xgb_model.py`'s fix) -- and that the picker
pipeline introduces no look-ahead of its own on top of the existing
feature-level guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.features import build_features_panel, feature_names
from stockml.labels import build_labels
from stockml.models.base import CLASS_ORDER, predict_labels
from stockml.models.sklearn_models import HGB
from stockml.picker.scores import PICKER_MODEL_REGISTRY, BinaryUp, compute_scores
from tests.conftest import make_synthetic_panel


def _tiny_xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    feat_cols = feature_names()
    X = pd.DataFrame(rng.normal(size=(n, len(feat_cols))), columns=feat_cols)
    y = pd.Series(rng.choice(["down", "stagnant", "up"], size=n))
    return X, y


def test_binary_up_proba_sums_to_one_and_p_down_is_always_zero():
    X, y = _tiny_xy()
    wrapped = BinaryUp(HGB(random_state=0))
    wrapped.fit(X, y)
    proba = wrapped.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(proba[:, CLASS_ORDER.index("down")], 0.0)


def test_binary_up_p_up_matches_the_wrapped_models_own():
    X, y = _tiny_xy()
    y_collapsed = y.where(y == "up", "stagnant")

    base = HGB(random_state=0)
    base.fit(X, y_collapsed)

    wrapped = BinaryUp(HGB(random_state=0))
    wrapped.fit(X, y)

    up = CLASS_ORDER.index("up")
    assert np.allclose(wrapped.predict_proba(X)[:, up], base.predict_proba(X)[:, up])


def test_binary_up_y_pred_is_up_iff_p_up_greater_than_half():
    X, y = _tiny_xy()
    wrapped = BinaryUp(HGB(random_state=0))
    wrapped.fit(X, y)
    proba = wrapped.predict_proba(X)
    y_pred = predict_labels(wrapped, X)
    p_up = proba[:, CLASS_ORDER.index("up")]
    assert ((y_pred == "up") == (p_up > 0.5)).all()


@pytest.mark.parametrize("model_name", ["logreg", "hgb", "xgb"])
def test_binary_up_works_for_every_model_family(model_name):
    # Regression test for the bug this wrapper first hit: XGBoost's sklearn
    # API requires 0..n_classes-1 contiguous labels, and encoding against
    # the *full* CLASS_ORDER (down=0, stagnant=1, up=2) gives {1, 2} for a
    # collapsed {stagnant, up} label -- not contiguous from 0. See
    # models/xgb_model.py's fit() for the fix this exercises.
    X, y = _tiny_xy()
    wrapped = BinaryUp(PICKER_MODEL_REGISTRY[model_name](seed=0))
    wrapped.fit(X, y)
    proba = wrapped.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(proba[:, CLASS_ORDER.index("down")], 0.0)


def _small_dataset() -> pd.DataFrame:
    panel = make_synthetic_panel(n_tickers=5, n_days=200)
    labeled = build_labels(panel, k=0.5, vol_window=20)
    featured = build_features_panel(labeled, k=0.5, vol_window=20)
    feat_cols = feature_names()
    return featured.dropna(subset=feat_cols).reset_index(drop=True)


def test_picks_for_day_d_unchanged_when_future_rows_deleted():
    """No look-ahead in the picker's own data flow (on top of the existing
    feature-level truncation test): scoring day d must not depend on
    whether later days' rows are even present in the eval frame.
    """
    dataset = _small_dataset()
    feat_cols = feature_names()
    unique_dates = sorted(dataset["date"].unique())
    train_end = unique_dates[100]
    train_df = dataset[dataset["date"] <= train_end]
    eval_full = dataset[dataset["date"] > train_end]
    day_d = sorted(eval_full["date"].unique())[5]

    scores_full, _, _ = compute_scores(
        HGB(random_state=0), train_df, eval_full, eval_full.iloc[0:0], feat_cols
    )
    eval_truncated = eval_full[eval_full["date"] <= day_d]
    scores_trunc, _, _ = compute_scores(
        HGB(random_state=0), train_df, eval_truncated, eval_truncated.iloc[0:0], feat_cols
    )

    a = scores_full[scores_full["date"] == day_d].sort_values("ticker").reset_index(drop=True)
    b = scores_trunc[scores_trunc["date"] == day_d].sort_values("ticker").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["date", "ticker", "p_up"]], b[["date", "ticker", "p_up"]])
