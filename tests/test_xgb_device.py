"""CPU and GPU xgb (evolution's "xgb" model_family) must agree on
predictions -- `device` (evolution/device.py) is purely a speed/host-
hardware choice, never something a genome's own fitness should depend on.
Skipped cleanly (not failed) when no GPU is present, since most CI/dev
machines won't have one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml import run as run_mod
from stockml.evolution.device import gpu_available
from stockml.features import feature_names
from stockml.models.base import predict_labels
from stockml.models.xgb_model import XGB

CONFIG = "configs/smoke.yaml"

# One representative set of hgb_*-grid values (see genome.py/xgb_model.py),
# fixed here rather than drawn from a genome -- this test is about the
# device, not about sweeping hyperparameters.
XGB_KWARGS = dict(
    random_state=0, max_depth=4, learning_rate=0.1, n_estimators=100,
    min_child_weight=20, reg_lambda=0.0,
)


@pytest.mark.skipif(
    not gpu_available(), reason="no GPU detected (evolution.device.gpu_available() is False)"
)
def test_cpu_and_gpu_xgb_agree_on_predictions():
    cfg = run_mod.load_config(CONFIG)
    _raw_panel, dataset = run_mod.build_dataset(cfg)
    feat_cols = feature_names()

    dates = pd.to_datetime(dataset["date"])
    split_at = dates.quantile(0.7)  # a plain time-ordered split -- no leak-testing machinery needed here
    train_df = dataset[dates <= split_at]
    test_df = dataset[dates > split_at]
    assert len(train_df) and len(test_df)

    cpu_model = XGB(**XGB_KWARGS, device="cpu")
    cpu_model.fit(train_df[feat_cols], train_df["label"])
    cpu_preds = predict_labels(cpu_model, test_df[feat_cols])

    gpu_model = XGB(**XGB_KWARGS, device="cuda")
    gpu_model.fit(train_df[feat_cols], train_df["label"])
    gpu_preds = predict_labels(gpu_model, test_df[feat_cols])

    agreement = float(np.mean(cpu_preds == gpu_preds))
    assert agreement > 0.99, f"CPU/GPU xgb predictions agree on only {agreement:.4%} of {len(test_df)} rows"
