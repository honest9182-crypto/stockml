"""`evaluate.summarize_daily_series` is the refactored-out core of
`day_level_paired_test` (CLAUDE.md's "Up-only picker" section: the picker's
precision-vs-base-rate edge shares this exact treatment). This proves the
refactor didn't change `day_level_paired_test`'s own numbers, on the real
step-1 smoke run, not just a synthetic series.
"""

from __future__ import annotations

import pandas as pd

from stockml import evaluate as eval_mod
from stockml import run as run_mod
from stockml import split as split_mod
from stockml.features import feature_names
from stockml.models.baselines import MajorityClass
from stockml.models.sklearn_models import HGB
from stockml.update import Frozen
from stockml.walk_forward import walk_forward_single_model

CONFIG = "configs/smoke.yaml"


def test_summarize_daily_series_reproduces_day_level_paired_test_on_smoke_run():
    cfg = run_mod.load_config(CONFIG)
    _raw_panel, dataset = run_mod.build_dataset(cfg)
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    split_dates = split_mod.compute_split_dates(
        unique_dates, cfg["split"]["train_years"], cfg["split"]["sanity_days"],
        embargo_days=cfg["split"].get("embargo_days", 1),
    )
    dataset = dataset.copy()
    dataset["split"] = split_mod.assign_split(dataset["date"], split_dates)
    train_df = dataset[dataset["split"] == "train"]
    test_df = dataset[dataset["split"] == "test"]
    feat_cols = feature_names()

    maj_preds, _, _, _ = walk_forward_single_model(
        train_df, test_df, test_df.iloc[0:0], MajorityClass(), Frozen(), feat_cols
    )
    hgb_preds, _, _, _ = walk_forward_single_model(
        train_df, test_df, test_df.iloc[0:0], HGB(random_state=0), Frozen(), feat_cols
    )

    maj_lookup = maj_preds.set_index(["date", "ticker"])["y_pred"]
    aligned_baseline = pd.Series(
        hgb_preds.set_index(["date", "ticker"]).index.map(maj_lookup), index=hgb_preds.index
    )

    direct = eval_mod.day_level_paired_test(
        hgb_preds["date"], hgb_preds["y_true"], hgb_preds["y_pred"], aligned_baseline, n_boot=200, seed=0
    )

    model_daily = eval_mod.daily_accuracy(hgb_preds["date"], hgb_preds["y_true"], hgb_preds["y_pred"])
    baseline_daily = eval_mod.daily_accuracy(hgb_preds["date"], hgb_preds["y_true"], aligned_baseline)
    edge = (model_daily - baseline_daily).dropna()
    via_helper = eval_mod.summarize_daily_series(edge, n_boot=200, seed=0)

    assert direct == via_helper
