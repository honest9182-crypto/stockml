"""End-to-end: the smoke config downloads (or reuses cached) data, runs the
full walk-forward pipeline, and writes a results folder. Needs network access
the first time; subsequent runs reuse the parquet cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockml import data as data_mod
from stockml import run as run_mod

CONFIG = "configs/smoke.yaml"


def test_smoke_end_to_end():
    cfg = run_mod.load_config(CONFIG)
    tickers = run_mod.resolve_tickers(cfg)
    end = run_mod.resolve_end_date(cfg)
    data_mod.download_prices(tickers, cfg["data"]["start"], end, cfg["data"]["cache_dir"])

    run_dir = run_mod.run(CONFIG)

    assert Path(run_dir).exists()
    assert (Path(run_dir) / "report.txt").exists()
    assert (Path(run_dir) / "metrics.json").exists()
    assert (Path(run_dir) / "predictions_test.parquet").exists()
    assert (Path(run_dir) / "predictions_sanity.parquet").exists()


def test_smoke_dataset_respects_configured_start():
    """`load_panel` must slice each ticker's cache to `data.start`, not just
    use whatever range happens to already be cached on disk -- a ticker's
    cache file can cover far more history than `configs/smoke.yaml` asks
    for (e.g. after a `configs/step1.yaml` run shares the same `data/cache/`
    directory), and the dataset actually used must match the config, not the
    cache's incidental contents.
    """
    cfg = run_mod.load_config(CONFIG)
    raw_panel, dataset = run_mod.build_dataset(cfg)

    configured_start = pd.Timestamp(cfg["data"]["start"])
    assert raw_panel["date"].min() >= configured_start
    assert dataset["date"].min() >= configured_start
