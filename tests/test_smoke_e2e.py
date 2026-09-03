"""End-to-end: the smoke config downloads (or reuses cached) data, runs the
full walk-forward pipeline, and writes a results folder. Needs network access
the first time; subsequent runs reuse the parquet cache.
"""

from __future__ import annotations

from pathlib import Path

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
