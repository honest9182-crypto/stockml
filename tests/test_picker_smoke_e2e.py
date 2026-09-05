"""End-to-end: `configs/picker_smoke.yaml` writes every expected output
file, and two runs with the same seed produce byte-identical `picks.parquet`
-- CLAUDE.md's "Up-only picker" section, same reproducibility bar as step 1
and the evolution layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockml.picker.run import run as pick_run

CONFIG = "configs/picker_smoke.yaml"

EXPECTED_FILES = [
    "config.yaml",
    "picks.parquet",
    "metrics.json",
    "report.txt",
    "k_sweep.csv",
    "k_sweep.png",
    "sector_mix.csv",
]


def test_picker_smoke_end_to_end_writes_every_file():
    run_dir = pick_run(CONFIG)
    assert Path(run_dir).exists()
    for name in EXPECTED_FILES:
        assert (Path(run_dir) / name).exists(), f"missing {name}"
    # one rolling_edge_<name>.png per model + deterministic baseline
    rolling_plots = list(Path(run_dir).glob("rolling_edge_*.png"))
    assert len(rolling_plots) >= 4  # 3 models (logreg/hgb/xgb) + at least top_vol


def test_two_picker_smoke_runs_same_seed_produce_identical_picks():
    run_dir_a = pick_run(CONFIG)
    run_dir_b = pick_run(CONFIG)
    a = pd.read_parquet(Path(run_dir_a) / "picks.parquet")
    b = pd.read_parquet(Path(run_dir_b) / "picks.parquet")
    pd.testing.assert_frame_equal(a, b)
