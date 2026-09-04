"""`STOCKML_RUNS_DIR`/`STOCKML_CACHE_DIR` must override `runs_dir`/
`data.cache_dir` without editing the config file -- this is what lets
kaggle/stage.ipynb reuse configs/evo.yaml unmodified (see README's
"Running on Kaggle" and `run.apply_env_overrides`'s own docstring).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from stockml import run as run_mod
from stockml.evolution.loop import load_evo_config


def test_apply_env_overrides_sets_runs_dir_and_cache_dir(monkeypatch):
    monkeypatch.setenv("STOCKML_RUNS_DIR", "/kaggle/working/runs")
    monkeypatch.setenv("STOCKML_CACHE_DIR", "/kaggle/input/stockml-price-cache")

    cfg = {"data": {"cache_dir": "data/cache"}, "runs_dir": "runs"}
    run_mod.apply_env_overrides(cfg)

    assert cfg["runs_dir"] == "/kaggle/working/runs"
    assert cfg["data"]["cache_dir"] == "/kaggle/input/stockml-price-cache"


def test_apply_env_overrides_is_a_noop_without_the_env_vars(monkeypatch):
    monkeypatch.delenv("STOCKML_RUNS_DIR", raising=False)
    monkeypatch.delenv("STOCKML_CACHE_DIR", raising=False)

    cfg = {"data": {"cache_dir": "data/cache"}, "runs_dir": "runs"}
    run_mod.apply_env_overrides(cfg)

    assert cfg["runs_dir"] == "runs"
    assert cfg["data"]["cache_dir"] == "data/cache"


def test_apply_env_overrides_tolerates_missing_data_key(monkeypatch):
    monkeypatch.setenv("STOCKML_CACHE_DIR", "/kaggle/input/stockml-price-cache")
    monkeypatch.delenv("STOCKML_RUNS_DIR", raising=False)

    cfg = {"runs_dir": "runs"}  # no "data" key at all
    run_mod.apply_env_overrides(cfg)

    assert cfg["data"]["cache_dir"] == "/kaggle/input/stockml-price-cache"


def test_load_config_applies_env_overrides(monkeypatch):
    monkeypatch.setenv("STOCKML_RUNS_DIR", "/kaggle/working/runs")
    monkeypatch.delenv("STOCKML_CACHE_DIR", raising=False)

    cfg = run_mod.load_config("configs/smoke.yaml")

    assert cfg["runs_dir"] == "/kaggle/working/runs"


def test_load_config_without_env_vars_keeps_config_values(monkeypatch):
    monkeypatch.delenv("STOCKML_RUNS_DIR", raising=False)
    monkeypatch.delenv("STOCKML_CACHE_DIR", raising=False)

    cfg = run_mod.load_config("configs/smoke.yaml")

    assert cfg["runs_dir"] == "runs"
    assert cfg["data"]["cache_dir"] == "data/cache"


def test_load_evo_config_applies_env_overrides(monkeypatch):
    monkeypatch.setenv("STOCKML_RUNS_DIR", "/kaggle/working/runs")
    monkeypatch.setenv("STOCKML_CACHE_DIR", "/kaggle/input/stockml-price-cache")

    cfg = load_evo_config("configs/evo.yaml")

    assert cfg["runs_dir"] == "/kaggle/working/runs"
    assert cfg["data"]["cache_dir"] == "/kaggle/input/stockml-price-cache"


def test_run_persists_pristine_cache_dir_and_runs_dir(tmp_path, monkeypatch):
    """A run created under Kaggle-style env-var overrides (kaggle/stage.ipynb)
    must still be usable locally afterward (e.g. `stockml leakcheck`): its
    own config.yaml snapshot has to record configs/smoke.yaml's real
    cache_dir/runs_dir, not the overridden ones -- see
    run.apply_env_overrides's and run.parse_config's docstrings.
    """
    override_runs_dir = tmp_path / "kaggle_working_runs"
    override_cache_dir = str(Path("data/cache").resolve())
    monkeypatch.setenv("STOCKML_RUNS_DIR", str(override_runs_dir))
    monkeypatch.setenv("STOCKML_CACHE_DIR", override_cache_dir)

    run_dir = run_mod.run("configs/smoke.yaml")

    # the run itself actually used the override...
    assert Path(run_dir).is_relative_to(override_runs_dir)

    # ...but what it wrote to disk is the source config's own values.
    with open(Path(run_dir) / "config.yaml", "r", encoding="utf-8") as f:
        saved_cfg = yaml.safe_load(f)
    assert saved_cfg["runs_dir"] == "runs"
    assert saved_cfg["data"]["cache_dir"] == "data/cache"
