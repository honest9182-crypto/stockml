"""End-to-end: `evolve --quick` writes every expected output file, and two
`--quick` runs with the same seed produce byte-identical `lineage.jsonl` --
the whole point of the deterministic RNG-derivation scheme in loop.py.

Slow (each `--quick` run fits ~15 genomes on 20 tickers): this is the one
test in the suite that's expected to take a few minutes, not a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from stockml.evolution.loop import evolve

CONFIG = "configs/evo.yaml"

EXPECTED_FILES = [
    "config.yaml",
    "progress.json",
    "lineage.jsonl",
    "generations.csv",
    "gene_frequency.csv",
    "champion_family_tree.txt",
    "fitness_by_generation.png",
    "diversity_by_generation.png",
    "feature_frequency_heatmap.png",
]


def test_quick_run_writes_every_output_file():
    run_dir = evolve(CONFIG, quick=True)
    assert Path(run_dir).exists()
    for name in EXPECTED_FILES:
        assert (Path(run_dir) / name).exists(), f"missing {name}"

    import json

    with open(Path(run_dir) / "progress.json", "r", encoding="utf-8") as f:
        progress = json.load(f)
    assert progress["status"] == "completed"
    assert progress["champion_id"] is not None


def test_two_quick_runs_same_seed_produce_identical_lineage():
    run_dir_a = evolve(CONFIG, quick=True)
    run_dir_b = evolve(CONFIG, quick=True)

    text_a = (Path(run_dir_a) / "lineage.jsonl").read_text(encoding="utf-8")
    text_b = (Path(run_dir_b) / "lineage.jsonl").read_text(encoding="utf-8")
    assert text_a == text_b

    gens_a = (Path(run_dir_a) / "generations.csv").read_text(encoding="utf-8")
    gens_b = (Path(run_dir_b) / "generations.csv").read_text(encoding="utf-8")
    assert gens_a == gens_b


def test_quick_run_persists_pristine_cache_dir_and_runs_dir(tmp_path, monkeypatch):
    """A run created under Kaggle-style env-var overrides (kaggle/stage.ipynb)
    must still be resumable/vault-able locally: its own config.yaml snapshot
    has to record configs/evo.yaml's real cache_dir/runs_dir, not the
    overridden ones -- see run.apply_env_overrides and
    evolution.loop.parse_evo_config's docstrings.
    """
    override_runs_dir = tmp_path / "kaggle_working_runs"
    override_cache_dir = str(Path("data/cache").resolve())
    monkeypatch.setenv("STOCKML_RUNS_DIR", str(override_runs_dir))
    monkeypatch.setenv("STOCKML_CACHE_DIR", override_cache_dir)

    run_dir = evolve(CONFIG, quick=True)

    # the run itself actually used the override...
    assert Path(run_dir).is_relative_to(override_runs_dir)

    # ...but what it wrote to disk is the source config's own values.
    with open(Path(run_dir) / "config.yaml", "r", encoding="utf-8") as f:
        saved_cfg = yaml.safe_load(f)
    assert saved_cfg["runs_dir"] == "runs"
    assert saved_cfg["data"]["cache_dir"] == "data/cache"
