"""evaluate_genome's timing breakdown (fit/predict/metrics/bootstrap --
see fitness.TIMING_PHASES) and device field (evolution/device.py).
"""

from __future__ import annotations

import pandas as pd

from stockml.evolution.device import resolve_device
from stockml.evolution.fitness import TIMING_PHASES, canonical_majority_daily, evaluate_genome
from stockml.evolution.genome import Genome
from stockml.evolution.loop import _fitness_result_from_dict
from stockml.evolution.zones import compute_evo_zones
from stockml.features import build_features_panel, feature_names
from stockml.labels import build_labels
from tests.conftest import make_synthetic_panel


def _small_dataset_and_zones():
    panel = make_synthetic_panel(n_tickers=4, n_days=1200, start="2015-01-01")
    labeled = build_labels(panel, k=0.5, vol_window=20)
    featured = build_features_panel(labeled, k=0.5, vol_window=20)
    feat_cols = feature_names()
    dataset = featured.dropna(subset=feat_cols).reset_index(drop=True)

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    split_idx = int(len(unique_dates) * 0.85)
    zones = compute_evo_zones(
        unique_dates, train_years=2, sanity_days=5, embargo_days=1,
        arena_end=unique_dates[split_idx], vault_start=unique_dates[split_idx + 1],
    )
    return dataset, zones


def test_evaluate_genome_reports_all_timing_phases():
    dataset, zones = _small_dataset_and_zones()
    majority_daily = canonical_majority_daily(dataset, zones, "arena")
    genome = Genome(model_family="hgb")

    result = evaluate_genome(genome, "arena", dataset, zones, majority_daily, seed=0, n_boot=20)

    assert set(result.timing_s.keys()) == set(TIMING_PHASES)
    assert all(v > 0 for v in result.timing_s.values())
    assert result.device == "cpu"  # hgb never uses a device


def test_evaluate_genome_records_device_for_xgb():
    dataset, zones = _small_dataset_and_zones()
    majority_daily = canonical_majority_daily(dataset, zones, "arena")
    genome = Genome(model_family="xgb")

    result = evaluate_genome(genome, "arena", dataset, zones, majority_daily, seed=0, n_boot=0)

    assert result.device == resolve_device("xgb")
    assert result.device in ("cpu", "cuda")
    assert result.timing_s["fit"] > 0
    assert result.timing_s["predict"] > 0


def test_fitness_result_to_dict_round_trips_device_but_not_timing():
    dataset, zones = _small_dataset_and_zones()
    majority_daily = canonical_majority_daily(dataset, zones, "arena")
    genome = Genome(model_family="hgb")

    result = evaluate_genome(genome, "arena", dataset, zones, majority_daily, seed=0, n_boot=0)
    d = result.to_dict()

    assert d["device"] == result.device
    # timing_s is deliberately NOT in to_dict()'s output -- see the field's
    # own docstring, and the next test for exactly why.
    assert "timing_s" not in d


def test_to_dict_is_identical_across_two_evaluations_of_the_same_genome():
    # This is the actual reproducibility invariant lineage.jsonl depends on
    # (loop.py's module docstring: "two runs with the same seed produce
    # byte-identical lineage.jsonl") -- exercised directly here rather than
    # only via a multi-minute full `evolve --quick` run. Two evaluations of
    # the identical genome/data/seed will still take measurably different
    # wall-clock time; to_dict()'s output must not.
    dataset, zones = _small_dataset_and_zones()
    majority_daily = canonical_majority_daily(dataset, zones, "arena")
    genome = Genome(model_family="hgb")

    result_a = evaluate_genome(genome, "arena", dataset, zones, majority_daily, seed=0, n_boot=20)
    result_b = evaluate_genome(genome, "arena", dataset, zones, majority_daily, seed=0, n_boot=20)

    assert result_a.to_dict() == result_b.to_dict()


def test_fitness_result_from_dict_defaults_device_and_timing_for_old_records():
    # A lineage.jsonl written before "device"/"timing_s" existed has no such
    # keys in its "fitness" dict -- must still decode (genome.py's own
    # "every field has a default" rule, applied here too).
    old_record = {
        "genome_hash": "abc123", "zone": "arena", "n_days": 10,
        "mean_edge_pp": 0.5, "se_pp": 0.1, "fitness": 0.4, "accuracy": 0.43,
        "ci95_low_pp": -0.1, "ci95_high_pp": 1.1, "prediction_mix": {"down": 0.3, "stagnant": 0.4, "up": 0.3},
    }
    result = _fitness_result_from_dict(old_record)
    assert result.device == "cpu"
    assert result.timing_s == {}
