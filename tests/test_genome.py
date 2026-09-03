"""Genome v1: encode/decode round-trip, hash stability, and that mutation
and crossover always produce valid genomes (grid values, >= 2 features on).
"""

from __future__ import annotations

import numpy as np
import pytest

from stockml.evolution.genome import (
    FEATURE_NAMES,
    GENE_GRIDS,
    MIN_FEATURES_ON,
    N_FEATURES,
    Genome,
    SEED_HGB,
    SEED_LOGREG,
    crossover,
    genetic_distance,
    mutate,
    random_genome,
)


def test_encode_decode_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = random_genome(rng)
        d = g.encode()
        g2 = Genome.decode(d)
        assert g == g2
        assert g.stable_hash() == g2.stable_hash()


def test_encode_feature_mask_is_a_bit_string():
    g = SEED_HGB
    d = g.encode()
    assert isinstance(d["feature_mask"], str)
    assert len(d["feature_mask"]) == N_FEATURES
    assert set(d["feature_mask"]) <= {"0", "1"}


def test_decode_missing_keys_use_defaults():
    partial = {"model_family": "logreg"}
    g = Genome.decode(partial)
    assert g.model_family == "logreg"
    assert g.hgb_max_iter == 100  # field default, not present in `partial`


def test_decode_ignores_unknown_keys():
    d = SEED_HGB.encode()
    d["some_future_gene"] = "whatever"
    g = Genome.decode(d)
    assert g == SEED_HGB


def test_hash_is_stable_and_content_addressed():
    a = SEED_LOGREG
    b = Genome.decode(SEED_LOGREG.encode())
    assert a.stable_hash() == b.stable_hash()
    assert SEED_LOGREG.stable_hash() != SEED_HGB.stable_hash()


def test_seeds_have_all_features_on():
    assert SEED_LOGREG.n_features_on() == N_FEATURES
    assert SEED_HGB.n_features_on() == N_FEATURES
    assert SEED_LOGREG.active_features() == FEATURE_NAMES


def test_random_genome_always_valid():
    rng = np.random.default_rng(2)
    for _ in range(500):
        g = random_genome(rng)
        assert g.n_features_on() >= MIN_FEATURES_ON
        for name, grid in GENE_GRIDS.items():
            assert getattr(g, name) in grid


def test_mutation_always_valid():
    rng = np.random.default_rng(3)
    g = random_genome(rng)
    for _ in range(1000):
        g, mutated = mutate(g, rng, rate=0.3)
        assert g.n_features_on() >= MIN_FEATURES_ON
        for name, grid in GENE_GRIDS.items():
            assert getattr(g, name) in grid
        assert isinstance(mutated, list)


def test_crossover_always_valid():
    rng = np.random.default_rng(4)
    for _ in range(500):
        a, b = random_genome(rng), random_genome(rng)
        child = crossover(a, b, rng)
        assert child.n_features_on() >= MIN_FEATURES_ON
        for name, grid in GENE_GRIDS.items():
            assert getattr(child, name) in grid


def test_direct_construction_rejects_invalid_genomes():
    with pytest.raises(ValueError):
        Genome(feature_mask=(True, False) + (False,) * (N_FEATURES - 2))  # only 1 on
    with pytest.raises(ValueError):
        Genome(model_family="not_a_family")
    with pytest.raises(ValueError):
        Genome(feature_mask=(True,) * (N_FEATURES - 1))  # wrong length


def test_genetic_distance_bounds_and_self_distance():
    rng = np.random.default_rng(5)
    a, b = random_genome(rng), random_genome(rng)
    assert genetic_distance(a, a) == 0.0
    d = genetic_distance(a, b)
    assert 0.0 <= d <= 1.0


def test_mutate_can_change_the_genome():
    # With rate=1.0 every gene attempts to change; over enough draws the
    # child must differ from the parent at least once.
    rng = np.random.default_rng(6)
    g = SEED_HGB
    changed_at_least_once = False
    for _ in range(20):
        child, mutated = mutate(g, rng, rate=1.0)
        if mutated:
            changed_at_least_once = True
    assert changed_at_least_once
