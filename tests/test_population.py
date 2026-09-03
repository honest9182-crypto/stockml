"""Population dynamics: rank-based first-parent selection is monotone in
rank, mate choice is biased toward genetic distance by `dissimilarity_power`,
the lottery draws from the bottom half, and storms fire on schedule and when
the diversity guard trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from stockml.evolution.fitness import FitnessResult
from stockml.evolution.genome import random_genome
from stockml.evolution.population import (
    Individual,
    choose_first_parent,
    choose_mate,
    lottery_parent,
    mean_pairwise_distance,
    next_generation,
    rank_weights,
)


def _fr(fitness: float) -> FitnessResult:
    return FitnessResult(
        genome_hash="x",
        zone="arena",
        n_days=10,
        mean_edge_pp=fitness,
        se_pp=0.0,
        fitness=fitness,
        accuracy=0.4,
        ci95_low_pp=fitness,
        ci95_high_pp=fitness,
        prediction_mix={"down": 0.3, "stagnant": 0.4, "up": 0.3},
    )


def _population(n: int, seed: int = 0, fitness_values: list[float] | None = None) -> list[Individual]:
    rng = np.random.default_rng(seed)
    if fitness_values is None:
        fitness_values = list(rng.normal(size=n))
    pop = []
    for i in range(n):
        pop.append(
            Individual(
                id=f"000_{i:03d}",
                generation=0,
                genome=random_genome(rng),
                parents=[],
                origin="random_init",
                fitness=_fr(fitness_values[i]),
            )
        )
    return pop


def test_rank_weights_best_gets_max_worst_gets_min():
    weights = rank_weights([10.0, 30.0, 20.0], selection_pressure=1.5)
    # index 1 (fitness 30) is best -> weight n**1.5; index 0 (fitness 10) worst -> 1**1.5
    assert weights[1] == pytest.approx(3**1.5)
    assert weights[0] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(2**1.5)


def test_first_parent_selection_monotone_in_rank():
    # Strictly increasing fitness by index -> higher index should be drawn
    # as first parent more often, over many draws.
    n = 10
    fitness_values = list(range(n))  # index i has fitness i; index n-1 is best
    pop = _population(n, seed=1, fitness_values=[float(v) for v in fitness_values])
    rng = np.random.default_rng(42)
    counts = np.zeros(n)
    n_draws = 20000
    for _ in range(n_draws):
        idx = choose_first_parent(pop, rng, selection_pressure=1.5)
        counts[idx] += 1
    # monotone non-decreasing in fitness rank
    diffs = np.diff(counts)
    assert (diffs >= -50).all(), f"counts not roughly monotone: {counts}"  # small noise tolerance
    assert counts[-1] > counts[0] * 5  # best drawn much more than worst


def test_mate_choice_biased_by_distance_when_power_positive():
    n = 30
    pop = _population(n, seed=2)
    rng = np.random.default_rng(7)

    def mean_chosen_distance(power: float) -> float:
        from stockml.evolution.genome import genetic_distance

        rng_local = np.random.default_rng(7)
        total = 0.0
        draws = 3000
        for _ in range(draws):
            first = rng_local.integers(0, n)
            second = choose_mate(pop, first, rng_local, selection_pressure=1.5, dissimilarity_power=power)
            total += genetic_distance(pop[first].genome, pop[second].genome)
        return total / draws

    dist_power4 = mean_chosen_distance(4.0)
    dist_power0 = mean_chosen_distance(0.0)
    assert dist_power4 > dist_power0


def test_mate_choice_never_self_mates():
    n = 5
    pop = _population(n, seed=3)
    rng = np.random.default_rng(11)
    for _ in range(1000):
        first = rng.integers(0, n)
        second = choose_mate(pop, first, rng, selection_pressure=1.5, dissimilarity_power=2.0)
        assert second != first


def test_lottery_draws_from_bottom_half_and_reproduces():
    n = 20
    fitness_values = [float(i) for i in range(n)]  # index n-1 best
    pop = _population(n, seed=4, fitness_values=fitness_values)
    rng = np.random.default_rng(13)
    bottom_half_ids = {pop[i].id for i in range(n // 2)}
    hits = 0
    draws = 5000
    for _ in range(draws):
        idx = lottery_parent(pop, rng)
        assert pop[idx].id in bottom_half_ids
        hits += 1
    assert hits == draws  # every draw must be from the bottom half


def test_storm_fires_on_schedule():
    n = 12
    pop = _population(n, seed=5)
    rng = np.random.default_rng(20)
    cfg = dict(
        population_size=n,
        n_elite=2,
        n_lottery=2,
        n_immigrants=2,
        selection_pressure=1.5,
        dissimilarity_power=2.0,
        mutation_rate=0.05,
        storm_every=5,
        storm_factor=5,
    )
    _, storm0 = next_generation(pop, generation=0, cfg=cfg, rng=rng, force_storm=False)
    _, storm5 = next_generation(pop, generation=5, cfg=cfg, rng=rng, force_storm=False)
    _, storm6 = next_generation(pop, generation=6, cfg=cfg, rng=rng, force_storm=False)
    assert storm0 is False  # gen 0 never storms on schedule
    assert storm5 is True
    assert storm6 is False


def test_diversity_guard_forces_storm():
    n = 12
    pop = _population(n, seed=6)
    rng = np.random.default_rng(21)
    cfg = dict(
        population_size=n,
        n_elite=2,
        n_lottery=2,
        n_immigrants=2,
        selection_pressure=1.5,
        dissimilarity_power=2.0,
        mutation_rate=0.05,
        storm_every=5,
        storm_factor=5,
    )
    _, storm = next_generation(pop, generation=1, cfg=cfg, rng=rng, force_storm=True)
    assert storm is True


def test_next_generation_produces_exactly_population_size():
    n = 15
    pop = _population(n, seed=8)
    rng = np.random.default_rng(30)
    cfg = dict(
        population_size=n,
        n_elite=2,
        n_lottery=3,
        n_immigrants=2,
        selection_pressure=1.5,
        dissimilarity_power=2.0,
        mutation_rate=0.05,
        storm_every=5,
        storm_factor=5,
    )
    new_pop, _ = next_generation(pop, generation=1, cfg=cfg, rng=rng, force_storm=False)
    assert len(new_pop) == n
    ids = [ind.id for ind in new_pop]
    assert len(ids) == len(set(ids))  # unique ids


def test_next_generation_elites_are_unchanged_and_free():
    n = 10
    fitness_values = [float(i) for i in range(n)]
    pop = _population(n, seed=9, fitness_values=fitness_values)
    rng = np.random.default_rng(31)
    cfg = dict(
        population_size=n,
        n_elite=2,
        n_lottery=1,
        n_immigrants=1,
        selection_pressure=1.5,
        dissimilarity_power=2.0,
        mutation_rate=0.05,
        storm_every=5,
        storm_factor=5,
    )
    new_pop, _ = next_generation(pop, generation=1, cfg=cfg, rng=rng, force_storm=False)
    elites = [ind for ind in new_pop if ind.origin == "elite"]
    assert len(elites) == 2
    best_two_ids = {pop[n - 1].id, pop[n - 2].id}
    assert {e.parents[0] for e in elites} == best_two_ids
    for e in elites:
        assert e.memoized is True
        assert e.fitness is not None


def test_mean_pairwise_distance_zero_for_identical_population():
    n = 5
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    pop = [
        Individual(id=f"000_{i:03d}", generation=0, genome=g, parents=[], origin="random_init", fitness=_fr(0.0))
        for i in range(n)
    ]
    assert mean_pairwise_distance(pop) == 0.0
