"""Population dynamics: selection, mate choice, the lottery, crossover,
mutation, storms, and the diversity guard. Every random decision here takes
an explicit `np.random.Generator` -- callers are responsible for handing it
a generator derived deterministically from the run's seed (see
`evolution/loop.py`), never global `random`/`np.random` state.

Selection is rank-based (not fitness-value-based) throughout, per the
owner's brief: "Rank-based so a single outlier can't take over the whole
gene pool in one generation."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stockml.evolution.fitness import FitnessResult
from stockml.evolution.genome import Genome, crossover, genetic_distance, mutate, random_genome

ORIGIN_SEED = "seed"
ORIGIN_RANDOM_INIT = "random_init"
ORIGIN_ELITE = "elite"
ORIGIN_CHILD = "child"
ORIGIN_LOTTERY_CHILD = "lottery_child"
ORIGIN_IMMIGRANT = "immigrant"


@dataclass
class Individual:
    id: str
    generation: int
    genome: Genome
    parents: list[str]
    origin: str
    mutated_genes: list[str] = field(default_factory=list)
    storm: bool = False
    fitness: FitnessResult | None = None
    fitness_ref: str | None = None  # id of the individual whose fitness this reuses (elites)
    memoized: bool = False


def rank_weights(fitness_values: list[float], selection_pressure: float = 1.5) -> np.ndarray:
    """Rank-based roulette weights: best (rank 1) -> N**pressure, worst
    (rank N) -> 1**pressure. Ties keep stable input order.
    """
    n = len(fitness_values)
    order = np.argsort(np.argsort([-v for v in fitness_values], kind="stable"), kind="stable")
    # `order[i]` is 0 for the best individual, N-1 for the worst.
    ranks_from_best = order + 1  # 1..N, 1 = best
    weights = (n - ranks_from_best + 1).astype(float) ** selection_pressure
    return weights


def _weighted_choice(rng: np.random.Generator, n: int, weights: np.ndarray, exclude: int | None = None) -> int:
    w = weights.copy()
    if exclude is not None:
        w[exclude] = 0.0
    total = w.sum()
    if total <= 0:
        # Degenerate case (e.g. population of 1, or excluded index was everything):
        # fall back to a uniform draw over whatever remains eligible.
        eligible = [i for i in range(n) if i != exclude]
        return int(rng.choice(eligible))
    p = w / total
    return int(rng.choice(n, p=p))


def choose_first_parent(population: list[Individual], rng: np.random.Generator, selection_pressure: float) -> int:
    weights = rank_weights([ind.fitness.fitness for ind in population], selection_pressure)
    return _weighted_choice(rng, len(population), weights)


def choose_mate(
    population: list[Individual],
    first_parent_idx: int,
    rng: np.random.Generator,
    selection_pressure: float,
    dissimilarity_power: float,
) -> int:
    """Second parent: weight ~ rank_weight(candidate) * (1+distance)**power,
    self-mating excluded.
    """
    base_weights = rank_weights([ind.fitness.fitness for ind in population], selection_pressure)
    first_genome = population[first_parent_idx].genome
    weights = np.array(
        [
            base_weights[i] * (1 + genetic_distance(first_genome, ind.genome)) ** dissimilarity_power
            for i, ind in enumerate(population)
        ]
    )
    return _weighted_choice(rng, len(population), weights, exclude=first_parent_idx)


def lottery_parent(population: list[Individual], rng: np.random.Generator) -> int:
    """A weak individual gets lucky: uniform draw from the bottom half by
    fitness rank (its partner is chosen by the normal `choose_mate` rule
    against the whole population, so it usually mates upward).
    """
    n = len(population)
    order = sorted(range(n), key=lambda i: population[i].fitness.fitness)  # ascending: worst first
    bottom_half = order[: max(1, n // 2)]
    return int(bottom_half[rng.integers(0, len(bottom_half))])


def mean_pairwise_distance(population: list[Individual]) -> float:
    n = len(population)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += genetic_distance(population[i].genome, population[j].genome)
            count += 1
    return total / count


def _breed_child(
    population: list[Individual],
    rng: np.random.Generator,
    cfg: dict[str, Any],
    generation: int,
    idx: int,
    mutation_rate: float,
    storm: bool,
    first_parent_idx: int,
    origin: str,
) -> Individual:
    second_parent_idx = choose_mate(
        population, first_parent_idx, rng, cfg["selection_pressure"], cfg["dissimilarity_power"]
    )
    a = population[first_parent_idx]
    b = population[second_parent_idx]
    child_genome = crossover(a.genome, b.genome, rng)
    child_genome, mutated = mutate(child_genome, rng, mutation_rate)
    return Individual(
        id=f"{generation:03d}_{idx:03d}",
        generation=generation,
        genome=child_genome,
        parents=[a.id, b.id],
        origin=origin,
        mutated_genes=mutated,
        storm=storm,
    )


def next_generation(
    population: list[Individual],
    generation: int,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    force_storm: bool,
) -> tuple[list[Individual], bool]:
    """Builds generation `generation`'s population from the previous
    (fitness-scored) `population`. All RNG draws happen here, serially, in a
    fixed order -- so the same seed always produces the same generation,
    regardless of how fitness evaluation is parallelized.

    Returns (new_population_without_fitness, is_storm_generation). Callers
    must evaluate fitness for every individual with `fitness is None` before
    the next call.
    """
    n = cfg["population_size"]
    n_elite = cfg["n_elite"]
    n_lottery = cfg["n_lottery"]
    n_immigrants = cfg["n_immigrants"]
    base_rate = cfg["mutation_rate"]

    is_storm = force_storm or (generation > 0 and generation % cfg["storm_every"] == 0)
    mutation_rate = base_rate * cfg["storm_factor"] if is_storm else base_rate

    ranked = sorted(population, key=lambda ind: ind.fitness.fitness, reverse=True)
    new_pop: list[Individual] = []
    idx = 0

    for elite in ranked[:n_elite]:
        new_pop.append(
            Individual(
                id=f"{generation:03d}_{idx:03d}",
                generation=generation,
                genome=elite.genome,
                parents=[elite.id],
                origin=ORIGIN_ELITE,
                mutated_genes=[],
                storm=False,
                fitness=elite.fitness,
                fitness_ref=elite.id,
                memoized=True,
            )
        )
        idx += 1

    for _ in range(n_lottery):
        first = lottery_parent(population, rng)
        new_pop.append(
            _breed_child(population, rng, cfg, generation, idx, mutation_rate, is_storm, first, ORIGIN_LOTTERY_CHILD)
        )
        idx += 1

    for _ in range(n_immigrants):
        genome = random_genome(rng)
        new_pop.append(
            Individual(
                id=f"{generation:03d}_{idx:03d}",
                generation=generation,
                genome=genome,
                parents=[],
                origin=ORIGIN_IMMIGRANT,
                mutated_genes=[],
                storm=False,
            )
        )
        idx += 1

    n_remaining = n - len(new_pop)
    for _ in range(max(0, n_remaining)):
        first = choose_first_parent(population, rng, cfg["selection_pressure"])
        new_pop.append(
            _breed_child(population, rng, cfg, generation, idx, mutation_rate, is_storm, first, ORIGIN_CHILD)
        )
        idx += 1

    return new_pop, is_storm
