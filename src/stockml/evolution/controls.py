"""The two controls every evolution run is checked against (CLAUDE.md):

- **Random search**: the same number of unique fitness evaluations the
  evolution actually performed, spent on uniformly random genomes. If
  evolution's best isn't clearly better than random search's best, the
  mating logic added nothing.
- **Shifted-label null**: the identical evolution (same config, same seed)
  run against labels that have been circularly shifted in time, per ticker,
  by a random offset of at least 250 trading days. Its best arena fitness is
  the luck ceiling -- what pure selection on noise achieves with this
  budget. A champion that doesn't clearly beat it is noise.

Both write into the *same* run directory the main evolution used
(`control_random/`, `control_null/`), so `vault.py` and `evo-report` can find
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from stockml import evaluate as eval_mod
from stockml.evolution import lineage as lineage_mod
from stockml.evolution.fitness import canonical_majority_daily, evaluate_genome
from stockml.evolution.genome import random_genome
from stockml.evolution.loop import build_evo_dataset, gen_rng, run_generation_loop, write_progress
from stockml.evolution.population import Individual
from stockml.evolution.zones import EvoZones, compute_evo_zones

MIN_SHUFFLE_OFFSET_DAYS = 250


def shift_labels_per_ticker(
    dataset: pd.DataFrame, rng: np.random.Generator, min_offset: int = MIN_SHUFFLE_OFFSET_DAYS
) -> pd.Series:
    """Per ticker, circularly shift the label sequence (ordered by date) by
    an offset drawn uniformly from `[min_offset, n - min_offset)`. A
    circular shift is a permutation of the same multiset of values, so each
    ticker's exact class counts are preserved by construction; the offset
    floor means no feature at day t can plausibly still "know" the label
    now sitting at t through short-range autocorrelation.

    Returns a new label Series aligned to `dataset`'s index (does not
    mutate `dataset`).
    """
    shifted = pd.Series(index=dataset.index, dtype=object)
    for ticker, g in dataset.groupby("ticker", sort=False):
        g_sorted = g.sort_values("date")
        n = len(g_sorted)
        if n <= 2 * min_offset:
            raise ValueError(
                f"ticker {ticker!r} has only {n} rows -- too few for a "
                f">= {min_offset}-day circular shift (need > {2 * min_offset})"
            )
        offset = int(rng.integers(min_offset, n - min_offset))
        labels = g_sorted["label"].to_numpy()
        shifted.loc[g_sorted.index] = np.roll(labels, offset)
    return shifted


# ---------------------------------------------------------------------------
# Random search control
# ---------------------------------------------------------------------------


def run_random_search_control(
    cfg: dict[str, Any], dataset: pd.DataFrame, zones: EvoZones, majority_arena: pd.Series,
    n_evals: int, run_dir: str | Path,
) -> Path:
    """Draws `n_evals` genomes from a dedicated RNG (independent of the main
    evolution's per-generation generators) and evaluates each on the arena.
    """
    control_dir = Path(run_dir) / "control_random"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    rng = gen_rng(cfg["seed"], "random_search_control")
    individuals = [
        Individual(id=f"rs_{i:05d}", generation=0, genome=random_genome(rng), parents=[], origin="random_search")
        for i in range(n_evals)
    ]

    ev = cfg["evolution"]
    to_compute: dict[str, Any] = {}
    for ind in individuals:
        h = ind.genome.stable_hash()
        if h not in to_compute:
            to_compute[h] = ind.genome
    items = list(to_compute.items())
    # See loop.py's identical comment: without this, N outer joblib threads
    # each also fitting an internally-multithreaded model oversubscribes
    # the machine's cores and turns "parallel" into "much slower than serial".
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=ev["n_jobs"], backend="threading")(
            delayed(evaluate_genome)(genome, "arena", dataset, zones, majority_arena, cfg["seed"], ev["fitness_n_boot"])
            for _, genome in items
        )
    threshold, min_days = cfg["leak_alarm"]["threshold"], cfg["leak_alarm"]["min_days"]
    cache = {}
    for (h, genome), result in zip(items, results):
        cache[h] = result
        if eval_mod.check_leak_alarm(result.accuracy, result.n_days, threshold, min_days):
            print(f"[LEAK ALARM] random_search genome {h} tripped the alarm -- see this control's summary")

    for ind in individuals:
        ind.fitness = cache[ind.genome.stable_hash()]
    lineage_mod.append_lineage(control_dir, individuals)

    champion = max(individuals, key=lambda ind: ind.fitness.fitness)
    summary = {
        "kind": "random_search",
        "n_evals": n_evals,
        "n_unique_genomes": len(to_compute),
        "champion_id": champion.id,
        "champion_genome": champion.genome.encode(),
        "champion_fitness": champion.fitness.to_dict(),
    }
    with open(control_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_progress(control_dir, status="completed")
    print(f"[evolve-control random] {n_evals} evals ({len(to_compute)} unique) -> {control_dir}")
    print(f"[evolve-control random] champion {champion.id}: fitness={champion.fitness.fitness:+.4f}pp")
    return control_dir


# ---------------------------------------------------------------------------
# Shifted-label null control
# ---------------------------------------------------------------------------


def run_null_control(cfg: dict[str, Any], run_dir: str | Path) -> Path:
    """Runs the identical evolution (same config, same seed, same
    `run_generation_loop`) against a label-shuffled copy of the dataset.
    """
    control_dir = Path(run_dir) / "control_null"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    raw_panel, dataset = build_evo_dataset(cfg)
    shuffle_rng = gen_rng(cfg["seed"], "null_shuffle")
    dataset = dataset.copy()
    dataset["label"] = shift_labels_per_ticker(dataset, shuffle_rng)

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    zones = compute_evo_zones(
        unique_dates, cfg["split"]["train_years"], cfg["split"]["sanity_days"],
        cfg["split"].get("embargo_days", 1), cfg["zones"]["arena_end"], cfg["zones"]["vault_start"],
    )
    # This run's own canonical majority baseline, computed on the shuffled
    # labels -- "same exam for everyone" *within* this shuffled universe,
    # consistent with how the main run's baseline is computed on real labels.
    majority_arena = canonical_majority_daily(dataset, zones, "arena")

    write_progress(
        control_dir, status="running", last_completed_generation=-1, champion_id=None,
        best_fitness_so_far=-float("inf"), gens_since_improvement=0,
        force_storm_next=False, config_name=cfg["name"],
    )
    return run_generation_loop(
        cfg, dataset, zones, majority_arena, raw_panel, control_dir,
        start_gen=0, cache={}, population=None, best_fitness_so_far=-float("inf"),
        gens_since_improvement=0, force_storm=False, champion_id=None,
        log_prefix="evolve-control null",
    )
