"""The evolve loop: builds generation 0, evaluates fitness, breeds the next
generation, and repeats -- writing `lineage.jsonl`, `generations.csv`,
`gene_frequency.csv`, and `progress.json` as it goes so a run is both
observable and resumable.

Reproducibility: every generation's random decisions come from a fresh
`np.random.Generator` derived deterministically from `(seed, "generation",
g)` (see `gen_rng`) -- not one mutating generator whose state would need to
be serialized to resume. Two runs with the same seed produce byte-identical
`lineage.jsonl`, whether or not either of them was interrupted and resumed,
because generation `g`'s RNG only ever depends on `g` and the seed, never on
timing or which individuals happened to finish evaluating first (all RNG
draws happen serially in the main process before any parallel dispatch).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from stockml import data as data_mod
from stockml import evaluate as eval_mod
from stockml import run as run_mod
from stockml.evolution import lineage as lineage_mod
from stockml.evolution import outputs as outputs_mod
from stockml.evolution.device import resolve_device
from stockml.evolution.fitness import FitnessResult, canonical_majority_daily, evaluate_genome
from stockml.evolution.genome import SEED_HGB, SEED_LOGREG, Genome, random_genome
from stockml.evolution.model_builder import build_model
from stockml.evolution.population import (
    ORIGIN_RANDOM_INIT,
    ORIGIN_SEED,
    Individual,
    mean_pairwise_distance,
    next_generation,
)
from stockml.evolution.zones import EvoZones, compute_evo_zones
from stockml.update import Frozen
from stockml.walk_forward import walk_forward_single_model

# 20 tickers for --quick: the 10 already cached by configs/smoke.yaml plus
# 10 more liquid large caps, so a quick run mostly avoids a fresh download.
QUICK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG", "KO", "WMT",
    "META", "NVDA", "V", "MA", "HD", "DIS", "BAC", "PFE", "CSCO", "INTC",
]

GENERATIONS_FIELDS = [
    "generation", "best_fitness", "mean_fitness", "median_fitness",
    "diversity", "storm", "n_evaluated", "n_memoized", "cumulative_evaluations",
]


class LeakAlarmTripped(Exception):
    """Raised (and caught by `evolve`) when a genome's arena accuracy trips
    the step-1 leak alarm -- the whole run halts, per CLAUDE.md.
    """

    def __init__(self, genome: Genome, result: FitnessResult) -> None:
        self.genome = genome
        self.result = result
        super().__init__(
            f"leak alarm: genome {genome.stable_hash()} scored accuracy={result.accuracy:.3f} "
            f"on {result.n_days} arena days"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_evo_config(path: str | Path) -> dict[str, Any]:
    """Parse an evolution config YAML and fill in defaults -- no env-var
    overrides. The pristine, portable version, exactly like
    `run.parse_config`: `evolve()`'s fresh-run path persists this (plus
    `--quick`'s overrides, which ARE a real semantic difference in the run
    and belong in the snapshot) into the run's own config.yaml, never the
    env-var-redirected version -- see `run.apply_env_overrides`'s docstring
    for why. `load_evo_config` (below) is what almost everything else
    should call.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", 0)
    cfg.setdefault("runs_dir", "runs")
    cfg.setdefault("name", "evo")
    cfg.setdefault("split", {}).setdefault("embargo_days", 1)
    cfg.setdefault("leak_alarm", {}).setdefault("threshold", 0.60)
    cfg["leak_alarm"].setdefault("min_days", 250)

    ev = cfg.setdefault("evolution", {})
    ev.setdefault("population_size", 40)
    ev.setdefault("generations", 25)
    ev.setdefault("patience", 10)
    ev.setdefault("n_elite", 2)
    ev.setdefault("n_lottery", 3)
    ev.setdefault("n_immigrants", 2)
    ev.setdefault("selection_pressure", 1.5)
    ev.setdefault("dissimilarity_power", 2.0)
    ev.setdefault("mutation_rate", 0.05)
    ev.setdefault("storm_every", 5)
    ev.setdefault("storm_factor", 5)
    ev.setdefault("min_diversity", 0.10)
    ev.setdefault("fitness_n_boot", 200)
    ev.setdefault("report_n_boot", 2000)
    ev.setdefault("n_jobs", -2)
    ev.setdefault("arena_ticker_subsample", None)
    return cfg


def load_evo_config(path: str | Path) -> dict[str, Any]:
    """`parse_evo_config` plus `STOCKML_RUNS_DIR`/`STOCKML_CACHE_DIR`
    overrides -- the fully resolved, ready-to-run config almost every
    caller wants.
    """
    return run_mod.apply_env_overrides(parse_evo_config(path))


def apply_quick_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["universe"] = {"tickers": list(QUICK_TICKERS)}
    cfg["evolution"] = dict(cfg["evolution"])
    cfg["evolution"]["population_size"] = 6
    cfg["evolution"]["generations"] = 3
    cfg["name"] = f"{cfg.get('name', 'evo')}_quick"
    return cfg


def resolve_evo_tickers(cfg: dict[str, Any]) -> list[str]:
    tickers = run_mod.resolve_tickers(cfg)
    sub = cfg["evolution"].get("arena_ticker_subsample")
    if sub:
        tickers = tickers[:sub]
    return tickers


def build_evo_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like `run.build_dataset`, but resolves the (possibly subsampled)
    evolution ticker list first and makes sure it's cached -- a
    `--quick` run should work without a separate `download` step.
    """
    tickers = resolve_evo_tickers(cfg)
    end = run_mod.resolve_end_date(cfg)
    report = data_mod.download_prices(tickers, cfg["data"]["start"], end, cfg["data"]["cache_dir"])
    if report.ok:
        print(f"[evolve] downloaded {len(report.ok)} new tickers")
    if report.failed:
        print(f"[evolve][WARN] failed to download: {report.failed}")

    cfg_for_build = dict(cfg)
    cfg_for_build["universe"] = {"tickers": tickers}
    return run_mod.build_dataset(cfg_for_build)


# ---------------------------------------------------------------------------
# Deterministic RNG derivation
# ---------------------------------------------------------------------------


def _tag_to_int(tag: str) -> int:
    return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16], 16)


def gen_rng(seed: int, *tags: int | str) -> np.random.Generator:
    """One seed, a deterministic tree of generators: every distinct random
    concern (a generation's reproduction step, the null control's label
    shuffle, the random-search control) gets its own child generator
    derived from `(seed, *tags)`. Two calls with the same arguments always
    produce the same generator -- this is what makes `--resume` need no
    persisted RNG state at all.
    """
    ints = [seed] + [t if isinstance(t, int) else _tag_to_int(t) for t in tags]
    return np.random.default_rng(ints)


# ---------------------------------------------------------------------------
# Population construction / fitness evaluation
# ---------------------------------------------------------------------------


def _initial_population(cfg: dict[str, Any], rng: np.random.Generator) -> list[Individual]:
    pop_size = cfg["evolution"]["population_size"]
    if pop_size < 2:
        raise ValueError("population_size must be >= 2 (the two seeded individuals)")
    pop = [
        Individual(id="000_000", generation=0, genome=SEED_LOGREG, parents=[], origin=ORIGIN_SEED),
        Individual(id="000_001", generation=0, genome=SEED_HGB, parents=[], origin=ORIGIN_SEED),
    ]
    for i in range(2, pop_size):
        pop.append(
            Individual(
                id=f"000_{i:03d}",
                generation=0,
                genome=random_genome(rng),
                parents=[],
                origin=ORIGIN_RANDOM_INIT,
            )
        )
    return pop


def _print_timing_breakdown(results: list[FitnessResult], log_prefix: str = "evolve") -> None:
    """Averages each `FitnessResult.timing_s` phase (see fitness.py's
    TIMING_PHASES) over `results` and prints the split -- e.g. to see what
    share of a genome's evaluation cost fit/predict actually are, and so
    what share GPU acceleration (the "xgb" model_family) could touch.
    """
    from stockml.evolution.fitness import TIMING_PHASES

    if not results:
        return
    avg = {p: float(np.mean([r.timing_s.get(p, 0.0) for r in results])) for p in TIMING_PHASES}
    total = sum(avg.values())
    if total <= 0:
        return
    parts = " ".join(f"{p}={avg[p]:.3f}s({100 * avg[p] / total:.0f}%)" for p in TIMING_PHASES)
    print(f"[{log_prefix}] timing (avg over {len(results)} genomes this generation): {parts} total={total:.3f}s")


def _evaluate_population(
    individuals_needing_eval: list[Individual],
    dataset: pd.DataFrame,
    zones: EvoZones,
    majority_arena: pd.Series,
    cfg: dict[str, Any],
    cache: dict[str, FitnessResult],
    log_prefix: str = "evolve",
) -> tuple[int, int]:
    """Evaluates every individual with `fitness is None`, deduped by genome
    hash and memoized in `cache` across the whole run. Returns
    (n_evaluated, n_memoized) for this call. Raises `LeakAlarmTripped` the
    first time any freshly-computed genome's arena accuracy trips the alarm.
    """
    to_compute: dict[str, Genome] = {}
    for ind in individuals_needing_eval:
        h = ind.genome.stable_hash()
        if h not in cache and h not in to_compute:
            to_compute[h] = ind.genome

    if to_compute:
        items = list(to_compute.items())
        ev = cfg["evolution"]
        # threadpoolctl.threadpool_limits(1): HistGradientBoostingClassifier
        # (and BLAS under LogReg) spawn their own internal thread pools: N
        # outer joblib threads each also fitting a many-threaded model would
        # oversubscribe the machine's cores by a factor of N, turning "run N
        # genomes in parallel" into "thrash". Limiting each genome's own fit
        # to one thread makes the outer Parallel(n_jobs=...) the only source
        # of parallelism, which is what it's there for.
        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=ev["n_jobs"], backend="threading")(
                delayed(evaluate_genome)(
                    genome, "arena", dataset, zones, majority_arena, cfg["seed"], ev["fitness_n_boot"]
                )
                for _, genome in items
            )
        _print_timing_breakdown(results, log_prefix)
        threshold = cfg["leak_alarm"]["threshold"]
        min_days = cfg["leak_alarm"]["min_days"]
        for (h, genome), result in zip(items, results):
            cache[h] = result
            if eval_mod.check_leak_alarm(result.accuracy, result.n_days, threshold, min_days):
                raise LeakAlarmTripped(genome, result)

    n_evaluated = 0
    n_memoized = 0
    credited: set[str] = set()
    for ind in individuals_needing_eval:
        h = ind.genome.stable_hash()
        ind.fitness = cache[h]
        if h in to_compute and h not in credited:
            ind.memoized = False
            credited.add(h)
            n_evaluated += 1
        else:
            ind.memoized = True
            n_memoized += 1
    return n_evaluated, n_memoized


def _handle_leak_alarm(
    err: LeakAlarmTripped,
    dataset: pd.DataFrame,
    zones: EvoZones,
    raw_panel: pd.DataFrame,
    cfg: dict[str, Any],
    run_dir: Path,
) -> None:
    """Refits the flagged genome (evaluate_genome doesn't keep fitted models
    around -- this cost is only ever paid on the rare halt path) and runs
    the step-1 leak diagnostics against it."""
    print(
        f"\n[LEAK ALARM] {err}\n"
        f"Halting the evolution run. Running leak diagnostics before anything else.\n"
    )
    dates = pd.to_datetime(dataset["date"])
    train_end = zones.train_end
    train_start = max(zones.train_start, train_end - pd.DateOffset(years=err.genome.train_years_used))
    train_df = dataset[(dates >= train_start) & (dates <= train_end)]
    arena_df = dataset[(dates >= zones.arena_start) & (dates <= zones.arena_end)]
    active = err.genome.active_features()
    model = build_model(err.genome, cfg["seed"])
    model.fit(train_df[active], train_df["label"])

    report = eval_mod.run_leak_diagnostics(
        arena_df, model, active, raw_panel,
        cfg["data"]["cache_dir"], cfg["labels"]["k"], cfg["labels"]["vol_window"],
        seed=cfg["seed"],
    )
    with open(run_dir / "leak_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"genome": err.genome.encode(), "fitness": err.result.to_dict(), "diagnostics": report},
            f, indent=2,
        )
    print(f"[LEAK ALARM] wrote {run_dir / 'leak_report.json'}")


# ---------------------------------------------------------------------------
# Progress / generations.csv persistence
# ---------------------------------------------------------------------------


def write_progress(run_dir: Path, **fields: Any) -> None:
    with open(run_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, default=str)


def read_progress(run_dir: str | Path) -> dict[str, Any]:
    with open(Path(run_dir) / "progress.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _append_generations_csv(run_dir: Path, row: dict[str, Any]) -> None:
    path = run_dir / "generations.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GENERATIONS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _fitness_result_from_dict(d: dict[str, Any]) -> FitnessResult:
    # .get(..., default) for "device"/"timing_s": a lineage.jsonl written
    # before those fields existed still decodes -- same rule as Genome's
    # own fields (see genome.py's module docstring).
    return FitnessResult(
        genome_hash=d["genome_hash"], zone=d["zone"], n_days=d["n_days"],
        mean_edge_pp=d["mean_edge_pp"], se_pp=d["se_pp"], fitness=d["fitness"],
        accuracy=d["accuracy"], ci95_low_pp=d["ci95_low_pp"], ci95_high_pp=d["ci95_high_pp"],
        prediction_mix=d["prediction_mix"],
        device=d.get("device", "cpu"), timing_s=d.get("timing_s", {}),
    )


def _individual_from_record(rec: dict[str, Any]) -> Individual:
    return Individual(
        id=rec["id"], generation=rec["generation"], genome=Genome.decode(rec["genome"]),
        parents=rec["parents"], origin=rec["origin"], mutated_genes=rec["mutated_genes"],
        storm=rec["storm"],
        fitness=_fitness_result_from_dict(rec["fitness"]) if rec["fitness"] is not None else None,
        fitness_ref=rec["fitness_ref"], memoized=rec["memoized"],
    )


# ---------------------------------------------------------------------------
# The generation loop core -- shared by `evolve()` and the null control
# (`evolution/controls.py`), which must run "the identical evolution" (same
# config, same seed, same code path) against a label-shuffled dataset.
# ---------------------------------------------------------------------------


def run_generation_loop(
    cfg: dict[str, Any],
    dataset: pd.DataFrame,
    zones: EvoZones,
    majority_arena: pd.Series,
    raw_panel: pd.DataFrame,
    run_dir: Path,
    start_gen: int,
    cache: dict[str, FitnessResult],
    population: list[Individual] | None,
    best_fitness_so_far: float,
    gens_since_improvement: int,
    force_storm: bool,
    champion_id: str | None,
    log_prefix: str = "evolve",
) -> Path:
    n_generations = cfg["evolution"]["generations"]
    patience = cfg["evolution"]["patience"]
    ev_cfg = cfg["evolution"]

    t_run_start = time.time()
    last_gen_completed = start_gen - 1
    try:
        for gen in range(start_gen, n_generations):
            gen_t0 = time.time()
            rng = gen_rng(cfg["seed"], "generation", gen)
            is_storm = False
            if gen == 0:
                population = _initial_population(cfg, rng)
            else:
                population, is_storm = next_generation(population, gen, ev_cfg, rng, force_storm)

            needing_eval = [ind for ind in population if ind.fitness is None]
            n_eval, n_memo = _evaluate_population(
                needing_eval, dataset, zones, majority_arena, cfg, cache, log_prefix
            )

            lineage_mod.append_lineage(run_dir, [ind for ind in population if ind.fitness is not None])

            diversity = mean_pairwise_distance(population)
            force_storm = diversity < ev_cfg["min_diversity"]
            if force_storm:
                print(
                    f"[{log_prefix}] gen {gen}: diversity {diversity:.4f} < min_diversity "
                    f"{ev_cfg['min_diversity']} -- forcing a storm next generation"
                )

            best_ind = max(population, key=lambda ind: ind.fitness.fitness)
            gen_best = best_ind.fitness.fitness
            gen_mean = float(np.mean([ind.fitness.fitness for ind in population]))
            gen_median = float(np.median([ind.fitness.fitness for ind in population]))

            if gen_best > best_fitness_so_far:
                best_fitness_so_far = gen_best
                champion_id = best_ind.id
                gens_since_improvement = 0
            else:
                gens_since_improvement += 1

            _append_generations_csv(run_dir, {
                "generation": gen, "best_fitness": gen_best, "mean_fitness": gen_mean,
                "median_fitness": gen_median, "diversity": diversity, "storm": is_storm,
                "n_evaluated": n_eval, "n_memoized": n_memo, "cumulative_evaluations": len(cache),
            })
            outputs_mod.write_gene_frequency_row(run_dir, outputs_mod.gene_frequency_row(population, gen))

            gen_elapsed = time.time() - gen_t0
            print(
                f"[{log_prefix}] gen {gen}: best={gen_best:+.4f}pp mean={gen_mean:+.4f}pp "
                f"diversity={diversity:.3f}{' [STORM]' if is_storm else ''} "
                f"evaluated={n_eval} memoized={n_memo} ({gen_elapsed:.1f}s)"
            )
            if gen == 0:
                eta_s = gen_elapsed * (n_generations - 1)
                print(
                    f"[{log_prefix}] generation 0 took {gen_elapsed:.1f}s; ETA for the remaining "
                    f"{n_generations - 1} generations: ~{eta_s / 60:.1f} min "
                    f"(shrink population_size/generations in the config if that's too long)"
                )

            last_gen_completed = gen
            write_progress(
                run_dir, status="running", last_completed_generation=gen, champion_id=champion_id,
                best_fitness_so_far=best_fitness_so_far, gens_since_improvement=gens_since_improvement,
                force_storm_next=force_storm, config_name=cfg["name"],
            )

            if gens_since_improvement >= patience:
                print(f"[{log_prefix}] early stop: no improvement for {patience} generations (gen {gen})")
                break

    except LeakAlarmTripped as err:
        _handle_leak_alarm(err, dataset, zones, raw_panel, cfg, run_dir)
        write_progress(
            run_dir, status="halted_leak_alarm", last_completed_generation=last_gen_completed,
            champion_id=champion_id, best_fitness_so_far=best_fitness_so_far,
            gens_since_improvement=gens_since_improvement, force_storm_next=force_storm,
            config_name=cfg["name"],
        )
        return run_dir

    write_progress(
        run_dir, status="completed", last_completed_generation=last_gen_completed,
        champion_id=champion_id, best_fitness_so_far=best_fitness_so_far,
        gens_since_improvement=gens_since_improvement, force_storm_next=force_storm,
        config_name=cfg["name"],
    )
    outputs_mod.render_all(run_dir)
    total_elapsed = time.time() - t_run_start
    print(
        f"[{log_prefix}] done in {total_elapsed / 60:.1f} min -> {run_dir} "
        f"(champion: {champion_id}, fitness={best_fitness_so_far:+.4f}pp, "
        f"{len(cache)} unique genomes evaluated)"
    )
    return run_dir


# ---------------------------------------------------------------------------
# The evolve loop
# ---------------------------------------------------------------------------


def evolve(config_path: str | Path | None, quick: bool = False, resume: str | Path | None = None) -> Path:
    if resume:
        run_dir = Path(resume)
        with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_mod.apply_env_overrides(cfg)
        progress = read_progress(run_dir)
        if progress["status"] == "completed":
            print(f"[evolve] {run_dir} is already completed -- nothing to resume")
            return run_dir
        start_gen = progress["last_completed_generation"] + 1
        records = lineage_mod.read_lineage(run_dir)
        cache: dict[str, FitnessResult] = {}
        for rec in records.values():
            if rec["fitness"] is not None and rec["genome_hash"] not in cache:
                cache[rec["genome_hash"]] = _fitness_result_from_dict(rec["fitness"])
        last_gen_records = sorted(
            (r for r in records.values() if r["generation"] == progress["last_completed_generation"]),
            key=lambda r: r["id"],
        )
        population = [_individual_from_record(r) for r in last_gen_records]
        best_fitness_so_far = progress["best_fitness_so_far"]
        gens_since_improvement = progress["gens_since_improvement"]
        force_storm = progress.get("force_storm_next", False)
        champion_id = progress.get("champion_id")
        print(f"[evolve] resuming {run_dir} from generation {start_gen}")
    else:
        # cfg_to_persist (pristine, plus --quick's real semantic overrides)
        # is what gets written to this run's own config.yaml snapshot; cfg
        # (env-override-applied) is what actually runs -- see
        # parse_evo_config's docstring for why they're not the same object.
        cfg_to_persist = parse_evo_config(config_path)
        if quick:
            cfg_to_persist = apply_quick_overrides(cfg_to_persist)
        # Informational, not load-bearing (unlike runs_dir/cache_dir): what
        # an "xgb" genome would actually run on, on this machine, right now.
        # Recorded (not just logged) so a run's own config.yaml always says
        # what device it evaluated xgb genomes on -- see FitnessResult's own
        # per-evaluation `device` field for the authoritative, always-live
        # record if this run gets resumed on different hardware later.
        cfg_to_persist.setdefault("evolution", {})["device"] = resolve_device("xgb")
        cfg = copy.deepcopy(cfg_to_persist)
        run_mod.apply_env_overrides(cfg)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["runs_dir"]) / f"evo_{ts}_{cfg['name']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_to_persist, f, sort_keys=False)
        start_gen = 0
        cache = {}
        population = None
        best_fitness_so_far = -float("inf")
        gens_since_improvement = 0
        force_storm = False
        champion_id = None
        write_progress(
            run_dir, status="running", last_completed_generation=-1, champion_id=None,
            best_fitness_so_far=best_fitness_so_far, gens_since_improvement=0,
            force_storm_next=False, config_name=cfg["name"],
        )

    raw_panel, dataset = build_evo_dataset(cfg)
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    zones = compute_evo_zones(
        unique_dates,
        cfg["split"]["train_years"],
        cfg["split"]["sanity_days"],
        cfg["split"].get("embargo_days", 1),
        cfg["zones"]["arena_end"],
        cfg["zones"]["vault_start"],
    )
    print(
        f"[evolve] train={zones.train_start.date()}..{zones.train_end.date()} "
        f"arena={zones.arena_start.date()}..{zones.arena_end.date()} "
        f"vault={zones.vault_start.date()}..{zones.vault_end.date()} (untouched)"
    )
    majority_arena = canonical_majority_daily(dataset, zones, "arena")

    return run_generation_loop(
        cfg, dataset, zones, majority_arena, raw_panel, run_dir,
        start_gen, cache, population, best_fitness_so_far,
        gens_since_improvement, force_storm, champion_id, log_prefix="evolve",
    )
