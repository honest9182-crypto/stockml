# stockml

A small, honest research framework for predicting the **next-day direction**
of US large-cap stocks as a three-class problem: **up / down / stagnant**.
This is step 1 of a staged experiment. The point of step 1 is not a good
model. The point is a framework where a good number *cannot be faked*, and
where later steps (online updating, position sizing) plug in without
rewriting anything.

## Non-negotiable principles

These override any convenience. If a shortcut conflicts with one of them,
take the slow route and say so.

1. **No look-ahead, ever.** Every feature at row `t` uses only data with a
   timestamp `<= t`. The label at row `t` is the only thing allowed to touch
   `t+1`. Time-based splits only; never a random split of days. This is why
   `split.py` embargoes (drops) the last training day by default
   (`embargo_days: 1`): that row's *label* is `r_next` computed from the
   first test day's close, so training on it directly touches test-period
   information -- feature-side care (shift tests, truncation tests) would
   never catch this, because it's the label doing the leaking, not a feature.
2. **Baselines are mandatory.** Every results table shows the model next to
   `majority-class` and `always-up` baselines computed on the *same* test
   rows. A number without its baseline is not a result.
3. **Leak alarm.** If any model's out-of-sample accuracy exceeds **60%** on
   more than 250 test days, print a loud warning and automatically run the
   leak diagnostics before reporting. Treat a high score as a bug until
   proven otherwise.
4. **Reproducible.** Fixed seeds, config-driven runs, every run writes its
   config + metrics + predictions to `runs/<timestamp>_<name>/`.
5. **Simple over clever.** No deep learning, no hyperparameter search, no
   feature engineering beyond the documented list. Minimal dependencies. The
   model is deliberately boring; the harness is the deliverable.

## Statistical honesty: day-level vs. row-level significance

A naive significance test treats every (ticker, day) row in the test set as
an independent trial. It isn't: on any single day, all ~500 tickers are
exposed to the same market-wide news and mostly move together, so their
correctness/incorrectness on that day is correlated, not independent. A
row-level test (e.g. a binomial test over hit/miss rows) therefore behaves
as if the test set had orders of magnitude more independent observations
than it really does -- with ~500 tickers x ~3000 test days that's the
difference between treating the sample size as ~1.5 million versus ~3000. The
row-level p-value comes out far smaller (far more "significant") than the
data actually supports.

`evaluate.py` computes both, on purpose:

- `binomial_test_vs_baseline` — the row-level test. Kept in `metrics.json`
  under the key `row_level_overstated` so it's there for comparison, but
  never shown in `report.txt` and never the number to trust.
- `day_level_paired_test` — the honest version. The unit of analysis is one
  number per trading day (that day's pooled accuracy, model vs. majority
  baseline), so the sample size is the number of independent-ish trading
  days, not the number of rows. It reports the mean daily edge in
  percentage points, a one-sided paired t-test of that edge against zero,
  and a 20-day block-bootstrap 95% CI (block, not i.i.d., because
  consecutive trading days are themselves autocorrelated -- an i.i.d.
  bootstrap over days would repeat the same overstatement one level up).

If you add a new significance test, ask what the *actual* unit of
independence is before trusting its p-value.

## Evolutionary search (step 1.5)

`src/stockml/evolution/` adds a genetic-algorithm search over discrete
(features x model family x hyperparameters) genomes on top of step 1. Every
principle above still applies -- the leak alarm runs per genome, baselines
are still mandatory, everything is still seeded and config-driven. The
scientific question this layer exists to answer is **not** "can evolution
find a good model" but **"does evolution find anything random search and
pure luck don't?"** Every design choice here should make that question
easier to answer honestly, not the headline fitness number look better.

**Same exam for everyone.** Labels (`k`, `vol_window`) stay global and
fixed. A genome may choose what it looks at (features) and how it learns
(model family, hyperparameters, how much training history) -- never what it
is judged on. Concretely: there is exactly *one* majority-class baseline per
run, trained on the global train window (independent of any genome's own
`train_years_used`), and every genome's fitness is measured against that
same series (`evolution/fitness.py:canonical_majority_daily`).

**Three time zones, hard-walled, never overlapping** (`evolution/zones.py`):
- **train**: the step-1 training window (start of data -> the train/test
  embargo). Every genome fits here.
- **arena**: test days from the start of test through a config-fixed
  `arena_end` date. Fitness is measured here, thousands of times over a
  run -- evolution *will* overfit to it. That's expected; it's why the
  vault exists.
- **vault**: a config-fixed `vault_start` date through the start of the
  sanity slice. **Never touched during evolution.** Opened exactly once, at
  the end of a run, for a fixed, pre-declared list of individuals
  (`evolution/vault.py`). The guard is filesystem state, not a runtime
  flag: `run_vault` reads the run's `progress.json` and raises if
  `status == "running"` -- it cannot be talked out of refusing.
  `arena_end`/`vault_start` are adjacent calendar dates (no gap), so the
  arena's last day's label technically depends on the vault's first day's
  close -- deliberately *not* embargoed, for the same reason step 1 never
  embargoed test/sanity: a zone's label being *evaluated against* doesn't
  leak the way a zone's label being *trained on* would.

**A vault result is never a reason to change the config and run again.**
If you do run again, the earlier vault results stay in
`runs/evolution/vault_log.jsonl` and in that run's `vault_report.txt` --
nothing is ever removed from the log, including after a later, "better" run.

**Two controls, same budget as the evolution:**
- **Random search** (`evolve-control --kind random`): the same number of
  unique fitness evaluations the evolution actually performed, spent on
  uniformly random genomes. If evolution's best isn't clearly better than
  random search's best, the mating logic added nothing.
- **Shifted-label null** (`evolve-control --kind null`): the identical
  evolution (same config, same seed) run against labels circularly shifted
  per ticker by a random offset of >= 250 trading days -- exact class
  counts preserved, but no feature at day t can know the label now sitting
  at t. Its best arena fitness is the *luck ceiling*: what pure selection on
  noise achieves with this budget. A champion that doesn't clearly beat it
  is noise.

**One seed, a deterministic tree of generators, not one mutable RNG
object.** Every distinct random concern (a generation's reproduction step,
the null control's label shuffle, the random-search control) gets its own
child `np.random.Generator` derived via `evolution/loop.py:gen_rng(seed,
*tags)` -- e.g. `gen_rng(seed, "generation", g)`. All draws for a given
generation happen serially in the main process, in a fixed order, *before*
fitness evaluation is dispatched to parallel workers -- so parallelism can
never perturb the sequence, two runs with the same seed produce
byte-identical `lineage.jsonl`, and `--resume` needs no persisted RNG state
at all (generation `g`'s RNG is simply re-derived the same way).

**Parallel fitness evaluation, without oversubscribing the machine.**
Genomes within a generation (or a random-search control batch) are
evaluated with `joblib.Parallel(backend="threading")` -- threads, not
processes, so the multi-million-row dataset never gets pickled into worker
processes; scikit-learn's fit/predict releases the GIL during the actual
numeric work anyway. But `HistGradientBoostingClassifier` (and BLAS under
`LogReg`) spawn their *own* internal thread pools, so without care, N outer
joblib threads each also fitting an internally-multithreaded model
oversubscribes the machine's cores by a factor of N -- this was measured
directly during development: 16 genomes evaluated one at a time serially
took ~3.3 minutes total; the same 16 through the naive parallel dispatch
didn't finish in 10. The fix is `threadpoolctl.threadpool_limits(limits=1)`
wrapped around every `Parallel(...)` dispatch site
(`evolution/loop.py`/`evolution/controls.py`): it caps each genome's own
fit to one thread, so the outer `Parallel(n_jobs=...)` is the only source of
parallelism. `threadpoolctl` ships as a scikit-learn dependency already, no
new dependency added. The `"xgb"` model family (`models/xgb_model.py`) has
the same failure mode via its own OpenMP thread pool, not necessarily one
`threadpoolctl` controls -- its `XGBClassifier` is constructed with an
explicit `n_jobs=1` for the same reason, rather than trusting
`threadpool_limits` to catch it too.

**Fitness-loop bootstrap cost.** Every genome's `FitnessResult` carries a
block-bootstrap CI, but running step 1's `n_boot=2000` for every one of
~1000 genome evaluations in a run is real, avoidable cost. The hot loop uses
a small `evolution.fitness_n_boot` (default 200); the vault protocol and
final champion reporting re-evaluate the short final list of individuals
with a much larger `evolution.report_n_boot` (default 2000) -- the numbers
that actually get written up get the expensive CI, everything else gets a
cheap approximation good enough for *ranking* genomes against each other.

**Full lineage.** Every individual ever created -- seed, random-init,
elite, bred, lottery, or immigrant -- is appended to `lineage.jsonl` with
its id, generation, genome, parents, which genes mutated, and whether it
came from a storm, the lottery, or immigration. Any champion is traceable
back to generation 0 (`stockml lineage ... --id <individual>`).

## Repo layout

```
stockml/
  pyproject.toml
  README.md
  CLAUDE.md
  configs/
    step1.yaml               # the default step-1 run
    smoke.yaml                # 10 tickers, 2 years, for fast tests
    evo.yaml                   # the default evolution run
  data/
    tickers/sp500.csv         # bundled constituent list
    cache/                    # parquet, git-ignored
  src/stockml/
    data.py                   # download + cache + calendar alignment
    labels.py                 # three-class labels
    features.py                # past-only features
    split.py                   # walk-forward splitting
    walk_forward.py             # the fit/predict-day-by-day loop (shared by run.py and evolution/fitness.py)
    models/
      base.py                  # Model protocol
      baselines.py              # MajorityClass, AlwaysUp
      sklearn_models.py         # LogReg, HGB wrappers (extended with the hyperparameters genomes choose between)
    update.py                  # UpdatePolicy protocol + Frozen (step-2 hook)
    evaluate.py                 # metrics, day-level significance test, leak diagnostics
    evolution/
      genome.py                 # Genome, gene grids, encode/decode/hash, mutate/crossover
      zones.py                   # train/arena/vault time zones
      model_builder.py            # Genome -> fittable Model
      fitness.py                   # evaluate_genome, the canonical majority baseline
      population.py                 # selection, mate choice, lottery, crossover, mutation, storms
      lineage.py                     # lineage.jsonl read/write, ancestor tracing
      controls.py                     # random-search + shifted-label-null controls
      vault.py                         # the vault protocol + its filesystem guard
      outputs.py                       # gene_frequency.csv, plots, family tree
      loop.py                           # the evolve loop, --quick, --resume
    backtest/
      __init__.py               # step-3 hook: Policy + Reward protocol stubs
    run.py                      # ties step 1 together
    cli.py
  tests/
  runs/                        # git-ignored
```

## How to run

```
uv venv --python 3.13
uv pip install -e ".[dev]"

# step 1
uv run stockml download  --config configs/step1.yaml [--refresh]
uv run stockml dataset   --config configs/step1.yaml   # features + labels, prints class balance
uv run stockml run       --config configs/step1.yaml   # walk-forward, writes runs/<ts>_<name>/
uv run stockml report    runs/<ts>_<name>/              # pretty-print the table + baselines + warnings
uv run stockml leakcheck runs/<ts>_<name>/               # leak diagnostics on demand

# step 1.5: evolutionary search
uv run stockml evolve         --config configs/evo.yaml [--quick] [--resume PATH]
uv run stockml evolve-control --config configs/evo.yaml --kind random [--run-dir PATH]
uv run stockml evolve-control --config configs/evo.yaml --kind null   [--run-dir PATH]
uv run stockml vault           runs/evo_<ts>_<name>/                  # one-time look; logged permanently
uv run stockml lineage          runs/evo_<ts>_<name>/ --id <individual>
uv run stockml evo-report        runs/evo_<ts>_<name>/                 # re-render tables + plots

uv run pytest -q
```

Use `configs/smoke.yaml` (10 tickers, 2 years) for fast step-1 iteration, and
`evolve --quick` (20 tickers, 6 genomes, 3 generations) for fast evolution
iteration — both should run in well under a minute once prices are cached.

## Known, accepted limitations (step 1)

- **Survivorship bias**: `data/tickers/sp500.csv` is the *current* S&P 500
  constituent list. Delisted losers are absent from the universe entirely.
  This inflates any aggregate result. Documented, not solved, in step 1.
- **Retroactive adjustment**: `yfinance` with `auto_adjust=True` returns
  adjusted closes that are rewritten by the provider when new splits/dividends
  occur (i.e. a re-download today can change history). This is acceptable for
  direction labels (up/down/stagnant is robust to it) but would NOT be
  acceptable for anything price-level based (step 3 backtesting must not
  reuse this cache naively). This is also why `data.py` caches a per-ticker
  `.meta.json` recording the earliest `start` date ever requested for it:
  a ticker whose real history starts after the configured start (e.g. an
  IPO) can never "cover" that request, so without remembering that a
  request was already made, it would hit the live API again on *every*
  run -- and that live API is not guaranteed to return byte-identical data
  twice. This was directly observed breaking the evolution layer's "same
  seed -> identical run" guarantee (two `--quick` runs, identical genome,
  different fitness) before the `.meta.json` fix.

## Explicitly out of scope (designed for, not built)

- Online / on-the-fly update policies beyond `Frozen` (step 2). The
  `UpdatePolicy` protocol and the walk-forward loop's call sites already
  support this.
- Position sizing, P&L simulation with transaction costs, and any
  path-dependent reward (step 3). `backtest/__init__.py` holds only the
  `Policy` and `Reward` protocol stubs.
- Intraday data, market-wide or cross-sectional features, deep learning,
  hyperparameter search (step 1's own model; the evolution layer's grid
  search is a separate, explicitly-scoped-in thing, not "hyperparameter
  search" in the tuning-against-results sense -- see "Do not tune GA
  parameters" below).
- Update-policy genes (how often to refit, recency weighting,
  refit-on-big-miss) and "focus" genes (which sectors/stocks to train on,
  still judged on the whole market) -- deferred to a later genome version.
  Every field on `Genome` has a default specifically so these can be added
  without breaking genomes already saved in an old `lineage.jsonl`.
- Per-ticker models inside a genome, neural nets, continuous
  hyperparameters (genes are discrete grid values on purpose).
- Position sizing, P&L simulation with transaction costs, and any
  path-dependent reward (step 3). `backtest/__init__.py` holds only the
  `Policy` and `Reward` protocol stubs.

## Working notes

- Do not tune the model. If the result is near baseline, that is the honest
  outcome of step 1 and it's fine. If the leak alarm fires, chase the leak,
  not the score.
- `labels.py` owns the single implementation of `rolling_sigma`/`band`;
  `features.py` imports it rather than recomputing, so the label's band and
  the model's `sigma`/`band` features can never silently diverge.
- **Do not tune GA parameters (population_size, mutation_rate, selection
  pressure, etc.) by looking at vault results.** That's the exact same sin
  as tuning a model on its test set, one level up. If a champion beats the
  luck ceiling only marginally, say so plainly in the report text -- don't
  go adjust `configs/evo.yaml` and run again hoping for a better vault
  number. (Adjusting the GA config for a *new, freshly-vaulted* run is
  fine; going back to reopen an *already-vaulted* run's config in response
  to its own vault result is not -- and the vault log makes sure the
  earlier attempt's number can't quietly disappear either way.)
