# stockml

A small, honest research framework for predicting the **next-day direction**
of US large-cap stocks as a three-class problem: **up / down / stagnant**.

This is step 1 of a staged experiment. The point of step 1 is not a good
model — the model here is deliberately boring (a default-hyperparameter
logistic regression and a default-hyperparameter gradient-boosted tree, no
tuning). The point is a framework where a good number *cannot be faked*, and
where later steps (online updating, position sizing) plug in without
rewriting anything. See `CLAUDE.md` for the full set of non-negotiable
design principles and the repo layout.

## Install

```
uv venv --python 3.13
uv pip install -e ".[dev]"
```

## Usage

```
uv run stockml download  --config configs/step1.yaml [--refresh]
uv run stockml dataset   --config configs/step1.yaml   # features + labels, prints class balance
uv run stockml run       --config configs/step1.yaml   # walk-forward, writes runs/<ts>_<name>/
uv run stockml report    runs/<ts>_<name>/              # pretty-print the table + baselines + warnings
uv run stockml leakcheck runs/<ts>_<name>/               # leak diagnostics on demand

# step 1.5: evolutionary search (see below)
uv run stockml evolve         --config configs/evo.yaml [--quick] [--resume PATH]
uv run stockml evolve-control --config configs/evo.yaml --kind random [--run-dir PATH]
uv run stockml evolve-control --config configs/evo.yaml --kind null   [--run-dir PATH]
uv run stockml vault           runs/evo_<ts>_<name>/
uv run stockml lineage          runs/evo_<ts>_<name>/ --id <individual>
uv run stockml evo-report        runs/evo_<ts>_<name>/

uv run pytest -q
```

Use `configs/smoke.yaml` (10 tickers, ~2 years) for fast step-1 iteration,
and `evolve --quick` (20 tickers, 6 genomes, 3 generations) for fast
evolution iteration — both run in well under a minute per generation once
prices are cached.

## How the pipeline fits together

```
data.py         download + cache OHLCV per ticker (parquet), align to the union trading calendar
labels.py       per-ticker up/down/stagnant labels from a volatility-scaled band
features.py     per-ticker past-only features (imports labels.rolling_sigma -- one implementation)
split.py        walk-forward train/test/sanity split by date, asserted at runtime, 1-day train/test embargo
walk_forward.py the fit/predict-day-by-day loop, shared by run.py and evolution/fitness.py
models/         Model protocol; MajorityClass/AlwaysUp baselines; LogReg/HGB (boring, untuned)
update.py       UpdatePolicy protocol + Frozen (step-2 hook; the loop calls it every test day)
evaluate.py     metrics, per-ticker/per-class breakdowns, day-level significance test, leak diagnostics
evolution/      step 1.5: genetic search over (features x model x hyperparams) genomes -- see below
run.py          orchestrates step 1, writes runs/<ts>_<name>/
backtest/       Policy + Reward protocol stubs only (step-3 hook)
```

## Known, accepted limitations (step 1)

- **Survivorship bias**: `data/tickers/sp500.csv` is a snapshot of the
  *current* S&P 500 constituent list (fetched once, `as_of` date recorded in
  the file). Stocks that were delisted, acquired, or dropped from the index
  during the sample period are simply absent from the universe. This
  inflates any aggregate result versus what an investor living through that
  period would actually have seen. It is a documented, *accepted* limitation
  of step 1 — not something a later re-fetch of the list would fix; a real
  fix needs a point-in-time constituent history, which is out of scope here.
- **Retroactive adjustment**: prices are downloaded with `yfinance`,
  `auto_adjust=True`. Adjusted closes are rewritten by the data provider
  whenever a new split or dividend occurs, so re-downloading the same date
  range next month can silently change historical values. This is fine for
  direction labels (up/down/stagnant is robust to a scale adjustment) but
  would **not** be acceptable for anything price-level based — a step-3
  backtest must not reuse this cache naively.

## Two design details worth knowing about

- **Train/test embargo**: `split.py` drops the last training day by default
  (`embargo_days: 1`). That row's label is `r_next`, computed from the
  *first test day's* close — training on it would leak test-period
  information through the label, not a feature, so no feature-side check
  would ever catch it.
- **Day-level, not row-level, significance**: a plain binomial test over
  (ticker, day) rows treats each row as an independent trial, but on any
  given day hundreds of tickers move together on market-wide news — so a
  row-level test overstates significance by pretending there are far more
  independent observations than there are. `report.txt` shows a day-level
  paired test instead (mean edge in points, t-test, 20-day block-bootstrap
  95% CI); the old row-level p-value is kept in `metrics.json` for
  comparison, explicitly labelled `row_level_overstated`. Full rationale in
  `CLAUDE.md`.

## Non-negotiable principles (see CLAUDE.md for the full text)

1. No look-ahead, ever — every feature at row `t` uses only data `<= t`.
2. Baselines are mandatory — every table shows majority-class and always-up
   on the same test rows.
3. Leak alarm — accuracy above 60% on more than 250 test days triggers
   automatic leak diagnostics before the result is trusted.
4. Reproducible — fixed seeds, config-driven runs, every run's config +
   metrics + predictions + fitted models land in `runs/<timestamp>_<name>/`.
5. Simple over clever — no deep learning, no hyperparameter search, no
   feature engineering beyond the documented list.

## Step-1 results: full S&P 500, 2010-01-01 → 2026-09-03

`configs/step1.yaml`, pooled model, `train_years: 3`, `k: 0.5`, `vol_window: 20`,
`embargo_days: 1`. 493 of 503 tickers survived the `min_history_days: 1000`
filter (10 too short — young listings/spinoffs: GEHC, GEV, KVUE, RDDT, SOLV,
VLTO, etc.). Run: `runs/20260903_172551_step1/`.

> **Corrected numbers.** An earlier version of this table (`runs/20260903_005859_step1/`)
> was built on a `load_panel` bug: it never sliced each ticker's cached
> parquet to `data.start`/`data.end`, so a ticker's dataset was silently
> whatever date range happened to be on disk rather than what the config
> asked for. In practice this meant **AAPL, MSFT, AMZN, GOOGL, JPM, XOM,
> JNJ, PG, KO, and WMT — ten S&P 500 blue chips — were silently excluded**
> from that run: their cache had only ever been widened for `smoke.yaml`
> (start `2024-09-01`), so under the bug they were loaded with under a year
> of history, failed the `min_history_days: 1000` filter, and were dropped
> with no warning distinguishing them from a genuine short-history ticker.
> Fixed by having `load_panel` slice each ticker's cached frame to
> `[start, end]` before alignment (`data.py`); the table below is the
> re-run with all ten correctly included.

**Class balance** (train / test, both healthy — no warning fired):

| split | down | stagnant | up |
|---|---|---|---|
| train | 26.9% | 42.5% | 30.5% |
| test  | 26.3% | 43.4% | 30.3% |

**Test set: 1,599,411 rows, 3,347 trading days** (pooled across all tickers,
walk-forward, one model frozen after the initial 3-year train window):

| model | accuracy | balanced acc. | macro F1 |
|---|---|---|---|
| always_up | 0.3031 | 0.3333 | 0.1551 |
| majority_class | 0.4337 | 0.3333 | 0.2017 |
| logreg | 0.4363 | 0.3485 | 0.2642 |
| hgb | 0.4345 | 0.3583 | 0.3018 |

**Day-level paired test vs. majority baseline** — the honest significance
test (see "Two design details worth knowing about" above): one observation
per trading day, not per row.

| model | mean daily edge | t-test p-value | 95% CI (block bootstrap) |
|---|---|---|---|
| always_up | −13.065pp | 1.0 | [−14.055, −12.042]pp |
| logreg | **+0.258pp** | **0.0082** | **[+0.057, +0.503]pp** |
| hgb | +0.084pp | 0.2550 | [−0.192, +0.406]pp |

This is the more interesting result than the headline accuracy table: at the
row level, both logreg and HGB looked "significant" (p < 0.01). At the
*day* level — the actual unit of independence — **logreg's edge survives
(barely): a real but tiny ~0.26 point/day advantage. HGB's does not**: its
day-level p-value is 0.255, indistinguishable from noise, and its per-year
edge is negative in 5 of the last 5 years (2022–2026, see `report.txt`).
The row-level test would have told you both models "worked"; it doesn't
survive contact with the actual number of independent trading days.

**Sanity slice** (final 10 trading days — 4,521 rows, not merged into the
table above, not statistically meaningful on its own, exists only to prove
the pipeline runs on the freshest data):

| model | accuracy |
|---|---|
| always_up | 0.2134 |
| majority_class | 0.4787 |
| logreg | 0.4727 |
| hgb | 0.4534 |

**Reading this honestly**: no leak alarm fired (max accuracy 43.6%, nowhere
near the 60% threshold). `stockml leakcheck` was run on this exact result as
a final check: the label-alignment audit and the truncation test (rebuilding
features from data truncated at each sampled day) both came back with **0
mismatches**, and the shift test (staling every feature by one extra day)
dropped accuracy only slightly (hgb −0.0043, logreg −0.0027) rather than
collapsing to baseline — so logreg's small edge looks real, not leaked. It's
also modest enough, and close enough to the honest baseline, that step 1 did
its job: it didn't hand back a suspiciously good number to chase down, and
the day-level test caught a case (HGB) where the row-level number would
have. The corrected universe (ten more, and more liquid, tickers) barely
moved these numbers — the framework's conclusion doesn't hinge on the bug.

## Step 1.5: evolutionary search

`src/stockml/evolution/` adds a genetic-algorithm search over discrete
(features x model family x hyperparameters) genomes on top of step 1. Full
design rationale — the three time zones, the vault protocol, the two
controls, the RNG/reproducibility scheme, the fitness-loop bootstrap-cost
tradeoff — is in `CLAUDE.md`'s "Evolutionary search (step 1.5)" section.
Short version:

- **The question this answers is not "can evolution find a good model."**
  It's **"does evolution find anything random search and pure luck don't?"**
- **Same exam for everyone**: one canonical majority-class baseline per run;
  a genome picks what it looks at and how it learns, never what it's judged
  on.
- **Three zones**: `train` (fit here) -> `arena` (fitness measured here,
  thousands of times — evolution *will* overfit to it) -> `vault` (never
  touched during evolution; opened exactly once, for a fixed pre-declared
  list of individuals, guarded by the run's own `progress.json`).
- **Two controls, same budget**: random search (same number of unique
  fitness evaluations, uniformly random genomes) and a shifted-label null
  (the identical evolution run on labels circularly shifted >= 250 trading
  days per ticker — the luck ceiling).
- **A vault result is never a reason to change the config and run again.**
  Every look is permanently logged to `runs/evolution/vault_log.jsonl`.

### Mechanism sanity check (`--quick`, 20 tickers, 6 genomes, 3 generations)

This is **not** the real answer — it's a fast, small run whose only purpose
is to prove the machinery produces the qualitatively right ordering before
committing a night to the real `configs/evo.yaml` run. It does:

| | arena fitness (mean edge - 1 SE, points/day) |
|---|---|
| evolution's champion | +0.4312 |
| random search's champion (same budget: 16 evals) | +0.0436 |
| shifted-label null's champion (luck ceiling) | +0.0077 |

Evolution's champion clearly beats random search, which clearly beats the
null — exactly the honest picture the harness is built to surface (or fail
to surface) at real scale. On this toy run the champion happens to be the
seeded `SEED_LOGREG` individual itself (unsurprising with 20 tickers and 3
generations); whether evolution actually discovers something *better* than
the step-1 seeds is precisely the open question the real overnight run
(`configs/evo.yaml`, `population_size: 40`, `generations: 25`, full S&P 500)
is for.

### A bug the `--quick` run caught before it could waste a night

Running genomes in parallel was initially measured to be *slower* than
serial: `HistGradientBoostingClassifier` spawns its own internal thread
pool, so N outer `joblib` threads each also fitting an internally
multithreaded model oversubscribed the machine's cores into thrashing (16
genomes: ~3.3 min serial vs. not finishing in 10 min "parallel"). Fixed with
`threadpoolctl.threadpool_limits(1)` around every parallel dispatch site —
already a scikit-learn dependency, no new one added. This is exactly why
`--quick` exists: catching this on a 3-minute run instead of during a
population-40/generations-25/full-S&P-500 overnight run.
