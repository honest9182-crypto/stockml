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

# running evolve on Kaggle (see "Running on Kaggle" below)
uv run stockml fetch-run <kernel-ref>   # pull a Kaggle notebook's run output into runs/

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

### Model families and GPU acceleration

Three model families a genome can choose (`model_family` gene): `logreg`
and `hgb` (both scikit-learn, matching step 1's own two models exactly —
the two seeded individuals, `SEED_LOGREG`/`SEED_HGB`, stay scikit-learn
always, regardless of what else is available), and `xgb`
(`models/xgb_model.py`). `xgb` reuses the `hgb_*` hyperparameter genes
rather than adding a parallel gene set (`max_depth`, `learning_rate`,
`n_estimators` <- `hgb_max_iter`, `min_child_weight` <- `hgb_min_samples_leaf`,
`reg_lambda` <- `hgb_l2`) — see that module's docstring for the exact
mapping and two small semantic differences worth knowing about (`max_depth
=None` and `min_child_weight`'s scale don't mean quite the same thing to
XGBoost as to `HistGradientBoostingClassifier`). `xgb` runs on
`device="cuda"` automatically when a GPU is present (`evolution/device.py`'s
`nvidia-smi`-based detection — e.g. a Kaggle notebook with its accelerator
set to GPU) and `"cpu"` otherwise, with no config change either way; which
device actually ran is recorded per-evaluation in `FitnessResult.device`
and, for the run as a whole, in its own `config.yaml`'s `evolution.device`.

Every genome evaluation's timing (`fit` / `predict` / `metrics` /
`bootstrap` — `evolution/fitness.py`'s `TIMING_PHASES`) is recorded into its
`FitnessResult` and the average is printed once per generation — `fit` and
`predict` are the only two phases a device could ever touch; `metrics` and
`bootstrap` are pure pandas/numpy on the resulting predictions and always
run on CPU. On a `--quick` run (20 tickers, generation 0's 6 fresh
evaluations, this machine's own GPU, no other load on the machine) adding
`xgb` to the drawable grid — some of generation 0's random genomes landing
on GPU-accelerated `xgb` — measurably dropped the average:

| | fit | predict | metrics | bootstrap | total (avg/genome) |
|---|---|---|---|---|---|
| before (`logreg`/`hgb` only) | 0.585s (47%) | 0.623s (50%) | 0.022s (2%) | 0.011s (1%) | 1.242s |
| after (`xgb` included, GPU auto-selected) | 0.468s (68%) | 0.183s (27%) | 0.027s (4%) | 0.012s (2%) | 0.691s |

**`predict`'s own cost dropped separately, for every model family, from a
different change**: `walk_forward_single_model`'s Frozen policy (every
genome uses it) used to predict its zone one day at a time — a leftover
of the loop's general shape, needed for policies that update *during* the
walk, which Frozen never does. It now predicts the whole zone in a single
batched `predict_proba` call when the policy is `Frozen` (verified
byte-identical to the old day-by-day output, same values *and* row order,
in `test_walk_forward_batch.py`; the day-by-day path stays exactly as
before for any future non-Frozen policy). On the arena zone's ~1,682
trading days, that's one `predict_proba` call instead of ~1,682 of them:
`predict`'s average dropped from **~18–24s to ~0.06–0.2s per genome** (the
numbers above are already post-optimization) — a large enough win on its
own that it's most of why the whole table above reads in seconds rather
than tens of seconds.

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

## Running on Kaggle

`configs/evo.yaml`'s real (non-`--quick`) run is a genuine overnight job —
too long, and too much CPU, for most laptops. Kaggle notebooks give a free
CPU (or, with the notebook's accelerator set to GPU, free GPU) environment
for exactly this, at the cost of a hard session limit. Everything below
lives in `kaggle/` and `scripts/upload_cache_dataset.py`; none of it
changes what a local run does. Setting the notebook's accelerator to GPU
needs no config change either — `xgb` genomes (see "Model families and GPU
acceleration" above) detect and use it automatically, `kaggle/requirements.txt`
already pins a CUDA-capable `xgboost` wheel, and `logreg`/`hgb` genomes are
unaffected either way (scikit-learn, CPU-only).

**The 12-hour session cap.** A Kaggle notebook session is killed at 12
hours regardless of progress. `configs/evo.yaml`'s default (population 40,
generations 25, full S&P 500) will not finish in one session — that's
expected. `evolve`'s own persistence (`progress.json`, `lineage.jsonl`,
generation-by-generation `--resume` — see "Evolutionary search" above) is
exactly what makes splitting a run across sessions lossless: a session that
gets killed mid-generation loses at most the generation in progress, never
anything already written.

**Two shortcut stages** (added after the batched-predict optimisation brought a
full evolution down to roughly 3 hours on a Kaggle CPU session): `STAGE = "full"`
runs evolve, then the random-search control, then the null control, all in one
session (~9 hours, inside the 12-hour cap with margin), and `STAGE = "controls"`
runs both controls back to back against an attached, already-completed run.
Nothing changes about what each stage does; they just run in succession so no
PC has to stay on between them. Only the vault stays local.

**The chaining order**, each step a separate Kaggle notebook session (via
`kaggle/stage.ipynb`, described below):

1. **`evolve`** — starts a fresh run against `configs/evo.yaml`.
2. **`resume`** — continues the same run from its last completed
   generation. Repeat across as many sessions as it takes until
   `progress.json` reports `"status": "completed"`.
3. **`random`** and **`null`** — the two controls (see "Evolutionary
   search" above), run against the now-completed evolution run, same
   budget each. Either order; each is its own session if it doesn't fit
   in the remainder of one.
4. **`vault`** — run **locally**, not on Kaggle, once you've pulled the
   completed run down with `fetch-run` (below). It's a few rows and a log
   write; it doesn't need a Kaggle session, and CLAUDE.md's vault
   guard (`progress.json` must say `"status": "completed"`) applies the
   same either way.

### One-time setup: upload the price cache

Kaggle notebooks can't see your local `data/cache/`. Build it locally first
(`stockml download --config configs/evo.yaml`, full S&P 500 back to 2010 —
this is the slow, one-time part), then package and upload it as a private
Kaggle Dataset:

```
pip install kaggle   # once; see https://www.kaggle.com/docs/api#authentication
                      # for ~/.kaggle/kaggle.json credential setup

uv run python scripts/upload_cache_dataset.py --kaggle-username <you> --dry-run   # review first
uv run python scripts/upload_cache_dataset.py --kaggle-username <you>              # then actually create it

# later, once the local cache has grown (more history, more tickers):
uv run python scripts/upload_cache_dataset.py --kaggle-username <you> --version
```

This never makes anything public — `kaggle datasets create` defaults to
private and the script never passes `-u`/`--public`. It packages
`data/cache/*.parquet` plus a pinned snapshot of `data/tickers/sp500.csv`
(a dataset-metadata.json is generated alongside them); see the script's
docstring for why the layout is deliberately flat.

### The notebook: `kaggle/stage.ipynb`

Upload this notebook to Kaggle (or paste its cells into a new one), attach
the price-cache dataset as an input, and edit the `STAGE` variable in its
"Configure" cell to one of `evolve` / `resume` / `random` / `null` per the
chaining order above (`resume`/`random`/`null` also need a previous run
attached as input — the notebook's own prior committed output works).
Its cells, in order:

1. Clone the repo (`REPO_URL`/`REPO_REF` from the Configure cell).
2. Authenticate the `kaggle` CLI from Kaggle Secrets: add `KAGGLE_USERNAME`
   and `KAGGLE_KEY` Secrets to the notebook (Add-ons → Secrets, the same two
   fields as your local `~/.kaggle/kaggle.json`) and this cell sets them as
   env vars for the session. Only needed for step 6 below — the main stages
   never call the `kaggle` CLI themselves — and harmless to leave in
   otherwise.
3. `pip install -r kaggle/requirements.txt` (exact versions, pinned from
   `uv.lock` — see below), then `pip install -e .` for the `stockml`
   command itself.
4. Symlink `/kaggle/input/<cache dataset>` onto `data/cache`, and set
   `STOCKML_CACHE_DIR`/`STOCKML_RUNS_DIR` (env-var overrides — see
   `run.apply_env_overrides`) so `/kaggle/working/runs` is where output
   lands regardless of where the repo got cloned, with no config file
   edited.
5. For `resume`/`random`/`null`: copy the previous run's folder from the
   attached input into `/kaggle/working/runs` (a real, writable copy —
   `--resume`/`--run-dir` need to write into it).
6. Run the configured stage with `PYTHONUNBUFFERED=1` so its progress
   prints live in the notebook instead of buffering silently for hours.
7. Optional, off by default (`REFRESH_PRICE_CACHE = False` in the Configure
   cell): download the full universe fresh into `data/cache_fresh/` and
   push it as a new version of the price-cache dataset via
   `scripts/upload_cache_dataset.py --version` — the same script "One-time
   setup" below runs locally, just running on Kaggle's network instead. A
   separate, self-contained maintenance action (normally its own session,
   not combined with a `STAGE` run — a version pushed mid-session doesn't
   remount into that same session; the next session benefits from it).

`kaggle/requirements.txt` is generated with `uv export` from `uv.lock` (see
the file's own header for the regenerate command) so the Kaggle environment
resolves to the exact same package versions a local `uv pip install -e .`
would, not whatever Kaggle's base image happens to have.

### Getting a run back off Kaggle

Commit the notebook version (Save Version → Save & Run All) so
`/kaggle/working` becomes that version's attachable output, then either
keep chaining on Kaggle (attach that output as the next session's input) or
pull it down locally:

```
uv run stockml fetch-run <your-username>/<notebook-slug>
```

This shells out to `kaggle kernels output` and folds any `runs/<...>/`
folder in the notebook's output up into your local `runs/`, so a run
produced on Kaggle can be resumed, controlled, vaulted, or reported on
exactly like a local one (`stockml evo-report`, `stockml vault`, ...).
