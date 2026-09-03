# Build prompt for Claude Code — `stockml` step 1 framework

Copy everything below the line into Claude Code, run from an empty directory (`git init` first).

---

## What we're building

A small, honest research framework for predicting the **next-day direction** of US large-cap stocks as a three-class problem: **up / down / stagnant**. This is step 1 of a staged experiment. The point of step 1 is not a good model. The point is a framework where a good number *cannot be faked*, and where later steps (online updating, position sizing) plug in without rewriting anything.

Call the package `stockml`. Python 3.11+.

## Non-negotiable principles

These override any convenience. If a shortcut conflicts with one of them, take the slow route and say so.

1. **No look-ahead, ever.** Every feature at row `t` uses only data with a timestamp `<= t`. The label at row `t` is the only thing allowed to touch `t+1`. Time-based splits only; never a random split of days.
2. **Baselines are mandatory.** Every results table shows the model next to `majority-class` and `always-up` baselines computed on the *same* test rows. A number without its baseline is not a result.
3. **Leak alarm.** If any model's out-of-sample accuracy exceeds **60%** on more than 250 test days, print a loud warning and automatically run the leak diagnostics (below) before reporting. Treat a high score as a bug until proven otherwise.
4. **Reproducible.** Fixed seeds, config-driven runs, every run writes its config + metrics + predictions to `runs/<timestamp>_<name>/`.
5. **Simple over clever.** No deep learning, no hyperparameter search, no feature engineering beyond the list below. Minimal dependencies. The model is deliberately boring; the harness is the deliverable.

## Stack

- `pandas`, `numpy`, `scikit-learn`, `pyarrow` (parquet cache), `yfinance` (data), `pyyaml` (config), `scipy` (binomial test), `typer` (CLI), `pytest`.
- Use `sklearn.ensemble.HistGradientBoostingClassifier` for the tree model — avoid adding LightGBM/XGBoost.
- Ask before adding any dependency not on this list.

## Repo layout

```
stockml/
  pyproject.toml
  README.md
  CLAUDE.md                  # the principles above + how to run; keep it current
  configs/
    step1.yaml               # the default run
    smoke.yaml               # 10 tickers, 2 years, for fast tests
  data/
    tickers/sp500.csv        # bundled constituent list (ticker, name, sector, as_of date)
    cache/                   # parquet, git-ignored
  src/stockml/
    data.py                  # download + cache + calendar alignment
    labels.py                # three-class labels
    features.py              # past-only features
    split.py                 # walk-forward splitting
    models/
      base.py                # Model protocol: fit(X, y), predict_proba(X), classes_
      baselines.py           # MajorityClass, AlwaysUp
      sklearn_models.py      # LogisticRegression, HistGradientBoosting wrappers
    update.py                # UpdatePolicy protocol + Frozen (step-2 hook, see below)
    evaluate.py              # metrics, per-class, per-ticker, binomial test, leak diagnostics
    backtest/
      __init__.py            # step-3 hook: Policy + Reward protocols, stubs only
    run.py                   # ties it together
    cli.py
  tests/
  runs/                      # git-ignored
```

## Data (`data.py`)

- Universe: the S&P 500. Bundle a static `sp500.csv` in the repo (fetch the current constituent list once, write it to the file, record the `as_of` date). Do **not** hit the web for the list on every run.
- Download daily OHLCV with `yfinance` for a configurable date range (default: 2010-01-01 to today), `auto_adjust=True`, and cache per ticker as parquet. Re-running must not re-download unless `--refresh` is passed.
- Align everything to the union trading calendar. Drop a ticker-day if close is missing. Drop tickers with less than `min_history_days` (default 1,000) of data.
- Document in the README that using the current constituent list introduces **survivorship bias** (delisted losers are missing). It is an accepted limitation of step 1; note it, don't solve it.
- Document that adjusted closes are retroactively rewritten by the provider. For direction labels this is acceptable; for anything price-level based later it isn't.

## Labels (`labels.py`)

For each ticker and day `t`:

```
r_next(t)  = close(t+1) / close(t) - 1
sigma(t)   = std of daily returns over the trailing `vol_window` days ending at t   (default 20, uses days <= t only)
band(t)    = k * sigma(t)                                                            (default k = 0.5)

label(t) = up        if r_next(t) >  band(t)
         = down      if r_next(t) < -band(t)
         = stagnant  otherwise
```

- `k` and `vol_window` are config values. The band is per-ticker and volatility-scaled on purpose: a flat percentage band would make the class balance depend on which stock you look at, and the band rather than the model would set the score.
- Print the class distribution of the labels for the train and test periods in every run. If any class is under 15% or over 60%, warn — it means `k` needs adjusting before the accuracy number means anything.
- The last row of each ticker has no label. Drop it, never fill it.

## Features (`features.py`)

Keep to this list. All computed per ticker, then explicitly shifted so that row `t` contains nothing from after `t`:

- log returns lagged 1, 2, 3, 5, 10 days
- rolling mean of returns over 5 and 20 days
- rolling std of returns over 5 and 20 days, and their ratio
- close / SMA(20) − 1, close / SMA(60) − 1, SMA(20) / SMA(60) − 1
- volume z-score over 20 days
- day of week (categorical → one-hot or ordinal, your call)
- the label's own `band(t)` and `sigma(t)` (they are past-only, so allowed)

Add a `feature_names()` function so the report can list what the model saw. No cross-sectional or market-wide features in step 1 (keep the design open for them: features should be built per ticker in a function that receives a single-ticker frame).

## Splits (`split.py`)

Walk-forward:

- `train_years` (default 3) from the start of the data → training set.
- Every subsequent day → test set, predicted one day at a time in chronological order (batch-predict is fine for a frozen model, but the loop must be structured so that a day-by-day update policy can slot in).
- Reserve the **final 10 trading days** as a separate `sanity` slice. Report it in its own line, never merge it into the test metrics, and never use it to decide anything. It exists only to prove the pipeline works on the freshest data.
- Pooled training across all tickers (one model, ticker-agnostic features) is the default. Provide `per_ticker: true` as a config option that trains one model per ticker instead, but implement pooled first.
- Assert in code, not just in tests: no train timestamp `>=` any test timestamp, and any scaler/normalizer is fit on the training rows only.

## Update policy hook (`update.py`) — step-2 placeholder

Define the interface now, implement only `Frozen`:

```python
class UpdatePolicy(Protocol):
    def should_update(self, day, y_pred, y_true, history) -> bool: ...
    def update(self, model, X_hist, y_hist) -> Model: ...
```

The walk-forward loop must call `should_update` after each test day's outcome is known and `update` when it returns true. `Frozen` always returns false. Later policies (daily refit, refit-on-big-miss, recency-weighted) will be added without touching the loop. Write one test that plugs in a dummy policy which counts calls, to prove the loop honours the interface.

## Models (`models/`)

- `Model` protocol: `fit(X, y)`, `predict_proba(X)` returning columns in a fixed class order `[down, stagnant, up]`, `classes_`.
- Baselines: `MajorityClass` (predicts the most common training label), `AlwaysUp`.
- `LogReg` (with standard scaling inside a pipeline), `HGB` (HistGradientBoosting, default params, fixed `random_state`).
- The run config lists which models to run; all run on identical rows.

## Evaluation (`evaluate.py`)

For every model, on the test slice:

- accuracy, balanced accuracy, macro F1
- per-class precision / recall / support
- confusion matrix
- accuracy per ticker: mean, median, 10th/90th percentile, and the count of tickers where the model beats the majority baseline
- rolling 60-day accuracy series (saved, plotted to PNG in the run folder) so a lucky stretch is visible as a stretch
- a one-sided **binomial test** of the model's hit count against the majority-baseline rate: report the p-value
- the `sanity` slice's accuracy on its own line, clearly marked as 10 days / not significant

**Leak diagnostics** (run automatically when the alarm fires, and available via CLI at any time):

1. *Shift test*: shift all features by one extra day (row `t` gets the features of `t-1`) and re-evaluate. A small drop is normal, because the features are now staler. A collapse to the baseline means the original features were carrying information from `t+1` — leakage in the alignment.
2. *Label alignment audit*: for 50 random (ticker, day) rows, recompute `r_next` by hand from the raw cached closes and assert the label matches.
3. *Truncation test*: for 20 random (ticker, day) rows, rebuild features using only data up to that day and assert they are identical to the features from the full build. This is the single most important test in the repo — make it a pytest test as well.

## CLI (`cli.py`)

```
stockml download  --config configs/step1.yaml [--refresh]
stockml dataset   --config configs/step1.yaml          # builds features + labels, caches to parquet, prints class balance
stockml run       --config configs/step1.yaml          # walk-forward, writes runs/<ts>_<name>/
stockml report    runs/<ts>_<name>/                     # pretty-print the table + baselines + warnings
stockml leakcheck runs/<ts>_<name>/                     # the diagnostics above, on demand
```

## Config (`configs/step1.yaml`)

Every number mentioned above lives here: date range, `train_years`, `k`, `vol_window`, `min_history_days`, `sanity_days`, models list, `per_ticker`, `seed`, `leak_alarm_threshold`. `smoke.yaml` is the same shape with 10 tickers and 2 years so tests and quick checks run in under a minute.

## Tests (`tests/`)

At minimum:

- truncation test (features past-only)
- label alignment vs. hand-computed returns
- band behaviour: the stagnant share moves in the expected direction when `k` changes
- split: strict temporal ordering, sanity slice excluded from test, no overlap
- update-policy loop honours the protocol (dummy counting policy)
- baselines produce the expected accuracy on a synthetic dataset with a known class mix
- end-to-end smoke run on `smoke.yaml` completes and writes a results folder

## Working style

- Start by writing `CLAUDE.md` with the principles and the layout, then a short plan. Then build in this order: data → labels → features → split → baselines → evaluate → run/cli → sklearn models → tests → README. Run the smoke config after each stage.
- Do the full S&P 500 download at the end as a real run (`step1.yaml`), and paste the results table (with baselines, class balance, p-value, and the sanity line) into the final summary.
- Do not tune the model. If the result is near the baseline, that is the honest outcome of step 1 and it's fine. If the alarm fires, chase the leak, not the score.
- Prefer boring, readable code over abstractions. Type hints, docstrings on public functions, no notebooks required.

## Explicitly out of scope (design for it, don't build it)

- Online / on-the-fly update policies beyond `Frozen` (step 2)
- Position sizing, P&L simulation with transaction costs, and any path-dependent reward (step 3 — `backtest/` holds only the `Policy` and `Reward` protocol stubs)
- Intraday data, market-wide or cross-sectional features, deep learning, hyperparameter search
