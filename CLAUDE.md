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

## Repo layout

```
stockml/
  pyproject.toml
  README.md
  CLAUDE.md
  configs/
    step1.yaml               # the default run
    smoke.yaml                # 10 tickers, 2 years, for fast tests
  data/
    tickers/sp500.csv         # bundled constituent list
    cache/                    # parquet, git-ignored
  src/stockml/
    data.py                   # download + cache + calendar alignment
    labels.py                 # three-class labels
    features.py                # past-only features
    split.py                   # walk-forward splitting
    models/
      base.py                  # Model protocol
      baselines.py              # MajorityClass, AlwaysUp
      sklearn_models.py         # LogReg, HGB wrappers
    update.py                  # UpdatePolicy protocol + Frozen (step-2 hook)
    evaluate.py                 # metrics, leak diagnostics
    backtest/
      __init__.py               # step-3 hook: Policy + Reward protocol stubs
    run.py                      # ties it together
    cli.py
  tests/
  runs/                        # git-ignored
```

## How to run

```
uv venv --python 3.13
uv pip install -e ".[dev]"

uv run stockml download  --config configs/step1.yaml [--refresh]
uv run stockml dataset   --config configs/step1.yaml   # features + labels, prints class balance
uv run stockml run       --config configs/step1.yaml   # walk-forward, writes runs/<ts>_<name>/
uv run stockml report    runs/<ts>_<name>/              # pretty-print the table + baselines + warnings
uv run stockml leakcheck runs/<ts>_<name>/               # leak diagnostics on demand

uv run pytest -q
```

Use `configs/smoke.yaml` (10 tickers, 2 years) for fast iteration — it should
run end-to-end in under a minute.

## Known, accepted limitations (step 1)

- **Survivorship bias**: `data/tickers/sp500.csv` is the *current* S&P 500
  constituent list. Delisted losers are absent from the universe entirely.
  This inflates any aggregate result. Documented, not solved, in step 1.
- **Retroactive adjustment**: `yfinance` with `auto_adjust=True` returns
  adjusted closes that are rewritten by the provider when new splits/dividends
  occur (i.e. a re-download today can change history). This is acceptable for
  direction labels (up/down/stagnant is robust to it) but would NOT be
  acceptable for anything price-level based (step 3 backtesting must not
  reuse this cache naively).

## Explicitly out of scope (designed for, not built)

- Online / on-the-fly update policies beyond `Frozen` (step 2). The
  `UpdatePolicy` protocol and the walk-forward loop's call sites already
  support this.
- Position sizing, P&L simulation with transaction costs, and any
  path-dependent reward (step 3). `backtest/__init__.py` holds only the
  `Policy` and `Reward` protocol stubs.
- Intraday data, market-wide or cross-sectional features, deep learning,
  hyperparameter search.

## Working notes

- Do not tune the model. If the result is near baseline, that is the honest
  outcome of step 1 and it's fine. If the leak alarm fires, chase the leak,
  not the score.
- `labels.py` owns the single implementation of `rolling_sigma`/`band`;
  `features.py` imports it rather than recomputing, so the label's band and
  the model's `sigma`/`band` features can never silently diverge.
