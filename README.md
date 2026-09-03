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

uv run pytest -q
```

Use `configs/smoke.yaml` (10 tickers, ~2 years) for fast iteration — it runs
end to end in well under a minute once prices are cached.

## How the pipeline fits together

```
data.py      download + cache OHLCV per ticker (parquet), align to the union trading calendar
labels.py    per-ticker up/down/stagnant labels from a volatility-scaled band
features.py  per-ticker past-only features (imports labels.rolling_sigma -- one implementation)
split.py     walk-forward train/test/sanity split by date, asserted at runtime, 1-day train/test embargo
models/      Model protocol; MajorityClass/AlwaysUp baselines; LogReg/HGB (boring, untuned)
update.py    UpdatePolicy protocol + Frozen (step-2 hook; the loop calls it every test day)
evaluate.py  metrics, per-ticker/per-class breakdowns, day-level significance test, leak diagnostics
run.py       orchestrates all of the above, writes runs/<ts>_<name>/
backtest/    Policy + Reward protocol stubs only (step-3 hook)
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
`embargo_days: 1`. 483 of 503 tickers survived the `min_history_days: 1000`
filter (20 too short — younger listings, mostly). Run: `runs/20260903_005859_step1/`.

**Class balance** (train / test, both healthy — no warning fired):

| split | down | stagnant | up |
|---|---|---|---|
| train | 26.9% | 42.5% | 30.5% |
| test  | 26.3% | 43.4% | 30.3% |

**Test set: 1,565,941 rows, 3,347 trading days** (pooled across all tickers,
walk-forward, one model frozen after the initial 3-year train window):

| model | accuracy | balanced acc. | macro F1 |
|---|---|---|---|
| always_up | 0.3030 | 0.3333 | 0.1550 |
| majority_class | 0.4338 | 0.3333 | 0.2017 |
| logreg | 0.4363 | 0.3486 | 0.2646 |
| hgb | 0.4346 | 0.3582 | 0.3015 |

**Day-level paired test vs. majority baseline** — the honest significance
test (see "Two design details worth knowing about" above): one observation
per trading day, not per row.

| model | mean daily edge | t-test p-value | 95% CI (block bootstrap) |
|---|---|---|---|
| always_up | −13.080pp | 1.0 | [−14.083, −12.056]pp |
| logreg | **+0.251pp** | **0.0102** | **[+0.052, +0.495]pp** |
| hgb | +0.085pp | 0.2496 | [−0.196, +0.398]pp |

This is the more interesting result than the headline accuracy table: at the
row level, both logreg and HGB looked "significant" (p < 0.01). At the
*day* level — the actual unit of independence — **logreg's edge survives
(barely): a real but tiny ~0.25 point/day advantage. HGB's does not**: its
day-level p-value is 0.25, indistinguishable from noise, and its per-year
edge is negative in 5 of the last 5 years (2022–2026, see `report.txt`).
The row-level test would have told you both models "worked"; it doesn't
survive contact with the actual number of independent trading days.

**Sanity slice** (final 10 trading days — 4,349 rows, not merged into the
table above, not statistically meaningful on its own, exists only to prove
the pipeline runs on the freshest data):

| model | accuracy |
|---|---|
| always_up | 0.2102 |
| majority_class | 0.4794 |
| logreg | 0.4737 |
| hgb | 0.4560 |

**Reading this honestly**: no leak alarm fired (max accuracy 43.6%, nowhere
near the 60% threshold). `stockml leakcheck` was run on this exact result as
a final check: the label-alignment audit and the truncation test (rebuilding
features from data truncated at each sampled day) both came back with **0
mismatches**, and the shift test (staling every feature by one extra day)
dropped accuracy only slightly rather than collapsing to baseline — so
logreg's small edge looks real, not leaked. It's also modest enough, and
close enough to the honest baseline, that step 1 did its job: it didn't hand
back a suspiciously good number to chase down, and the day-level test caught
a case (HGB) where the row-level number would have.
