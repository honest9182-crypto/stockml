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
split.py     walk-forward train/test/sanity split by date, asserted at runtime
models/      Model protocol; MajorityClass/AlwaysUp baselines; LogReg/HGB (boring, untuned)
update.py    UpdatePolicy protocol + Frozen (step-2 hook; the loop calls it every test day)
evaluate.py  metrics, per-ticker/per-class breakdowns, binomial test, leak diagnostics
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

`configs/step1.yaml`, pooled model, `train_years: 3`, `k: 0.5`, `vol_window: 20`.
483 of 503 tickers survived the `min_history_days: 1000` filter (20 too short
— younger listings, mostly). Run: `runs/20260903_003136_step1/`.

**Class balance** (train / test, both healthy — no warning fired):

| split | down | stagnant | up |
|---|---|---|---|
| train | 26.9% | 42.5% | 30.6% |
| test  | 26.3% | 43.4% | 30.3% |

**Test set: 1,565,941 rows** (pooled across all tickers, walk-forward, one
model frozen after the initial 3-year train window):

| model | accuracy | balanced acc. | macro F1 | p-value vs. majority | n beating majority (of 483 tickers) |
|---|---|---|---|---|---|
| always_up | 0.3030 | 0.3333 | 0.1550 | 1.0 | 0 |
| majority_class | 0.4338 | 0.3333 | 0.2017 | — (baseline) | 0 |
| logreg | 0.4363 | 0.3487 | 0.2647 | 6.7e-11 | 331 |
| hgb | 0.4349 | 0.3589 | 0.3032 | 2.3e-3 | 271 |

**Sanity slice** (final 10 trading days — 4,349 rows, not merged into the
table above, not statistically meaningful on its own, exists only to prove
the pipeline runs on the freshest data):

| model | accuracy |
|---|---|
| always_up | 0.2102 |
| majority_class | 0.4794 |
| logreg | 0.4732 |
| hgb | 0.4553 |

**Reading this honestly**: logreg and HGB beat the majority-class baseline by
0.3–1.1 percentage points. With 1.5M+ test rows that margin is statistically
significant (p < 0.01 for both), but it is *tiny* — this is not a tradeable
edge, and no leak alarm fired (max accuracy 43.6%, nowhere near the 60%
threshold). `stockml leakcheck` was run on this exact result as a final
check: the label-alignment audit and the truncation test (rebuilding
features from data truncated at each sampled day) both came back with **0
mismatches**, and the shift test (staling every feature by one extra day)
dropped accuracy only slightly rather than collapsing to baseline — so this
small edge looks real, not leaked. It's also modest enough, and close enough
to the honest baseline, that step 1 did its job: it didn't hand back a
suspiciously good number to chase down.
