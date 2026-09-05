# Build prompt for Claude Code — `stockml` up-only picker (step 1.6)

Copy everything below the line into Claude Code, started inside `D:\stockml`. Tell it to read `CLAUDE.md` first.

---

## What we're adding

Read `CLAUDE.md`, the step-1 code (`run.py`, `walk_forward.py`, `evaluate.py`) and `evolution/fitness.py` first; every principle in them still applies. We are adding a **picker**: a model that can only ever say "up". Every trading day it is **forced to name exactly `n_picks` stocks as "up"** — never fewer, never more, never "down" or "stagnant" — and its whole job is choosing *which* ones. In the owner's words: it never predicts anything else, but it can choose which stocks it says up to.

This turns the question from "what will this stock do tomorrow?" into "**which stocks are the most likely to be up tomorrow?**" — a ranking problem, judged by how many of the day's picks actually went up. The step-1/1.5 models already lean this way (the evolution champion says "stagnant" 85–91% of the time and "down" almost never), so this makes that behaviour the design instead of a side effect.

Default `n_picks` = **10**, about 2% of the ~490-ticker universe. Reasoning, for the README: with one pick each day's score is all-or-nothing; with 50 the picker is little more than an index tilt; 10 is a buy-list a person could actually act on, gives 10-point precision granularity per day, and over the ~3,300 test days of a step-1 split pins the mean edge down to roughly half a point. `n_picks` lives in config, and the report always shows a sweep (below), so the choice stays visible instead of baked in.

## Non-negotiables (in addition to CLAUDE.md)

1. **Forced output.** Exactly `n_picks` tickers per day, chosen among the tickers that have a row that day (all of them if fewer exist — log it). No abstaining, no probability threshold, no variable count. The only decision is *which*.
2. **The score is judged against the day's own base rate.** On day `d`: `precision_d` = share of the picks whose true label is `up`; `base_d` = share of *all* tickers with a row on `d` whose label is `up` — which is exactly the expected precision of picking `n_picks` at random that day. `edge_d = precision_d − base_d`, in percentage points. On a day when the whole market jumps, every pick is "right", and so is every random pick; the paired edge is what's left. Report mean edge, day-level t-test and 20-day block-bootstrap CI, per-year table and rolling series — exactly what step 1 does for accuracy. Refactor `evaluate.day_level_paired_test` so both flows share one "summarise a daily series" helper rather than copying it.
3. **Volatility is not a signal.** The label band is `k·σ`, so a stock with a high `sigma` crosses it more often in *both* directions. A picker can therefore beat `base_d` by choosing volatile names while knowing nothing about direction. Every report shows the picks' `down` rate next to their `up` rate, and the `top_vol` baseline (below) is the bar the picker actually has to clear.
4. **Baselines are mandatory** (below), all evaluated on identical days with identical `n_picks`. A picker number without them is not a result.
5. **Same walk-forward, same zones.** Fit on the training window, score every later day with the frozen model (one batched predict, as `walk_forward_single_model` already does), embargo day dropped, final 10 days reported as `sanity` on their own line and never merged. No look-ahead: the picks for day `d` come from features at `d` only. Ties in the score are broken with the run's seeded RNG — same seed, byte-identical `picks.parquet`.
6. **Leak alarm.** If any picker's mean edge exceeds **+15 points** over more than 250 days, print the warning, run the existing leak diagnostics, and treat the number as a bug until proven otherwise. Random picks score about 30% precision here; a next-day picker sitting at 45% is not a discovery, it's a leak.
7. **Simple.** No new dependencies, no new features, no per-day retraining (step 2), no money (step 3).

## Scores (`picker/scores.py`)

The picker is a **decision rule on top of a per-row score `p_up`**. Two ways to get the score, selectable in config:

- `three_class` — reuse the existing models unchanged and take the `p_up` column that `walk_forward_single_model` already emits. Costs nothing, and it means every step-1 model and every evolution genome can be evaluated as a picker.
- `binary` — collapse the label to `up` vs `not_up` and fit the same model classes (`logreg`, `hgb`, `xgb`, through the existing wrappers) on that. A model trained for "is this one of the ups?" may rank better than one trained to tell three classes apart. If the fixed `[down, stagnant, up]` proba order in `models/base.py` makes this awkward, wrap it: `BinaryUp(model)` fits on the collapsed label and returns three columns with `p_down = 0` and `p_stagnant = 1 − p_up`, so every downstream tool keeps working unchanged.

Scores are pooled across tickers (one model, ticker-agnostic features), as in step 1. Do not add cross-sectional features in this step; keep the design open for them (rank-within-day of an existing feature is the obvious first one — note it in `CLAUDE.md` as a later gene).

## Selection (`picker/select.py`)

`select_top_k(scores_df, n_picks, rng) -> picks_df`: per day, sort by `p_up` descending, seeded-random tie-break, take the first `n_picks`. Output columns: `date, ticker, score, rank, y_true, r_next, sigma`. A pure function of its inputs — this is what evolution will call later.

## Evaluation (`picker/evaluate.py`)

For every picker and every baseline, on the test slice:

- daily `precision_d`, `base_d`, `edge_d`; mean precision and mean base rate in absolute terms, then mean edge with t-test p and block-bootstrap 95% CI; per-year edge table; rolling 60-day edge series → PNG
- **hit mix:** share of picks labelled `up` / `stagnant` / `down`, next to the universe's shares on the same days
- **return check** (secondary, labelled "not a P&L" in the report): mean `r_next` of the picks minus the mean `r_next` of all tickers that day, in basis points, with the same daily-series test. Volatility alone cannot move this number, so it either corroborates the precision edge or contradicts it. Equal weight, no costs, nothing path-dependent.
- **concentration:** number of distinct tickers ever picked; share of all pick-slots taken by the 10 most-picked tickers (list them); mean overlap between consecutive days' pick sets (turnover); picks by sector (from `sp500.csv`) against the universe's sector shares. A picker that names the same ten stocks every day has found a pattern *in the names*, not in the days — that is worth seeing, not hiding.
- **`n_picks` sweep:** re-run the selection (the scores don't change) for `k_sweep` = [1, 5, 10, 25, 50, 100] and tabulate/plot mean edge with CI against `k`. Real ranking skill shows up as an edge that *rises* as `k` shrinks; noise shows up flat. The sweep is a diagnostic: if the owner changes `n_picks` after seeing it, that is a new named run, and the old run's sweep stays in its folder.
- the `sanity` slice on its own line, marked as 10 days / not significant

## Baselines (`picker/baselines.py`) — same days, same `n_picks`

| name | picks each day | what it tests |
|---|---|---|
| `random` | `n_picks` uniformly at random, repeated for `random_draws` = 200 seeds | the luck band: mean edge should be ~0; report its 2.5th/97.5th percentiles as the band every picker has to leave |
| `top_vol` | the `n_picks` highest trailing `sigma` | "volatile stocks cross the band more often" — the real bar |
| `momentum` | the `n_picks` highest `log_ret_lag_1` | yesterday's winners |
| `reversal` | the `n_picks` lowest `log_ret_lag_1` | yesterday's losers |
| `frequent` | the `n_picks` tickers with the highest `up` rate in the training window, every day | a static list; should sit near zero out of sample |

All of these come from columns the dataset already has. The pickers (`logreg` / `hgb` / `xgb`, `three_class` or `binary`) appear in the same table, and the report text says plainly which baselines each picker beats and whether its CI clears `top_vol`'s.

## Evolution readiness (design for it, don't wire it)

Write `evaluate_picker(scores_df, zone_days, n_picks, seed, n_boot) -> PickerResult` with the same shape as `FitnessResult` (mean edge, SE, fitness = mean edge − 1 SE, CI, n_days, hit mix), so `evolution/fitness.py` can later offer `fitness_kind: picker` with `n_picks` and `score_source` as genes. Do not add the genes or touch the evolution loop in this step. Arena/vault zones must work through the same function — test it on an arena-shaped slice.

## Outputs

`runs/<timestamp>_<name>/` with `config.yaml`, `picks.parquet` (all pickers and baselines, tagged by name), `metrics.json`, `report.txt`, `k_sweep.csv`, `sector_mix.csv`, `rolling_edge_<name>.png`, `k_sweep.png`.

CLI:

```
stockml pick        --config configs/picker.yaml      # walk-forward, writes runs/<ts>_<name>/
stockml pick-report runs/<ts>_<name>/                  # re-print the table, baselines, sweep, warnings
```

## Config (`configs/picker.yaml`, plus `picker_smoke.yaml`)

Same shape as `step1.yaml`, plus: `n_picks: 10`, `k_sweep: [1, 5, 10, 25, 50, 100]`, `score_source: binary` (`three_class` | `binary`), `models: [logreg, hgb, xgb]`, `random_draws: 200`, `leak_alarm: {edge_threshold_pp: 15, min_days: 250}`. The smoke config is 10 tickers, 2 years, `n_picks: 2`, `random_draws: 20`, and must run in under a minute.

## Tests

- selection: exactly `n_picks` per day (all tickers when fewer exist), only "up" is ever emitted, deterministic under a seed including ties, `rank` matches score order
- `base_d` equals the day's true `up` share exactly; with an oracle score (`p_up` = 1 for the true ups) precision is 1 and edge is `1 − base_d`; with a constant score the mean edge is ~0 and inside the `random` band
- the `k` sweep is monotone non-increasing in `k` under an oracle score
- `BinaryUp`: proba rows sum to 1, `p_up` equals the underlying binary model's, `y_pred` is `up` iff `p_up > 0.5`
- no look-ahead: the existing truncation test passes through the binary pipeline, and the picks for day `d` are unchanged when every row after `d` is deleted
- the daily-series helper reproduces `day_level_paired_test`'s numbers on the step-1 smoke run
- return check: on synthetic data where the picks have a known mean-return offset, the reported bp edge matches
- concentration: a picker that always picks the same tickers reports overlap 1.0 and a 10-ticker share of 1.0
- end-to-end `picker_smoke.yaml` run writes every output file; two runs with the same seed produce identical `picks.parquet`

## Working style

- Start by adding a short "Up-only picker (step 1.6)" section to `CLAUDE.md`: forced `n_picks`, the base-rate-paired edge, the volatility caveat and the `top_vol` bar, the alarm. Then a short plan.
- Build order: `select_top_k` + daily edge on synthetic data → the daily-series helper refactor → `BinaryUp` → baselines → `pick` run/CLI → sweep, concentration, return check → report/plots → tests → smoke → README.
- Run the smoke config after each stage. Then run `configs/picker.yaml` on the full S&P 500 (frozen model — a step-1-sized job, not an overnight) and paste the results table — pickers, all five baselines with the random band, the hit mix, the sweep and the sanity line — into the final summary.
- Do not tune `n_picks`, the models or the features by looking at the results. If the picker sits inside the random band or below `top_vol`, that is the honest outcome, and the report says so.
- Prefer boring, readable code. Type hints, docstrings on public functions, no notebooks.

## Explicitly out of scope (design for it, don't build it)

- Abstaining or a variable number of picks (the owner chose a forced `n_picks`; a later variant may add "may pass on a day")
- Money: position sizing, P&L with costs, path-dependent or sequence-dependent reward (step 3)
- Per-day retraining and update policies (step 2)
- Cross-sectional or market-wide features, per-ticker models, anything intraday
- Making `n_picks` / `score_source` genes and `fitness_kind: picker` in the evolution loop (the next prompt)
