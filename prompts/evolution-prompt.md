# Build prompt for Claude Code — `stockml` evolution layer

Copy everything below the line into Claude Code, started inside `D:\stockml`. Tell it to read `CLAUDE.md` first.

---

## What we're adding

Read `CLAUDE.md` and the step-1 code first; every principle in it still applies. We are adding an **evolutionary search layer** on top of the step-1 framework. The idea, in the owner's words: each model's focus is its DNA. Every generation, the strongest models have the most babies, with each other. But like in real life, models seek partners that are genetically *different* from themselves, so the gene pool stays varied. Sometimes a weak model gets lucky and mates with a strong one. Every few generations there is a semi-random mutation storm. Everything has a bit of randomness, and everything is still aimed at the same target: next-day up / down / stagnant on the stock market.

The scientific question is not "can evolution find a great model" but "**does evolution find anything that random search and pure luck don't**". The build must make that question answerable.

## Non-negotiables (in addition to CLAUDE.md)

1. **Same exam for everyone.** Labels (`k`, `vol_window`) stay global and fixed. Every genome in a run is fitted and scored on identical rows. A genome may choose what it looks at and how it learns, never what it is judged on.
2. **Three time zones, hard-coded in config, never overlapping:**
   - `train`: the existing training window (start of data → embargo).
   - `arena`: test days from the start of test to **2019-12-31**. Fitness is measured here. Evolution sees the arena thousands of times, so it *will* overfit to it. That's expected and is exactly why the third zone exists.
   - `vault`: **2020-01-01** → start of the sanity slice. Never touched during evolution. Opened once at the end of a run, for a fixed, pre-declared list of individuals (below). The code must enforce this: the vault loader raises if called while an evolution is in progress.
3. **Two controls run with the same budget as every evolution:**
   - **Random search**: the same number of unique fitness evaluations the evolution actually performed, spent on uniformly random genomes. If evolution's best isn't clearly better than random search's best, the mating logic added nothing.
   - **Shifted-label null**: the identical evolution (same config, same seed) run on labels that have been *circularly shifted in time, per ticker, by a random offset of at least 250 trading days*. Each ticker keeps its exact class counts and its label autocorrelation, but no feature at day `t` can know anything about the label now sitting at `t`. Its best arena fitness is the **luck ceiling**: what pure selection on noise achieves with this budget. A champion that doesn't clearly beat the luck ceiling is noise. (Do not shuffle within a date across tickers: that preserves each day's market-wide class mix, so market-wide predictability would survive the shuffle and the null would not be a null.)
4. **One RNG, seeded from config,** for every random decision: initial population, parent draws, mate choice, lottery, crossover, mutation, storms, immigration, label shuffling. Same seed → identical run.
5. **Full lineage.** Every individual ever created gets an id, its generation, its parents, which genes mutated, and whether it came from a storm, the lottery, or immigration. Any champion must be traceable back to generation 0.
6. **The step-1 leak alarm still applies** to every genome. Any genome above the alarm threshold on the arena halts the run and triggers the diagnostics.
7. **Simple.** Genes are discrete choices or values on small grids. No neural nets, no continuous hyperparameter spaces, no new heavy dependencies (`joblib` ships with scikit-learn and is allowed).

## Genome v1

A frozen dataclass `Genome` with encode/decode to a flat dict (for logging) and a stable hash (for memoization). Genes:

| gene | values | meaning |
|---|---|---|
| `feature_mask` | one bit per name in `feature_names()` | which features the model sees; at least 2 must be on |
| `model_family` | `logreg`, `hgb` | which step-1 model class |
| `train_years_used` | 1, 2, 3 | how far back it trains (always ending at the embargo) |
| `stagnant_bias` | −0.10, −0.05, 0, +0.05, +0.10 | added to `p_stagnant` before argmax, so a genome can choose to call direction more or less often |
| `lr_C` | 0.01, 0.1, 1, 10 | logreg regularisation |
| `class_weight` | `none`, `balanced` | both families |
| `hgb_max_depth` | none, 2, 3, 4, 6 | `none` = scikit-learn's default, so the step-1 seed is representable |
| `hgb_learning_rate` | 0.03, 0.1, 0.3 | |
| `hgb_max_iter` | 50, 100, 200, 400 | |
| `hgb_min_samples_leaf` | 20, 100, 500, 2000 | |
| `hgb_l2` | 0, 1, 10 | |

Genes that don't apply to the current `model_family` are still carried and inherited (recessive genes: a logreg genome's HGB settings can resurface in its children). Design `Genome` so future genes can be added as fields with defaults without breaking saved lineages — **deferred for later versions:** update-policy genes (how often to refit, recency weighting, refit-on-big-miss) and "focus" genes (which sectors/stocks to train on, still judged on the whole market).

## Fitness

`evaluate_genome(genome, zone)` builds the model from the genome, fits it on the genome's training window, predicts the requested zone walk-forward (frozen model, exactly as step 1), and returns the **day-level paired edge vs. the majority baseline** using the existing evaluate.py function.

**fitness = mean daily edge − 1 standard error** (in percentage points). This rewards consistency; a single lucky year is penalised. Also store plain mean edge, accuracy, the block-bootstrap CI, and the prediction mix.

- Full S&P 500 universe on the arena (the owner accepted an overnight budget). Provide `arena_ticker_subsample: null | int` in config for fast debugging runs.
- Build the dataset once, keep it in memory, evaluate a generation's genomes in parallel with `joblib` (`n_jobs` in config, default `-2`).
- Memoize fitness by genome hash. Clones and elites cost nothing.
- After generation 0 finishes, print the time it took and an ETA for the whole run, so the owner can shrink `generations` or `population_size` before committing a night to it. Write progress to the run folder as it goes; a run must be resumable from its last completed generation (`--resume runs/evo_...`).

## Population dynamics

All numbers in config; defaults given.

- `population_size` 40, `generations` 25, `patience` 10 (stop early if the best fitness hasn't improved for that many generations).
- **Generation 0:** random genomes, plus two seeded individuals that reproduce the step-1 models exactly (logreg and hgb with all features on, default settings). We must be able to see whether evolution ever beats where we started.
- **Elitism:** the top `n_elite` = 2 are copied unchanged into the next generation.
- **First parent — the strongest have the most babies.** Rank-based roulette: weight ∝ `(N − rank + 1) ** selection_pressure`, `selection_pressure` = 1.5. Rank-based so a single outlier can't take over the whole gene pool in one generation.
- **Mate choice — seek a different partner.** Given the first parent, the second parent is drawn with probability ∝ `rank_weight × (1 + distance) ** dissimilarity_power`, where `distance` is the fraction of genes that differ (feature mask counted bit by bit, normalised so distance is in [0, 1]) and `dissimilarity_power` = 2. Self-mating is not allowed.
- **Lottery — a weak one gets lucky.** `n_lottery` = 3 children per generation have their first parent drawn uniformly from the *bottom half* of the population; their partner is drawn by the normal mate-choice rule, so the lucky one usually mates upward.
- **Immigration.** `n_immigrants` = 2 fresh random genomes per generation.
- **Crossover.** Uniform, gene by gene (each gene from either parent with p = 0.5; feature mask bit by bit). For grid-valued genes, with p = 0.25 the child instead gets a value drawn uniformly from the grid positions *between* the two parents (inclusive).
- **Mutation.** Each gene mutates with `mutation_rate` = 0.05: feature bits flip, categorical genes resample, grid genes step one position up or down. Re-validate (min 2 features on).
- **Mutation storm — every few generations.** Every `storm_every` = 5 generations, the mutation rate is multiplied by `storm_factor` = 5 for that generation's children (elites are exempt). Storms are logged.
- **Diversity guard.** Each generation, compute mean pairwise genetic distance. If it drops below `min_diversity` = 0.10, force a storm next generation. Log it.
- The remaining slots (`population_size − n_elite − n_lottery − n_immigrants`) are filled by normal children.

## Vault protocol

When an evolution run finishes, evaluate in the vault **exactly this list and nothing else**: the champion (best arena fitness), the top 5 by arena fitness, the two step-1 seeds, the random-search champion, the null run's champion, and the majority baseline. Report each one's arena fitness next to its vault edge with the bootstrap CI. Append every vault look to `runs/evolution/vault_log.jsonl` with the run id and timestamp, and write in `CLAUDE.md`: **a vault result is never a reason to change the config and run again. If you do run again, the earlier vault results stay in the log and in the report.**

The expected honest picture: arena fitness of the champion > seeds, and the vault edge of the champion shrinks toward zero. How much it shrinks is the finding.

## Outputs

`runs/evo_<timestamp>_<name>/` containing: `config.yaml`, `lineage.jsonl`, `generations.csv` (best / mean / median fitness, diversity, storm flag, n_evaluated, n_memoized per generation), `gene_frequency.csv` (per generation, the share of the population with each feature bit on and each categorical value — **this is the most interesting output: which genes get selected**), `champion_family_tree.txt` (ancestors back to generation 0, with the mutation that produced each), `vault_report.txt`, and PNGs: best & mean fitness by generation with the random-search and null curves overlaid; diversity by generation with storms marked; feature-frequency heatmap (features × generations).

CLI:

```
stockml evolve         --config configs/evo.yaml [--quick]      # --quick: 20 tickers, 6 genomes, 3 generations
stockml evolve-control --config configs/evo.yaml --kind random   # same budget, random genomes
stockml evolve-control --config configs/evo.yaml --kind null     # same evolution, shuffled labels
stockml vault          runs/evo_<ts>_<name>/                     # one-time look; prints a warning and logs the look
stockml lineage        runs/evo_<ts>_<name>/ --id <individual>   # trace one individual back to generation 0
stockml evo-report     runs/evo_<ts>_<name>/                     # re-render tables and plots
```

The controls are separate commands so an overnight can be split: evolution one night, controls the next. `evo-report` must work with whichever of the three have finished, and clearly mark missing ones.

## Tests

- genome encode/decode round-trip; hash stable; mutation and crossover always produce valid genomes (grid values, ≥ 2 features)
- mate choice: on a synthetic population, with `dissimilarity_power` = 4 chosen partners are on average farther than random draws; with 0 they are not
- lottery: over many generations on a synthetic population, bottom-half individuals reproduce a non-zero share of the time; with `n_lottery` = 0 they almost never do
- first-parent selection: higher rank → more children (monotone over many draws)
- storms fire on schedule and when the diversity guard trips
- memoization: an identical genome is not re-evaluated
- time zones: arena and vault never overlap, both exclude train and sanity; the vault loader raises while `evolving` is set
- shifted-label null: every ticker's class counts are preserved exactly, every offset is ≥ 250 days, and the shifted labels differ from the originals
- `--quick` end-to-end run writes every output file
- reproducibility: two `--quick` runs with the same seed produce identical `lineage.jsonl`

## Working style

- Start by updating `CLAUDE.md` with the three time zones, the vault rule, and the two controls. Then a short plan.
- Build order: genome → `evaluate_genome` (refactor `run.py`'s walk-forward pieces into reusable functions instead of copying them) → selection / mate choice / lottery / crossover / mutation / storms → the evolve loop → controls → vault → outputs and plots → tests → `--quick` run → README.
- Run `--quick` after each stage. Don't start the overnight run yourself; leave the exact commands in the summary.
- Do not tune GA parameters by looking at vault results. If the champion beats the luck ceiling only marginally, say so plainly in the report text.
- Prefer boring, readable code. Type hints, docstrings, no notebooks.

## Explicitly out of scope (design for it, don't build it)

- Update-policy genes and "focus" genes (later versions of the genome)
- Position sizing, P&L simulation, and any path-dependent reward
- Per-ticker models inside the genome, neural nets, continuous hyperparameters
