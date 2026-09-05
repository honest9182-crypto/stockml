"""Turns a per-row score into the day's forced picks.

The only decision a picker makes is *which* `n_picks` tickers to name "up"
each day -- see CLAUDE.md's "Up-only picker" section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_COL = "p_up"
PICKS_COLUMNS = ["date", "ticker", "score", "rank", "y_true", "r_next", "sigma"]


def select_top_k(scores_df: pd.DataFrame, n_picks: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per day, sort by `scores_df[SCORE_COL]` descending, breaking ties
    with `rng`, and take the first `n_picks` rows -- or every row that day,
    if fewer than `n_picks` tickers have one (never an error: the picker is
    forced to pick among what exists).

    `scores_df` must have columns `date, ticker, p_up, y_true, r_next,
    sigma` -- `p_up` is the ranking key regardless of whether it came from
    an actual model (`picker/scores.py`) or a baseline's proxy score
    (`picker/baselines.py` populates a `p_up`-named column too, so this
    function never needs to special-case them).

    Deterministic given `rng`'s state: two calls with a freshly-seeded,
    identical `rng` produce byte-identical output, ties included -- what
    makes two runs with the same seed produce identical `picks.parquet`.
    """
    if not len(scores_df):
        return pd.DataFrame(columns=PICKS_COLUMNS)

    picks = []
    for day, day_df in scores_df.groupby("date", sort=True):
        day_df = day_df.reset_index(drop=True)
        n = len(day_df)
        k = min(n_picks, n)
        # Primary key: score, descending. Secondary (tie-break): a fresh
        # random draw per day, so ties resolve deterministically given rng's
        # state but never favor e.g. row order or ticker name.
        tie = rng.random(n)
        order = np.lexsort((-tie, -day_df[SCORE_COL].to_numpy()))
        top = day_df.iloc[order[:k]].copy()
        top["rank"] = np.arange(1, k + 1)
        top["score"] = top[SCORE_COL]
        picks.append(top[PICKS_COLUMNS])

    return pd.concat(picks, ignore_index=True)


def n_picks_shortfall_days(scores_df: pd.DataFrame, n_picks: int) -> dict[str, int]:
    """Days where fewer than `n_picks` tickers had a row that day, mapped
    to how many were actually available -- `select_top_k` silently picks
    all of them on such a day (per CLAUDE.md: "all of them if fewer exist
    -- log it"); this is that log.
    """
    counts = scores_df.groupby("date").size()
    short = counts[counts < n_picks]
    return {str(d.date()): int(n) for d, n in short.items()}
