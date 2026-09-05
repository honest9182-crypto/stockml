"""Mandatory comparison pickers, same days and same `n_picks` as every real
picker (CLAUDE.md's "Up-only picker" section -- a picker number without
these next to it is not a result):

| name | picks each day | what it tests |
|---|---|---|
| `random` | `n_picks` uniformly at random, `random_draws` seeds | the luck band |
| `top_vol` | highest trailing `sigma` | the real bar (volatility, not direction) |
| `momentum` | highest `log_ret_lag_1` | yesterday's winners |
| `reversal` | lowest `log_ret_lag_1` | yesterday's losers |
| `frequent` | highest training-window `up` rate, same list every day | a static roster |

Each (except `random`, which is a distribution, not a single ranking --
see `random_baseline_edges`) produces a `scores_df` shaped exactly like a
real model's (`date, ticker, y_true, p_up, r_next, sigma`), so
`select_top_k`/`evaluate_picker` never need to know the difference between
a baseline's proxy score and an actual model's `p(up)`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from stockml.picker.evaluate import daily_precision_and_base
from stockml.picker.select import select_top_k

_OUT_COLS = ["date", "ticker", "y_true", "p_up", "r_next", "sigma"]


def _standardize(df: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    out = df[["date", "ticker", "label", "r_next", "sigma"]].rename(columns={"label": "y_true"}).copy()
    out["p_up"] = np.asarray(score)
    return out[_OUT_COLS].reset_index(drop=True)


def top_vol_scores(eval_df: pd.DataFrame) -> pd.DataFrame:
    return _standardize(eval_df, eval_df["sigma"])


def momentum_scores(eval_df: pd.DataFrame) -> pd.DataFrame:
    return _standardize(eval_df, eval_df["log_ret_lag_1"])


def reversal_scores(eval_df: pd.DataFrame) -> pd.DataFrame:
    return _standardize(eval_df, -eval_df["log_ret_lag_1"])


def frequent_scores(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Static list: each ticker's historical `up` rate over `train_df`,
    the same score every eval day -- should sit near zero out of sample.
    """
    up_rate = train_df.groupby("ticker")["label"].apply(lambda s: (s == "up").mean())
    score = eval_df["ticker"].map(up_rate).fillna(-np.inf)
    return _standardize(eval_df, score)


# Baselines that reduce to "some deterministic score, ranked like any
# model" -- `random` doesn't fit this shape (see below) and is handled
# separately by callers.
DETERMINISTIC_BASELINES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    "top_vol": lambda train_df, eval_df: top_vol_scores(eval_df),
    "momentum": lambda train_df, eval_df: momentum_scores(eval_df),
    "reversal": lambda train_df, eval_df: reversal_scores(eval_df),
    "frequent": frequent_scores,
}


def random_baseline_edges(eval_df: pd.DataFrame, n_picks: int, random_draws: int, seed: int) -> np.ndarray:
    """`random_draws` independent seeds, each picking `n_picks` uniformly at
    random every day (a fresh uniform score per row makes `select_top_k`'s
    ranking exactly that) and scored for its mean daily edge (pp). The
    resulting array's 2.5th/97.5th percentiles are the luck band every
    other picker/baseline has to clear -- report those, not a CI on a
    single run, since there's no "the" random picker, only the distribution
    of them.
    """
    edges = np.empty(random_draws)
    for i in range(random_draws):
        rng = np.random.default_rng([seed, i])
        noise = rng.random(len(eval_df))
        scores = _standardize(eval_df, pd.Series(noise, index=eval_df.index))
        picks = select_top_k(scores, n_picks, rng)
        precision_d, base_d = daily_precision_and_base(picks, scores)
        edge = (precision_d - base_d).dropna()
        edges[i] = float(edge.mean() * 100) if len(edge) else 0.0
    return edges
