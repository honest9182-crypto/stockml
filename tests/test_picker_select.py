"""select_top_k: the forced-output contract (CLAUDE.md's "Up-only picker"
section) -- exactly n_picks per day, never fewer unless the day genuinely
has fewer tickers, never a probability threshold, deterministic under a
seed including ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.picker.select import n_picks_shortfall_days, select_top_k


def _scores(n_days: int = 5, n_tickers: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append(
                {
                    "date": d, "ticker": f"T{i}", "p_up": rng.random(),
                    "y_true": rng.choice(["down", "stagnant", "up"]),
                    "r_next": rng.normal(0, 0.01), "sigma": rng.random() * 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_exactly_n_picks_per_day():
    scores = _scores(n_days=4, n_tickers=8)
    picks = select_top_k(scores, n_picks=3, rng=np.random.default_rng(0))
    counts = picks.groupby("date").size()
    assert (counts == 3).all()


def test_all_tickers_picked_when_fewer_than_n_picks_exist():
    scores = _scores(n_days=2, n_tickers=4)
    picks = select_top_k(scores, n_picks=10, rng=np.random.default_rng(0))
    counts = picks.groupby("date").size()
    assert (counts == 4).all()  # never more than what exists, no error


def test_never_abstains_even_with_uniformly_low_scores():
    # No threshold, ever -- even if every score is near zero, n_picks tickers
    # are still named "up" that day.
    scores = _scores(n_days=2, n_tickers=6)
    scores["p_up"] = 1e-9
    picks = select_top_k(scores, n_picks=3, rng=np.random.default_rng(0))
    assert (picks.groupby("date").size() == 3).all()


def test_rank_matches_score_order():
    scores = _scores(n_days=3, n_tickers=6)
    picks = select_top_k(scores, n_picks=6, rng=np.random.default_rng(0))
    for _, day_picks in picks.groupby("date"):
        ordered = day_picks.sort_values("rank")
        assert (ordered["score"].to_numpy() == np.sort(ordered["score"].to_numpy())[::-1]).all()
        assert list(ordered["rank"]) == list(range(1, len(ordered) + 1))


def test_deterministic_given_same_rng_state_including_ties():
    scores = _scores(n_days=5, n_tickers=6)
    scores["p_up"] = 0.5  # force every row to tie -- exercises the tie-break path
    picks_a = select_top_k(scores, n_picks=3, rng=np.random.default_rng(42))
    picks_b = select_top_k(scores, n_picks=3, rng=np.random.default_rng(42))
    pd.testing.assert_frame_equal(picks_a, picks_b)


def test_different_seeds_break_ties_differently():
    scores = _scores(n_days=5, n_tickers=6)
    scores["p_up"] = 0.5
    picks_a = select_top_k(scores, n_picks=3, rng=np.random.default_rng(1))
    picks_b = select_top_k(scores, n_picks=3, rng=np.random.default_rng(2))
    assert not picks_a["ticker"].reset_index(drop=True).equals(picks_b["ticker"].reset_index(drop=True))


def test_empty_scores_returns_empty_with_expected_columns():
    empty = pd.DataFrame(columns=["date", "ticker", "p_up", "y_true", "r_next", "sigma"])
    picks = select_top_k(empty, n_picks=5, rng=np.random.default_rng(0))
    assert len(picks) == 0
    assert list(picks.columns) == ["date", "ticker", "score", "rank", "y_true", "r_next", "sigma"]


def test_n_picks_shortfall_days_reports_only_short_days():
    scores = _scores(n_days=3, n_tickers=6)
    short_day = scores["date"].unique()[1]
    scores = scores[~((scores["date"] == short_day) & (scores["ticker"].isin(["T4", "T5"])))]
    shortfalls = n_picks_shortfall_days(scores, n_picks=6)
    assert len(shortfalls) == 1
    assert list(shortfalls.values())[0] == 4
