"""picker/evaluate.py: base_d exactness, oracle-score precision/edge,
constant-score noise, k-sweep monotonicity, the return check, and
concentration -- CLAUDE.md's "Up-only picker" section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.picker.evaluate import (
    concentration_stats,
    daily_precision_and_base,
    evaluate_picker,
    k_sweep,
    return_check,
)


def _oracle_scores(n_days: int, n_tickers: int, n_up: int, seed: int = 0) -> pd.DataFrame:
    """p_up = 1.0 for exactly `n_up` "up"-labeled tickers per day, 0.0 for
    the rest (labeled "stagnant") -- a perfect ranker, so precision at any
    `k <= n_up` is exactly 1.0.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    rows = []
    for d in dates:
        labels = np.array(["up"] * n_up + ["stagnant"] * (n_tickers - n_up))
        rng.shuffle(labels)
        for i, lbl in enumerate(labels):
            rows.append(
                {
                    "date": d, "ticker": f"T{i}", "y_true": lbl,
                    "p_up": 1.0 if lbl == "up" else 0.0,
                    "r_next": 0.0, "sigma": 0.01,
                }
            )
    return pd.DataFrame(rows)


def _random_scores(n_days: int, n_tickers: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append(
                {
                    "date": d, "ticker": f"T{i}",
                    "y_true": rng.choice(["down", "stagnant", "up"]),
                    "p_up": 0.5,  # constant -- forces random tie-break selection
                    "r_next": rng.normal(0, 0.01), "sigma": 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_base_d_equals_true_up_share_exactly():
    n_tickers = 8
    dates = pd.bdate_range("2021-01-01", periods=3)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append({"date": d, "ticker": f"T{i}", "y_true": "up" if i < 3 else "stagnant"})
    universe = pd.DataFrame(rows)
    _precision_d, base_d = daily_precision_and_base(universe, universe)
    assert np.allclose(base_d.to_numpy(), 3 / 8)


def test_oracle_score_precision_is_one_and_edge_is_one_minus_base():
    n_days, n_tickers, n_up = 15, 8, 3
    scores = _oracle_scores(n_days, n_tickers, n_up)
    result = evaluate_picker(scores, scores["date"].unique(), n_picks=n_up, seed=0, n_boot=0)
    expected_base = n_up / n_tickers
    assert result.mean_precision == pytest.approx(1.0)
    assert result.mean_base_rate == pytest.approx(expected_base)
    assert result.mean_edge_pp == pytest.approx((1.0 - expected_base) * 100)


def test_constant_score_edge_ci_contains_zero():
    # No real signal (every row scored identically -> select_top_k reduces
    # to "n_picks uniformly at random per day") -- the 95% CI on its own
    # daily edge should contain zero.
    scores = _random_scores(n_days=150, n_tickers=10, seed=1)
    result = evaluate_picker(scores, scores["date"].unique(), n_picks=3, seed=0, n_boot=1000)
    assert result.ci95_low_pp <= 0.0 <= result.ci95_high_pp


def test_k_sweep_monotone_non_increasing_under_oracle_score():
    n_days, n_tickers, n_up = 20, 10, 3
    scores = _oracle_scores(n_days, n_tickers, n_up)
    sweep = k_sweep(scores, k_values=[1, 2, 3, 5, 8, 10], seed=0, n_boot=0).sort_values("k")
    edges = sweep["mean_edge_pp"].to_numpy()
    assert all(edges[i] >= edges[i + 1] - 1e-9 for i in range(len(edges) - 1))
    # k <= n_up should all be identical (every pick is a true "up"); k > n_up strictly worse.
    assert edges[0] == pytest.approx(edges[2])  # k=1 and k=3 (both <= n_up=3)
    assert edges[3] < edges[2] - 1e-9  # k=5 > n_up: strictly worse


def test_return_check_matches_known_offset():
    n_days, n_tickers, n_picked = 10, 6, 2
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    offset = 0.002  # 20 bp on the picked tickers, deterministic (no noise)
    rows = []
    for d in dates:
        for i in range(n_tickers):
            r_next = offset if i < n_picked else 0.0
            rows.append({"date": d, "ticker": f"T{i}", "r_next": r_next, "y_true": "stagnant"})
    universe = pd.DataFrame(rows)
    picks = universe[universe["ticker"].isin([f"T{i}" for i in range(n_picked)])].copy()

    expected_universe_mean = (n_picked * offset) / n_tickers
    expected_bp = (offset - expected_universe_mean) * 10000

    rc = return_check(picks, universe, n_boot=0, seed=0)
    assert rc["mean_edge_bp"] == pytest.approx(expected_bp)


def test_concentration_same_tickers_every_day_reports_full_overlap_and_share():
    dates = pd.bdate_range("2021-01-01", periods=5)
    rows = [{"date": d, "ticker": t} for d in dates for t in ["T0", "T1", "T2"]]
    picks = pd.DataFrame(rows)
    stats = concentration_stats(picks, top_n=10)
    assert stats["top_n_share"] == pytest.approx(1.0)
    assert stats["mean_consecutive_day_overlap"] == pytest.approx(1.0)
    assert stats["n_distinct_tickers"] == 3


def test_concentration_disjoint_daily_picks_reports_zero_overlap():
    dates = pd.bdate_range("2021-01-01", periods=4)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "ticker": f"T{i}"})  # a different single ticker each day
    picks = pd.DataFrame(rows)
    stats = concentration_stats(picks, top_n=10)
    assert stats["mean_consecutive_day_overlap"] == pytest.approx(0.0)
    assert stats["n_distinct_tickers"] == 4
