"""Shifted-label null control: every ticker's exact class counts are
preserved by the circular shift, every offset is >= the configured minimum,
and the shuffled labels differ from the originals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.evolution.controls import MIN_SHUFFLE_OFFSET_DAYS, shift_labels_per_ticker


def _synthetic_labeled_dataset(n_tickers: int = 3, n_days: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    frames = []
    for i in range(n_tickers):
        labels = rng.choice(["down", "stagnant", "up"], size=n_days, p=[0.25, 0.5, 0.25])
        frames.append(pd.DataFrame({"ticker": f"T{i}", "date": dates, "label": labels}))
    return pd.concat(frames, ignore_index=True)


def test_class_counts_preserved_per_ticker():
    ds = _synthetic_labeled_dataset()
    rng = np.random.default_rng(1)
    shifted = shift_labels_per_ticker(ds, rng)
    for ticker, g in ds.groupby("ticker"):
        orig_counts = g["label"].value_counts().sort_index()
        shifted_counts = shifted.loc[g.index].value_counts().sort_index()
        pd.testing.assert_series_equal(orig_counts, shifted_counts, check_names=False)


def test_shifted_labels_differ_from_original():
    ds = _synthetic_labeled_dataset()
    rng = np.random.default_rng(2)
    shifted = shift_labels_per_ticker(ds, rng)
    # With a >=250-day offset on 600-day series and 3-way random labels,
    # it would be astronomically unlikely for the shift to reproduce the
    # original sequence exactly for every row.
    assert not (shifted.to_numpy() == ds["label"].to_numpy()).all()


def test_offset_is_at_least_the_minimum():
    ds = _synthetic_labeled_dataset(n_tickers=1, n_days=1000)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        shifted = shift_labels_per_ticker(ds, rng, min_offset=MIN_SHUFFLE_OFFSET_DAYS)
        g = ds.sort_values("date")
        labels = g["label"].to_numpy()
        shifted_labels = shifted.loc[g.index].to_numpy()
        # Find the actual offset by brute force and confirm it's >= min_offset
        # (and <= n - min_offset, i.e. not "trivially" a near-zero wrap the
        # other way either).
        n = len(labels)
        found = None
        for off in range(n):
            if (np.roll(labels, off) == shifted_labels).all():
                found = off
                break
        assert found is not None
        assert MIN_SHUFFLE_OFFSET_DAYS <= found <= n - MIN_SHUFFLE_OFFSET_DAYS


def test_too_short_ticker_raises():
    ds = _synthetic_labeled_dataset(n_tickers=1, n_days=100)  # < 2*250
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        shift_labels_per_ticker(ds, rng)


def test_deterministic_given_same_rng_state():
    ds = _synthetic_labeled_dataset()
    shifted_a = shift_labels_per_ticker(ds, np.random.default_rng(42))
    shifted_b = shift_labels_per_ticker(ds, np.random.default_rng(42))
    assert (shifted_a.to_numpy() == shifted_b.to_numpy()).all()
