"""Label correctness: hand-computed alignment and band/k behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.labels import build_labels, class_distribution, compute_labels_one_ticker
from tests.conftest import make_synthetic_panel


def test_label_matches_hand_computed_returns():
    panel = make_synthetic_panel(n_tickers=1, n_days=100)
    labeled = compute_labels_one_ticker(panel, k=0.5, vol_window=20)

    close = panel["close"].to_numpy()
    for t in range(len(panel) - 1):
        row = labeled.iloc[t]
        if pd.isna(row["label"]):
            continue
        r_next_hand = close[t + 1] / close[t] - 1
        assert np.isclose(r_next_hand, row["r_next"])
        band = row["band"]
        if r_next_hand > band:
            expected = "up"
        elif r_next_hand < -band:
            expected = "down"
        else:
            expected = "stagnant"
        assert row["label"] == expected


def test_last_row_per_ticker_has_no_label():
    panel = make_synthetic_panel(n_tickers=2, n_days=50)
    labeled = build_labels(panel, k=0.5, vol_window=20)
    last_dates = panel.groupby("ticker")["date"].max()
    for ticker, last_date in last_dates.items():
        assert not (
            (labeled["ticker"] == ticker) & (labeled["date"] == last_date)
        ).any(), "the last row of a ticker must never receive a label"


def test_band_widens_stagnant_share_as_k_increases():
    panel = make_synthetic_panel(n_tickers=3, n_days=400)
    small_k = build_labels(panel, k=0.1, vol_window=20)
    large_k = build_labels(panel, k=2.0, vol_window=20)

    small_dist = class_distribution(small_k["label"])
    large_dist = class_distribution(large_k["label"])

    assert large_dist["stagnant"] > small_dist["stagnant"], (
        "a larger k should widen the band and increase the stagnant share"
    )
    assert large_dist["up"] < small_dist["up"]
    assert large_dist["down"] < small_dist["down"]
