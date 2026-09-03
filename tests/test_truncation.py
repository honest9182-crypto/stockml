"""The single most important test in the repo: rebuilding a ticker's features
from data truncated at day t must exactly reproduce the features computed
from the full series at day t. If it doesn't, a feature is reading future data.
"""

from __future__ import annotations

from stockml.evaluate import truncation_test
from stockml.features import build_features_panel
from tests.conftest import make_synthetic_panel


def test_truncated_rebuild_matches_full_build():
    panel = make_synthetic_panel(n_tickers=4, n_days=200)
    full_features = build_features_panel(panel, k=0.5, vol_window=20)

    result = truncation_test(panel, full_features, k=0.5, vol_window=20, n=30, seed=1)

    assert result["n_mismatch"] == 0, result["mismatches"]
    assert result["n_checked"] == 30
